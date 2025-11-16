import glob
import os
from collections import OrderedDict

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from modules.sync_batchnorm import convert_model
from utils import video_augmentation


# ---------------------------------------------------------------------------
# User settings – edit these paths to match your environment, no CLI args.
# ---------------------------------------------------------------------------
SETTINGS = {
    "config_path": "./configs/baseline.yaml",
    "weights_path": r"C:\Users\Tugce\Documents\github\VAC_CSLR\resnet18_vac_smkd_dev_19.80_epoch35_model.pt",
    "data_root": r"C:\Users\Tugce\Documents\Datasets\PHOENIX-2014-T\features\fullFrame-210x260px\train",
    "output_dir": "./features/custom_train",
    "device": "cuda",
    "mode": "test",  # "train" applies random crop/flip/rescale like training loader
}


def resolve_device(device_str):
    device_str = device_str or "cpu"
    if device_str.lower().startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        return torch.device("cpu")
    try:
        return torch.device(device_str)
    except RuntimeError:
        print(f"Unknown device '{device_str}', falling back to CPU.")
        return torch.device("cpu")


def import_class(name):
    components = name.rsplit('.', 1)
    mod = __import__(components[0], fromlist=[components[1]])
    return getattr(mod, components[1])


def load_config(config_path):
    with open(config_path, "r") as f:
        try:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
        except AttributeError:
            cfg = yaml.load(f)
    return cfg


def build_model(cfg, weights_path, device):
    model_args = cfg.get("model_args", {}).copy()
    weight_norm = model_args.get("weight_norm", True)
    share_classifier = model_args.get("share_classifier", True)
    loss_weights = cfg.get("loss_weights", {"SeqCTC": 1.0})
    model_class = import_class(cfg["model"])
    gloss_dict = None
    if "num_classes" not in model_args:
        checkpoint = torch.load(weights_path, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        num_classes = infer_num_classes(state_dict)
        model_args["num_classes"] = num_classes
    model = model_class(gloss_dict=gloss_dict, loss_weights=loss_weights, **model_args)
    state = torch.load(weights_path, map_location=device)
    weights = state.get("model_state_dict", state)
    weights = OrderedDict([(k.replace(".module", ""), v) for k, v in weights.items()])
    if not weight_norm:
        if "classifier.bias" not in weights and hasattr(model.classifier, "bias") and model.classifier.bias is not None:
            model.classifier.bias.data.zero_()
        if "conv1d.fc.bias" not in weights and hasattr(model.conv1d.fc, "bias") and model.conv1d.fc.bias is not None:
            model.conv1d.fc.bias.data.zero_()
    missing, unexpected = model.load_state_dict(weights, strict=False)
    if missing:
        print(f"Missing parameters from checkpoint: {missing}")
    if unexpected:
        print(f"Unexpected parameters in checkpoint: {unexpected}")
    model = convert_model(model)
    model = model.to(device)
    model.eval()
    return model


def infer_num_classes(state_dict):
    for key in ["classifier.weight", "conv1d.fc.weight"]:
        if key in state_dict:
            return state_dict[key].shape[0]
    raise ValueError("Unable to infer number of classes from checkpoint.")


def build_transforms(mode):
    if mode == "train":
        transforms = video_augmentation.Compose([
            video_augmentation.RandomCrop(224),
            video_augmentation.RandomHorizontalFlip(0.5),
            video_augmentation.ToTensor(),
            video_augmentation.TemporalRescale(0.2),
        ])
    else:
        transforms = video_augmentation.Compose([
            video_augmentation.CenterCrop(224),
            video_augmentation.ToTensor(),
        ])
    return transforms


def load_sequence_images(seq_dir):
    frame_paths = sorted(glob.glob(os.path.join(seq_dir, "*.png")))
    if not frame_paths:
        frame_paths = sorted(glob.glob(os.path.join(seq_dir, "*.jpg")))
    frames = []
    for img_path in frame_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frames.append(img)
    return frames, frame_paths


def normalize_video(video_tensor):
    return video_tensor.float() / 127.5 - 1


def pad_video(video_tensor):
    original_len = video_tensor.shape[0]
    if original_len == 0:
        raise ValueError("Encountered empty video sequence.")
    left_pad = 6
    right_pad = int(np.ceil(original_len / 4.0)) * 4 - original_len + 6
    total_len = original_len + left_pad + right_pad
    left = video_tensor[0:1].expand(left_pad, -1, -1, -1)
    if right_pad > 0:
        right = video_tensor[-1:].expand(right_pad, -1, -1, -1)
        padded = torch.cat((left, video_tensor, right), dim=0)
    else:
        padded = torch.cat((left, video_tensor), dim=0)
    video_length = int(np.ceil(original_len / 4.0) * 4 + 12)
    return padded, video_length


def process_sequence(seq_dir, model, device, transforms, output_dir):
    frames, frame_paths = load_sequence_images(seq_dir)
    if not frames:
        print(f"Skipped {seq_dir}: no frames found.")
        return
    video_tensor, _ = transforms(frames, [])
    video_tensor = normalize_video(video_tensor)
    padded_video, vid_len = pad_video(video_tensor)
    batch_video = padded_video.unsqueeze(0).to(device)
    vid_len_tensor = torch.LongTensor([vid_len]).to(device)
    with torch.no_grad():
        ret_dict = model(batch_video, vid_len_tensor)
    features = ret_dict["framewise_features"][0][:, :vid_len].T.cpu().numpy()
    save_dict = {
        "label": np.array([], dtype=np.int64),
        "features": features,
    }
    os.makedirs(output_dir, exist_ok=True)
    sample_name = os.path.basename(seq_dir.rstrip("/\\"))
    np.save(os.path.join(output_dir, f"{sample_name}_features.npy"), save_dict)


def main():
    cfg = load_config(SETTINGS["config_path"])
    device = resolve_device(SETTINGS["device"])
    model = build_model(cfg, SETTINGS["weights_path"], device)
    transforms = build_transforms(SETTINGS["mode"])
    sequence_dirs = sorted([p for p in glob.glob(os.path.join(SETTINGS["data_root"], "*")) if os.path.isdir(p)])
    if not sequence_dirs:
        raise ValueError(f"No sequence folders found under {SETTINGS['data_root']}")
    for seq_dir in tqdm(sequence_dirs, desc="Extracting features"):
        process_sequence(seq_dir, model, device, transforms, SETTINGS["output_dir"])


if __name__ == "__main__":
    main()
