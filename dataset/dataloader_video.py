import os
import cv2
import sys
import pdb
import six
import glob
import time
import torch
import random
import pandas
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import numpy as np
# import pyarrow as pa
from PIL import Image
import torch.utils.data as data
import matplotlib.pyplot as plt
from utils import video_augmentation
from torch.utils.data.sampler import Sampler

sys.path.append("..")


class BaseFeeder(data.Dataset):
    def __init__(self, prefix, gloss_dict, drop_ratio=1, num_gloss=-1, mode="train", transform_mode=True,
                 datatype="lmdb"):
        self.mode = mode
        self.ng = num_gloss
        self.prefix = prefix
        self.frame_root = self._resolve_frame_root()
        self.dict = gloss_dict or dict()
        self.data_type = datatype
        self.transform_mode = "train" if transform_mode else "test"
        self.inputs_list = self._load_inputs_list(mode)
        # self.inputs_list = np.load(f"{prefix}/annotations/manual/{mode}.corpus.npy", allow_pickle=True).item()
        # self.inputs_list = np.load(f"{prefix}/annotations/manual/{mode}.corpus.npy", allow_pickle=True).item()
        # self.inputs_list = dict([*filter(lambda x: isinstance(x[0], str) or x[0] < 10, self.inputs_list.items())])
        print(mode, len(self))
        self.data_aug = self.transform()
        print("")

    def __getitem__(self, idx):
        if self.data_type == "video":
            input_data, label, fi = self.read_video(idx)
            input_data, label = self.normalize(input_data, label)
            # input_data, label = self.normalize(input_data, label, fi['fileid'])
            return input_data, torch.LongTensor(label), self.inputs_list[idx]['original_info']
        elif self.data_type == "lmdb":
            input_data, label, fi = self.read_lmdb(idx)
            input_data, label = self.normalize(input_data, label)
            return input_data, torch.LongTensor(label), self.inputs_list[idx]['original_info']
        else:
            input_data, label = self.read_features(idx)
            return input_data, label, self.inputs_list[idx]['original_info']

    def read_video(self, index, num_glosses=-1):
        # load file info
        fi = self.inputs_list[index]
        img_folder = os.path.join(self.frame_root, fi['folder'])
        img_list = sorted(glob.glob(img_folder))
        label_list = []
        for phase in fi['label'].split(" "):
            if phase == '':
                continue
            if self.dict and phase in self.dict:
                label_list.append(self.dict[phase][0])
        return [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in img_list], label_list, fi

    def read_features(self, index):
        # load file info
        fi = self.inputs_list[index]
        data = np.load(f"./features/{self.mode}/{fi['fileid']}_features.npy", allow_pickle=True).item()
        return data['features'], data['label']

    def normalize(self, video, label, file_id=None):
        video, label = self.data_aug(video, label, file_id)
        video = video.float() / 127.5 - 1
        return video, label

    def transform(self):
        if self.transform_mode == "train":
            print("Apply training transform.")
            return video_augmentation.Compose([
                # video_augmentation.CenterCrop(224),
                # video_augmentation.WERAugment('/lustre/wangtao/current_exp/exp/baseline/boundary.npy'),
                video_augmentation.RandomCrop(224),
                video_augmentation.RandomHorizontalFlip(0.5),
                video_augmentation.ToTensor(),
                video_augmentation.TemporalRescale(0.2),
                # video_augmentation.Resize(0.5),
            ])
        else:
            print("Apply testing transform.")
            return video_augmentation.Compose([
                video_augmentation.CenterCrop(224),
                # video_augmentation.Resize(0.5),
                video_augmentation.ToTensor(),
            ])

    def byte_to_img(self, byteflow):
        unpacked = pa.deserialize(byteflow)
        imgbuf = unpacked[0]
        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        img = Image.open(buf).convert('RGB')
        return img

    @staticmethod
    def collate_fn(batch):
        batch = [item for item in sorted(batch, key=lambda x: len(x[0]), reverse=True)]
        video, label, info = list(zip(*batch))
        if len(video[0].shape) > 3:
            max_len = len(video[0])
            video_length = torch.LongTensor([np.ceil(len(vid) / 4.0) * 4 + 12 for vid in video])
            left_pad = 6
            right_pad = int(np.ceil(max_len / 4.0)) * 4 - max_len + 6
            max_len = max_len + left_pad + right_pad
            padded_video = [torch.cat(
                (
                    vid[0][None].expand(left_pad, -1, -1, -1),
                    vid,
                    vid[-1][None].expand(max_len - len(vid) - left_pad, -1, -1, -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video)
        else:
            max_len = len(video[0])
            video_length = torch.LongTensor([len(vid) for vid in video])
            padded_video = [torch.cat(
                (
                    vid,
                    vid[-1][None].expand(max_len - len(vid), -1),
                )
                , dim=0)
                for vid in video]
            padded_video = torch.stack(padded_video).permute(0, 2, 1)
        label_length = torch.LongTensor([len(lab) for lab in label])
        if max(label_length) == 0:
            padded_label = torch.LongTensor([])
        else:
            padded_label = []
            for lab in label:
                padded_label.extend(lab)
            padded_label = torch.LongTensor(padded_label)
        return padded_video, video_length, padded_label, label_length, info

    def __len__(self):
        return len(self.inputs_list) - 1

    def _resolve_frame_root(self):
        candidates = [
            os.path.join(self.prefix, "features/fullFrame-256x256px"),
            os.path.join(self.prefix, "features/fullFrame-210x260px"),
        ]
        for cand in candidates:
            if os.path.exists(cand):
                print(f"Using frames from {cand}")
                return cand
        raise FileNotFoundError(
            f"Could not find resized frames under {candidates[0]} or {candidates[1]}. "
            f"Please verify your dataset path or run the preprocessing step."
        )

    def _load_inputs_list(self, mode):
        cache_dir = "./preprocess/phoenix2014"
        cache_path = os.path.join(cache_dir, f"{mode}_info.npy")
        if os.path.exists(cache_path):
            return np.load(cache_path, allow_pickle=True).item()
        info = self._build_inputs_list_from_annotations(mode)
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_path, info)
        return info

    def _annotation_csv_path(self, mode):
        csv_path = os.path.join(self.prefix, "annotations", "manual", f"{mode}.corpus.csv")
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Annotation file {csv_path} not found. Download the RWTH-PHOENIX annotations or "
                f"run the preprocessing script to generate metadata."
            )
        return csv_path

    def _build_inputs_list_from_annotations(self, mode):
        csv_path = self._annotation_csv_path(mode)
        info_dict = {'prefix': self.frame_root.replace("\\", "/")}
        skip_idx = 2390 if mode == "train" else None
        with open(csv_path, "r", encoding="utf-8") as f:
            header = f.readline()
            sample_idx = 0
            for line_idx, line in enumerate(f):
                record = line.strip()
                if not record:
                    continue
                if skip_idx is not None and line_idx == skip_idx:
                    continue
                parts = record.split("|")
                if len(parts) < 4:
                    continue
                fileid, folder, signer = parts[:3]
                label = "|".join(parts[3:]).strip()
                folder_rel = f"{mode}/{folder}"
                frame_glob = os.path.join(self.frame_root, folder_rel)
                num_frames = len(glob.glob(frame_glob))
                info_dict[sample_idx] = {
                    'fileid': fileid,
                    'folder': folder_rel,
                    'signer': signer,
                    'label': label,
                    'num_frames': num_frames,
                    'original_info': record,
                }
                sample_idx += 1
        if sample_idx == 0:
            raise ValueError(f"No samples parsed from {csv_path}.")
        print(f"Built {sample_idx} samples for {mode} directly from annotations.")
        return info_dict

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time


if __name__ == "__main__":
    feeder = BaseFeeder()
    dataloader = torch.utils.data.DataLoader(
        dataset=feeder,
        batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0,
    )
    for data in dataloader:
        pdb.set_trace()
