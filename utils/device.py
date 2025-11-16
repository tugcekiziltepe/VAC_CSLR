import os
import pdb
import torch
import torch.nn as nn


class GpuDataParallel(object):
    def __init__(self):
        self.gpu_list = []
        self.output_device = "cpu"

    def set_device(self, device):
        device = 'None' if device is None else str(device)
        self.gpu_list = []
        self.output_device = "cpu"

        if device.lower() in ('none', 'cpu'):
            return

        requested = [d.strip() for d in device.split(',') if d.strip() != '']
        if not requested:
            return

        try:
            cuda_available = torch.cuda.is_available()
        except RuntimeError as err:
            print(f"CUDA initialization failed ({err}). Falling back to CPU.")
            return
        if not cuda_available:
            print("CUDA is not available. Falling back to CPU.")
            return

        available_gpu = torch.cuda.device_count()
        valid_gpu = []
        for d in requested:
            try:
                idx = int(d)
            except ValueError:
                print(f"Invalid GPU id '{d}', ignoring.")
                continue
            if idx >= available_gpu:
                print(f"Requested GPU {idx} but only {available_gpu} device(s) are available. Ignoring.")
                continue
            valid_gpu.append(d)
        if not valid_gpu:
            print("No valid GPU ids found. Falling back to CPU.")
            return

        prev_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        new_visible = ",".join(valid_gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = new_visible

        def restore_visible():
            if prev_visible is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = prev_visible

        self.gpu_list = [i for i in range(len(valid_gpu))]
        output_device = self.gpu_list[0]
        try:
            self.occupy_gpu(self.gpu_list)
        except RuntimeError as err:
            print(f"Unable to initialize requested GPU(s): {err}. Falling back to CPU.")
            self.gpu_list = []
            self.output_device = "cpu"
            restore_visible()
            return
        self.output_device = output_device

    def model_to_device(self, model):
        # model = convert_model(model)
        model = model.to(self.output_device)
        if len(self.gpu_list) > 1:
            model = nn.DataParallel(
                model,
                device_ids=self.gpu_list,
                output_device=self.output_device)
        return model

    def data_to_device(self, data):
        if isinstance(data, torch.FloatTensor):
            return data.to(self.output_device)
        elif isinstance(data, torch.DoubleTensor):
            return data.float().to(self.output_device)
        elif isinstance(data, torch.ByteTensor):
            return data.long().to(self.output_device)
        elif isinstance(data, torch.LongTensor):
            return data.to(self.output_device)
        elif isinstance(data, list) or isinstance(data, tuple):
            return [self.data_to_device(d) for d in data]
        else:
            raise ValueError(data.shape, "Unknown Dtype: {}".format(data.dtype))

    def criterion_to_device(self, loss):
        return loss.to(self.output_device)

    def occupy_gpu(self, gpus=None):
        """
            make program appear on nvidia-smi.
        """
        if not torch.cuda.is_available():
            return
        gpus = [] if gpus is None else gpus
        if len(gpus) == 0:
            torch.zeros(1).cuda()
        else:
            gpus = [gpus] if isinstance(gpus, int) else list(gpus)
            for g in gpus:
                torch.zeros(1).cuda(g)
