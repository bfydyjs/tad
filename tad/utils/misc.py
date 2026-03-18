import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist


def set_seed(seed, disable_deterministic=False):
    """Set randon seed for pytorch and numpy"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if disable_deterministic:
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)


def update_workdir(cfg, exp_id, gpu_num):
    cfg.work_dir = Path(cfg.work_dir) / f"gpu{gpu_num}_id{exp_id}"
    return cfg


def create_folder(folder_path):
    path = Path(folder_path).expanduser()
    path.mkdir(mode=0o777, parents=True, exist_ok=True)


def save_config(cfg, folder_path):
    shutil.copy2(cfg, folder_path)


def reduce_loss(loss_dict):
    # reduce loss when distributed training, only for logging
    if not (dist.is_available() and dist.is_initialized()):
        return loss_dict

    for loss_name, loss_value in loss_dict.items():
        loss_value = loss_value.detach().clone()
        dist.all_reduce(loss_value.div_(dist.get_world_size()))
        loss_dict[loss_name] = loss_value
    return loss_dict


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
