import os
from collections import OrderedDict
from pathlib import Path

import torch


def load_checkpoint(model, filename, map_location="cpu", strict=False, logger=None):
    """Load checkpoint from a file or url and load it into the model."""
    if filename.startswith("http://") or filename.startswith("https://"):
        checkpoint = torch.hub.load_state_dict_from_url(
            filename, map_location=map_location, check_hash=True
        )
    else:
        if not os.path.isfile(filename):
            raise OSError(f"Checkpoint file not found: {filename}")
        checkpoint = torch.load(filename, map_location=map_location, weights_only=False)

    # Get state_dict from checkpoint
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Strip prefix of state_dict if it comes from DDP wrapper but model is not
    if state_dict and next(iter(state_dict)).startswith("module."):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    # Load state_dict
    if hasattr(model, "module"):
        model.module.load_state_dict(state_dict, strict=strict)
    else:
        model.load_state_dict(state_dict, strict=strict)

    if logger is not None:
        logger.info(f"Successfully loaded checkpoint from {filename}")

    return checkpoint


def save_checkpoint(save_states, work_dir, mode):

    save_dir = Path(work_dir) / "checkpoint"
    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = Path(save_dir) / f"{mode}.pt"
    temp_path = Path(str(checkpoint_path) + ".tmp")

    # 原子写入
    torch.save(save_states, temp_path)
    os.replace(temp_path, checkpoint_path)
