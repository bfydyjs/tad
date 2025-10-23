import os
import torch

def save_checkpoint(model, model_ema, optimizer, scheduler, epoch, work_dir=None, mode=None, **kwargs):
    save_dir = os.path.join(work_dir, "checkpoint")
    save_states = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }
    save_states.update(kwargs)

    if model_ema is not None:
        save_states.update({"state_dict_ema": model_ema.module.state_dict()})

    os.makedirs(save_dir, exist_ok=True)

    checkpoint_path = os.path.join(save_dir, mode)# mode: 'last' or 'best'
    temp_path = checkpoint_path + ".tmp"
    torch.save(save_states, temp_path)
    os.replace(temp_path, checkpoint_path)