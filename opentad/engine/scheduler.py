import torch
from torch.optim.lr_scheduler import (
    LinearLR,
    CosineAnnealingLR,
    SequentialLR,
)


def build_scheduler(cfg, optimizer, dataloader_len):
    cfg.pop("type", None)
    max_epoch = cfg["max_epoch"]

    # Convert epochs to iterations
    total_iters = int(max_epoch * dataloader_len)
    warmup_iters = int(cfg.get("warmup_epoch", 0) * dataloader_len)

    # Calculate start factor for LinearLR
    # LinearLR scales the initial lr by start_factor.
    # We want the first step to be warmup_start_lr.
    base_lr = optimizer.param_groups[0]["lr"]
    warmup_start_lr = cfg.get("warmup_start_lr", 0.0)
    start_factor = warmup_start_lr / base_lr if base_lr > 0 else 0.0
    
    # LinearLR requires start_factor > 0
    if start_factor == 0:
        start_factor = 1e-9

    # Create Warmup Scheduler
    if warmup_iters > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=start_factor,
            end_factor=1.0,
            total_iters=warmup_iters,
        )
    else:
        warmup_scheduler = None

    # Create Main Scheduler
    main_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_iters - warmup_iters,
        eta_min=cfg.get("eta_min", 1e-8),
    )

    # Combine Schedulers
    if warmup_scheduler:
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_iters],
        )
    else:
        scheduler = main_scheduler

    return scheduler, max_epoch
