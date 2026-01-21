from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR


def build_scheduler(cfg, optimizer, dataloader_len):
    max_epoch = cfg["max_epoch"]

    # Convert epochs to iterations
    total_iters = int(max_epoch * dataloader_len)
    warmup_iters = int(cfg.get("warmup_epoch", 0) * dataloader_len)

    # Calculate start factor for LinearLR
    start_factor = cfg.get("start_factor", 0.001)

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
        eta_min=cfg.get("eta_min", 1.0e-8),
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