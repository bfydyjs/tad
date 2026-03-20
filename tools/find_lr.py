import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from tad.datasets import build_dataloader, build_dataset
from tad.engine import build_optimizer
from tad.models import build_detector
from tad.utils import Config, DictAction, LRFinder, set_seed, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Find optimal learning rate using LR Range Test")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument(
        "--disable_deterministic",
        action="store_true",
        help="disable deterministic for faster speed",
    )
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    return parser.parse_args()


def init_distributed(args):
    if "WORLD_SIZE" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.rank = int(os.environ["RANK"])
        dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
        torch.cuda.set_device(args.local_rank)
        args.distributed = True
    else:
        args.local_rank = 0
        args.world_size = 1
        args.rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
        args.distributed = False


def setup_env(cfg, args):
    set_seed(args.seed, args.disable_deterministic)
    if args.rank == 0:
        Path(cfg.work_dir).expanduser().mkdir(mode=0o777, parents=True, exist_ok=True)
    logger = setup_logger("FindLR", save_dir=cfg.work_dir, distributed_rank=args.rank)
    if "common" in cfg.dataset:
        del cfg.dataset["common"]
    return logger


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    init_distributed(args)
    # Give find_lr its own sub-directory inside work_dir to prevent conflicts with training outputs
    cfg.work_dir = Path(cfg.work_dir) / "find_lr"
    logger = setup_env(cfg, args)

    try:
        # Build Dataset & Dataloader
        train_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=logger))
        train_loader = build_dataloader(
            train_dataset,
            rank=args.rank,
            world_size=args.world_size,
            shuffle=True,
            drop_last=True,
            **cfg.dataloader.train,
        )

        # Build Model
        model = build_detector(cfg.model)
        model = model.to(args.local_rank)

        use_static_graph = getattr(cfg.dataloader, "static_graph", False)
        if args.distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=False if use_static_graph else True,
                static_graph=use_static_graph,
            )

        # Build Optimizer
        optimizer = build_optimizer(cfg.optimizer, model, logger)

        # Run LR Finder
        logger.info("Running LR Range Test...")
        lr_finder = LRFinder(model, optimizer, device=args.local_rank)
        lr_finder.range_test(train_loader)

        if args.rank == 0:
            save_path = Path(cfg.work_dir) / "lr_finder_curve.png"
            lr_finder.plot(save_path=str(save_path))
            logger.info(f"LR Range Test finished. Plot saved to {save_path}")

    finally:
        if getattr(args, "distributed", False) and dist.is_initialized():
            dist.destroy_process_group()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
