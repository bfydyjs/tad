import argparse
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from tad.datasets import build_dataloader, build_dataset
from tad.engine import inference_and_eval_one_epoch
from tad.models import build_detector
from tad.utils import Config, DictAction, set_seed, setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Test a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--checkpoint", type=str, default="none", help="the checkpoint path")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument(
        "--skip-eval", action="store_true", help="whether to not to eval, only do inference"
    )
    parser.add_argument(
        "--disable-deterministic",
        action="store_true",
        help="disable deterministic for faster speed",
    )
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    return parser.parse_args()


def init_distributed(args):
    """Initialize distributed testing environment."""
    if "WORLD_SIZE" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.rank = int(os.environ["RANK"])
        print(
            f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})"
        )
        dist.init_process_group("nccl", rank=args.rank, world_size=args.world_size)
        torch.cuda.set_device(args.local_rank)
        args.distributed = True
    else:
        args.local_rank = 0
        args.world_size = 1
        args.rank = 0
        if torch.cuda.is_available():
            print("Non-distributed init: Running on single GPU")
            torch.cuda.set_device(args.local_rank)
        else:
            print("Non-distributed init: Running on CPU")
        args.distributed = False


def setup_env(cfg, args):
    """Setup random seed, work directory and logger."""
    set_seed(args.seed, args.disable_deterministic)
    cfg.work_dir = Path(cfg.work_dir) / f"gpu{args.world_size}_id{args.id}"
    if args.rank == 0:
        Path(cfg.work_dir).expanduser().mkdir(mode=0o777, parents=True, exist_ok=True)

    # setup logger
    logger = setup_logger("Test", save_dir=cfg.work_dir, distributed_rank=args.rank)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    logger.info(f"Config: \n{cfg.pretty_text}")
    return cfg, logger


def build_test_loader(cfg, args, logger):
    """Build test dataset and loader."""
    test_dataset = build_dataset(cfg.dataset.test, default_args=dict(logger=logger))
    test_loader = build_dataloader(
        test_dataset,
        rank=args.rank,
        world_size=args.world_size,
        shuffle=False,
        drop_last=False,
        **cfg.dataloader.test,
    )
    return test_loader


def build_and_wrap_model(cfg, args, logger):
    """Build model and wrap with DDP."""
    # build model
    model = build_detector(cfg.model)

    # DDP
    if torch.cuda.is_available():
        model = model.to(args.local_rank)
        if args.distributed:
            model = DistributedDataParallel(
                model, device_ids=[args.local_rank], output_device=args.local_rank
            )
            logger.info(f"Using DDP with total {args.world_size} GPUS...")
        else:
            logger.info("Running on single GPU (No DDP)...")
    else:
        logger.info("Running on CPU...")
    return model


def load_weights(cfg, args, logger, model):
    """Load model weights from checkpoint or skip if using raw predictions."""
    if (
        cfg.inference.load_from_raw_predictions
    ):  # if load with saved predictions, no need to load checkpoint
        logger.info(f"Loading from raw predictions: {cfg.inference.fuse_list}")
        return

    # load checkpoint: args -> config -> best
    work_dir = Path(cfg.work_dir)
    if args.checkpoint != "none":
        checkpoint_path = args.checkpoint
    elif "test_epoch" in cfg.inference.keys():
        checkpoint_path = work_dir / "checkpoint" / f"epoch_{cfg.inference.test_epoch}.pt"
    else:
        checkpoint_path = work_dir / "checkpoint" / "best.pt"

    logger.info(f"Loading checkpoint from: {checkpoint_path}")
    if torch.cuda.is_available():
        device = f"cuda:{args.rank % torch.cuda.device_count()}"
    else:
        device = "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    logger.info(f"Checkpoint is epoch {checkpoint['epoch']}.")

    # Model EMA
    # If model is wrapped by DistributedDataParallel, load into the underlying
    # module to avoid mismatched `module.` prefixes in state_dict keys.
    target_model = model.module if isinstance(model, DistributedDataParallel) else model
    use_ema = getattr(cfg.dataloader, "ema", True)
    if use_ema:
        target_model.load_state_dict(checkpoint["state_dict_ema"])
        logger.info("Using Model EMA...")
    else:
        target_model.load_state_dict(checkpoint["state_dict"])


def run_evaluation(cfg, args, logger, test_loader, model):
    """Run evaluation for one epoch."""
    # AMP: automatic mixed precision
    use_amp = getattr(cfg.dataloader, "amp", False)
    if use_amp:
        logger.info("Using Automatic Mixed Precision...")

    # test the detector
    logger.info("Testing Starts...\n")
    inference_and_eval_one_epoch(
        test_loader,
        model,
        cfg,
        logger,
        args.rank,
        model_ema=None,  # since we have loaded the ema model above
        use_amp=use_amp,
        world_size=args.world_size,
        skip_eval=args.skip_eval,
    )
    logger.info("Testing Over...\n")


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Initialize environment
    init_distributed(args)
    cfg, logger = setup_env(cfg, args)

    # Build and setup model using helper functions
    test_loader = build_test_loader(cfg, args, logger)
    model = build_and_wrap_model(cfg, args, logger)

    # Load Weights
    load_weights(cfg, args, logger, model)

    # Run Eval
    run_evaluation(cfg, args, logger, test_loader, model)

    if args.distributed:
        if dist.is_initialized():
            dist.destroy_process_group()
            logger.info("Destroyed process group.")


if __name__ == "__main__":
    main()
