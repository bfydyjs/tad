# --------------------------------------------------------
# This file is modified from OpenTAD
# (https://github.com/sming256/OpenTAD)
# Copyright (c) OpenTAD Authors. All rights reserved.
# --------------------------------------------------------

import argparse
import datetime
import os
import shutil
import time
from pathlib import Path

import torch
import torch.distributed as dist
import wandb
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
from torch.nn.parallel import DistributedDataParallel

from tad.datasets import build_dataloader, build_dataset
from tad.engine import (
    build_optimizer,
    build_scheduler,
    inference_and_eval_one_epoch,
    train_one_epoch,
)
from tad.models import build_detector
from tad.utils import (
    Config,
    DictAction,
    ModelEma,
    calculate_params_gflops,
    get_custom_config,
    save_checkpoint,
    set_seed,
    setup_logger,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--resume", type=str, default=None, help="resume from a checkpoint")
    parser.add_argument(
        "--skip_eval", action="store_true", help="whether not to eval, only do inference"
    )
    parser.add_argument(
        "--disable_deterministic",
        action="store_true",
        help="disable deterministic for faster speed",
    )
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    return parser.parse_args()


def init_distributed(args):
    """Initialize distributed training environment."""
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
        print("Non-distributed init: Running on single GPU")
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
        args.distributed = False


def setup_env(cfg, args):
    """Setup random seed, work directory and logger."""
    set_seed(args.seed, args.disable_deterministic)

    if args.rank == 0:
        Path(cfg.work_dir).expanduser().mkdir(mode=0o777, parents=True, exist_ok=True)
        shutil.copy2(args.config, cfg.work_dir)

    # setup logger
    logger = setup_logger("Train", save_dir=cfg.work_dir, distributed_rank=args.rank)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    if "common" in cfg.dataset:
        del cfg.dataset["common"]
    logger.info(f"Config: \n{cfg.pretty_text}")
    return logger


class TADTrainer:
    def __init__(self, args, cfg, logger):
        self.args = args
        self.cfg = cfg
        self.logger = logger
        self.val_map_best = 0.0
        self.resume_epoch = -1
        self.use_amp = getattr(cfg.dataloader, "amp", False)

        self.train_loader, self.val_loader = self._build_loaders()
        self.model, self.model_ema, self.optimizer, self.scheduler, self.scaler, self.max_epoch = (
            self._build_model_and_optimizer()
        )

        self.max_epoch = self.cfg.workflow.get("end_epoch", self.max_epoch)
        self._resume_training()

    def _build_loaders(self):
        """Build train and validation dataloaders."""
        train_dataset = build_dataset(self.cfg.dataset.train, default_args=dict(logger=self.logger))
        train_loader = build_dataloader(
            train_dataset,
            rank=self.args.rank,
            world_size=self.args.world_size,
            shuffle=True,
            drop_last=True,
            **self.cfg.dataloader.train,
        )

        val_dataset = build_dataset(self.cfg.dataset.val, default_args=dict(logger=self.logger))
        val_loader = build_dataloader(
            val_dataset,
            rank=self.args.rank,
            world_size=self.args.world_size,
            shuffle=False,
            drop_last=False,
            **self.cfg.dataloader.val,
        )
        return train_loader, val_loader

    def _build_model_and_optimizer(self):
        """Build model, optimizer, scheduler, and configure DDP/AMP/EMA/WandB."""
        # build model
        model = build_detector(self.cfg.model)
        params, gflops = calculate_params_gflops(model, self.cfg)
        self.logger.info(f"Params: {params / 1e6:.2f} M\n")

        # wandb: init project
        custom_config = get_custom_config(self.cfg, params=params, gflops=gflops)
        if self.args.rank == 0:
            wandb.init(
                project="tad",
                name=f"{datetime.datetime.now().strftime('%m%d_%H%M')}",
                config=custom_config,
                dir=self.cfg.work_dir,
                resume="allow",
            )

        # DDP
        use_static_graph = getattr(self.cfg.dataloader, "static_graph", False)
        model = model.to(self.args.local_rank)

        if self.args.distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[self.args.local_rank],
                output_device=self.args.local_rank,
                find_unused_parameters=False if use_static_graph else True,
                static_graph=use_static_graph,
            )
            self.logger.info(f"Using DDP with total {self.args.world_size} GPUS...")
        else:
            self.logger.info("Running on single GPU (No DDP)...")

        # FP16 compression
        use_fp16_compress = getattr(self.cfg.dataloader, "fp16_compress", False)
        if use_fp16_compress:
            if self.args.distributed:
                self.logger.info("Using FP16 compression ...")
                model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
            else:
                self.logger.warning("FP16 compression is ignored in non-distributed mode.")

        # Model EMA
        self.logger.info("Using Model EMA...")
        model_ema = ModelEma(model)

        # AMP
        if self.use_amp:
            self.logger.info("Using Automatic Mixed Precision...")
        else:
            scaler = None

        # build optimizer and scheduler
        optimizer = build_optimizer(self.cfg.optimizer, model, self.logger)
        scheduler, max_epoch = build_scheduler(
            self.cfg.scheduler, optimizer, len(self.train_loader)
        )

        return model, model_ema, optimizer, scheduler, scaler, max_epoch

    def _resume_training(self):
        """Resume training from checkpoint if provided."""
        if self.args.resume is not None:
            self.logger.info(f"Resume training from: {self.args.resume}")
            device = f"cuda:{self.args.local_rank}"
            checkpoint = torch.load(self.args.resume, map_location=device)
            self.resume_epoch = checkpoint["epoch"]
            if "val_map_best" in checkpoint:
                self.val_map_best = checkpoint["val_map_best"]
            self.logger.info(
                f"Resume epoch is {self.resume_epoch}, best mAP is {self.val_map_best}"
            )
            self.model.load_state_dict(checkpoint["state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
            self.scheduler.load_state_dict(checkpoint["scheduler"])
            if hasattr(self.model_ema, "module"):
                self.model_ema.module.load_state_dict(checkpoint["state_dict_ema"])

            del checkpoint
            torch.cuda.empty_cache()

    def run(self):
        """Run the main training loop."""
        self.logger.info("Training Starts...\n")
        start_time = time.time()
        val_start_epoch = self.cfg.workflow.get("val_start_epoch", 0)

        for epoch in range(self.resume_epoch + 1, self.max_epoch):
            log_dict = {}
            if hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(epoch)

            # train for one epoch
            global_step = train_one_epoch(
                self.train_loader,
                self.model,
                self.optimizer,
                self.scheduler,
                epoch,
                self.logger,
                rank=self.args.rank,
                model_ema=self.model_ema,
                clip_grad_l2norm=self.cfg.dataloader.clip_grad_norm,
                logging_interval=self.cfg.workflow.logging_interval,
                scaler=self.scaler,
            )

            # save checkpoint
            if (epoch == self.max_epoch - 1) or (
                (epoch + 1) % self.cfg.workflow.checkpoint_interval == 0
            ):
                if self.args.rank == 0:
                    save_checkpoint(
                        self.model,
                        self.model_ema,
                        self.optimizer,
                        self.scheduler,
                        epoch,
                        work_dir=self.cfg.work_dir,
                        mode="last.pt",
                        val_map_best=self.val_map_best,
                    )

            # val_eval
            if epoch >= val_start_epoch:
                if (self.cfg.workflow.val_eval_interval > 0) and (
                    (epoch + 1) % self.cfg.workflow.val_eval_interval == 0
                ):
                    val_map = inference_and_eval_one_epoch(
                        self.val_loader,
                        self.model,
                        self.cfg,
                        self.logger,
                        self.args.rank,
                        model_ema=self.model_ema,
                        use_amp=self.use_amp,
                        world_size=self.args.world_size,
                        skip_eval=self.args.skip_eval,
                    )
                    log_dict["val/mAP (%)"] = round(val_map * 100, 2)
                    # save the best checkpoint
                    if val_map > self.val_map_best:
                        self.logger.info(f"New best mAP {val_map * 100:.2f} % at epoch {epoch}")
                        self.val_map_best = val_map
                        if self.args.rank == 0:
                            save_checkpoint(
                                self.model,
                                self.model_ema,
                                self.optimizer,
                                self.scheduler,
                                epoch,
                                work_dir=self.cfg.work_dir,
                                mode="best.pt",
                                val_map_best=self.val_map_best,
                            )
            log_dict["epoch"] = epoch
            if self.args.rank == 0:
                wandb.log(log_dict, step=global_step)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        self.logger.info(f"Training Over, total time: {total_time_str}\n")


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # Initialize environment
    init_distributed(args)
    cfg.work_dir = Path(cfg.work_dir) / f"gpu{args.world_size}_id{args.id}"
    logger = setup_env(cfg, args)

    try:
        trainer = TADTrainer(args, cfg, logger)
        trainer.run()

    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    finally:
        if getattr(args, "rank", -1) == 0:
            if wandb.run is not None:
                wandb.finish()
        if getattr(args, "distributed", False) and dist.is_initialized():
            dist.destroy_process_group()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
