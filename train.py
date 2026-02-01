import os
import time
import datetime
import argparse
import wandb
import torch
import torch.distributed as dist
from pathlib import Path
from torch.distributed.algorithms.ddp_comm_hooks import default as comm_hooks
from torch.nn.parallel import DistributedDataParallel
from torch.amp import GradScaler
from tad.models import build_detector
from tad.datasets import build_dataset, build_dataloader
from tad.engine import train_one_epoch, eval_one_epoch, build_optimizer, build_scheduler
from tad.utils import set_seed,update_workdir,create_folder,save_config,setup_logger,ModelEma,save_checkpoint,Config,DictAction,get_custom_config,LRFinder,calculate_params_gflops


def parse_args():
    parser = argparse.ArgumentParser(description="Train a Temporal Action Detector")
    parser.add_argument("config", metavar="FILE", type=str, help="path to config file")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--id", type=int, default=0, help="repeat experiment id")
    parser.add_argument("--resume", type=str, default=None, help="resume from a checkpoint")
    parser.add_argument("--skip_eval", action="store_true", help="whether not to eval, only do inference")
    parser.add_argument("--disable_deterministic", action="store_true", help="disable deterministic for faster speed")
    parser.add_argument("--lr-range-test", action="store_true", help="run lr range test")
    parser.add_argument("--cfg-options", nargs="+", action=DictAction, help="override settings")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # load config
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # DDP init
    if "WORLD_SIZE" in os.environ:
        args.local_rank = int(os.environ["LOCAL_RANK"])
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.rank = int(os.environ["RANK"])
        print(f"Distributed init (rank {args.rank}/{args.world_size}, local rank {args.local_rank})")
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

    # set random seed, create work_dir, and save config
    set_seed(args.seed, args.disable_deterministic)
    cfg = update_workdir(cfg, args.id, args.world_size)
    if args.rank == 0:
        create_folder(cfg.work_dir)
        save_config(args.config, cfg.work_dir)

    # setup logger
    logger = setup_logger("Train", save_dir=cfg.work_dir, distributed_rank=args.rank)
    logger.info(f"Using torch version: {torch.__version__}, CUDA version: {torch.version.cuda}")
    if "common" in cfg.dataset:
        del cfg.dataset["common"]
    logger.info(f"Config: \n{cfg.pretty_text}")

    try:
        # build dataset
        train_dataset = build_dataset(cfg.dataset.train, default_args=dict(logger=logger))
        train_loader = build_dataloader(
            train_dataset,
            rank=args.rank,
            world_size=args.world_size,
            shuffle=True,
            drop_last=True,
            **cfg.dataloader.train,
        )
    
        val_dataset = build_dataset(cfg.dataset.val, default_args=dict(logger=logger))
        val_loader = build_dataloader(
            val_dataset,
            rank=args.rank,
            world_size=args.world_size,
            shuffle=False,
            drop_last=False,
            **cfg.dataloader.val,
        )
    
        # build model
        model = build_detector(cfg.model)
        params, gflops = calculate_params_gflops(model, cfg)
        logger.info(f"Params: {params / 1e6:.2f} M\n")
        # wandb: init project
        custom_config = get_custom_config(cfg, params=params, gflops=gflops)
        if args.rank == 0:
            wandb.init(
                project="tad",
                name=f"{datetime.datetime.now().strftime('%m%d_%H%M')}",
                config=custom_config,
                dir=cfg.work_dir,
                resume="allow"
            )

        # DDP
        use_static_graph = getattr(cfg.dataloader, "static_graph", False)
        model = model.to(args.local_rank)

        if args.distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[args.local_rank],
                output_device=args.local_rank,
                find_unused_parameters=False if use_static_graph else True,
                static_graph=use_static_graph,  # default is False, should be true when use activation checkpointing in E2E
            )
            logger.info(f"Using DDP with total {args.world_size} GPUS...")
        else:
            logger.info("Running on single GPU (No DDP)...")
    
        # FP16 compression
        use_fp16_compress = getattr(cfg.dataloader, "fp16_compress", False)
        if use_fp16_compress:
            if args.distributed:
                logger.info("Using FP16 compression ...")
                model.register_comm_hook(state=None, hook=comm_hooks.fp16_compress_hook)
            else:
                logger.warning("FP16 compression is ignored in non-distributed mode.")
    
        # Model EMA
        logger.info("Using Model EMA...")
        model_ema = ModelEma(model)
    
        # AMP: automatic mixed precision
        use_amp = getattr(cfg.dataloader, "amp", False)
        if use_amp:
            logger.info("Using Automatic Mixed Precision...")
            scaler = GradScaler()
        else:
            scaler = None
    
        # build optimizer and scheduler
        optimizer = build_optimizer(cfg.optimizer, model, logger)
        scheduler, max_epoch = build_scheduler(cfg.scheduler, optimizer, len(train_loader))
    
        # LR Range Test
        if args.lr_range_test:
            logger.info("Running LR Range Test...")
            lr_finder = LRFinder(model, optimizer, device=args.local_rank)
            lr_finder.range_test(train_loader)
            if args.rank == 0:
                save_path = Path(cfg.work_dir) / "lr_finder_curve.png"
                lr_finder.plot(save_path=str(save_path))
                logger.info(f"LR Range Test finished. Plot saved to {save_path}")
            return
    
        # override the max_epoch
        max_epoch = cfg.workflow.get("end_epoch", max_epoch)
    
        # resume: reset epoch, load checkpoint / best rmse
        val_map_best = 0.0
        if args.resume is not None:
            logger.info(f"Resume training from: {args.resume}")
            device = f"cuda:{args.local_rank}"
            checkpoint = torch.load(args.resume, map_location=device)
            resume_epoch = checkpoint["epoch"]
            if "val_map_best" in checkpoint:
                val_map_best = checkpoint["val_map_best"]
            logger.info(f"Resume epoch is {resume_epoch}, best mAP is {val_map_best}")
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["scheduler"])
            if model_ema is not None:
                model_ema.module.load_state_dict(checkpoint["state_dict_ema"])
    
            del checkpoint  #  save memory if the model is very large such as ViT-g
            torch.cuda.empty_cache()
        else:
            resume_epoch = -1
    
        # train the detector
        logger.info("Training Starts...\n")
        start_time = time.time()
        val_start_epoch = cfg.workflow.get("val_start_epoch", 0)
        
        for epoch in range(resume_epoch + 1, max_epoch):
            log_dict = {}
            train_loader.sampler.set_epoch(epoch)
    
            # train for one epoch (returns last_global_step for exact alignment)
            global_step = train_one_epoch(
                train_loader,
                model,
                optimizer,
                scheduler,
                epoch,
                logger,
                rank=args.rank,
                model_ema=model_ema,
                clip_grad_l2norm=cfg.dataloader.clip_grad_norm,
                logging_interval=cfg.workflow.logging_interval,
                scaler=scaler,
            )
    
            # save checkpoint
            if (epoch == max_epoch - 1) or ((epoch + 1) % cfg.workflow.checkpoint_interval == 0):
                if args.rank == 0:
                    save_checkpoint(
                        model,
                        model_ema,
                        optimizer,
                        scheduler,
                        epoch,
                        work_dir=cfg.work_dir,
                        mode='last.pt',
                        val_map_best=val_map_best
                    )
    
            # val_eval for one epoch
            if epoch >= val_start_epoch:
                if (cfg.workflow.val_eval_interval > 0) and ((epoch + 1) % cfg.workflow.val_eval_interval == 0):
                    val_map = eval_one_epoch(
                        val_loader,
                        model,
                        cfg,
                        logger,
                        args.rank,
                        model_ema=model_ema,
                        use_amp=use_amp,
                        world_size=args.world_size,
                        skip_eval=args.skip_eval,
                    )
                    log_dict["val_mAP (%)"] = round(val_map * 100, 2)
                    # save the best checkpoint
                    if val_map > val_map_best:
                        logger.info(f"New best mAP {val_map*100:.2f} % at epoch {epoch}")
                        val_map_best = val_map
                        if args.rank == 0:
                            save_checkpoint(
                                model,
                                model_ema,
                                optimizer,
                                scheduler,
                                epoch,
                                work_dir=cfg.work_dir,
                                mode='best.pt',
                                val_map_best=val_map_best
                            )
            log_dict["epoch"] = epoch
            if args.rank == 0:               
                wandb.log(log_dict, step=global_step)
    
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logger.info(f"Training Over, total time: {total_time_str}\n")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    finally:
        if args.rank == 0:
            wandb.finish()
        # Safe check for distributed attribute
        if hasattr(args, "distributed") and args.distributed:
            if dist.is_initialized():
                dist.destroy_process_group()
    
    
if __name__ == "__main__":
    main()