import json
import time
from pathlib import Path

import torch
import torch.distributed as dist
import tqdm
import wandb

from tad.datasets.dataset import SlidingWindowDataset
from tad.metrics import build_evaluator
from tad.models.post_processing import batched_nms, build_classifier
from tad.utils.meters import AverageMeter


def move_to_device(data_dict, device):
    """Move data dictionary to device."""
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            data_dict[k] = v.to(device, non_blocking=True)
        elif isinstance(v, list) and len(v) > 0 and isinstance(v[0], torch.Tensor):
            data_dict[k] = [x.to(device, non_blocking=True) for x in v]


def _log_training_info(
    logger,
    curr_epoch,
    iter_idx,
    num_iters,
    losses_tracker,
    grad_norm_tracker,
    curr_det_lr,
    curr_backbone_lr,
    interval_data_time,
    start_time,
    rank,
):
    """Log training information to logger and wandb."""
    # print to terminal
    block1 = f"[{curr_epoch:03d}][{iter_idx:05d}/{num_iters - 1:05d}]"
    block2 = f"Loss={losses_tracker['loss'].avg:.4f}"
    block3 = [f"{key}={value.avg:.4f}" for key, value in losses_tracker.items() if key != "loss"]
    block4 = f"flr_det={curr_det_lr:.1e}"
    if curr_backbone_lr is not None:
        block4 = f"lr_backbone={curr_backbone_lr:.1e}" + "  " + block4
    block5 = f"mem={torch.cuda.max_memory_allocated() / 1024.0 / 1024.0:.0f}MB"
    block7 = f"data_time={interval_data_time:.2f}s"
    block8 = f"time={time.time() - start_time:.2f}s"
    logger.info("  ".join([block1, block2, "  ".join(block3), block4, block5, block7, block8]))

    # log to wandb at the same logging granularity (use global step)
    current_step = int(curr_epoch) * int(num_iters) + int(iter_idx)
    try:
        if rank == 0 and wandb.run is not None:
            log_dict = {
                "train/lr": curr_det_lr,
                "train/grad_norm": grad_norm_tracker.avg,
                **{f"train/{key}": value.avg for key, value in losses_tracker.items()},
            }
            if curr_backbone_lr is not None:
                log_dict["train/lr_backbone"] = curr_backbone_lr

            wandb.log(log_dict, step=current_step)
    except Exception:
        pass


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    logger,
    rank=0,
    model_ema=None,
    clip_grad_l2norm=-1,
    logging_interval=200,
    scaler=None,
):
    """Training the model for one epoch"""

    logger.info(f"[Train]: Epoch {curr_epoch} started")
    losses_tracker = {}
    grad_norm_tracker = AverageMeter()
    num_iters = len(train_loader)

    # get the device of the model
    device = next(model.parameters()).device

    model.train()
    interval_start_time = end = time.time()
    interval_data_time = 0.0
    for iter_idx, data_dict in enumerate(train_loader):
        data_time = time.time() - end
        interval_data_time += data_time

        # move data to device
        move_to_device(data_dict, device)

        optimizer.zero_grad()

        # Compat for both DDP (model.module) and Single GPU (model)
        raw_model = model.module if hasattr(model, "module") else model

        # current learning rate
        curr_backbone_lr = None
        if hasattr(raw_model, "backbone") and not raw_model.backbone.freeze_backbone:
            curr_backbone_lr = scheduler.get_last_lr()[0]
        curr_det_lr = scheduler.get_last_lr()[-1]

        # Use amp if scaler is provided and enabled
        is_amp_enabled = scaler is not None and scaler.is_enabled()

        # forward pass
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=is_amp_enabled):
            losses = model(**data_dict, return_loss=True)

        # compute the gradients
        if is_amp_enabled:
            scaler.scale(losses["loss"]).backward()
        else:
            losses["loss"].backward()

        # gradient clipping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            if is_amp_enabled:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))

        # update parameters
        if is_amp_enabled:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # update scheduler
        scheduler.step()

        # update ema
        if model_ema is not None:
            model_ema.update(model)

        # track all losses locally
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

        # track grad_norm separately
        grad_norm_tracker.update(grad_norm.item())

        # printing each logging_interval
        if ((iter_idx != 0) and (iter_idx % logging_interval) == 0) or (
            (iter_idx + 1) == num_iters
        ):
            # reduce local averages across distributed GPUs, only for logging
            if dist.is_available() and dist.is_initialized():
                loss_names = list(losses_tracker.keys())
                if len(loss_names) > 0:
                    # gather locally accumulated averages
                    loss_avgs = torch.tensor(
                        [losses_tracker[name].avg for name in loss_names], device=device
                    )
                    dist.all_reduce(loss_avgs, op=dist.ReduceOp.AVG)
                    # temporarily override avg for logging
                    for i, name in enumerate(loss_names):
                        losses_tracker[name].avg = loss_avgs[i].item()

            _log_training_info(
                logger,
                curr_epoch,
                iter_idx,
                num_iters,
                losses_tracker,
                grad_norm_tracker,
                curr_det_lr,
                curr_backbone_lr,
                interval_data_time,
                interval_start_time,
                rank,
            )
            interval_start_time = time.time()
            interval_data_time = 0.0

        end = time.time()

    # Calculate final global step for this epoch to ensure it's defined for all ranks
    return int(curr_epoch) * int(num_iters) + int(num_iters) - 1


def _setup_inference_resources(cfg, test_loader):
    cfg.inference["folder"] = Path(cfg.work_dir) / "outputs"
    if cfg.inference.save_raw_prediction:
        Path(cfg.inference["folder"]).expanduser().mkdir(mode=0o777, parents=True, exist_ok=True)

    # external classifier
    if "external_cls" in cfg.post_processing:
        if cfg.post_processing.external_cls is not None:
            external_cls = build_classifier(cfg.post_processing.external_cls)
        else:
            external_cls = test_loader.dataset.class_map
    else:
        external_cls = test_loader.dataset.class_map

    # whether the testing dataset is sliding window
    cfg.post_processing.sliding_window = isinstance(test_loader.dataset, SlidingWindowDataset)
    return external_cls, cfg


def _save_and_evaluate(result_dict, cfg, logger, skip_eval):
    result_eval = dict(results=result_dict)
    if cfg.post_processing.save_dict:
        result_path = Path(cfg.work_dir) / "result_detection.json"
        with open(result_path, "w") as out:
            json.dump(result_eval, out)

    if not skip_eval:
        # build evaluator
        evaluator = build_evaluator(dict(prediction_file=result_eval, **cfg.evaluation))
        # evaluate and output
        logger.info("[Evaluation]:")
        metrics_dict = evaluator.evaluate()
        evaluator.logging(logger)
        if "average_mAP" in metrics_dict:
            return metrics_dict["average_mAP"]
    return 0.0


def inference_and_eval_one_epoch(
    test_loader,
    model,
    cfg,
    logger,
    rank,
    model_ema=None,
    use_amp=False,
    world_size=0,
    skip_eval=False,
):
    """Inference and Evaluation the model"""

    # Use EMA model directly for evaluation to save memory, or fallback to standard model
    eval_model = model_ema.module if model_ema is not None else model

    external_cls, cfg = _setup_inference_resources(cfg, test_loader)

    # get the device of the model
    device = next(eval_model.parameters()).device

    # determine if amp is enabled based on scaler if provided, or default to checking use_amp param
    # Note: inference usually uses the manual use_amp flag since test doesn't necessarily get scaler

    # model forward
    eval_model.eval()
    result_dict = {}
    for data_dict in tqdm.tqdm(test_loader, disable=(rank != 0)):  # inference + NMS
        # move data to device
        move_to_device(data_dict, device)
        # forward pass
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                results = eval_model(
                    **data_dict,
                    return_loss=False,
                    infer_cfg=cfg.inference,
                    post_cfg=cfg.post_processing,
                    ext_cls=external_cls,
                )

        # update the result dict
        for k, v in results.items():
            if k in result_dict:
                result_dict[k].extend(v)
            else:
                result_dict[k] = v

    result_dict = gather_ddp_results(world_size, result_dict, cfg.post_processing)

    if rank == 0:
        return _save_and_evaluate(result_dict, cfg, logger, skip_eval)
    return 0.0


def gather_ddp_results(world_size, result_dict, post_cfg):
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        gather_dict_list = [None for _ in range(world_size)]
        dist.all_gather_object(gather_dict_list, result_dict)
        result_dict = {}
        for i in range(world_size):  # update the result dict
            for k, v in gather_dict_list[i].items():
                if k in result_dict:
                    result_dict[k].extend(v)
                else:
                    result_dict[k] = v

    # do nms for sliding window, if needed
    if post_cfg.sliding_window is True and post_cfg.nms is not None:
        # assert sliding_window=True
        tmp_result_dict = {}
        for k, v in result_dict.items():
            segments = torch.Tensor([data["segment"] for data in v])
            scores = torch.Tensor([data["score"] for data in v])
            labels = []
            class_idx = []
            for data in v:
                if data["label"] not in class_idx:
                    class_idx.append(data["label"])
                labels.append(class_idx.index(data["label"]))
            labels = torch.Tensor(labels)

            segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)

            results_per_video = []
            for segment, label, score in zip(segments, labels, scores, strict=True):
                # convert to python scalars
                results_per_video.append(
                    dict(
                        segment=[round(seg.item(), 2) for seg in segment],
                        label=class_idx[int(label.item())],
                        score=round(score.item(), 4),
                    )
                )
            tmp_result_dict[k] = results_per_video
        result_dict = tmp_result_dict
    return result_dict
