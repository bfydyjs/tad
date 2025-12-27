import os
import time
import copy
import json
import tqdm
import torch
import torch.distributed as dist

from opentad.utils import create_folder
from opentad.utils.misc import AverageMeter, reduce_loss
from opentad.models.utils.post_processing import build_classifier, batched_nms
from opentad.evaluation import build_evaluator
from opentad.datasets.base import SlidingWindowDataset


def train_one_epoch(
    train_loader,
    model,
    optimizer,
    scheduler,
    curr_epoch,
    logger,
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
    use_amp = False if scaler is None else True

    model.train()
    interval_start_time = end = time.time()
    interval_data_time = 0.0
    for iter_idx, data_dict in enumerate(train_loader):
        data_time = time.time() - end
        interval_data_time += data_time
        end = time.time()
        optimizer.zero_grad()

        # current learning rate
        curr_backbone_lr = None
        if hasattr(model.module, "backbone") and not model.module.backbone.freeze_backbone:
            curr_backbone_lr = scheduler.get_last_lr()[0]
        curr_det_lr = scheduler.get_last_lr()[-1]

        # forward pass
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            losses = model(**data_dict, return_loss=True)

        # compute the gradients
        (scaler.scale(losses["cost"]) if use_amp else losses["cost"]).backward()

        # gradient clipping (to stabilize training if necessary)
        if clip_grad_l2norm > 0.0:
            if use_amp:
                scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_l2norm)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float('inf'))

        # update parameters
        if use_amp:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # update scheduler
        scheduler.step()

        # update ema
        if model_ema is not None:
            model_ema.update(model)

        # track all losses
        losses = reduce_loss(losses)  # only for log
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

        # track grad_norm separately
        grad_norm_tracker.update(grad_norm.item())

        # printing each logging_interval
        if ((iter_idx != 0) and (iter_idx % logging_interval) == 0) or ((iter_idx + 1) == num_iters):
            # print to terminal
            block1 = f"[Train]: [{curr_epoch:03d}][{iter_idx:05d}/{num_iters - 1:05d}]"
            block2 = f"Loss={losses_tracker['cost'].avg:.4f}"
            block3 = [f"{key}={value.avg:.4f}" for key, value in losses_tracker.items() if key != "cost"]
            block4 = f"flr_det={curr_det_lr:.1e}"
            if curr_backbone_lr is not None:
                block4 = f"lr_backbone={curr_backbone_lr:.1e}" + "  " + block4
            block5 = f"mem={torch.cuda.max_memory_allocated() / 1024.0 / 1024.0:.0f}MB"
            block7 = f"data_time={interval_data_time:.2f}s"
            block8 = f"time={time.time() - interval_start_time:.2f}s"
            logger.info("  ".join([block1, block2, "  ".join(block3), block4, block5, block7, block8]))
            interval_start_time = end = time.time()
            interval_data_time = 0.0

    # return average loss and grad_norm
    return losses_tracker["cost"].avg, grad_norm_tracker.avg


def val_one_epoch(
    val_loader,
    model,
    logger,
    rank,
    curr_epoch,
    model_ema=None,
    use_amp=False,
):
    """Validating the model for one epoch: compute the loss"""

    # load the ema dict for evaluation
    if model_ema is not None:
        current_dict = copy.deepcopy(model.state_dict())
        model.load_state_dict(model_ema.module.state_dict())

    logger.info(f"[Val]: Epoch {curr_epoch} Loss")
    losses_tracker = {}

    model.eval()
    for data_dict in tqdm.tqdm(val_loader, disable=(rank != 0)):
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                losses = model(**data_dict, return_loss=True)

        # track all losses
        losses = reduce_loss(losses)  # only for log
        for key, value in losses.items():
            if key not in losses_tracker:
                losses_tracker[key] = AverageMeter()
            losses_tracker[key].update(value.item())

    # print to terminal
    block1 = f"[Val]: [{curr_epoch:03d}]"
    block2 = f"Loss={losses_tracker['cost'].avg:.4f}"
    block3 = [f"{key}={value.avg:.4f}" for key, value in losses_tracker.items() if key != "cost"]
    logger.info("  ".join([block1, block2, "  ".join(block3)]))

    # load back the normal model dict
    if model_ema is not None:
        model.load_state_dict(current_dict)
    return losses_tracker["cost"].avg


def eval_one_epoch(
    test_loader,
    model,
    cfg,
    logger,
    rank,
    model_ema=None,
    use_amp=False,
    world_size=0,
    not_eval=False,
):
    """Inference and Evaluation the model"""

    # load the ema dict for evaluation
    if model_ema is not None:
        current_dict = copy.deepcopy(model.state_dict())
        model.load_state_dict(model_ema.module.state_dict())

    cfg.inference["folder"] = os.path.join(cfg.work_dir, "outputs")
    if cfg.inference.save_raw_prediction:
        create_folder(cfg.inference["folder"])

    # external classifier
    if "external_cls" in cfg.post_processing:
        if cfg.post_processing.external_cls is not None:
            external_cls = build_classifier(cfg.post_processing.external_cls)
    else:
        external_cls = test_loader.dataset.class_map

    # whether the testing dataset is sliding window
    cfg.post_processing.sliding_window = isinstance(test_loader.dataset, SlidingWindowDataset)

    # model forward
    model.eval()
    result_dict = {}
    for data_dict in tqdm.tqdm(test_loader, disable=(rank != 0)):
        with torch.amp.autocast('cuda', dtype=torch.float16, enabled=use_amp):
            with torch.no_grad():
                results = model(
                    **data_dict,
                    return_loss=False,
                    infer_cfg=cfg.inference,
                    post_cfg=cfg.post_processing,
                    ext_cls=external_cls,
                )

        # update the result dict
        for k, v in results.items():
            if k in result_dict.keys():
                result_dict[k].extend(v)
            else:
                result_dict[k] = v

    result_dict = gather_ddp_results(world_size, result_dict, cfg.post_processing)

    # load back the normal model dict
    if model_ema is not None:
        model.load_state_dict(current_dict)

    if rank == 0:
        result_eval = dict(results=result_dict)
        if cfg.post_processing.save_dict:
            result_path = os.path.join(cfg.work_dir, "result_detection.json")
            with open(result_path, "w") as out:
                json.dump(result_eval, out)

        if not not_eval:
            # build evaluator
            evaluator = build_evaluator(dict(prediction_filename=result_eval, **cfg.evaluation))
            # evaluate and output
            logger.info("Evaluation starts...")
            metrics_dict = evaluator.evaluate()
            evaluator.logging(logger)
            return metrics_dict["average_mAP"]
    return 0.0


def gather_ddp_results(world_size, result_dict, post_cfg):
    if world_size > 1 and dist.is_available() and dist.is_initialized():
        gather_dict_list = [None for _ in range(world_size)]
        dist.all_gather_object(gather_dict_list, result_dict)
        result_dict = {}
        for i in range(world_size):  # update the result dict
            for k, v in gather_dict_list[i].items():
                if k in result_dict.keys():
                    result_dict[k].extend(v)
                else:
                    result_dict[k] = v

    # do nms for sliding window, if needed
    if post_cfg.sliding_window == True and post_cfg.nms is not None:
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
            for segment, label, score in zip(segments, labels, scores):
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
