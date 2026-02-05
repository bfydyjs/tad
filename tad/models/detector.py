import torch

from .builder import DETECTORS, build_backbone, build_head, build_neck, build_projection
from .utils.post_processing import batched_nms, convert_to_seconds, load_predictions, save_predictions


@DETECTORS.register_module()
class Detector(torch.nn.Module):
    def __init__(self, backbone=None, projection=None, neck=None, rpn_head=None):
        super().__init__()

        if backbone is not None:
            self.backbone = build_backbone(backbone)

        if projection is not None:
            self.projection = build_projection(projection)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            self.rpn_head = build_head(rpn_head)

    def forward(
        self,
        inputs,
        masks,
        metas,
        gt_segments=None,
        gt_labels=None,
        return_loss=True,
        infer_cfg=None,
        post_cfg=None,
        **kwargs
    ):
        if return_loss:
            return self.forward_train(inputs, masks, metas, gt_segments=gt_segments, gt_labels=gt_labels, **kwargs)
        else:
            return self.forward_detection(inputs, masks, metas, infer_cfg, post_cfg, **kwargs)

    def forward_detection(self, inputs, masks, metas, infer_cfg, post_cfg, **kwargs):
        # step1: inference the model
        if infer_cfg.load_from_raw_predictions:  # easier and faster to tune the hyper parameter in postprocessing
            predictions = load_predictions(metas, infer_cfg)
        else:
            predictions = self.forward_test(inputs, masks, metas, infer_cfg)

            if infer_cfg.save_raw_prediction:  # save the predictions to disk
                save_predictions(predictions, metas, infer_cfg.folder)

        # step2: detection post processing
        results = self.post_processing(predictions, metas, post_cfg, **kwargs)
        return results

    @property
    def with_backbone(self):
        """bool: whether the detector has backbone"""
        return hasattr(self, "backbone") and self.backbone is not None

    @property
    def with_projection(self):
        """bool: whether the detector has projection"""
        return hasattr(self, "projection") and self.projection is not None

    @property
    def with_neck(self):
        """bool: whether the detector has neck"""
        return hasattr(self, "neck") and self.neck is not None

    @property
    def with_rpn_head(self):
        """bool: whether the detector has localization head"""
        return hasattr(self, "rpn_head") and self.rpn_head is not None

    def extract_feat(self, inputs, masks):
        if self.with_backbone:
            x = self.backbone(inputs, masks)
        else:
            x = inputs

        if self.with_projection:
            x, masks = self.projection(x, masks)

        if self.with_neck:
            x, masks = self.neck(x, masks)
        return x, masks

    def forward_train(self, inputs, masks, metas, gt_segments, gt_labels, **kwargs):
        x, masks = self.extract_feat(inputs, masks)
        losses = dict()

        if self.with_rpn_head:
            rpn_losses = self.rpn_head.forward_train(
                x,
                masks,
                gt_segments=gt_segments,
                gt_labels=gt_labels,
                **kwargs,
            )
            losses.update(rpn_losses)

        # only key has loss will be record
        losses["loss"] = sum(losses.values())
        return losses

    def forward_test(self, inputs, masks, metas=None, infer_cfg=None, **kwargs):
        x, masks = self.extract_feat(inputs, masks)

        if self.with_rpn_head:
            rpn_proposals, rpn_scores = self.rpn_head.forward_test(x, masks)
        else:
            rpn_proposals = rpn_scores = None

        predictions = rpn_proposals, rpn_scores
        return predictions

    @torch.no_grad()
    def post_processing(self, predictions, metas, post_cfg, ext_cls, **kwargs):
        rpn_proposals, rpn_scores = predictions
        # rpn_proposals,  # [B,K,2]
        # rpn_scores,  # [B,K,num_classes] after sigmoid

        pre_nms_thresh = post_cfg.get("pre_nms_thresh", 0.001)
        pre_nms_topk = post_cfg.get("pre_nms_topk", 2000)
        num_classes = rpn_scores[0].shape[-1]

        results = {}
        for i in range(len(metas)):  # processing each video
            segments = rpn_proposals[i].detach()  # [N,2]
            scores = rpn_scores[i].detach()  # [N,class]

            if num_classes == 1:
                scores = scores.squeeze(-1)
                labels = torch.zeros(scores.shape[0], device=scores.device).long()
            else:
                pred_prob = scores.flatten()  # [N*class]

                # Apply filtering to make NMS faster following detectron2
                # 1. Keep seg with confidence score > a threshold
                keep_idxs1 = pred_prob > pre_nms_thresh
                pred_prob = pred_prob[keep_idxs1]
                topk_idxs = keep_idxs1.nonzero(as_tuple=True)[0]

                # 2. Keep top k top scoring boxes only
                num_topk = min(pre_nms_topk, topk_idxs.size(0))
                pred_prob, idxs = pred_prob.sort(descending=True)
                pred_prob = pred_prob[:num_topk]
                topk_idxs = topk_idxs[idxs[:num_topk]]

                # 3. gather predicted proposals
                pt_idxs = torch.div(topk_idxs, num_classes, rounding_mode="floor")
                cls_idxs = torch.fmod(topk_idxs, num_classes)

                segments = segments[pt_idxs]
                scores = pred_prob
                labels = cls_idxs

            segments = segments.cpu()
            scores = scores.cpu()
            labels = labels.cpu()

            # if not sliding window, do nms
            if post_cfg.sliding_window is False and post_cfg.nms is not None:
                segments, scores, labels = batched_nms(segments, scores, labels, **post_cfg.nms)

            video_id = metas[i]["video_name"]

            # convert segments to seconds
            segments = convert_to_seconds(segments, metas[i])

            # merge with external classifier
            if isinstance(ext_cls, list):  # own classification results
                labels = [ext_cls[label.item()] for label in labels]
            else:
                segments, labels, scores = ext_cls(video_id, segments, scores)

            results_per_video = []
            for segment, label, score in zip(segments, labels, scores):
                # convert to python scalars
                results_per_video.append(
                    dict(
                        segment=[round(seg.item(), 2) for seg in segment],
                        label=label,
                        score=round(score.item(), 4),
                    )
                )

            if video_id in results.keys():
                results[video_id].extend(results_per_video)
            else:
                results[video_id] = results_per_video

        return results
