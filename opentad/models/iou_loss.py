import torch
import torch.nn as nn
from .builder import LOSSES
from .utils.iou_tools import compute_diou_torch, compute_giou_torch


@LOSSES.register_module()
class DIOULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        reduction: str = "none",
        eps: float = 1e-8,
    ) -> torch.Tensor:
        loss = 1 - torch.diag(compute_diou_torch(target_bboxes, input_bboxes))

        if reduction == "mean":
            loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif reduction == "sum":
            loss = loss.sum()

        return loss


@LOSSES.register_module()
class GIOULoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        input_bboxes: torch.Tensor,
        target_bboxes: torch.Tensor,
        reduction: str = "none",
        eps: float = 1e-8,
    ) -> torch.Tensor:
        loss = 1 - torch.diag(compute_giou_torch(target_bboxes, input_bboxes))

        if reduction == "mean":
            loss = loss.mean() if loss.numel() > 0 else 0.0 * loss.sum()
        elif reduction == "sum":
            loss = loss.sum()
        return loss


@LOSSES.register_module()
class IOULoss(nn.Module):
    def __init__(self, reduction="none"):
        super().__init__()
        self.reduction = reduction

    def forward(self, pred, target, reduction=None):
        # pred/target: [N,2] (start, end)
        start = torch.max(pred[:, 0], target[:, 0])
        end = torch.min(pred[:, 1], target[:, 1])
        inter = (end - start).clamp(min=0)
        union = (pred[:, 1] - pred[:, 0]) + (target[:, 1] - target[:, 0]) - inter
        iou = inter / (union + 1e-6)
        if reduction is None:
            reduction = self.reduction
        if reduction == "mean":
            return iou.mean()
        elif reduction == "sum":
            return iou.sum()
        else:
            return iou