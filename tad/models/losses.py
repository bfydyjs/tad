import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits

from .builder import LOSSES
from .utils import compute_diou_torch, compute_giou_torch


@LOSSES.register_module()
class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(
        self, inputs: torch.Tensor, targets: torch.Tensor, reduction: str = "none"
    ) -> torch.Tensor:
        """
        Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
        Taken from
        https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
        # Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

        Args:
            inputs: A float tensor of arbitrary shape.
                    The predictions for each example.
            targets: A float tensor with the same shape as inputs. Stores the binary
                    classification label for each element in inputs
                    (0 for the negative class and 1 for the positive class).
            alpha: (optional) Weighting factor in range (0,1) to balance
                    positive vs negative examples. Default = 0.25.
            gamma: Exponent of the modulating factor (1 - p_t) to
                balance easy vs hard examples.
            reduction: 'none' | 'mean' | 'sum'
                    'none': No reduction will be applied to the output.
                    'mean': The output will be averaged.
                    'sum': The output will be summed.
        Returns:
            Loss tensor with the reduction option applied.
        """
        loss = sigmoid_focal_loss(inputs, targets, self.alpha, self.gamma)

        if reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()

        return loss

    def __repr__(self):
        return f"{self.__class__.__name__}(alpha={self.alpha}, gamma={self.gamma})"


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    inputs = inputs.float()
    targets = targets.float()
    p = torch.sigmoid(inputs)
    ce_loss = binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = p * targets + (1 - p) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    return loss


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
