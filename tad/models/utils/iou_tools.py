import torch


def compute_iou_torch(gt_boxes, anchors):
    """Compute IoU between gt_boxes and anchors.
    gt_boxes: shape [N, 2]
    anchors:  shape [M, 2]
    """

    gt_areas = gt_boxes[:, 1] - gt_boxes[:, 0]
    anchors_areas = anchors[:, 1] - anchors[:, 0]

    inter_min = torch.max(anchors[:, 0].unsqueeze(1), gt_boxes[:, 0].unsqueeze(0))
    inter_max = torch.min(anchors[:, 1].unsqueeze(1), gt_boxes[:, 1].unsqueeze(0))
    inter = (inter_max - inter_min).clamp(min=0)

    scores = inter / (anchors_areas.unsqueeze(1) + gt_areas.unsqueeze(0) - inter).clamp(
        min=1e-6
    )  # shape [M, N]
    return scores.to(anchors.dtype)


def compute_ioa_torch(gt_boxes, anchors):
    """Compute Intersection between gt_boxes and anchors.
    gt_boxes: np.array shape [N, 2]
    anchors:  np.array shape [M, 2]
    """

    anchors_areas = anchors[:, 1] - anchors[:, 0]

    inter_min = torch.max(anchors[:, 0].unsqueeze(1), gt_boxes[:, 0].unsqueeze(0))
    inter_max = torch.min(anchors[:, 1].unsqueeze(1), gt_boxes[:, 1].unsqueeze(0))
    inter = (inter_max - inter_min).clamp(min=0)

    scores = inter / anchors_areas.unsqueeze(1).clamp(min=1e-6)  # shape [M, N]
    return scores.to(anchors.dtype)


def compute_batched_iou_torch(gt_boxes, anchors):
    """Compute IoU between gt_boxes and anchors.
    gt_boxes: shape [B, N, 2]
    anchors:  shape [B, N, 2]
    gt_boxes has been aligned with anchors
    """
    # ...existing code...
    gt_areas = gt_boxes[..., 1] - gt_boxes[..., 0]
    anchors_areas = anchors[..., 1] - anchors[..., 0]

    inter_min = torch.max(anchors[..., 0], gt_boxes[..., 0])
    inter_max = torch.min(anchors[..., 1], gt_boxes[..., 1])
    inter = (inter_max - inter_min).clamp(min=0)

    scores = inter / (anchors_areas + gt_areas - inter).clamp(min=1e-6)  # [B,N]
    return scores.to(anchors.dtype)


def compute_giou_torch(gt_boxes, anchors):
    """Compute GIoU between gt_boxes and anchors.
    gt_boxes: shape [N, 2]
    anchors:  shape [M, 2]
    """

    gt_areas = gt_boxes[:, 1] - gt_boxes[:, 0]
    anchors_areas = anchors[:, 1] - anchors[:, 0]

    inter_min = torch.max(anchors[:, 0].unsqueeze(1), gt_boxes[:, 0].unsqueeze(0))
    inter_max = torch.min(anchors[:, 1].unsqueeze(1), gt_boxes[:, 1].unsqueeze(0))
    inter = (inter_max - inter_min).clamp(min=0)

    union = anchors_areas.unsqueeze(1) + gt_areas.unsqueeze(0) - inter

    iou = inter / union.clamp(min=1e-6)  # shape [M, N]

    x1_enclosing = torch.min(anchors[:, 0].unsqueeze(1), gt_boxes[:, 0].unsqueeze(0))
    x2_enclosing = torch.max(anchors[:, 1].unsqueeze(1), gt_boxes[:, 1].unsqueeze(0))
    area = (x2_enclosing - x1_enclosing).clamp(min=1e-7)

    # GIOU
    giou = iou - (area - union) / (area + 1e-6)
    return giou.to(anchors.dtype)  # [M,N]


def compute_diou_torch(gt_boxes, anchors, eps=1e-7):
    """
    Compute DIoU (Distance Intersection over Union) between pairs of 1D boxes.

    Encourages maximizing the overlap and minimizing the distance between the centers of the boxes.

    Tensor broadcasting and repeating are used to efficiently compute overlaps and distances
    between all pairs of boxes.

    Args:
    gt_boxes (torch.Tensor): Ground truth boxes, shape (N, 2)
    anchors (torch.Tensor): Anchor boxes, shape (M, 2)
    eps (float, optional): A small number to prevent division by zero.
    Returns:
    torch.Tensor: The DIoU between each pair of boxes, shape (M, N)
    """

    gt_areas = gt_boxes[:, 1] - gt_boxes[:, 0]
    anchors_areas = anchors[:, 1] - anchors[:, 0]

    # overlap
    inter_min = torch.max(anchors[:, 0].unsqueeze(1), gt_boxes[:, 0].unsqueeze(0))
    inter_max = torch.min(anchors[:, 1].unsqueeze(1), gt_boxes[:, 1].unsqueeze(0))
    inter = (inter_max - inter_min).clamp(min=0)

    # union
    union = anchors_areas.unsqueeze(1) + gt_areas.unsqueeze(0) - inter

    # IoU
    iou = inter / union.clamp(min=eps)  # shape [M, N]

    # enclose area
    x1_enclosing = torch.min(anchors[:, 0].unsqueeze(1), gt_boxes[:, 0].unsqueeze(0))
    x2_enclosing = torch.max(anchors[:, 1].unsqueeze(1), gt_boxes[:, 1].unsqueeze(0))
    area = x2_enclosing - x1_enclosing

    # center distance
    c1 = (anchors[:, 0].unsqueeze(1) + anchors[:, 1].unsqueeze(1)) / 2
    c2 = (gt_boxes[:, 0].unsqueeze(0) + gt_boxes[:, 1].unsqueeze(0)) / 2
    c_dist = (c2 - c1).abs()

    # DIoU
    diou = iou - c_dist * c_dist / (area * area).clamp(min=eps)
    return diou.to(anchors.dtype)  # [M,N]
