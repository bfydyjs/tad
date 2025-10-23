import torch
import torch.nn as nn
import torch.nn.functional as F

from .builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead
from .bricks import ConvModule, Scale

@HEADS.register_module()
class DecoupledIoUHead(AnchorFreeHead):
    def __init__(
        self,
        num_classes,
        in_channels,
        feat_channels,
        num_convs=3,
        prior_generator=None,
        loss=None,
        loss_normalizer=100,
        loss_normalizer_momentum=0.9,
        center_sample="radius",
        center_sample_radius=1.5,
        label_smoothing=0,
        cls_prior_prob=0.01,
        loss_weight=1.0,
        iou_loss_weight=1.0,  # 新增 IoU 损失权重
    ):
        super().__init__(
            num_classes,
            in_channels,
            feat_channels,
            num_convs=num_convs,
            prior_generator=prior_generator,
            loss=loss,
            loss_normalizer=loss_normalizer,
            loss_normalizer_momentum=loss_normalizer_momentum,
            center_sample=center_sample,
            center_sample_radius=center_sample_radius,
            label_smoothing=label_smoothing,
            cls_prior_prob=cls_prior_prob,
            loss_weight=loss_weight,
        )
        self.iou_loss_weight = iou_loss_weight
        # 用于计算 IoU 损失的辅助工具
        self.iou_calculator = build_loss(dict(type="IOULoss")) 

    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList([])
        self.reg_convs = nn.ModuleList([])
        
        # 解耦的两个卷积塔
        for i in range(self.num_convs):
            self.cls_convs.append(
                ConvModule(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type="LN"),
                    act_cfg=dict(type="relu"),
                )
            )
            self.reg_convs.append(
                ConvModule(
                    self.in_channels if i == 0 else self.feat_channels,
                    self.feat_channels,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    norm_cfg=dict(type="LN"),
                    act_cfg=dict(type="relu"),
                )
            )

        # 预测头
        self.cls_head = nn.Conv1d(self.feat_channels, self.num_classes, kernel_size=3, padding=1)
        self.reg_head = nn.Conv1d(self.feat_channels, 2, kernel_size=3, padding=1)
        # 新增：IoU 预测头 (共享回归特征)
        self.iou_head = nn.Conv1d(self.feat_channels, 1, kernel_size=3, padding=1)
        
        self.scale = nn.ModuleList([Scale() for _ in range(len(self.prior_generator.strides))])

        # 初始化分类偏置
        if self.cls_prior_prob > 0:
            import math
            bias_value = -(math.log((1 - self.cls_prior_prob) / self.cls_prior_prob))
            nn.init.constant_(self.cls_head.bias, bias_value)

    def forward_train(self, feat_list, mask_list, gt_segments, gt_labels, **kwargs):
        cls_pred, reg_pred, iou_pred = [], [], []

        for l, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            cls_feat = feat
            reg_feat = feat

            # 通过解耦的卷积塔
            for i in range(self.num_convs):
                cls_feat, mask = self.cls_convs[i](cls_feat, mask)
                reg_feat, mask = self.reg_convs[i](reg_feat, mask)

            cls_pred.append(self.cls_head(cls_feat))
            reg_pred.append(F.relu(self.scale[l](self.reg_head(reg_feat))))
            iou_pred.append(self.iou_head(reg_feat)) # IoU 分支

        points = self.prior_generator(feat_list)

        losses = self.losses(cls_pred, reg_pred, iou_pred, mask_list, points, gt_segments, gt_labels)
        return losses

    def forward_test(self, feat_list, mask_list, **kwargs):
        cls_pred, reg_pred, iou_pred = [], [], []

        for l, (feat, mask) in enumerate(zip(feat_list, mask_list)):
            cls_feat = feat
            reg_feat = feat

            for i in range(self.num_convs):
                cls_feat, mask = self.cls_convs[i](cls_feat, mask)
                reg_feat, mask = self.reg_convs[i](reg_feat, mask)

            cls_pred.append(self.cls_head(cls_feat))
            reg_pred.append(F.relu(self.scale[l](self.reg_head(reg_feat))))
            iou_pred.append(self.iou_head(reg_feat))

        points = self.prior_generator(feat_list)
        
        # 获取 proposals 和 scores (包含 IoU 修正)
        return self.get_valid_proposals_scores(points, reg_pred, cls_pred, iou_pred, mask_list)

    def losses(self, cls_pred, reg_pred, iou_pred, mask_list, points, gt_segments, gt_labels):
        gt_cls, gt_reg = self.prepare_targets(points, gt_segments, gt_labels)

        # 准备 mask
        gt_cls = torch.stack(gt_cls)
        valid_mask = torch.cat(mask_list, dim=1)
        pos_mask = torch.logical_and((gt_cls.sum(-1) > 0), valid_mask)
        num_pos = pos_mask.sum().item()

        # Loss Normalizer 更新
        if self.training:
            self.loss_normalizer = self.loss_normalizer_momentum * self.loss_normalizer + (
                1 - self.loss_normalizer_momentum
            ) * max(num_pos, 1)
            loss_normalizer = self.loss_normalizer
        else:
            loss_normalizer = max(num_pos, 1)

        # 1. Classification Loss
        cls_pred_cat = [x.permute(0, 2, 1) for x in cls_pred]
        cls_pred_cat = torch.cat(cls_pred_cat, dim=1)[valid_mask]
        gt_target = gt_cls[valid_mask]
        
        # Label Smoothing
        gt_target *= 1 - self.label_smoothing
        gt_target += self.label_smoothing / (self.num_classes + 1)

        cls_loss = self.cls_loss(cls_pred_cat, gt_target, reduction="sum")
        cls_loss /= loss_normalizer

        # 2. Regression Loss & 3. IoU Loss
        split_size = [reg.shape[-1] for reg in reg_pred]
        gt_reg_split = torch.stack(gt_reg).permute(0, 2, 1).split(split_size, dim=-1)
        
        # 获取预测框和 GT 框
        pred_segments = self.get_refined_proposals(points, reg_pred)[pos_mask]
        gt_segments_decoded = self.get_refined_proposals(points, gt_reg_split)[pos_mask]
        
        # 处理 IoU 预测
        iou_pred_cat = [x.permute(0, 2, 1) for x in iou_pred]
        iou_pred_cat = torch.cat(iou_pred_cat, dim=1) # [B, T, 1]
        iou_pred_pos = iou_pred_cat[pos_mask].squeeze(-1) # [N_pos]

        if num_pos == 0:
            reg_loss = pred_segments.sum() * 0
            iou_loss = iou_pred_cat.sum() * 0
        else:
            # Reg Loss (DIoU/GIoU)
            reg_loss = self.reg_loss(pred_segments, gt_segments_decoded, reduction="sum")
            reg_loss /= loss_normalizer

            # 计算真实的 IoU 作为 Target
            # 使用 iou_calculator 计算 pred 和 gt 之间的 IoU
            with torch.no_grad():
                iou_targets = self.iou_calculator(pred_segments, gt_segments_decoded, reduction="none")
                iou_targets = iou_targets.clamp(min=0, max=1.0)
            
            # IoU Loss (Binary Cross Entropy)
            iou_loss = F.binary_cross_entropy_with_logits(iou_pred_pos, iou_targets, reduction="sum")
            iou_loss /= loss_normalizer

        # 动态调整 Loss Weight
        if self.loss_weight > 0:
            loss_weight = self.loss_weight
        else:
            loss_weight = cls_loss.detach() / max(reg_loss.item(), 0.01)

        return {
            "cls_loss": cls_loss, 
            "reg_loss": reg_loss * loss_weight,
            "iou_loss": iou_loss * self.iou_loss_weight # 通常 IoU loss 权重设为 1.0 或 0.5
        }

    def get_valid_proposals_scores(self, points, reg_pred, cls_pred, iou_pred, mask_list):
        # 获取基础 Proposals
        proposals = self.get_refined_proposals(points, reg_pred)  # [B,T,2]
        
        # 获取分类分数
        cls_scores = torch.cat(cls_pred, dim=-1).permute(0, 2, 1).sigmoid()  # [B,T,num_classes]
        
        # 获取 IoU 分数
        iou_scores = torch.cat(iou_pred, dim=-1).permute(0, 2, 1).sigmoid() # [B,T,1]
        
        # 融合分数：Score = Cls * IoU
        final_scores = cls_scores * iou_scores

        # Mask out invalid
        masks = torch.cat(mask_list, dim=1)  # [B,T]
        new_proposals, new_scores = [], []
        for proposal, score, mask in zip(proposals, final_scores, masks):
            new_proposals.append(proposal[mask])
            new_scores.append(score[mask])
        return new_proposals, new_scores