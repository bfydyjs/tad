import copy

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from torch.nn.modules.batchnorm import _BatchNorm

from tad.datasets.builder import Pipeline
from tad.utils import load_checkpoint
from tad.utils.registry import Registry

BACKBONES = Registry("model")


class BackboneWrapper(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        custom_cfg = cfg.custom
        model_cfg = copy.deepcopy(cfg)
        model_cfg.pop("custom")

        # build the backbone
        self.model = BACKBONES.build(model_cfg)

        # custom settings: pretrained checkpoint, post_processing_pipeline,
        # norm_eval, freeze_backbone
        # 1. load the pretrained model
        if getattr(custom_cfg, "pretrain", None):
            load_checkpoint(self.model, custom_cfg.pretrain, map_location="cpu")
        else:
            print(
                "Warning: no pretrain path is provided, the backbone will be randomly initialized, "
                "unless you have initialized the weights in the model.py."
            )

        # 2. pre_processing_pipeline
        pre_pipeline_cfg = getattr(custom_cfg, "pre_processing_pipeline", None)
        self.pre_processing_pipeline = Pipeline(pre_pipeline_cfg) if pre_pipeline_cfg else None

        # 3. post_processing_pipeline for pooling and other operations
        post_pipeline_cfg = getattr(custom_cfg, "post_processing_pipeline", None)
        self.post_processing_pipeline = Pipeline(post_pipeline_cfg) if post_pipeline_cfg else None

        # 4. norm_eval: set all norm layers to eval mode
        self.norm_eval = getattr(custom_cfg, "norm_eval", True)

        # 5. freeze_backbone: whether to freeze the backbone, default is False
        self.freeze_backbone = getattr(custom_cfg, "freeze_backbone", False)

        print(f"freeze_backbone: {self.freeze_backbone}, norm_eval: {self.norm_eval}")

        # 6. whether to use temporal activation checkpointing
        self.use_temporal_checkpointing = getattr(custom_cfg, "temporal_checkpointing", False)
        if self.use_temporal_checkpointing:
            self.temporal_checkpointing_chunk_num = custom_cfg.temporal_checkpointing_chunk_num
            self.temporal_checkpointing_chunk_dim = custom_cfg.temporal_checkpointing_chunk_dim

        # apply norm_eval setting immediately
        self.set_norm_layer()

    def forward(self, frames, masks=None):
        # two types: snippet or frame

        # snippet: 3D backbone, [bs, T, 3, clip_len, H, W]
        # frame: 3D backbone, [bs, 1, 3, T, H, W]

        # data preprocessing: normalize mean and std
        frames, _ = self.model.data_preprocessor.preprocess(
            list(frames),  # need list input
            data_samples=None,
            training=False,  # for blending, which is not used in openTAD
        )

        # pre_processing_pipeline:
        if self.pre_processing_pipeline is not None:
            frames = self.pre_processing_pipeline(dict(frames=frames))["frames"]

        # flatten the batch dimension and num_segs dimension
        batches, num_segs = frames.shape[0:2]
        frames = frames.flatten(0, 1).contiguous()  # [bs*num_seg, ...]

        # go through the video backbone
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.freeze_backbone):
            if self.use_temporal_checkpointing:
                features = self.temporal_checkpointing(
                    frames,
                    self.temporal_checkpointing_chunk_num,
                    self.temporal_checkpointing_chunk_dim,
                )
            else:
                features = self.model.backbone(frames)

        # unflatten and pool the features
        if isinstance(features, (tuple, list)):
            features = torch.cat(
                [self.unflatten_and_pool_features(f, batches, num_segs) for f in features], dim=1
            )
        else:
            features = self.unflatten_and_pool_features(features, batches, num_segs)

        # apply mask
        if masks is not None and features.dim() == 3:
            features = features * masks.unsqueeze(1).detach().float()

        # make sure detector has the float32 input
        features = features.to(torch.float32)
        return features

    def train(self, mode=True):
        """Override train to set norm layers to eval mode if norm_eval is True."""
        super().train(mode)
        self.set_norm_layer()
        return self

    def unflatten_and_pool_features(self, features, batches, num_segs):
        # unflatten the batch dimension and num_segs dimension
        features = features.unflatten(dim=0, sizes=(batches, num_segs))  # [bs, num_seg, ...]

        # convert the feature to [B,C,T]: pooling and other operations
        if self.post_processing_pipeline is not None:
            features = self.post_processing_pipeline(dict(feats=features))["feats"]
        return features

    def set_norm_layer(self):
        if self.norm_eval:
            for m in self.modules():
                if isinstance(m, (nn.LayerNorm, nn.GroupNorm, _BatchNorm)):
                    m.eval()

                    for param in m.parameters():
                        param.requires_grad = False

    def temporal_checkpointing(self, frames, chunk_num, chunk_dim):
        """Temporal Checkpointing for Video Backbone.

        Temporal checkpointing will 1) split the video frames along the temporal dimension and
        sequentially forward each chunk with no gradients. 2) The backward pass will recompute the
        intermediate activations and compute each chunk's gradient. 3) Backbone's gradients will be
        accumulated along different chunks.

        Args:
            frames (Tensor): input frames, [B*N,3,T,H,W]
            chunk_num (int): number of chunks to split the temporal dimension
            chunk_dim (int): input shape is [B*N,3,T,H,W], so either dim=0 or 2 is fine
        """

        def _inner_forward(frames):
            return self.model.backbone(frames)

        video_feat = []
        for mini_frames in torch.chunk(frames, chunk_num, dim=chunk_dim):  # B*N is chunked
            # we can use torch.cp.checkpoint to implement an efficient
            # temporal checkpointing mechanism
            mini_feat = cp.checkpoint(
                _inner_forward,
                mini_frames,
                use_reentrant=False,
            )
            video_feat.append(mini_feat)

        if isinstance(video_feat[0], (tuple, list)):
            video_feat = [
                torch.cat([f[idx] for f in video_feat], dim=chunk_dim)
                for idx in range(len(video_feat[0]))
            ]
        else:
            video_feat = torch.cat(video_feat, dim=chunk_dim)
        return video_feat
