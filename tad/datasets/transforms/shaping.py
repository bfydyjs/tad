import warnings

import numpy as np
import scipy
import torch
import torchvision
from torch.nn.functional import interpolate

from ..builder import TRANSFORMS


@TRANSFORMS.register_module()
class ResizeFeat:
    def __init__(self, tool, channel_first=False):
        self.tool = tool
        self.channel_first = channel_first

    @torch.no_grad()
    def torchvision_align(self, feat, tscale):
        # input feat shape [c,t]
        pseudo_input = feat.unsqueeze(0).unsqueeze(3)  # [1,c,t,1]
        pseudo_bbox = torch.Tensor([[0, 0, 0, 1, feat.shape[1]]])
        # output feat shape [c,tscale]
        output = torchvision.ops.roi_align(
            pseudo_input.half().double(),
            pseudo_bbox.half().double(),
            output_size=(tscale, 1),
            aligned=True,
        ).to(pseudo_input.dtype)
        output = output.squeeze(0).squeeze(-1)
        return output

    @torch.no_grad()
    def gtad_align(self, feat):
        raise NotImplementedError("Not implemented yet")

    @torch.no_grad()
    def bmn_align(self, feat, tscale, num_bin=1, num_sample_bin=3, pool_type="mean"):
        feat = feat.numpy()
        c, t = feat.shape

        # x is the temporal location corresponding to each location  in feature sequence
        x = [0.5 + ii for ii in range(t)]
        f = scipy.interpolate.interp1d(x, feat, axis=1)

        video_feature = []
        zero_sample = np.zeros(num_bin * c)
        tmp_anchor_xmin = [1.0 / tscale * i for i in range(tscale)]
        tmp_anchor_xmax = [1.0 / tscale * i for i in range(1, tscale + 1)]

        num_sample = num_bin * num_sample_bin
        for idx in range(tscale):
            xmin = max(x[0] + 0.0001, tmp_anchor_xmin[idx] * t)
            xmax = min(x[-1] - 0.0001, tmp_anchor_xmax[idx] * t)
            if xmax < x[0]:
                video_feature.append(zero_sample)
                continue
            if xmin > x[-1]:
                video_feature.append(zero_sample)
                continue

            plen = (xmax - xmin) / (num_sample - 1)
            x_new = [xmin + plen * ii for ii in range(num_sample)]
            y_new = f(x_new)
            y_new_pool = []
            for b in range(num_bin):
                tmp_y_new = y_new[:, num_sample_bin * b : num_sample_bin * (b + 1)]
                if pool_type == "mean":
                    tmp_y_new = np.mean(tmp_y_new, axis=1)
                elif pool_type == "max":
                    tmp_y_new = np.max(tmp_y_new, axis=1)
                y_new_pool.append(tmp_y_new)
            y_new_pool = np.stack(y_new_pool, axis=1).reshape(-1)
            # y_new_pool = np.reshape(y_new_pool, [-1])
            video_feature.append(y_new_pool)
        video_feature = np.stack(video_feature, axis=1)
        return torch.from_numpy(video_feature)

    @torch.no_grad()
    def torch_interpolate(self, feat, tscale):
        # input feat shape [c,t]
        feats = interpolate(
            feat.unsqueeze(0), size=tscale, mode="linear", align_corners=False
        ).squeeze(0)
        return feats

    def __call__(self, results):
        assert "resize_length" in results.keys(), "should have resize_length as a key"
        tscale = results["resize_length"]

        if not self.channel_first:
            feats = results["feats"].permute(1, 0)  # [t,c] -> [c,t]
        else:
            feats = results["feats"]

        assert isinstance(feats, torch.Tensor)
        assert feats.ndim == 2  # [c,t]

        if self.tool == "torchvision_align":
            resized_feat = self.torchvision_align(feats, tscale)
        elif self.tool == "gtad_align":
            resized_feat = self.gtad_align(feats, tscale)
        elif self.tool == "bmn_align":
            resized_feat = self.bmn_align(feats, tscale)
        elif self.tool == "interpolate":
            resized_feat = self.torch_interpolate(feats, tscale)

        assert resized_feat.shape[0] == feats.shape[0]
        assert resized_feat.shape[1] == tscale

        if "gt_segments_feat" in results:
            # convert gt seconds to feature grid
            # clamp using 1e-8 to avoid division by zero
            dur = max(results["duration"], 1e-8)
            results["gt_segments_feat"] = (results["gt_segments_feat"] / dur).clamp(
                min=0.0, max=1.0
            )
            results["gt_segments_feat"] *= tscale

        results["feats_len_ori"] = results["feats"].shape[1]  # for future usage
        if not self.channel_first:
            results["feats"] = resized_feat.permute(1, 0)  # [c,t] -> [t,c]
        else:
            results["feats"] = resized_feat
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(tool='{self.tool}', channel_first={self.channel_first})"


@TRANSFORMS.register_module()
class Padding:
    def __init__(self, length, pad_value=0, channel_first=False):
        self.length = length
        self.pad_value = pad_value
        self.channel_first = channel_first

    def __call__(self, results):
        assert "feats" in results.keys(), "should have feats as a key"
        assert results["feats"].ndim == 2, "feats should be 2 dim"

        if self.channel_first:
            feats = results["feats"].permute(1, 0)
        else:
            feats = results["feats"]

        feat_len = feats.shape[0]
        if feat_len < self.length:
            pad = torch.ones((self.length - feat_len, feats.shape[1])) * self.pad_value
            new_feats = torch.cat((feats, pad), dim=0)

            if self.channel_first:
                results["feats"] = new_feats.permute(1, 0)
            else:
                results["feats"] = new_feats

            pad_masks = torch.zeros(self.length - feat_len).bool()
            if "masks" in results:
                results["masks"] = torch.cat((results["masks"], pad_masks), dim=0)
            else:
                results["masks"] = torch.cat((torch.ones(feat_len).bool(), pad_masks), dim=0)
        else:
            warnings.warn(
                f"Feature length {feat_len} is larger than padding length. "
                f"Will be resized to {self.length}.",
                stacklevel=2,
            )
            results["snippet_stride"] = results["snippet_stride"] * feat_len / self.length
            results["offset_frames"] = results["offset_frames"] * feat_len / self.length
            new_feats = interpolate(
                feats.permute(1, 0)[
                    None
                ].float(),  # [b,c,t] Cast to float for stability in linear mode
                size=self.length,
                mode="linear",
                align_corners=False,
            ).squeeze(0)
            # new_feats [c,t]
            results["feats"] = new_feats if self.channel_first else new_feats.permute(1, 0)
            results["masks"] = torch.ones(self.length).bool()
        return results

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"length={self.length}, "
            f"pad_value={self.pad_value}, "
            f"channel_first={self.channel_first})"
        )


@TRANSFORMS.register_module()
class ChannelReduction:
    """Select features along the channel dimension."""

    def __init__(self, in_channels, index):
        self.in_channels = in_channels
        self.index = index
        assert len(self.index) == 2

    def __call__(self, results):
        assert isinstance(results["feats"], torch.Tensor)
        assert results["feats"].shape[1] == self.in_channels  # [t,c]

        # select the features
        results["feats"] = results["feats"][:, self.index[0] : self.index[1]]
        return results

    def __repr__(self):
        return f"{self.__class__.__name__}(in_channels={self.in_channels}, index={self.index})"
