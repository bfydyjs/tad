from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import interpolate

from tad.datasets.builder import TRANSFORMS


@TRANSFORMS.register_module()
class LoadFeats:
    def __init__(self, feat_format, prefix="", suffix=""):
        self.feat_format = feat_format
        self.prefix = prefix
        self.suffix = suffix
        # check feat format
        if isinstance(self.feat_format, str):
            self.check_feat_format(self.feat_format)
        elif isinstance(self.feat_format, list):
            for feat_format in self.feat_format:
                self.check_feat_format(feat_format)

    def check_feat_format(self, feat_format):
        assert feat_format in ["npy", "npz", "pt"], f"not support {feat_format}"

    def read_from_tensor(self, file_path):
        feats = torch.load(file_path).float().numpy()
        return feats

    def read_from_npy(self, file_path):
        feats = np.load(file_path).astype(np.float32)
        return feats

    def read_from_npz(self, file_path):
        feats = np.load(file_path)["feats"].astype(np.float32)
        return feats

    def load_single_feat(self, file_path, feat_format):
        try:
            if feat_format == "npy":
                feats = self.read_from_npy(file_path)
            elif feat_format == "npz":
                feats = self.read_from_npz(file_path)
            elif feat_format == "pt":
                feats = self.read_from_tensor(file_path)
        except Exception as e:
            raise RuntimeError(f"Missing data: {file_path}, error: {e}") from e
        return feats

    def __call__(self, results):
        video_name = results["video_name"]

        if isinstance(results["data_path"], str):
            file_path = (
                Path(results["data_path"])
                / f"{self.prefix}{video_name}{self.suffix}.{self.feat_format}"
            )
            feats = self.load_single_feat(file_path, self.feat_format)
        elif isinstance(results["data_path"], list):
            feats = []

            # check if the feat_format is a list
            if isinstance(self.feat_format, str):
                self.feat_format = [self.feat_format] * len(results["data_path"])

            for data_path, feat_format in zip(results["data_path"], self.feat_format, strict=True):
                file_path = (
                    Path(data_path) / f"{self.prefix}{video_name}{self.suffix}.{feat_format}"
                )
                feats.append(self.load_single_feat(file_path, feat_format))

            max_len = max([feat.shape[0] for feat in feats])
            for i in range(len(feats)):
                if feats[i].shape[0] != max_len:
                    # assume the first dimension is T
                    tmp_feat = interpolate(
                        torch.Tensor(feats[i]).permute(1, 0).unsqueeze(0),
                        size=max_len,
                        mode="linear",
                        align_corners=False,
                    ).squeeze(0)
                    feats[i] = tmp_feat.permute(1, 0).numpy()
            feats = np.concatenate(feats, axis=1)

        # sample the feature
        sample_stride = results.get("sample_stride", 1)
        if sample_stride > 1:
            feats = feats[::sample_stride]

        results["feats"] = feats
        return results

    def __repr__(self):
        repr_str = f"{self.__class__.__name__}(feat_format={self.feat_format})"
        return repr_str
