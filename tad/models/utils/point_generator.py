import torch


class PointGenerator:
    def __init__(
        self,
        strides,  # strides of fpn levels
        regression_range,  # regression range (on feature grids)
        use_offset=False,  # if to align the points at grid centers
    ):
        super().__init__()
        self.strides = strides
        self.regression_range = regression_range
        self.use_offset = use_offset

    def __call__(self, feat_list):
        # feat_list: list[B,C,T]

        pts_list = []
        for i, feat in enumerate(feat_list):
            t = feat.shape[-1]
            device = feat.device

            points = torch.arange(t, dtype=torch.float, device=device) * self.strides[i]

            if self.use_offset:
                points += 0.5 * self.strides[i]

            points = points[:, None]  # [T, 1]
            reg_range = torch.tensor(self.regression_range[i], device=device).expand(t, 2)
            stride = torch.tensor(self.strides[i], device=device).expand(t, 1)

            pts_list.append(torch.cat((points, reg_range, stride), dim=1))  # [T, 4]
        return pts_list
