import torch.nn as nn

from ..bricks import AffineDropPath, Scale
from ..builder import DETECTORS
from .detector import Detector


@DETECTORS.register_module()
class DyFADet(Detector):
    def __init__(self, projection, rpn_head, neck=None, backbone=None):
        super().__init__(
            backbone=backbone,
            neck=neck,
            projection=projection,
            rpn_head=rpn_head,
        )

    def get_optim_groups(self, cfg):
        # separate out all parameters that with / without weight decay
        # see https://github.com/karpathy/minGPT/blob/master/mingpt/model.py#L134
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, nn.Conv1d)
        blacklist_weight_modules = (nn.LayerNorm, nn.GroupNorm)

        # loop over all modules / params
        for mn, m in self.named_modules():
            for pn, _ in m.named_parameters():
                fpn = f"{mn}.{pn}" if mn else pn  # full param name

                # exclude the backbone parameters
                if fpn.startswith("backbone"):
                    continue

                if pn.endswith("bias"):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith("weight") and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif pn.endswith("scale") and isinstance(m, (Scale, AffineDropPath)):
                    # corner case of our scale layer
                    no_decay.add(fpn)
                elif pn.endswith("scale_weight"):
                    # corner case for relative position encoding
                    no_decay.add(fpn)
                elif pn.endswith("output_weight"):
                    # corner case for relative position encoding
                    no_decay.add(fpn)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters() if not pn.startswith("backbone")}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, f"parameters {inter_params} made it into both decay/no_decay sets!"
        assert len(param_dict.keys() - union_params) == 0, (
            f"parameters {param_dict.keys() - union_params} were not separated into either decay/no_decay set!"
        )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": cfg["weight_decay"]},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups
