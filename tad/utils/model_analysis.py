# 移动到与库同一目录下

import sys
import torch
from fvcore.nn import FlopCountAnalysis
from typing import Tuple, Union
from pathlib import Path
from typing import Tuple, Optional

# Calculate Params and GFLOPs
def calculate_params_gflops(
    model: torch.nn.Module,
    cfg = None,
    device: str = "cpu"
) -> Tuple[int, Optional[float]]:
    """
    Calculate per-sample FLOPs and total trainable parameters of a model.

    Args:
        model: PyTorch model.
        input_shape: Input shape tuple (e.g., (3, 224, 224) for image, (2048, 100) for video features).
                     Batch dimension is NOT included.
        device: Device to run dummy forward pass ('cpu' or 'cuda').

    Returns:
        flops_per_sample: FLOPs for one sample (float)
        params: Number of trainable parameters (int)
    """

    params = sum(p.numel() for p in model.parameters() if p.requires_grad) 


    input_shape = (cfg.model.projection.in_channels, cfg.model.projection.max_seq_len)

    dummy_metas = {}

    gflops = None

    return params, gflops