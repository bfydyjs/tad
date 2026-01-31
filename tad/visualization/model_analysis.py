# 移动到与库同一目录下

import sys
import torch
from fvcore.nn import FlopCountAnalysis
from typing import Tuple, Union
from pathlib import Path
# Add the parent directory of the 'tad' package to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))
def calculate_flops_params(
    model: torch.nn.Module,
    input_shape: Tuple[int, ...],
    device: str = "cpu"
) -> Tuple[float, int]:
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
    # Add batch dimension = 1
    dummy_input = torch.randn(1, *input_shape, device=device)

    # Create dummy masks (all True, assuming no padding)
    dummy_masks = torch.ones(1, input_shape[-1], device=device).bool()

    # Create dummy metas (empty dict)
    dummy_metas = {}

    # Set model to evaluation mode and move to device
    model.eval()
    model.to(device)

    # Create a simple nn.Module wrapper that directly calls forward_test to ensure all model components are traced
    # class ModelWrapper(torch.nn.Module):
    #     def __init__(self, model):
    #         super().__init__()
    #         self.model = model
    #     def forward(self, inputs, masks, metas):
    #         # Directly call the forward_test method to trace all model components
    #         return self.model.forward_test(inputs, masks, metas, None)
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, inputs, masks, metas):
            # 根据模型类型选择不同的前向传播方法
            model_name = self.model.__class__.__name__
            
            # 首先检查是否有 forward_test 方法（如 TAD 模型）
            if hasattr(self.model, 'forward_test'):
                return self.model.forward_test(inputs, masks, metas, None)
            
            # 检查是否是 DyFADet 类型
            elif model_name == 'DyFADet':
                # Try different approaches for DyFADet model
                try:
                    # First try the original approach with backbone and neck
                    batched_inputs = inputs
                    # Ensure mask has the correct shape [B, 1, T] for interpolate
                    batched_masks = masks.unsqueeze(1)  # Add channel dimension
                    
                    # Forward through backbone and neck only
                    feats, masks = self.model.backbone(batched_inputs, batched_masks)
                    fpn_feats, fpn_masks = self.model.neck(feats, masks)
                    return fpn_feats
                except AttributeError:
                    # If backbone/neck attributes don't exist, try simple forward with inputs
                    # Create video_list format input
                    video_list = [{
                        'feats': inputs.squeeze(0),  # Remove batch dimension
                        'segments': torch.tensor([[0, inputs.shape[-1]]], device=inputs.device),  # Dummy segments
                        'labels': torch.tensor([0], device=inputs.device),  # Dummy label
                        'video_id': 'dummy',
                        'fps': 30.0,
                        'duration': inputs.shape[-1],
                        'feat_stride': 1.0,
                        'feat_num_frames': 1
                    }]
                    return self.model(video_list)
            
            # 检查是否是 Detector 类型
            elif model_name == 'Detector':
                # For Detector model, use forward with masks and metas
                return self.model(inputs, masks, metas)
            
            # 其他模型类型
            else:
                # Try default forward method with different parameter combinations
                try:
                    # First try with all three parameters
                    return self.model(inputs, masks, metas)
                except TypeError:
                    try:
                        # Try with just inputs
                        return self.model(inputs)
                    except TypeError:
                        # Try with inputs and masks
                        return self.model(inputs, masks)

    # Create wrapper instance
    wrapper = ModelWrapper(model)

    # Calculate FLOPs (total for batch=1 → per-sample)
    flops = FlopCountAnalysis(wrapper, (dummy_input, dummy_masks, dummy_metas)).total()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    return flops, params


def format_number(x: Union[int, float]) -> str:
    """Format large numbers to human-readable strings."""
    if x >= 1e9:
        return f"{x / 1e9:.2f}G"
    elif x >= 1e6:
        return f"{x / 1e6:.2f}M"
    elif x >= 1e3:
        return f"{x / 1e3:.2f}K"
    else:
        return f"{x:.2f}"


if __name__ == "__main__":
    from tad.tad.models import build_detector
    from tad.tad.utils import Config
    config_file = Path(__file__).resolve().parent / "tad" / "configs" / "ddiou" / "thumos_videomaev2_g.yaml"
    print(config_file)
    cfg = Config.fromfile(config_file)
    model = build_detector(cfg.model)
    flops, params = calculate_flops_params(model, input_shape=(1408, 2304))
    # Parameter counting requires only the model architecture, not input data.
    print(f"FLOPs: {format_number(flops)}")
    print(f"Params: {format_number(params)}\n")


    # pip install opentad/models/roi_heads/roi_extractors/align1d --no-build-isolation
    # pip install opentad/models/roi_heads/roi_extractors/boundary_pooling --no-build-isolation
    from OpenTAD.opentad.models import build_detector
    from mmengine.config import Config
    config_file = Path(__file__).resolve().parent / "OpenTAD" / "configs" / "dyfadet" / "thumos_videomaev2_g.py"
    print(config_file)
    cfg = Config.fromfile(config_file)
    model = build_detector(cfg.model)
    flops, params = calculate_flops_params(model, input_shape=(1408, 2304))
    # Parameter counting requires only the model architecture, not input data.
    print(f"FLOPs: {format_number(flops)}")
    print(f"Params: {format_number(params)}\n")

    
    # rename DyFADet_pytorch to DyFADet-pytorch
    from DyFADet_pytorch.libs.modeling import make_meta_arch
    from DyFADet_pytorch.libs.core import load_config
    config_file = Path(__file__).resolve().parent / "DyFADet_pytorch" / "configs" / "thumos_mae.yaml"
    print(config_file)
    cfg = load_config(config_file)
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    flops, params = calculate_flops_params(model, input_shape=(1408, 2304))
    # Parameter counting requires only the model architecture, not input data.
    print(f"FLOPs: {format_number(flops)}")
    print(f"Params: {format_number(params)}\n")

    