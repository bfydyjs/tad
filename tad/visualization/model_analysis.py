import sys
import torch
from fvcore.nn import FlopCountAnalysis
from typing import Tuple, Union
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

def calculate_flops(
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

    # Calculate FLOPs (total for batch=1 → per-sample)
    flops = FlopCountAnalysis(model, (dummy_input, dummy_masks, dummy_metas)).total()
    return flops


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
    from tad.models import build_detector
    from tad.utils import update_workdir, set_seed, create_folder, setup_logger, Config, DictAction
    config_file = Path(__file__).resolve().parent.parent.parent / "configs" / "ddiou" / "thumos_videomaev2_g.yaml"
    print(config_file)
    cfg = Config.fromfile(config_file)
    model = build_detector(cfg.model)
    flops = calculate_flops(model, input_shape=(1408, 2304))
    # Parameter counting requires only the model architecture, not input data.
    params = sum(p.numel() for p in model.parameters() if p.requires_grad) 
    print(f"FLOPs: {format_number(flops)}")
    print(f"Params: {format_number(params)}\n")