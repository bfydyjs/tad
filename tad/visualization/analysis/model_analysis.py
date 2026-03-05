# 移动到与库同一目录下

import sys
import warnings
from pathlib import Path

import torch
from fvcore.nn import FlopCountAnalysis

warnings.filterwarnings("ignore", category=FutureWarning)

# Note on FLOPs vs. MACs:
# Although commonly referred to as "FLOPs", the value returned by fvcore's FlopCountAnalysis
# actually represents MACs (Multiply-Accumulate Operations).
# As stated in the FlopCountAnalysis docstring (line 53):
# "We count one fused multiply-add as one flop."
#
# Both fvcore and thop report MACs, but fvcore is generally preferred because:
#   - It supports a broader range of operations and modules.
#   - It provides more accurate and comprehensive counting.
#   - Its results tend to be slightly higher and closer to real-world computational cost.
#
# Terminology clarification across fields:
#   - In scientific computing / HPC:
#       1 FLOP = 1 floating-point operation (either addition or multiplication),
#       hence 1 MAC (multiply-add) = 2 FLOPs.
#   - In deep learning / computer vision:
#       1 FLOP ≈ 1 MAC (i.e., one fused multiply-add is counted as a single operation).
#
# Therefore, when comparing FLOPs across domains or tools, always verify the underlying definition.


# Add the parent directory of the 'tad' package to Python path
sys.path.insert(0, str(Path(__file__).resolve().parent))


def calculate_flops_params(
    model: torch.nn.Module, input_shape: tuple[int, ...], device: str = "cpu"
) -> tuple[float, int]:
    """
    Calculate per-sample FLOPs and total trainable parameters of a model.

    Args:
        model: PyTorch model.
        input_shape: Input shape tuple (e.g., (3, 224, 224) for image,
                     (2048, 100) for video features).
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

    # Create a simple nn.Module wrapper that directly calls forward_test
    # to ensure all model components are traced
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
            model_name = self.model.__class__.__name__

            if hasattr(self.model, "forward_test"):
                return self.model.forward_test(inputs, masks, metas, None)

            # 检查是否是 DyFADet 类型
            elif model_name == "DyFADet":
                # For DyFADet model, use direct forward through all components
                # This ensures all submodules are called during FLOPs calculation
                try:
                    # Ensure mask has the correct shape [B, 1, T] for interpolate
                    batched_masks = masks.unsqueeze(1)  # Add channel dimension

                    # Forward through backbone and neck
                    feats, masks = self.model.backbone(inputs, batched_masks)
                    fpn_feats, fpn_masks = self.model.neck(feats, masks)

                    # Forward through point generator, cls_head and reg_head
                    points = self.model.point_generator(fpn_feats)
                    out_cls_logits = self.model.cls_head(fpn_feats, fpn_masks)
                    out_offsets = self.model.reg_head(fpn_feats, fpn_masks)

                    return out_cls_logits, out_offsets, points
                except Exception:
                    # If any error occurs, fall back to backbone and neck only
                    batched_masks = masks.unsqueeze(1)
                    feats, masks = self.model.backbone(inputs, batched_masks)
                    fpn_feats, fpn_masks = self.model.neck(feats, masks)
                    return fpn_feats

            elif model_name == "Detector":
                # For Detector model, use forward with masks and metas
                return self.model(inputs, masks, metas)

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
    # Parameter counting requires only the model architecture, not input data.
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return flops, params


if __name__ == "__main__":
    # Define output file path
    output_file = (
        Path(__file__).resolve().parent / "tad" / "output" / "analysis" / "model_complexity.txt"
    )
    # Overwrite the file at start
    with open(output_file, "w") as f:
        pass

    def log_and_print(msg):
        print(msg)
        with open(output_file, "a") as f:
            f.write(msg + "\n")

    from tad.tad.models import build_detector
    from tad.tad.utils import Config

    ddiou_models = [
        ("thumos_i3d.yaml", (2048, 2304)),
        ("thumos_videomaev2_g.yaml", (1408, 2304)),
    ]

    base_ddiou_path = Path(__file__).resolve().parent / "tad" / "configs" / "ddiou"

    for config_name, input_shape in ddiou_models:
        config_file = base_ddiou_path / config_name
        log_and_print(str(config_file))

        cfg = Config.fromfile(config_file)
        model = build_detector(cfg.model)
        flops, params = calculate_flops_params(model, input_shape=input_shape)

        log_and_print(f"DDIOU: {config_name}")
        log_and_print(f"GFLOPs: {flops / 1e9:.2f}")
        log_and_print(f"Params: {params / 1e6:.2f}M\n")

    # pip install opentad/models/roi_heads/roi_extractors/align1d --no-build-isolation
    # pip install opentad/models/roi_heads/roi_extractors/boundary_pooling --no-build-isolation
    from mmengine.config import Config
    from OpenTAD.opentad.models import build_detector

    models_info = [
        ("ActionFormer", "actionformer", "thumos_i3d.py", (2048, 2304)),
        # ("AFSD", "afsd", "thumos_i3d.py", (2048, 2304)),
        # ("BMN", "bmn", "thumos_i3d.py", (2048, 2304)),
        # ("CausalTAD", "causaltad", "thumos_i3d.py", (2048, 2304)),
        ("DyFADet", "dyfadet", "thumos_videomaev2_g.py", (1408, 2304)),
        ("ETAD", "etad", "thumos_i3d.py", (2048, 2304)),
        ("GTAD", "gtad", "thumos_i3d.py", (2048, 2304)),
        # ("TadTR", "tadtr", "thumos_i3d.py", (2048, 2304)),
        ("Temporalmaxer", "temporalmaxer", "thumos_i3d.py", (2048, 2304)),
        ("Tridet", "tridet", "thumos_i3d.py", (2048, 2304)),
        # ("TSI", "tsi", "thumos_i3d.py", (2048, 2304)),
        # ("VSGN", "vsgn", "thumos_i3d.py", (2048, 2304)),
    ]

    base_path = Path(__file__).resolve().parent / "OpenTAD" / "configs"

    for name, folder, config_name, input_shape in models_info:
        config_file = base_path / folder / config_name
        log_and_print(str(config_file))

        cfg = Config.fromfile(config_file)
        model = build_detector(cfg.model)

        device = "cuda" if torch.cuda.is_available() and name == "GTAD" else "cpu"

        flops, params = calculate_flops_params(model, input_shape=input_shape, device=device)

        log_and_print(f"{name}: {config_name}")
        log_and_print(f"GFLOPs: {flops / 1e9:.2f}")
        log_and_print(f"Params: {params / 1e6:.2f}M\n")

    # rename DyFADet_pytorch to DyFADet-pytorch
    from DyFADet_pytorch.libs.core import load_config
    from DyFADet_pytorch.libs.modeling import make_meta_arch

    dyfadet_pt_models = [
        ("thumos_i3d.yaml", (2048, 2304)),
        ("thumos_mae.yaml", (1408, 2304)),
    ]

    base_path = Path(__file__).resolve().parent / "DyFADet_pytorch" / "configs"

    for config_name, input_shape in dyfadet_pt_models:
        config_file = base_path / config_name
        log_and_print(str(config_file))

        cfg = load_config(config_file)
        model = make_meta_arch(cfg["model_name"], **cfg["model"])
        flops, params = calculate_flops_params(model, input_shape=input_shape)

        display_name = config_name.replace(".yaml", "")
    log_and_print(f"DyFADet-pytorch: {display_name}")
    log_and_print(f"GFLOPs: {flops / 1e9:.2f}")
    log_and_print(f"Params: {params / 1e6:.2f}M\n")
