"""Visualize cosine similarity heatmaps from temporal features.

Usage:
    python tad/visualization/analysis/cosine_similarity_heatmap.py \
        configs/ddiou/thumos_videomaev2_g.yaml \
        exps/thumos/videomaev2_g/gpu1_id1/checkpoint/best.pt

    python -m tad.visualization.analysis.cosine_similarity_heatmap \
        configs/ddiou/thumos_videomaev2_g.yaml \
        exps/thumos/videomaev2_g/gpu1_id1/checkpoint/best.pt
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from matplotlib.patches import Rectangle

from tad.datasets import build_dataset
from tad.models import build_detector
from tad.utils import Config
from tad.visualization.plot.setup_paper_style import setup_paper_style

# Add project root to sys.path to allow absolute imports
_project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(_project_root))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize Cosine Similarity Heatmap from Real Model Features"
    )
    parser.add_argument("config", help="Path to config file (e.g., configs/anet_i3d.yaml)")
    parser.add_argument("checkpoint", help="Path to checkpoint file (e.g., work_dirs/xxx/best.pt)")
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Index of the video sample in validation set to visualize",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help=("0 -> raw input features ; 1..N select FPN levels from model outputs."),
    )
    return parser.parse_args()


def _patch_config_for_gt(cfg):
    """Forcefully enable GT loading for the validation dataset."""
    print("Patching config to enable GT loading...")
    cfg.dataset.val.test_mode = False
    for transform in cfg.dataset.val.pipeline:
        if transform["type"] == "ConvertToTensor" and "gt_segments" not in transform["keys"]:
            transform["keys"].append("gt_segments")
        if transform["type"] == "Collect" and "gt_segments" not in transform["keys"]:
            transform["keys"].append("gt_segments")
    return cfg


def _get_feature_stride(cfg):
    """Get feature stride from config."""
    if "common" in cfg.dataset and "feature_stride" in cfg.dataset.common:
        return cfg.dataset.common.feature_stride
    if "feature_stride" in cfg.dataset.val:
        return cfg.dataset.val.feature_stride
    return 1


def _extract_features(args, model, inputs, masks):
    """Extract features based on the specified level."""
    if args.level == 0:
        print("Using RAW INPUT features (level=0).")
        return inputs[0]  # [C, T]

    print(f"Extracting MODEL OUTPUT features for level {args.level}...")
    with torch.no_grad():
        feats, _ = model.extract_feat(inputs, masks)

    if isinstance(feats, (list, tuple)):
        level_idx = args.level - 1  # map 1..N -> 0..N-1
        if not (0 <= level_idx < len(feats)):
            raise ValueError(
                f"Requested level {args.level} is out of range. Model returned {len(feats)} levels."
            )
        print(
            f"Model returned {len(feats)} feature levels. "
            f"Selecting level {args.level} (index {level_idx})."
        )
        feature_tensor = feats[level_idx]
    else:
        if args.level != 1:
            raise ValueError(
                f"Requested level {args.level}, but model is single-scale. "
                "Use --level 0 for inputs or --level 1 for output."
            )
        feature_tensor = feats

    return feature_tensor[0] if feature_tensor.dim() == 3 else feature_tensor


def plot_heatmap(similarity_matrix, gt_intervals_indices, seconds_per_step, video_name):
    """Plot and save the heatmap and timeline."""
    t = similarity_matrix.shape[0]
    setup_paper_style(
        440 / 2, ratio=1.1, fraction=0.98, font_size_tex=10, font_size_main=7, line_width_axis=0.5
    )
    print("Plotting heatmap...")
    fig = plt.figure()
    gs = fig.add_gridspec(
        2, 2, width_ratios=[50, 1], height_ratios=[20, 1], wspace=0.02, hspace=0.05
    )

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    cbar_ax = fig.add_subplot(gs[0, 1])

    # Heatmap
    sns.heatmap(similarity_matrix, cmap="viridis", ax=ax1, cbar_ax=cbar_ax)
    cbar_ax.tick_params(which="both", length=0)

    # Ticker setup
    import matplotlib.ticker as ticker

    def index_to_seconds(x, pos):
        return f"{x * seconds_per_step:.1f}"

    locator = ticker.MaxNLocator(nbins=10, integer=True)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(index_to_seconds))
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10, integer=True))

    # Ticks and labels
    ax1.tick_params(axis="both", which="both", length=0)
    ax2.tick_params(axis="both", which="both", length=0)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=True)
    ax1.set_ylabel("Time (s)")

    # GT rectangles on heatmap
    for start, end in gt_intervals_indices:
        start, end = max(0, start), min(t, end)
        if end > start:
            ax1.add_patch(
                Rectangle(
                    (start, start),
                    end - start,
                    end - start,
                    linewidth=1,
                    edgecolor="#FF3333",
                    facecolor="none",
                )
            )

    # Timeline bar
    ax2.set_xlim(0, t)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("GT", rotation=0, labelpad=10, va="center")
    ax2.set_yticks([])
    for start, end in gt_intervals_indices:
        start, end = max(0, start), min(t, end)
        ax2.fill_between([start, end], 0, 1, color="#32CD32", alpha=0.8)

    # Save figure
    output_path = _project_root / "output" / "figures" / "heatmap.pdf"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Saving figure to: {output_path}")


def main():
    args = parse_args()
    # Device selection: use GPU if available else CPU (no CLI option)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # 1. 加载配置并打补丁
    cfg = Config.fromfile(args.config)
    cfg = _patch_config_for_gt(cfg)

    # 2. 构建数据集
    print("Building dataset...")
    dataset = build_dataset(cfg.dataset.val)
    print(f"Dataset loaded with {len(dataset)} samples.")

    # 3. 构建模型 (如果需要)
    model = None
    if args.level != 0:
        print(f"Building model and loading checkpoint from {args.checkpoint}...")
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: checkpoint file not found: {checkpoint_path}")
            print("Tip: provide a valid path or set --level 0 to use raw inputs.")
            sys.exit(1)
        model = build_detector(cfg.model)
        print("Loading checkpoint with weights_only=False (UNSAFE)...")
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

    # 4. 获取样本数据
    print(f"Processing sample index: {args.index}")
    data_sample = dataset[args.index]
    inputs = data_sample["inputs"].to(device).unsqueeze(0)
    masks = data_sample["masks"].to(device).unsqueeze(0)
    metas_obj = data_sample.get("metas", {})
    metas = metas_obj.data if hasattr(metas_obj, "data") else metas_obj
    video_name = metas.get("video_name", f"sample_{args.index}")

    # 5. 提取特征
    feature_tensor = _extract_features(args, model, inputs, masks)
    features = feature_tensor.transpose(0, 1).cpu().numpy()
    print(f"Feature shape for heatmap: Time={features.shape[0]}, Dim={features.shape[1]}")

    # 6. 计算相似度矩阵
    print("Computing cosine similarity...")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features_norm = features / (norms + 1e-8)
    similarity_matrix = np.dot(features_norm, features_norm.T)

    # 7. 计算GT和时间缩放
    gt_segments = data_sample["gt_segments"].cpu().numpy()
    feature_stride = _get_feature_stride(cfg)
    fps = metas.get("fps")
    seconds_per_step = feature_stride / fps if fps else 1.0
    if not fps:
        print("Warning: FPS not found. Assuming 1:1 mapping (Index=Seconds).")
    gt_intervals_indices = gt_segments / seconds_per_step
    print(f"Video: {video_name}, FPS: {fps}, Stride: {feature_stride}")

    # 8. 绘图
    plot_heatmap(similarity_matrix, gt_intervals_indices, seconds_per_step, video_name)


if __name__ == "__main__":
    main()
