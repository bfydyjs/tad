"""Visualize cosine similarity heatmaps from temporal features.
Usage:

1. for Linux/Mac:
python -m tad.visualization.plot.cosine_similarity_heatmap \
    configs/ddiou/thumos_videomaev2_g.yaml \
    exps/thumos/videomaev2_g/gpu1_id0/checkpoint/best.pt

2. for Windows PowerShell:
python -m tad.visualization.plot.cosine_similarity_heatmap `
    configs/ddiou/thumos_videomaev2_g.yaml `
    exps/thumos/videomaev2_g/gpu1_id0/checkpoint/best.pt
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
from tad.visualization.utils import save_figure, setup_paper_style


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
    keys_to_add = ["gt_segments_feat", "gt_segments_s", "gt_segments_frame"]
    for transform in cfg.dataset.val.pipeline:
        if transform["type"] == "ConvertToTensor":
            for k in keys_to_add:
                if k not in transform["keys"]:
                    transform["keys"].append(k)
        if transform["type"] == "Collect":
            for k in keys_to_add:
                if k not in transform["keys"]:
                    transform["keys"].append(k)
    return cfg


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
        feature_tensor = feats[level_idx]
        print(
            f"Model returned {len(feats)} feature levels. "
            f"Selecting level {args.level} (index {level_idx})."
        )
    else:
        if args.level != 1:
            raise ValueError(
                f"Requested level {args.level}, but model is single-scale. "
                "Use --level 0 for inputs or --level 1 for output."
            )
        feature_tensor = feats

    return feature_tensor[0] if feature_tensor.dim() == 3 else feature_tensor


def plot_heatmap(
    args, similarity_matrix, gt_segments, snippet_stride, offset_frames, fps, video_name
):
    """Plot and save the heatmap and timeline."""
    t = similarity_matrix.shape[0]
    setup_paper_style(
        440 / 2,
        ratio=1.1,
        fraction=0.98,
        font_size_tex=10,
        font_size_main=7,
        line_width_axis=0.5,
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

    max_sec = (t * snippet_stride + offset_frames) / fps

    if max_sec < 10:
        locator = ticker.MultipleLocator(base=2)
        sec_ticks = locator.tick_values(0, max_sec)
    else:
        target_step = max_sec / 5.0
        if max_sec <= 50:
            step = max(5, round(target_step / 5.0) * 5)
        else:
            step = max(10, round(target_step / 10.0) * 10)

        locator = ticker.MultipleLocator(base=step)
        sec_ticks = locator.tick_values(0, max_sec)

    sec_ticks = [s for s in sec_ticks if 0 <= (s * fps - offset_frames) / snippet_stride <= t]
    x_ticks = [(s * fps - offset_frames) / snippet_stride for s in sec_ticks]

    ax1.set_xticks(x_ticks)
    ax1.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{round((x * snippet_stride + offset_frames) / fps)}")
    )
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10, integer=True))

    # Ticks and labels
    ax1.tick_params(axis="both", which="both", length=0)
    ax2.tick_params(axis="x", which="both", length=1)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.setp(ax2.get_xticklabels(), visible=True)
    ax1.set_ylabel("Time (s)")

    # GT rectangles on heatmap
    print(f"Heatmap size (timesteps): {t}")
    for i, (start, end) in enumerate(gt_segments):
        start = int(max(0, min(t, start)))
        end = int(max(0, min(t, end)))
        if end > start:
            print(f"  GT #{i + 1}: Drawing rectangle at [{start}, {end}] (span={end - start})")
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
        else:
            print(f"  GT #{i + 1}: SKIPPED (start={start}, end={end}, invalid span)")

    # Timeline bar
    ax2.set_xlim(0, t)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("GT", rotation=0, labelpad=10, va="center")
    ax2.set_yticks([])
    for start, end in gt_segments:
        start = int(max(0, min(t, start)))
        end = int(max(0, min(t, end)))
        if end > start:
            ax2.fill_between([start, end], 0, 1, color="#32CD32", alpha=0.8)

    return fig


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
    gt_segments_feat = data_sample["gt_segments_feat"]  # [N_gt, 2]
    metas = data_sample.get("metas", {})
    video_name = metas.get("video_name", f"sample_{args.index}")
    data_path = metas.get("data_path", "N/A")
    fps = metas.get("fps", "N/A")
    duration = metas.get("duration", "N/A")
    snippet_stride = metas.get("snippet_stride", "N/A")
    offset_frames = metas.get("offset_frames", "N/A")
    print("data_sample.keys()", data_sample.keys())
    print(f"inputs.shape: {data_sample['inputs'].shape}")
    print(f"masks.shape: {data_sample['masks'].shape}")
    print(f"gt segments shape: {gt_segments_feat.shape}")
    print(f"gt segments: {gt_segments_feat}")
    print(f"metas.keys(): {metas.keys()}")
    print(f"video_name: {video_name}")
    print(f"data_path: {data_path}")
    print(f"fps: {fps}")
    print(f"duration: {duration}")
    print(f"snippet_stride: {snippet_stride}")
    print(f"offset_frames: {offset_frames}")
    # 5. 提取特征
    feature_tensor = _extract_features(args, model, inputs, masks)
    features = feature_tensor.transpose(0, 1).cpu().numpy()
    print(f"Feature shape for heatmap: Time={features.shape[0]}, Dim={features.shape[1]}")

    # 6. 计算相似度矩阵
    print("Computing cosine similarity...")
    # Normalize features
    features_norm = features / np.linalg.norm(features, axis=1, keepdims=True).clip(min=1e-8)
    # Compute cosine similarity matrix [T, T]
    similarity_matrix = np.dot(features_norm, features_norm.T)

    # 7. 计算 GT 和时间缩放
    print(f"Ground truth file: {cfg.evaluation.ground_truth_file}")
    print("=====================================================\n")

    # 8. 绘图
    fig = plot_heatmap(
        args,
        similarity_matrix,
        gt_segments_feat,
        snippet_stride,
        offset_frames,
        fps,
        video_name,
    )
    try:
        save_figure(f"cosine_similarity_heatmap_{args.index}_{args.level}")
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
