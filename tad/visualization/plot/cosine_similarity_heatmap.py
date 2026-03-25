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
        nargs="+",
        default=[0],
        help="Index(es) of the video sample(s) in validation set to visualize. e.g. --index 0 1 2",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="If set, visualize all samples in the dataset, ignoring --index.",
    )
    parser.add_argument(
        "--level",
        type=int,
        default=0,
        help=("0 -> raw input features ; 1..N select FPN levels from model outputs."),
    )
    return parser.parse_args()


def _extract_features(args, model, inputs, masks):
    """Extract features based on the specified level."""
    if args.level == 0:
        print("Using RAW INPUT features (level=0).")
        return inputs[0]  # [C, T]
    with torch.no_grad():
        feats, _ = model.extract_feat(inputs, masks)
    if isinstance(feats, (list, tuple)):
        level_idx = args.level - 1  # map 1..N -> 0..N-1
        feature_tensor = feats[level_idx]
        print(f"[{args.level}/{len(feats)}] (index {level_idx})")
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
    rectangles = []
    for i, (start, end) in enumerate(gt_segments):
        start = int(max(0, min(t, start)))
        end = int(max(0, min(t, end)))
        if end > start:
            rectangles.append((start, end))
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
    print(f"Drawing rectangle at {rectangles}")
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

    # 2. 构建数据集
    dataset = build_dataset(cfg.dataset.val)

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

        # 尝试像 eval.py 一样优先加载 EMA 权重
        use_ema = getattr(cfg.dataloader, "ema", True) if hasattr(cfg, "dataloader") else True
        if use_ema and "state_dict_ema" in checkpoint:
            print("Loading Model EMA weights...")
            state_dict = checkpoint["state_dict_ema"]
        else:
            print("Loading standard model weights...")
            state_dict = checkpoint.get("state_dict", checkpoint)

        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

    # 4. 获取样本数据
    indices_to_process = range(len(dataset)) if args.all else args.index

    for idx in indices_to_process:
        print("================================================================================")
        data_sample = dataset[idx]
        inputs = data_sample["inputs"].to(device).unsqueeze(0)
        masks = data_sample["masks"].to(device).unsqueeze(0)
        metas = data_sample.get("metas", {})

        # Manually extract GTs since test_mode=True bypasses them in pipeline
        video_name = metas.get("video_name", f"sample_{idx}")
        video_info = None
        for name, info, anno in dataset.data_list:
            if name == video_name:
                video_info = info
                break

        if video_info is not None:
            # We must load GT manually here
            video_anno = dataset.get_gt(video_info)
            if video_anno:
                gt_segments_feat = (
                    video_anno["gt_segments_frame"] - dataset.offset_frames
                ) / dataset.snippet_stride
            else:
                gt_segments_feat = []
        else:
            gt_segments_feat = []

        fps = metas.get("fps", "N/A")
        duration = metas.get("duration", "N/A")
        snippet_stride = metas.get("snippet_stride", "N/A")
        offset_frames = metas.get("offset_frames", "N/A")
        print(
            f"[{idx}/{indices_to_process[-1]}] | video_name: {video_name} | fps: {fps} | duration: {duration} | "
            f"snippet_stride: {snippet_stride} | offset_frames: {offset_frames}"
        )
        # print(f"gt segments: {gt_segments_feat}")
        # 5. 提取特征
        feature_tensor = _extract_features(args, model, inputs, masks)  # [C, T]

        # 6. 计算相似度矩阵 (直接在 GPU 上计算更高效)
        with torch.no_grad():
            # 沿通道维度 (dim=0) 归一化
            features_norm = torch.nn.functional.normalize(feature_tensor, p=2, dim=0)
            # 矩阵乘法计算相似度矩阵 [T, C] @ [C, T] -> [T, T]
            similarity_matrix = torch.mm(features_norm.t(), features_norm).cpu().numpy()

        features_shape = feature_tensor.shape
        print(f"Feature shape for heatmap: Time={features_shape[1]}, Dim={features_shape[0]}")

        # 7. 计算 GT 和时间缩放
        print(f"Ground truth file: {cfg.evaluation.ground_truth_file}")
        print("================================================================================")

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
            save_figure(
                f"cosine_similarity_heatmap_{idx}_{args.level}", extensions=["png"], fig=fig
            )
        finally:
            plt.close(fig)


if __name__ == "__main__":
    main()
