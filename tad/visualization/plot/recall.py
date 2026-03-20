"""Visualize cosine similarity heatmaps from temporal features.

Requires setting save_dict to true in YAML config, then running eval.py
to generate result_detection.json file.

Usage:

1. for Linux/Mac:
python -m tad.visualization.plot.recall \
    --ground-truth-file data/thumos-14/annotations/thumos_14_anno.json \
    --prediction-file exps/thumos/videomaev2_g/gpu1_id0/result_detection.json \
    --subset validation \
    --tiou-thresholds 0.3,0.4,0.5,0.6,0.7

2. for Windows PowerShell:
python -m tad.visualization.plot.recall `
    --ground-truth-file data/thumos-14/annotations/thumos_14_anno.json `
    --prediction-file exps/thumos/videomaev2_g/gpu1_id0/result_detection.json `
    --subset validation `
    --tiou-thresholds 0.3,0.4,0.5,0.6,0.7
"""

import argparse

import matplotlib.pyplot as plt

from tad.metrics.recall import Recall
from tad.visualization.utils import save_figure, setup_paper_style


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot recall curves from ground truth and prediction json files"
    )
    parser.add_argument(
        "--ground-truth-file", type=str, required=True, help="Ground truth json file"
    )
    parser.add_argument("--prediction-file", type=str, required=True, help="Prediction json file")
    parser.add_argument(
        "--subset", type=str, required=True, help="Evaluation subset, e.g. validation/test"
    )
    parser.add_argument(
        "--tiou-thresholds",
        type=parse_float_list,
        required=True,
        help="Comma-separated tIoU thresholds, e.g. 0.3,0.4,0.5,0.6,0.7",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. Default: <project_root>/output/figures",
    )
    parser.add_argument("--show", action="store_true", help="Show figures interactively")
    return parser.parse_args()


def parse_float_list(value):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def plot_ar_an_curve(evaluator):
    """Plot Average Recall vs Average Number of proposals curve.

    The curve shows average recall over all tIoU thresholds as a function
    of the average number of proposals per video.
    """
    setup_paper_style(
        440 / 2,
        ratio=1.618,
        fraction=0.98,
        font_size_tex=5,
        font_size_main=4.5,
        line_width_axis=0.5,
    )
    fig, ax = plt.subplots()
    ax.plot(
        evaluator.proposals_per_video,
        evaluator.avg_recall,
        label=f"Average Recall (AUC={evaluator.average_auc * 100:.2f}%)",
        color="tab:blue",
    )
    ax.set_xlabel("Average Number of Proposals per Video")
    ax.set_ylabel("Average Recall")
    ax.set_title("AR-AN Curve")
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()

    return fig


def plot_recall_k_tiou(evaluator):
    """Plot Recall@K vs tIoU thresholds for different K values.

    Each curve shows recall at a specific K (number of top proposals)
    across different tIoU thresholds.
    """
    setup_paper_style(
        textwidth=440,
        ratio=1.618,
        fraction=0.98,
        font_size_tex=5,
        font_size_main=4.5,
        line_width_axis=0.5,
    )
    fig, ax = plt.subplots()

    num_points = evaluator.recall.shape[1]
    for k in evaluator.topk:
        if k < 1 or k > num_points:
            print(f"Skip Recall@{k}: valid range is [1, {num_points}]")
            continue
        # recall[i, j] is recall at ith tiou threshold and jth proposal count
        # k-1 index corresponds to the k-th proposal count (0-indexed)
        recall_values = evaluator.recall[:, k - 1]
        ax.plot(evaluator.tiou_thresholds, recall_values, marker="o", label=f"Recall@{k}")

    ax.set_xlabel("tIoU Threshold")
    ax.set_ylabel("Recall")
    ax.set_title("Recall@K vs tIoU")
    ax.set_ylim(0, 1.05)
    ax.legend()
    return fig


def load_recall_data(ground_truth_file, prediction_file, subset, tiou_thresholds):
    """Load and evaluate recall data from files.

    Args:
        ground_truth_file: Path to ground truth JSON file
        prediction_file: Path to prediction JSON file
        subset: Evaluation subset name
        tiou_thresholds: List of tIoU thresholds

    Returns:
        Recall evaluator object
    """
    evaluator = Recall(
        ground_truth_file=ground_truth_file,
        prediction_file=prediction_file,
        subset=subset,
        tiou_thresholds=tiou_thresholds,
    )
    evaluator.evaluate()
    return evaluator


def plot_recall_from_files(
    ground_truth_file,
    prediction_file,
    subset,
    tiou_thresholds,
    output_dir=None,
    show=False,
):
    """Plot recall curves from ground truth and prediction files.

    Args:
        ground_truth_file: Path to ground truth JSON file
        prediction_file: Path to prediction JSON file
        subset: Evaluation subset name
        tiou_thresholds: List of tIoU thresholds
        output_dir: Output directory for saving plots
        show: Whether to display plots interactively
    """
    evaluator = load_recall_data(ground_truth_file, prediction_file, subset, tiou_thresholds)

    fig_ar = plot_ar_an_curve(evaluator)
    try:
        save_figure("recall_ar_an_curve", output_dir=output_dir, fig=fig_ar)
        if show:
            plt.show()
    finally:
        plt.close(fig_ar)

    fig_k = plot_recall_k_tiou(evaluator)
    try:
        save_figure("recall_k_tiou_curve", output_dir=output_dir, fig=fig_k)
        if show:
            plt.show()
    finally:
        plt.close(fig_k)


def main():
    """Main entry point for recall visualization."""
    import sys

    if len(sys.argv) > 1:
        args = parse_args()
        plot_recall_from_files(
            ground_truth_file=args.ground_truth_file,
            prediction_file=args.prediction_file,
            subset=args.subset,
            tiou_thresholds=args.tiou_thresholds,
            output_dir=args.output_dir,
            show=args.show,
        )
    else:
        print("\nUsage: python -m tad.visualization.plot.recall \\")
        print("    --ground-truth-file <gt.json> \\")
        print("    --prediction-file <pred.json> \\")
        print("    --subset validation/test \\")
        print("    --tiou-thresholds 0.3,0.4,0.5,0.6,0.7")
        print()


if __name__ == "__main__":
    main()
