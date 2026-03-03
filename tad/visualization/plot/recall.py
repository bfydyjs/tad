# python eval.py configs/ddiou/thumos_videomaev2_g.yaml --checkpoint
# exps/thumos/videomaev2_g/gpu1_id0/checkpoint/best.pt --plot-recall
import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from tad.metrics.recall import Recall

from .setup_paper_style import setup_paper_style


def parse_args():
    parser = argparse.ArgumentParser(description="Plot recall curves from ground truth and prediction json files")
    parser.add_argument("--ground-truth-file", type=str, required=True, help="Ground truth json file")
    parser.add_argument("--prediction-file", type=str, required=True, help="Prediction json file")
    parser.add_argument("--subset", type=str, required=True, help="Evaluation subset, e.g. validation/test")
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


def resolve_output_dir(output_dir):
    if output_dir is not None:
        out_dir = Path(output_dir)
    else:
        project_root = Path(__file__).resolve().parents[3]
        out_dir = project_root / "output" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def plot_ar_an_curve(evaluator, out_dir):
    setup_paper_style(440 / 2, ratio=1.618, fraction=0.98, font_size_tex=5, font_size_main=4.5, line_width_axis=0.5)
    plt.figure()
    plt.plot(
        evaluator.proposals_per_video,
        evaluator.avg_recall,
        label=f"Average Recall (AUC={evaluator.average_auc * 100:.2f}%)",
        color="tab:blue",
    )
    plt.xlabel("Average Number of Proposals per Video")
    plt.ylabel("Average Recall")
    plt.title("AR-AN Curve")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()

    output_path = out_dir / "recall_ar_an_curve.pdf"
    plt.savefig(output_path)
    print(f"Saved: {output_path}")


def plot_recall_k_tiou(evaluator, out_dir):
    setup_paper_style(440, ratio=1.618, fraction=0.98, font_size_tex=5, font_size_main=4.5, line_width_axis=0.5)
    plt.figure()

    num_points = evaluator.recall.shape[1]
    for k in evaluator.topk:
        if k < 1 or k > num_points:
            print(f"Skip Recall@{k}: valid range is [1, {num_points}]")
            continue
        recall_values = evaluator.recall[:, k - 1]
        plt.plot(evaluator.tiou_thresholds, recall_values, marker="o", label=f"Recall@{k}")

    plt.xlabel("tIoU Threshold")
    plt.ylabel("Recall")
    plt.title("Recall@K vs tIoU")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.tight_layout()

    output_path = out_dir / "recall_k_tiou_curve.pdf"
    plt.savefig(output_path)
    print(f"Saved: {output_path}")


def plot_recall_from_files(
    ground_truth_file,
    prediction_file,
    subset,
    tiou_thresholds,
    output_dir=None,
    show=False,
):
    evaluator = Recall(
        ground_truth_file=ground_truth_file,
        prediction_file=prediction_file,
        subset=subset,
        tiou_thresholds=tiou_thresholds,
    )
    evaluator.evaluate()

    out_dir = resolve_output_dir(output_dir)
    plot_ar_an_curve(evaluator, out_dir)
    plot_recall_k_tiou(evaluator, out_dir)

    if show:
        plt.show()
    else:
        plt.close("all")

    return out_dir


def maybe_plot_recall(cfg, args, logger):
    if not args.plot_recall:
        return

    result_path = Path(cfg.work_dir) / "result_detection.json"
    if not result_path.exists():
        logger.warning(f"Skip recall plotting: prediction file not found: {result_path}")
        return

    if "evaluation" not in cfg:
        logger.warning("Skip recall plotting: config missing evaluation section")
        return

    evaluation_cfg = cfg.evaluation
    required_keys = ["ground_truth_file", "subset", "tiou_thresholds"]
    missing_keys = [key for key in required_keys if key not in evaluation_cfg]
    if missing_keys:
        logger.warning(f"Skip recall plotting: missing evaluation keys: {missing_keys}")
        return

    try:
        out_dir = plot_recall_from_files(
            ground_truth_file=evaluation_cfg.ground_truth_file,
            prediction_file=str(result_path),
            subset=evaluation_cfg.subset,
            tiou_thresholds=evaluation_cfg.tiou_thresholds,
            output_dir=args.plot_output_dir,
            show=False,
        )
        logger.info(f"Recall plots saved to: {out_dir}")
    except Exception as exc:
        logger.warning(f"Failed to plot recall curves: {exc}")


def main():
    args = parse_args()

    plot_recall_from_files(
        ground_truth_file=args.ground_truth_file,
        prediction_file=args.prediction_file,
        subset=args.subset,
        tiou_thresholds=args.tiou_thresholds,
        output_dir=args.output_dir,
        show=args.show,
    )


if __name__ == "__main__":
    main()
