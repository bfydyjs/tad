from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from setup_paper_style import setup_paper_style


def main():
    df = pd.read_csv(r"C:\Users\yanho\Downloads\wandb_export_2026-02-03T22_06_26.613+08_00.csv")
    step_col = "epoch"
    loss_col = "0125_1101 - train/loss"

    if loss_col not in df.columns:
        for col in df.columns:
            if "train/loss" in col:
                loss_col = col
                break
        else:
            raise KeyError("未能找到包含 'train/loss' 的列")

    setup_paper_style(
        textwidth=440 / 2,
        ratio=1.618,
        fraction=0.98,
        font_size_tex=5,
        font_size_main=4.5,
        line_width_axis=0.5,
    )
    plt.figure()
    plt.plot(df[step_col], df[loss_col], label="Training Loss", color="tab:blue")  # linewidth=2.0
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss Curve")
    plt.grid(True, linestyle="--")
    plt.legend()
    plt.tight_layout()

    base_output_dir = Path(__file__).resolve().parents[3] / "output" / "figures"

    for ext in ["pdf", "png"]:
        output_dir = base_output_dir / ext
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"training_loss_curve.{ext}"
        print(f"Saving figure to: {output_path}")
        plt.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    main()
