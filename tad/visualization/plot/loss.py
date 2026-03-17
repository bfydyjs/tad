from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ..utils import save_figure, setup_paper_style


def load_loss_data(
    file_path: str,
    step_col: str,
    loss_col: str | None = None,
) -> pd.DataFrame | None:

    df = pd.read_csv(file_path)

    if loss_col is None:
        for col in df.columns:
            if "train/loss" in col:
                loss_col = col
                break
        else:
            print(f"Warning: No 'train/loss' column found in {file_path}")
            return None

    if loss_col not in df.columns:
        print(f"Warning: Column '{loss_col}' not found in {file_path}")
        return None

    return df[[step_col, loss_col]].rename(columns={loss_col: "loss"})


def plot_training_loss(
    data_list: list[dict],
    xlabel: str = "Epoch",
    ylabel: str = "Loss",
    title: str = "Training Loss Comparison",
):

    setup_paper_style(
        textwidth=440 / 2,
        ratio=1.618,
        fraction=0.98,
        font_size_tex=5,
        font_size_main=4.5,
        line_width_axis=0.5,
    )
    fig, ax = plt.subplots()

    for data_config in data_list:
        ax.plot(
            data_config["data"][data_config["step_col"]],
            data_config["data"]["loss"],
            label=data_config["label"],
            color=data_config["color"],
            linestyle=data_config["linestyle"],
            linewidth=1.5,
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend(loc="upper right")
    return fig


def process_and_save(data_list: list[dict]) -> None:
    """处理数据并保存图表。

    Args:
        data_list: 包含数据和配置的列表
    """
    if not data_list:
        print("Error: No valid data to plot")
        return
    # plt.show()
    fig = plot_training_loss(data_list)
    try:
        save_figure("loss", fig=fig)
    finally:
        plt.close(fig)


loss_output_dir = Path(__file__).resolve().parents[3] / "assets"


def main():
    loss_configs = [
        {
            "file": loss_output_dir / "wandb_export_2026-02-03T22_06_26.613+08_00.csv",
            "step_col": "epoch",
            "loss_col": "0130_1147 - train/loss",
            "label": "Model A",
            "color": "tab:blue",
            "linestyle": "-",
        },
        # {
        #     "file": r"path/to/second_experiment.csv",
        #     "step_col": "epoch",
        #     "loss_col": None,
        #     "label": "Model B",
        #     "color": "tab:orange",
        #     "linestyle": "--",
        # },
        # ...
    ]

    data_list = []
    for config in loss_configs:
        df = load_loss_data(
            config["file"],
            config["step_col"],
            config.get("loss_col"),
        )
        if df is not None:
            data_list.append(
                {
                    "data": df,
                    "label": config["label"],
                    "color": config["color"],
                    "linestyle": config["linestyle"],
                    "step_col": config["step_col"],
                }
            )

    process_and_save(data_list)


if __name__ == "__main__":
    main()
