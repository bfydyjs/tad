import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..utils import save_figure, setup_paper_style


def load_map_data() -> tuple[list[int], list[int], np.ndarray]:
    kernel_sizes_1 = [3, 5, 7, 9, 11, 13, 15]
    kernel_sizes_2 = [3, 5, 7, 9, 11, 13, 15]

    map_values = np.array(
        [
            [71.3, 72.1, 72.0, 71.6, 71.7, 71.6, 71.6],  # kernel_sizes_1=3
            [71.8, 72.6, 72.9, 73.4, 73.8, 73.0, 72.1],  # kernel_sizes_1=5
            [71.9, 72.6, 72.9, 73.3, 72.5, 72.1, 72.0],  # kernel_sizes_1=7
            [72.6, 72.8, 73.1, 72.6, 72.4, 72.2, 72.0],  # kernel_sizes_1=9
            [72.6, 72.8, 73.1, 72.6, 72.4, 72.2, 72.0],  # kernel_sizes_1=11
            [72.6, 72.8, 73.1, 72.6, 72.4, 72.2, 72.0],  # kernel_sizes_1=13
            [72.6, 72.8, 73.1, 72.6, 72.4, 72.2, 72.0],  # kernel_sizes_1=15
        ]
    )

    return kernel_sizes_1, kernel_sizes_2, map_values


def plot_map_heatmap(
    kernel_sizes_1: list[int],
    kernel_sizes_2: list[int],
    map_values: np.ndarray,
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
    sns.heatmap(
        map_values,
        annot=True,
        annot_kws={"size": 4.5},
        fmt=".1f",
        cmap="YlOrRd",
        cbar_kws={"label": "mAP (%)"},
        xticklabels=kernel_sizes_1,
        yticklabels=kernel_sizes_2,
        ax=ax,
    )
    ax.grid(False)
    cbar = ax.collections[0].colorbar
    ax.set_xlabel("Kernel Size A")
    ax.set_ylabel("Kernel Size B")
    # ax.tick_params(left=False, bottom=False)

    cbar.ax.tick_params(length=0)
    return fig


def main():
    kernel_sizes_1, kernel_sizes_2, map_values = load_map_data()
    fig = plot_map_heatmap(kernel_sizes_1, kernel_sizes_2, map_values)
    # plt.show()
    try:
        save_figure("map_heatmap", fig=fig)
    finally:
        plt.close(fig)


if __name__ == "__main__":
    main()
