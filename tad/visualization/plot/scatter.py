import matplotlib.pyplot as plt

from ..utils import save_figure, setup_paper_style


def get_data_params():
    """准备参数量对比数据。"""
    return [
        {
            "name": "ActionFormer",
            "metric": 45.41,
            "mAP": 66.8,
            "marker": "o",
            "color": "lightblue",
            "s": 30,
        },
        {
            "name": "DyFADet",
            "metric": 30.87,
            "mAP": 69.2,
            "marker": "s",
            "color": "lightcoral",
            "s": 30,
        },
        {"name": "TriDet", "metric": 18.8, "mAP": 69.3, "marker": "D", "color": "gold", "s": 20},
        {"name": "HSFPN", "metric": 2.6, "mAP": 31.2, "marker": "p", "color": "plum", "s": 30},
        {"name": "AFPN", "metric": 2.8, "mAP": 30.5, "marker": "v", "color": "lightgreen", "s": 30},
        {"name": "FreqGFPN", "metric": 2.5, "mAP": 30.8, "marker": "X", "color": "orange", "s": 30},
        {"name": "GFPN", "metric": 3.0, "mAP": 32.0, "marker": "^", "color": "gray", "s": 30},
        {"name": "CGFPRN", "metric": 3.4, "mAP": 30.2, "marker": "<", "color": "tan", "s": 30},
        {"name": "MAFPN", "metric": 3.2, "mAP": 31.8, "marker": ">", "color": "teal", "s": 30},
        {"name": "Ours", "metric": 17.56, "mAP": 69.4, "marker": "*", "color": "red", "s": 60},
    ]


def get_data_gflops():
    """准备 GFLOPs 对比数据。"""
    return [
        {
            "name": "ActionFormer",
            "metric": 45.41,
            "mAP": 66.8,
            "marker": "o",
            "color": "lightblue",
            "s": 30,
        },
        {
            "name": "DyFADet",
            "metric": 90.87,
            "mAP": 69.2,
            "marker": "s",
            "color": "lightcoral",
            "s": 30,
        },
        {"name": "TriDet", "metric": 43.7, "mAP": 69.3, "marker": "D", "color": "gold", "s": 20},
        {"name": "HSFPN", "metric": 2.6, "mAP": 31.2, "marker": "p", "color": "plum", "s": 30},
        {"name": "AFPN", "metric": 2.8, "mAP": 30.5, "marker": "v", "color": "lightgreen", "s": 30},
        {"name": "FreqGFPN", "metric": 2.5, "mAP": 30.8, "marker": "X", "color": "orange", "s": 30},
        {"name": "GFPN", "metric": 3.0, "mAP": 32.0, "marker": "^", "color": "gray", "s": 30},
        {"name": "CGFPRN", "metric": 3.4, "mAP": 30.2, "marker": "<", "color": "tan", "s": 30},
        {"name": "MAFPN", "metric": 3.2, "mAP": 31.8, "marker": ">", "color": "teal", "s": 30},
        {"name": "Ours", "metric": 38.88, "mAP": 69.4, "marker": "*", "color": "red", "s": 60},
    ]


def plot_single_scatter(ax, data, metric):
    for item in data:
        ax.scatter(
            item["metric"],
            item["mAP"],
            s=item["s"],
            marker=item["marker"],
            color=item["color"],
            label=item["name"],
            edgecolors="k",
            linewidth=0.8,
            alpha=0.9,
            zorder=3,
        )

    best_model = next((item for item in data if "Ours" in item["name"]), None)

    if best_model:
        ax.annotate(
            "Best Performance",
            xy=(best_model["metric"] + 0.05, best_model["mAP"] + 0.05),
            xytext=(
                best_model["metric"] + (0.2 if metric == "Parameters (M)" else 1.0),
                best_model["mAP"] + 0.2,
            ),
            arrowprops=dict(
                arrowstyle="->",
                color="red",
                lw=1.2,
                connectionstyle="arc3,rad=-0.2",
            ),
            fontsize=3,
            color="#d62728",
            weight="bold",
            bbox=dict(
                boxstyle="round,pad=0.4",
                facecolor="white",
                edgecolor="#d62728",
                alpha=0.9,
            ),
        )

    ax.set_xlabel(metric)
    ax.tick_params(left=False, bottom=False)
    ax.grid(zorder=0)

    all_params = [d["metric"] for d in data]
    all_maps = [d["mAP"] for d in data]
    p_margin = (max(all_params) - min(all_params)) * 0.1
    m_margin = (max(all_maps) - min(all_maps)) * 0.1

    ax.set_xlim(min(all_params) - p_margin, max(all_params) + p_margin)
    ax.set_ylim(min(all_maps) - m_margin, max(all_maps) + m_margin * 2.0)
    ax.legend(loc="upper right", ncol=2, handlelength=1)


def setup_plot_style():
    """Set up plotting style for paper."""
    setup_paper_style(
        textwidth=440 / 2,
        ratio=1.618 * 2,
        fraction=0.98 * 2,
        font_size_tex=5,
        font_size_main=4.5,
        line_width_axis=0.5,
    )


def main():
    metrics = ["Parameters (M)", "GFLOPs"]
    data_list = [get_data_params(), get_data_gflops()]

    setup_plot_style()

    _, axes = plt.subplots(1, 2)

    for ax, data, metric in zip(axes, data_list, metrics, strict=True):
        plot_single_scatter(ax, data, metric)
        ax.set_ylabel("mAP@0.5 (%)") if ax == axes[0] else ax.set_ylabel("")

    plt.tight_layout()
    save_figure("scatter_mAP_vs_Params")
    plt.show()


if __name__ == "__main__":
    main()
