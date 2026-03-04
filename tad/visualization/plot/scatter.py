from pathlib import Path

import matplotlib.pyplot as plt
from setup_paper_style import setup_paper_style


# Performance vs. Efficiency Comparison
def main():
    metric = "GFLOPs"  # Parameters (M) / GFLOPs
    # Parameters (M)
    # data = [
    #     {"name": "ActionFormer", "metric": 45.41, "mAP": 66.8, "marker": 'o', "color": 'lightblue',  's': 30},
    #     {"name": "DyFADet",      "metric": 90.87, "mAP": 69.2, "marker": 's', "color": 'lightcoral', 's': 30},
    #     {"name": "TriDet",       "metric": 43.7,  "mAP": 69.3, "marker": 'D', "color": 'gold',       's': 20},
    #     # {"name": "HSFPN",        "metric": 2.6, "mAP": 31.2, "marker": 'p', "color": 'plum',       's': 30},
    #     # {"name": "AFPN",         "metric": 2.8, "mAP": 30.5, "marker": 'v', "color": 'lightgreen', 's': 30},
    #     # {"name": "FreqGFPN",     "metric": 2.5, "mAP": 30.8, "marker": 'X', "color": 'orange',     's': 30},
    #     # {"name": "GFPN",         "metric": 3.0, "mAP": 32.0, "marker": '^', "color": 'gray',       's': 30},
    #     # {"name": "CGFPRN",       "metric": 3.4, "mAP": 30.2, "marker": '<', "color": 'tan',        's': 30},
    #     # {"name": "MAFPN",        "metric": 3.2, "mAP": 31.8, "marker": '>', "color": 'teal',       's': 30},
    #     {"name": "Ours",         "metric": 17.56, "mAP": 69.4, "marker": '*', "color": 'red',        's': 60},
    #     # The best method (ours) is highlighted in red with an asterisk (*).
    # ]
    # GFLOPs
    data = [
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
        # {"name": "HSFPN",        "metric": 2.6, "mAP": 31.2, "marker": 'p', "color": 'plum',       's': 30},
        # {"name": "AFPN",         "metric": 2.8, "mAP": 30.5, "marker": 'v', "color": 'lightgreen', 's': 30},
        # {"name": "FreqGFPN",     "metric": 2.5, "mAP": 30.8, "marker": 'X', "color": 'orange',     's': 30},
        # {"name": "GFPN",         "metric": 3.0, "mAP": 32.0, "marker": '^', "color": 'gray',       's': 30},
        # {"name": "CGFPRN",       "metric": 3.4, "mAP": 30.2, "marker": '<', "color": 'tan',        's': 30},
        # {"name": "MAFPN",        "metric": 3.2, "mAP": 31.8, "marker": '>', "color": 'teal',       's': 30},
        {"name": "Ours", "metric": 38.88, "mAP": 69.4, "marker": "*", "color": "red", "s": 60},
        # The best method (ours) is highlighted in red with an asterisk (*).
    ]

    setup_paper_style(
        440 / 2,
        ratio=1.618,
        fraction=0.98,
        font_size_tex=5,
        font_size_main=4.5,
        line_width_axis=0.5,
    )
    plt.figure()

    for item in data:
        plt.scatter(
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
        plt.annotate(
            "Best Performance",
            xy=(
                best_model["metric"] + 0.01,
                best_model["mAP"] - 0.01,
            ),
            xytext=(
                best_model["metric"] + 0.2,
                best_model["mAP"] - 0.2,
            ),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.2, connectionstyle="arc3,rad=-0.2"),
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
    plt.xlabel(metric)
    plt.ylabel("mAP@0.5 (%)")
    plt.tick_params(left=False, bottom=False)
    plt.grid(zorder=0)

    all_params = [d["metric"] for d in data]
    all_maps = [d["mAP"] for d in data]
    p_margin = (max(all_params) - min(all_params)) * 0.1
    m_margin = (max(all_maps) - min(all_maps)) * 0.1

    plt.xlim(min(all_params) - p_margin, max(all_params) + p_margin)
    plt.ylim(min(all_maps) - m_margin, max(all_maps) + m_margin * 2.0)
    plt.legend(loc="upper right", ncol=2, handlelength=1)
    plt.tight_layout()

    output_path = (
        Path(__file__).resolve().parent.parent.parent.parent / "output" / "figures" / "scatter.pdf"
    ).resolve()

    print(f"Saving figure to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    plt.show()


if __name__ == "__main__":
    main()
