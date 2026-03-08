import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

from ..utils import save_figure, setup_paper_style

# ==========================================
# 1. Insert the numerical results obtained from running average_cosine_similarity.py here.
# 2. Arial font is not available on the server;
#    do not run this script on the server to generate figures.
# ==========================================

# Format: [Level 0 (Raw), Level 1, Level 2, ..., Level N]
MODEL_A_NAME = "ActionFormer"
MODEL_A_DATA = [0.7027, 0.4786, 0.4723, 0.4839, 0.5170, 0.5823, 0.6913]
# ActionFormer: [0.7027, 0.5592, 0.4740, 0.4414, 0.4833, 0.5496, 0.6631]
# iou_weight=0: [0.7027, 0.4786, 0.4723, 0.4839, 0.5170, 0.5823, 0.6913]
MODEL_B_NAME = "Ours"
MODEL_B_DATA = [0.7027, 0.4801, 0.4716, 0.4826, 0.5144, 0.5796, 0.6849]


def get_data():
    model_a_name = MODEL_A_NAME
    model_a_data = MODEL_A_DATA
    model_b_name = MODEL_B_NAME
    model_b_data = MODEL_B_DATA

    baseline_value = (model_a_data[0] + model_b_data[0]) / 2

    return model_a_name, model_a_data, model_b_name, model_b_data, baseline_value


def plot_cosine_similarity():

    model_a_name, model_a_data, model_b_name, model_b_data, baseline_value = get_data()

    min_len = min(len(model_a_data), len(model_b_data))
    y_a = model_a_data[:min_len]
    y_b = model_b_data[:min_len]
    x = list(range(min_len))
    x_labels = [str(i) for i in x]

    fig = plt.figure()

    plt.hlines(
        y=baseline_value,
        xmin=min(x),
        xmax=max(x),
        color="gray",
        linewidth=0.9,
        linestyle="--",
        label="Raw Baseline",
    )
    plt.plot(x, y_a, marker="o", linestyle="-", label=model_a_name, color="#1f77b4")
    plt.plot(x, y_b, marker="o", linestyle="-", label=model_b_name, color="#ff7f0e")

    all_values = y_a + y_b + [baseline_value]

    y_min = max(0.0, min(all_values) - 0.05)
    y_max = min(1.05, max(all_values) + 0.05)

    plt.ylim(y_min, y_max)
    plt.xlim(min(x) - 0.2, max(x) + 0.2)
    plt.xlabel("Levels", labelpad=0)
    plt.ylabel("Cosine Similarity", labelpad=2.7)
    plt.xticks(x, labels=x_labels)
    plt.yticks()

    ax = plt.gca()
    ax.yaxis.set_major_locator(MultipleLocator(0.05))

    plt.legend(loc="center", edgecolor="#bfbfbf", handlelength=2, fontsize=8, frameon=False)
    plt.tight_layout()

    return fig


def setup_plot_style():
    setup_paper_style(
        textwidth=440 / 2,
        ratio=1.618,
        fraction=0.98,
        font_size_tex=10,
        font_size_main=9,
        line_width_axis=0.5,
    )


def main():
    setup_plot_style()
    fig = plot_cosine_similarity()
    save_figure("cosine_similarity_comparison", fig=fig)
    plt.show()


if __name__ == "__main__":
    main()
