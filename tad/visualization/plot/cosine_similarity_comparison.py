import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import MultipleLocator
from setup_paper_style import setup_paper_style

# ==========================================
# 1. Insert the numerical results obtained from running average_cosine_similarity.py here.
# 2. Arial font is not available on the server; do not run this script on the server to generate figures.
# ==========================================

# Format: [Level 0 (Raw), Level 1, Level 2, ..., Level N]
model_a_name = "ActionFormer"
model_a_data = [0.7027, 0.4786, 0.4723, 0.4839, 0.5170, 0.5823, 0.6913]
# ActionFormer: [0.7027, 0.5592, 0.4740, 0.4414, 0.4833, 0.5496, 0.6631]
# iou_weight=0: [0.7027, 0.4786, 0.4723, 0.4839, 0.5170, 0.5823, 0.6913]
model_b_name = "Ours"
model_b_data = [0.7027, 0.4801, 0.4716, 0.4826, 0.5144, 0.5796, 0.6849]

baseline_value = (model_a_data[0] + model_b_data[0]) / 2

# ========================================
# Apply paper-style plotting configuration
# ========================================


def plot_comparison():
    setup_paper_style()
    min_len = min(len(model_a_data), len(model_b_data))
    y_a = model_a_data[:min_len]
    y_b = model_b_data[:min_len]
    x = list(range(min_len))
    x_labels = [str(i) for i in x]

    plt.figure(figsize=(10, 6))

    plt.plot(x, y_a, marker="o", linestyle="-", label=model_a_name, color="#1f77b4")
    plt.plot(x, y_b, marker="o", linestyle="-", label=model_b_name, color="#ff7f0e")

    plt.hlines(
        y=baseline_value,
        xmin=min(x),
        xmax=max(x),
        color="gray",
        linewidth=1.5,
        linestyle="--",
        label="Raw Baseline",
    )

    plt.grid(True, color="#bdbdbd")

    all_values = y_a + y_b + [baseline_value]

    y_min = max(0.0, min(all_values) - 0.05)
    y_max = min(1.05, max(all_values) + 0.05)

    plt.ylim(y_min, y_max)
    plt.xlim(min(x) - 0.2, max(x) + 0.2)

    plt.xlabel("Levels")
    plt.ylabel("Cosine Similarity")

    plt.xticks(x, labels=x_labels)

    plt.legend(loc="lower right", frameon=True, edgecolor="#bfbfbf", fancybox=False)

    plt.tight_layout()
    output_path = (
        Path(__file__).resolve().parent.parent.parent.parent
        / "output"
        / "figures"
        / "cosine_similarity_comparison.pdf"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_comparison()
