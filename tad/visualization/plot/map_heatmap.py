from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from setup_paper_style import setup_paper_style

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
setup_paper_style(
    440 / 2, ratio=1.618, fraction=0.98, font_size_tex=5, font_size_main=4.5, line_width_axis=0.5
)
plt.figure()
ax = sns.heatmap(
    map_values,
    annot=True,
    annot_kws={"size": 4.5},
    fmt=".1f",
    cmap="YlOrRd",
    cbar_kws={"label": "mAP (%)"},
    xticklabels=kernel_sizes_1,
    yticklabels=kernel_sizes_2,
)
ax.grid(False)
cbar = ax.collections[0].colorbar
ax.set_xlabel("Kernel Size A")
ax.set_ylabel("Kernel Size B")
ax.set_xticks(np.arange(len(kernel_sizes_1)) + 0.5, labels=kernel_sizes_1)
ax.set_yticks(np.arange(len(kernel_sizes_2)) + 0.5, labels=kernel_sizes_2)
ax.tick_params(left=False, bottom=False)

cbar.ax.tick_params(length=0)
plt.tight_layout()

output_path = (
    Path(__file__).resolve().parent.parent.parent.parent / "output" / "figures" / "map_heatmap.pdf"
)
print(f"Saving figure to: {output_path}")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path)
plt.show()
