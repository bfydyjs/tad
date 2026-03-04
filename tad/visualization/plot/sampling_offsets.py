from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from setup_paper_style import setup_paper_style

# ----------------------------
# 1. 模拟数据：假设我们有 N=50 个 action queries
N = 50

# 模拟 TE-TAD 的采样偏移（只有一个分支）
te_tad_offsets = np.random.normal(0.2, 0.1, N)  # 平均偏移为 0.2，集中在中心附近
te_tad_std = np.random.uniform(0.05, 0.15, N)  # 标准差

# 模拟 DiGIT 的两个分支
central_offsets = np.random.normal(0.3, 0.1, N)  # 中心区域关注偏移
adjacent_offsets = np.random.normal(-0.4, 0.12, N)  # 边缘区域关注偏移

central_std = np.random.uniform(0.05, 0.15, N)
adjacent_std = np.random.uniform(0.05, 0.15, N)

# 排序以使图更清晰
sort_idx = np.argsort(te_tad_offsets)
te_tad_offsets = te_tad_offsets[sort_idx]
te_tad_std = te_tad_std[sort_idx]

sort_idx_c = np.argsort(central_offsets)
central_offsets = central_offsets[sort_idx_c]
central_std = central_std[sort_idx_c]

sort_idx_a = np.argsort(adjacent_offsets)
adjacent_offsets = adjacent_offsets[sort_idx_a]
adjacent_std = adjacent_std[sort_idx_a]

# ----------------------------
setup_paper_style(
    440, ratio=1.618, fraction=0.98, font_size_tex=10, font_size_main=9, line_width_axis=0.5
)
# 2. 绘图
fig, (ax1, ax2) = plt.subplots(2, 1)
xticks_vals = np.arange(-1.0, 1.01, 0.25)
xtick_labels = [f"{x:.2f}" for x in xticks_vals]

# (a) TE-TAD
ax1.errorbar(
    te_tad_offsets,
    range(N),
    xerr=te_tad_std,
    fmt="-o",
    ecolor="blue",
    capsize=2,
    markersize=4,
    color="blue",
    linewidth=1,
    label="Cross-Attention",
)
ax1.set_xlim(-1.0, 1.0)
ax1.set_ylim(-2, N + 1)
ax1.set_xticks(xticks_vals)
ax1.set_xticklabels(xtick_labels)
ax1.set_yticks([])  # 隐藏纵轴索引
ax1.set_xlabel("(a) TE-TAD [15]", fontsize=11)
ax1.grid(axis="x", linestyle="--")  # 仅开启 X 轴网格
ax1.legend(loc="upper left")

# (b) DiGIT
ax2.errorbar(
    central_offsets,
    range(N),
    xerr=central_std,
    fmt="-o",
    ecolor="blue",
    capsize=2,
    markersize=4,
    color="blue",
    linewidth=1,
    label="Central-Region Cross-Attention",
)
ax2.errorbar(
    adjacent_offsets,
    range(N),
    xerr=adjacent_std,
    fmt="-o",
    ecolor="red",
    capsize=2,
    markersize=4,
    color="red",
    linewidth=1,
    label="Adjacent-Region Cross-Attention",
)
ax2.set_xlim(-1.0, 1.0)
ax2.set_ylim(-2, N + 1)
ax2.set_xticks(xticks_vals)
ax2.set_xticklabels(xtick_labels)
ax2.set_yticks([])  # 隐藏纵轴索引
ax2.set_xlabel("(b) DiGIT", fontsize=11)
ax2.grid(axis="x", linestyle="--")  # 仅开启 X 轴网格
ax2.legend(loc="upper left")

plt.tight_layout()
output_path = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "output"
    / "figures"
    / "sampling_offsets.png"
)
print(f"Saving figure to: {output_path}")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()
