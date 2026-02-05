from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

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
# 2. 绘图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# (a) TE-TAD
ax1.errorbar(range(N), te_tad_offsets, yerr=te_tad_std, fmt="o", ecolor="blue", capsize=3, markersize=4, color="blue")
ax1.set_xlim(-1, N)
ax1.set_ylim(-1.0, 1.0)
ax1.set_ylabel("Query Index")
ax1.set_title("(a) TE-TAD [15]")
ax1.grid(True, alpha=0.3)
ax1.axvline(x=-0.5, color="gray", linestyle="--", alpha=0.5)  # 可选：辅助线
ax1.text(-0.7, 0.9, "Cross-Attention", fontsize=10, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))

# (b) DiGIT
ax2.errorbar(
    range(N),
    central_offsets,
    yerr=central_std,
    fmt="o",
    ecolor="blue",
    capsize=3,
    markersize=4,
    color="blue",
    label="Central-Region Cross-Attention",
)
ax2.errorbar(
    range(N),
    adjacent_offsets,
    yerr=adjacent_std,
    fmt="o",
    ecolor="red",
    capsize=3,
    markersize=4,
    color="red",
    label="Adjacent-Region Cross-Attention",
)
ax2.set_xlim(-1, N)
ax2.set_ylim(-1.0, 1.0)
ax2.set_xlabel("Sampling Offset (normalized)")
ax2.set_ylabel("Query Index")
ax2.set_title("(b) DiGIT")
ax2.grid(True, alpha=0.3)
ax2.legend(loc="upper left")

plt.tight_layout()
output_path = Path(__file__).resolve().parent.parent.parent / "output" / "figures" / "sampling_offsets.png"
print(f"Saving figure to: {output_path}")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()
