from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 模拟数据：mAP values for different combinations
kernel_sizes_1 = [3, 5, 7, 9, 11, 13, 15]
kernel_sizes_2 = [3, 5, 7, 9, 11, 13, 15]

# 构造 mAP 矩阵（行：layers，列：kernel_size）
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

# 绘图
plt.figure(figsize=(10, 6))

# 创建热图
heatmap = sns.heatmap(
    map_values,
    annot=True,
    fmt=".1f",
    cmap="YlOrRd",
    cbar_kws={"label": "mAP (%)"},
    xticklabels=kernel_sizes_1,
    yticklabels=kernel_sizes_2,
)

ax = plt.gca()

# 获取颜色条
cbar = heatmap.collections[0].colorbar

# 设置标签
ax.set_xlabel("Kernel Size A")
ax.set_ylabel("kernel size B")

# 关键：将刻度位置调整到单元格中心
# 热图的单元格从0到n，刻度应该在每个单元格的中心位置：i + 0.5
ax.set_xticks(np.arange(len(kernel_sizes_1)) + 0.5)
ax.set_yticks(np.arange(len(kernel_sizes_2)) + 0.5)
ax.set_xticklabels(kernel_sizes_1)
ax.set_yticklabels(kernel_sizes_2, va="center")  # 现在标签会居中显示

# 隐藏刻度线
ax.tick_params(left=False, bottom=False)

# 隐藏颜色条的刻度线
cbar.ax.tick_params(length=0)

plt.tight_layout()

output_path = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "output"
    / "figures"
    / "map_heatmap.png"
)
print(f"Saving figure to: {output_path}")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path)
plt.show()
