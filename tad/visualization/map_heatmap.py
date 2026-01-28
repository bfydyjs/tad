import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 模拟数据：mAP values for different combinations
kernel_sizes = [3, 5, 7, 9, 11, 13, 15]
num_layers = [1, 2, 3, 4]

# 构造 mAP 矩阵（行：layers，列：kernel_size）
map_values = np.array([
    [71.3, 72.1, 72.0, 71.6, 71.7, 71.6, 71.6],  # layers=1
    [71.8, 72.6, 72.9, 73.4, 73.8, 73.0, 72.1],  # layers=2
    [71.9, 72.6, 72.9, 73.3, 72.5, 72.1, 72.0],  # layers=3
    [72.6, 72.8, 73.1, 72.6, 72.4, 72.2, 72.0]   # layers=4
])

# 绘图
plt.figure(figsize=(10, 6))

# 创建热图
heatmap = sns.heatmap(map_values, annot=True, fmt='.1f', cmap='YlOrRd', 
                      cbar_kws={'label': 'mAP (%)'}, 
                      xticklabels=kernel_sizes, 
                      yticklabels=num_layers)

ax = plt.gca()

# 获取颜色条
cbar = heatmap.collections[0].colorbar

# 设置标签
ax.set_xlabel('Kernel Size')
ax.set_ylabel('Number of Dilated Convs.')

# 关键：将刻度位置调整到单元格中心
# 热图的单元格从0到n，刻度应该在每个单元格的中心位置：i + 0.5
ax.set_xticks(np.arange(len(kernel_sizes)) + 0.5)
ax.set_yticks(np.arange(len(num_layers)) + 0.5)
ax.set_xticklabels(kernel_sizes)
ax.set_yticklabels(num_layers, va='center')  # 现在标签会居中显示

# 隐藏刻度线
ax.tick_params(left=False, bottom=False)

# 隐藏颜色条的刻度线
cbar.ax.tick_params(length=0)

plt.tight_layout()

output_path = (Path(__file__).resolve().parent.parent.parent / "output" / "figures" / "map_heatmap.png")
print(f"Saving figure to: {output_path}")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path, dpi=300)
plt.show()