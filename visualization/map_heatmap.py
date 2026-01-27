import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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
ax = sns.heatmap(map_values, annot=True, fmt='.1f', cmap='YlOrRd', cbar_kws={'label': 'mAP (%)'})
ax.set_xlabel('Kernel Size')
ax.set_ylabel('Number of Dilated Convs.')
ax.set_xticks(range(len(kernel_sizes)))
ax.set_xticklabels(kernel_sizes)
ax.set_yticks(range(len(num_layers)))
ax.set_yticklabels(num_layers)

plt.title('Ablation Study: mAP vs Kernel Size and Number of Dilated Convolutions')
plt.tight_layout()
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "..", "output", "figures", "map_heatmap.png")
output_path = os.path.normpath(output_path)

print(f"Saving figure to: {output_path}")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300)
plt.show()