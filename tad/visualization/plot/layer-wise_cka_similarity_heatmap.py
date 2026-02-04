import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 模拟数据（CKA 矩阵）
cka_a_pre = np.array([
    [1.000, 0.925, 0.832, 0.669, 0.557],
    [0.925, 1.000, 0.925, 0.721, 0.603],
    [0.832, 0.925, 1.000, 0.810, 0.679],
    [0.669, 0.721, 0.810, 1.000, 0.877],
    [0.557, 0.603, 0.679, 0.877, 1.000]
])

cka_a_post = np.array([
    [1.000, 0.896, 0.801, 0.762, 0.752],
    [0.896, 1.000, 0.928, 0.844, 0.736],
    [0.801, 0.928, 1.000, 0.927, 0.722],
    [0.762, 0.844, 0.927, 1.000, 0.749],
    [0.752, 0.736, 0.722, 0.749, 1.000]
])

# 绘图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

sns.heatmap(cka_a_pre, annot=True, fmt='.3f', cmap='YlOrRd', cbar=False, ax=axes[0,0])
axes[0,0].set_title('(a) Pre-Encoder')
axes[0,0].set_xlabel('Scale Level')
axes[0,0].set_ylabel('Scale Level')

sns.heatmap(cka_a_post, annot=True, fmt='.3f', cmap='YlOrRd', cbar=False, ax=axes[0,1])
axes[0,1].set_title('(a) Post-Encoder')
axes[0,1].set_xlabel('Scale Level')
axes[0,1].set_ylabel('Scale Level')

plt.tight_layout()
output_path = (Path(__file__).resolve().parent.parent.parent.parent / "output" / "figures" / "cka_comparison.png")
print(f"Saving figure to: {output_path}")
output_path.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(output_path)
plt.show()