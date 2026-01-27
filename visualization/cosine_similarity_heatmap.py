import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# ----------------------------
# 1. 模拟数据：假设我们有 T=50 个时间步的特征向量 (T, D)
T = 50
D = 128
np.random.seed(42)

# 模拟编码器输出特征（这里用随机数据 + 添加动作结构）
features = np.random.randn(T, D) * 0.1
# 假设 [17, 25] 和 [35, 40] 是动作时间段，增强这些区间的特征一致性
for i in range(17, 26):
    features[i] += np.array([1.0] * D)  # 加强特征
for i in range(35, 41):
    features[i] += np.array([0.8] * D)

# ----------------------------
# 2. 计算余弦相似度矩阵
similarity_matrix = np.zeros((T, T))
for i in range(T):
    for j in range(T):
        sim = np.dot(features[i], features[j]) / (
            np.linalg.norm(features[i]) * np.linalg.norm(features[j])
        )
        similarity_matrix[i, j] = sim

# ----------------------------
# 3. 设置 ground truth 动作区间（时间索引）
gt_intervals = [(17, 25), (35, 40)]  # 时间步范围（对应秒数可缩放）

# ----------------------------
# 4. 绘图
plt.figure(figsize=(12, 8))

# Top: Cosine Similarity Heatmap
ax1 = plt.subplot(2, 2, 1)
sns.heatmap(similarity_matrix, cmap='YlGnBu', cbar=False, ax=ax1)
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Time (s)')
ax1.set_title('(a) TE-TAD')

# 添加红色虚线框标记 ground truth 区间
for start, end in gt_intervals:
    rect = Rectangle((start, start), end - start, end - start,
                     linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    ax1.add_patch(rect)

# Bottom: Ground Truth Timeline
ax2 = plt.subplot(2, 2, 3)
ax2.set_xlim(0, T)
ax2.set_ylim(0, 1)
ax2.set_xlabel('Time (s)')
ax2.set_ylabel('Action')
ax2.set_title('Ground Truth')

# 绘制绿色条形表示动作
for start, end in gt_intervals:
    ax2.fill_between([start, end], 0, 1, color='lightgreen', alpha=0.5)

# ----------------------------
# 重复上述步骤画 DiGIT 版本（可稍调参数模拟差异）
# 比如让 DiGIT 的相似性更集中或更强
features_digit = features.copy()
# 可以加一些扰动或增强局部相似性来模拟不同模型行为
for i in range(17, 26):
    features_digit[i] *= 1.5
for i in range(35, 40):
    features_digit[i] *= 1.3

similarity_digit = np.zeros((T, T))
for i in range(T):
    for j in range(T):
        sim = np.dot(features_digit[i], features_digit[j]) / (
            np.linalg.norm(features_digit[i]) * np.linalg.norm(features_digit[j])
        )
        similarity_digit[i, j] = sim

# Plot DiGIT
ax3 = plt.subplot(2, 2, 2)
sns.heatmap(similarity_digit, cmap='YlGnBu', cbar=False, ax=ax3)
ax3.set_xlabel('Time (s)')
ax3.set_ylabel('Time (s)')
ax3.set_title('(b) DiGIT')

for start, end in gt_intervals:
    rect = Rectangle((start, start), end - start, end - start,
                     linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
    ax3.add_patch(rect)

ax4 = plt.subplot(2, 2, 4)
ax4.set_xlim(0, T)
ax4.set_ylim(0, 1)
ax4.set_xlabel('Time (s)')
ax4.set_ylabel('Action')
ax4.set_title('Ground Truth')

for start, end in gt_intervals:
    ax4.fill_between([start, end], 0, 1, color='lightgreen', alpha=0.5)

plt.tight_layout()
script_dir = os.path.dirname(os.path.abspath(__file__))
output_path = os.path.join(script_dir, "..", "output", "figures", "heatmap.png")
output_path = os.path.normpath(output_path)

print(f"Saving figure to: {output_path}")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.savefig(output_path, dpi=300, bbox_inches='tight')
plt.show()