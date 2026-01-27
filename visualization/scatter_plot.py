# 导入绘图库 matplotlib 和数值计算库 numpy
import matplotlib.pyplot as plt
import numpy as np

# 数据定义：模型名称、参数量（单位：百万）、mAP@0.5指标值（百分比）
models = [
    "FPN(yolo11n)", "BIFPN", "BIFPN-GLSA", "HSFPN", "AFPN",
    "FreqGFPN", "GFPN", "CGFPRN", "MAFPN", "EHMPFN (Ours)"
]
params = [2.1, 2.3, 2.4, 2.6, 2.8, 2.5, 3.0, 3.4, 3.2, 2.0]  # 参数量（M）
mAPs = [32.8, 33.9, 31.5, 31.2, 30.5, 30.8, 32.0, 30.2, 31.8, 35.5]  # mAP@0.5性能指标(%)

# 每个模型对应的标记样式和颜色
markers = ['o', 's', 'D', 'p', '*', 'x', '^', '<', '>', '*']
colors = ['lightblue', 'lightcoral', 'gold', 'plum', 'lightgreen', 'yellow', 'lightgray', 'tan', 'teal', 'black']

# 创建一个图形窗口，并设置尺寸为宽10英寸，高7英寸
plt.figure(figsize=(10, 7))

# 循环遍历所有模型，绘制散点图
for i in range(len(models)):
    plt.scatter(
        params[i], mAPs[i], s=100, marker=markers[i], color=colors[i], label=models[i],
        edgecolors='black', linewidth=1  # 设置边框颜色为黑色，线宽为1
    )

# 标注“最佳”模型（这里假设最后一个模型"EHMPFN (Ours)"是最好的）
best_idx = 9  # EHMPFN 的索引
plt.annotate(
    'Best',  # 要显示的文本
    xy=(params[best_idx] - 0.005, mAPs[best_idx] - 0.005),  # 箭头指向的位置（数据点）
    xytext=(params[best_idx] - 0.2, mAPs[best_idx] - 0.2),  # 文本位置
    arrowprops=dict(
        arrowstyle='->',  # 箭头样式
        color='red',  # 箭头颜色
        lw=2,  # 线宽
        connectionstyle='arc3,rad=0.3'  # 弯曲箭头的曲率半径
    ),
    fontsize=12,  # 字体大小
    color='red',  # 文本颜色
    weight='bold',  # 加粗
    bbox=dict(boxstyle="round,pad=0.3", facecolor="red", alpha=0.3)  # 文本框样式
)

# 设置坐标轴标签及标题
plt.xlabel("Parameters (M)", fontsize=12)  # X轴标签：参数量（单位：百万）
plt.ylabel("mAP@0.5 (%)", fontsize=12)  # Y轴标签：mAP@0.5性能指标（百分比）
plt.title("(a) Performance vs Parameters", fontsize=14, pad=20)  # 图表标题

# 设置坐标轴范围及网格线
plt.xlim(1.8, 3.6)  # X轴范围
plt.ylim(29.5, 36.0)  # Y轴范围
plt.grid(True, alpha=0.3)  # 显示浅灰色网格线

# 自定义图例，确保每个模型在图例中都有相应的标记样式和颜色
legend_elements = [plt.Line2D([0], [0], marker=markers[i], color='w', markerfacecolor=colors[i], markersize=10, label=models[i]) for i in range(len(models))]
plt.legend(handles=legend_elements, loc='upper right', frameon=True, fancybox=True, shadow=True)

# 自动调整子图参数，防止标签被裁剪
plt.tight_layout()

# 显示图表
plt.show()