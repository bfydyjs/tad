import matplotlib.pyplot as plt
from pathlib import Path
# ==========================================
# 1. 在这里填入你运行 average_cosine_similarity.py 得到的数字
# ==========================================

# 模型 A 的数据 (例如: ActionFormer)
# 格式: [Raw Sim, Level 0 Sim, Level 1 Sim, ..., Level N Sim]
model_a_name = "ActionFormer"
model_a_data = [0.7027, 0.4786, 0.4723, 0.4839, 0.5170, 0.5823, 0.6913]
# ActionFormer: [0.7027, 0.5592, 0.4740, 0.4414, 0.4833, 0.5496, 0.6631]
# iou_weight=0: [0.7027, 0.4786, 0.4723, 0.4839, 0.5170, 0.5823, 0.6913]
# 模型 B 的数据 (例如: Ours)
model_b_name = "Ours"
model_b_data = [0.7027, 0.4801, 0.4716, 0.4826, 0.5144, 0.5796, 0.6849]

# 基准线 (Raw Features)
# 通常取两个模型 Raw Sim 的平均值，或者如果一样就取任意一个
baseline_value = (model_a_data[0] + model_b_data[0]) / 2

# ==========================================
# 2. 绘图配置 (Paper Style)
# ==========================================
# 设置无衬线字体 (Journal Standard: Arial/Helvetica)，drawio默认的Helvetica实际上是Arial
plt.rcParams['font.family'] = 'sans-serif'
# 优先顺序
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans'] 

# 全局参数设置
plt.rcParams.update({
    'font.size': 11,           # 基础字体大小
    'axes.labelsize': 12,      # 坐标轴标签大小
    'axes.titlesize': 12,      # 标题大小
    'xtick.labelsize': 11,     # x轴刻度标签大小
    'ytick.labelsize': 11,     # y轴刻度标签大小
    'legend.fontsize': 11,     # 图例字体大小
    'legend.frameon': True,    # 图例边框
    'legend.framealpha': 0.8,  # 图例透明度
    'legend.edgecolor': 'black', # 图例边框颜色
    'lines.linewidth': 2.0,    # 线条宽度
    'lines.markersize': 8,     # 标记大小
    'axes.linewidth': 1.2,     # 坐标轴线宽
    'grid.linewidth': 0.8,     # 网格线宽
    'grid.alpha': 0.3,         # 网格透明度
    'savefig.dpi': 600,        # 输出分辨率
    'savefig.bbox': 'tight',   # 紧凑边框
    'savefig.pad_inches': 0.1, # 内边距
})

def plot_comparison():
    # 检查数据长度是否一致
    # 如果不一致，取最短的长度
    min_len = min(len(model_a_data), len(model_b_data))
    
    # 截取数据
    y_a = model_a_data[:min_len]
    y_b = model_b_data[:min_len]
    
    # 生成 X 轴坐标 (0, 1, 2...)
    x = list(range(min_len))
    x_labels = [str(i) for i in x]
    
    plt.figure(figsize=(10, 6))
    
    # 绘制 Model A
    plt.plot(x, y_a, marker='o', linestyle='-', label=model_a_name, color='#1f77b4') # Blue
    
    # 绘制 Model B
    plt.plot(x, y_b, marker='o', linestyle='-', label=model_b_name, color='#ff7f0e') # Orange
    
    # 绘制 Baseline (黑色实线)
    # 起点: Raw (Index 0), 终点: Last Layer (Index max)
    plt.hlines(y=baseline_value, xmin=min(x), xmax=max(x), 
               color='gray', linewidth=1.5, linestyle='--', label='Raw Baseline')

    # Grid
    plt.grid(True, linestyle='--', linewidth=1, alpha=0.6, color='#bdbdbd')
    
    # Y-Axis Limits (自动调整 + buffer)
    all_values = y_a + y_b + [baseline_value]
    y_min = max(0.0, min(all_values) - 0.05)
    y_max = min(1.05, max(all_values) + 0.05)
    plt.ylim(y_min, y_max)
    plt.xlim(min(x)-0.5, max(x)+0.5)
    
    # Labels
    plt.xlabel('LevelsGRQat')
    plt.ylabel('Cosine Similarity')
    
    # Ticks
    plt.xticks(x, labels=x_labels)
    
    # Legend
    # 强制放在右下角
    plt.legend(loc='lower right', frameon=True, edgecolor='#bfbfbf', fancybox=False)
    
    plt.tight_layout()
    output_path = (Path(__file__).resolve().parent.parent.parent / "output" / "figures" / "cosine_similarity_comparison.pdf")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_comparison()
