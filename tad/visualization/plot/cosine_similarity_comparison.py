import matplotlib.pyplot as plt
from pathlib import Path
from setup_paper_style import setup_paper_style



# ==========================================
# 1. 在这里填入你运行 average_cosine_similarity.py 得到的数字
# 2. 服务器上不支持Arial字体，不要在服务器上运行此脚本生成图表
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
# 样式配置已移至 setup_paper_style.py

def plot_comparison():
    setup_paper_style()
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
    plt.grid(True, linestyle='--', color='#bdbdbd')
    
    # Y-Axis Limits (自动调整 + buffer)
    all_values = y_a + y_b + [baseline_value]
    y_min = max(0.0, min(all_values) - 0.05)
    y_max = min(1.05, max(all_values) + 0.05)
    plt.ylim(y_min, y_max)
    plt.xlim(min(x)-0.5, max(x)+0.5)
    
    # Labels
    plt.xlabel('Levels')
    plt.ylabel('Cosine Similarity')
    
    # Ticks
    plt.xticks(x, labels=x_labels)
    
    # Legend
    # 强制放在右下角
    plt.legend(loc='lower right', frameon=True, edgecolor='#bfbfbf', fancybox=False)
    
    plt.tight_layout()
    output_path = (Path(__file__).resolve().parent.parent.parent.parent / "output" / "figures" / "cosine_similarity_comparison.pdf")
    # 确保目录存在
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_comparison()
