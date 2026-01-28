import matplotlib.pyplot as plt
from pathlib import Path

def main():
    # 1. 数据结构优化：将数据组织在一起，方便管理和修改
    data = [
        {"name": "FPN(yolo11n)", "params": 2.1, "mAP": 32.8, "marker": 'o', "color": 'lightblue'},
        {"name": "BIFPN",        "params": 2.3, "mAP": 33.9, "marker": 's', "color": 'lightcoral'},
        {"name": "BIFPN-GLSA",   "params": 2.4, "mAP": 31.5, "marker": 'D', "color": 'gold'},
        {"name": "HSFPN",        "params": 2.6, "mAP": 31.2, "marker": 'p', "color": 'plum'},
        {"name": "AFPN",         "params": 2.8, "mAP": 30.5, "marker": 'v', "color": 'lightgreen'}, # 修改 marker 为 'v' 避免重复
        {"name": "FreqGFPN",     "params": 2.5, "mAP": 30.8, "marker": 'X', "color": 'orange'},     # 修改 yellow -> orange 提高对比度
        {"name": "GFPN",         "params": 3.0, "mAP": 32.0, "marker": '^', "color": 'gray'},       # 修改 lightgray -> gray 提高可见性
        {"name": "CGFPRN",       "params": 3.4, "mAP": 30.2, "marker": '<', "color": 'tan'},
        {"name": "MAFPN",        "params": 3.2, "mAP": 31.8, "marker": '>', "color": 'teal'},
        {"name": "EHMPFN (Ours)","params": 2.0, "mAP": 35.5, "marker": '*', "color": 'red'},        # 标红显示自己的方法
    ]

    # 设置字体样式（可选，更像论文风格）
    plt.rcParams['font.family'] = 'sans-serif' 
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']

    # Create a figure
    plt.figure(figsize=(10, 7))

    # 2. 绘制散点图
    for item in data:
        # 增加 zorder=3 确保点在网格线上面
        plt.scatter(
            item["params"], item["mAP"],
            s=130, # 稍微加大点的大小
            marker=item["marker"],
            color=item["color"],
            label=item["name"],
            edgecolors='k', # 黑色边框
            linewidth=0.8,
            alpha=0.9,
            zorder=3
        )

    # 3. 标注最佳模型
    # 自动查找包含 "Ours" 的模型数据，避免硬编码索引
    best_model = next((item for item in data if "Ours" in item["name"]), None)
    
    if best_model:
        plt.annotate(
            'Best Performance',
            xy=(best_model["params"] + 0.01, best_model["mAP"] - 0.01), # 箭头指向位置、微调以避免遮挡
            xytext=(best_model["params"] + 0.2, best_model["mAP"] - 0.2), # 调整文本位置到右下方
            arrowprops=dict(
                arrowstyle='->',
                color='red',
                lw=2,
                connectionstyle='arc3,rad=-0.2'
            ),
            fontsize=12,
            color='#d62728', # 较深的红色
            weight='bold',
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#d62728", alpha=0.9)
        )

    # 4. 图表装饰
    plt.xlabel("Parameters (M)", fontsize=14, fontweight='bold')
    plt.ylabel("mAP@0.5 (%)", fontsize=14, fontweight='bold')
    plt.title("Performance vs. Efficiency Comparison", fontsize=16, pad=20)

    # 增加网格线，设为虚线
    plt.grid(True, linestyle='--', alpha=0.5, zorder=0)

    # 自动计算边界并增加一点留白
    all_params = [d["params"] for d in data]
    all_maps = [d["mAP"] for d in data]
    p_margin = (max(all_params) - min(all_params)) * 0.1
    m_margin = (max(all_maps) - min(all_maps)) * 0.1
    
    plt.xlim(min(all_params) - p_margin, max(all_params) + p_margin)
    plt.ylim(min(all_maps) - m_margin, max(all_maps) + m_margin * 2.0) # 顶部留更多空间给图例

    # Legend (分两列显示，避免太长)
    plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=True, ncol=2, fontsize=10)

    plt.tight_layout()

    # 5. 保存文件
    script_dir = Path(__file__).resolve().parent
    output_path = (script_dir / ".." / "output" / "figures" / "scatter_plot.png").resolve()
    
    print(f"Saving figure to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    main()
