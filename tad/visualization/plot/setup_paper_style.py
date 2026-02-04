import matplotlib.pyplot as plt

def setup_paper_style():
    """配置论文绘图风格"""
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
