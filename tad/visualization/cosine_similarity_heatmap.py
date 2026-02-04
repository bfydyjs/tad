# python tad/visualization/cosine_similarity_heatmap.py configs/ddiou/thumos_videomaev2_g.yaml exps/thumos/videomaev2_g/gpu1_id1/checkpoint/best.pt

import argparse
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from tad.models import build_detector
from tad.datasets import build_dataset
from tad.utils import Config

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize Cosine Similarity Heatmap from Real Model Features")
    parser.add_argument("config", help="Path to config file (e.g., configs/anet_i3d.yaml)")
    parser.add_argument("checkpoint", help="Path to checkpoint file (e.g., work_dirs/xxx/best.pt)")
    parser.add_argument("--index", type=int, default=0, help="Index of the video sample in validation set to visualize")
    parser.add_argument("--output", default="heatmap_features.png", help="Output image filename")
    parser.add_argument("--use_input", action="store_true", help="If set, visualize input features instead of encoder output")
    parser.add_argument("--level", type=int, default=0, help="FPN feature level to visualize (default: 0)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. 加载配置
    print(f"Loading config from {args.config}...")
    cfg = Config.fromfile(args.config)
    
    # --- PATCH: 强制开启 GT 加载 ---
    # 因为验证集配置通常 test_mode=True 会跳过加载 GT，且 pipeline 中也不包含 GT
    print("Patching config to enable GT loading...")
    cfg.dataset.val.test_mode = False 
    
    # 修改 pipeline 以包含 gt_segments
    for transform in cfg.dataset.val.pipeline:
        if transform['type'] == 'ConvertToTensor':
             if 'gt_segments' not in transform['keys']:
                transform['keys'].append('gt_segments')
        if transform['type'] == 'Collect':
            if 'gt_segments' not in transform['keys']:
                transform['keys'].append('gt_segments')

    # 2. 构建数据集 (使用验证集以获取GT)
    print("Building dataset...")
    # 为了简化，只构建验证集
    dataset = build_dataset(cfg.dataset.val)
    print(f"Dataset loaded with {len(dataset)} samples.")

    # 3. 构建模型并加载权重
    if not args.use_input:
        print(f"Building model and loading checkpoint from {args.checkpoint}...")
        model = build_detector(cfg.model)
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        model.to(args.device)
        model.eval()
    
    # 4. 获取样本数据
    print(f"Processing sample index: {args.index}")
    data_sample = dataset[args.index]
    
    # 获取输入特征和掩码
    # dataset通常返回 [C, T] 格式的tensor
    inputs = data_sample['inputs'].to(args.device).unsqueeze(0) # 增加Batch维度 -> [1, C, T]
    masks = data_sample['masks'].to(args.device).unsqueeze(0)   # -> [1, T]

    # 获取元数据 (用于时间对齐)
    metas = data_sample['metas'].data if hasattr(data_sample['metas'], 'data') else data_sample['metas']
    video_name = metas.get('video_name', f'sample_{args.index}')
    fps = metas.get('fps', None)
    
    # 获取 Feature Stride (通常在配置中)
    feature_stride = 1
    if 'common' in cfg.dataset and 'feature_stride' in cfg.dataset.common:
        feature_stride = cfg.dataset.common.feature_stride
    elif 'feature_stride' in cfg.dataset.val:
        feature_stride = cfg.dataset.val.feature_stride
    
    print(f"Video: {video_name}, FPS: {fps}, Stride: {feature_stride}")

    # 5. 获取待分析的特征
    if args.use_input:
        print("Using RAW INPUT features.")
        feature_tensor = inputs[0] # [C, T]
    else:
        print("Extracting MODEL OUTPUT features...")
        with torch.no_grad():
            # extract_feat 通常返回 (feats, masks)
            # feats 可能是单个Tensor或Tensor列表(多尺度FPN)
            feats, _ = model.extract_feat(inputs, masks)
        
        if isinstance(feats, (list, tuple)):
            print(f"Model returned {len(feats)} feature levels. Selecting level {args.level}.")
            feature_tensor = feats[args.level] 
        else:
            feature_tensor = feats
        
        # 移除Batch维度 [1, C, T] -> [C, T]
        if feature_tensor.dim() == 3:
            feature_tensor = feature_tensor[0]

    # 转置为 [T, C] 用于计算相似度
    features = feature_tensor.transpose(0, 1).cpu().numpy() # [T, C]
    T, D = features.shape
    print(f"Feature shape for heatmap: Time={T}, Dim={D}")

    # 6. 计算余弦相似度矩阵
    print("Computing cosine similarity...")
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features_norm = features / (norms + 1e-8) # 避免除零
    similarity_matrix = np.dot(features_norm, features_norm.T)

    # 7. 处理 Ground Truth 时间区间
    # GT 通常是秒为单位 [K, 2]
    gt_segments = data_sample['gt_segments'].cpu().numpy()
    
    # 计算缩放因子: 1个时间步对应多少秒?
    # Index = Seconds * FPS / Stride  => Seconds = Index * Stride / FPS
    # 所以 Seconds_per_step = Stride / FPS
    seconds_per_step = 1.0
    if fps:
        seconds_per_step = feature_stride / fps
    else:
        print("Warning: FPS not found. Assuming 1:1 mapping (Index=Seconds).")

    gt_intervals = gt_segments # GT 本身就是秒，不需要再乘 scale_factor 了，只需确认 dataset 返回的是秒即可
    # 注意：如果之前代码 gt_segments 是秒，那就不变。
    # 之前代码写的是: gt_intervals = gt_segments * scale_factor，但这取决于 scale_factor 定义。
    # 让我们重新理一下：
    # dataset 返回的 gt_segments 通常是秒。
    # 绘图时，我们的底座是 heatmap，它的坐标是 0, 1, 2... T (indices)。
    # 要把 GT 画在 heatmap 上，我们需要把 GT(秒) 转换成 Index。
    # Index = Seconds / Seconds_per_step
    
    gt_intervals_indices = gt_segments / seconds_per_step

    # 8. 绘图
    print("Plotting heatmap...")
    
    # --- PAPER STYLE CONFIG ---
    # 设置适合论文发表的字体和大小
    plt.rcParams.update({
        'font.family': 'serif',          # 使用衬线体 (这是论文标准，接近 Times New Roman)
        'font.size': 14,                 # 全局字体大小
        'axes.labelsize': 16,            # 坐标轴标签大小
        'xtick.labelsize': 14,           # 刻度标签大小
        'ytick.labelsize': 14,           # 刻度标签大小
        'lines.linewidth': 2,            # 线宽
    })

    # 使用 GridSpec 将 Colorbar 放在单独的列，确保 ax1 和 ax2 左侧和右侧严格对齐
    # 调整 figsize 为 (11, 11) 以获得接近正方形的视觉效果 (1:1 比例)，避免热力图变形
    fig = plt.figure(figsize=(10, 10)) # 稍微调小一点尺寸，字体会显得更大，适合插入文档
    # 调小 hspace (0.1 -> 0.05) 让上下图靠得更近
    gs = fig.add_gridspec(2, 2, width_ratios=[50, 1], height_ratios=[20, 1], wspace=0.02, hspace=0.05)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[1, 0], sharex=ax1)
    cbar_ax = fig.add_subplot(gs[0, 1])
    
    # (a) Similarity Heatmap
    # cbar_ax 参数让 colorbar 绘制在指定的轴上，不挤压 ax1
    # 推荐移除 Colorbar 标签，保持图面整洁，相关含义应在论文 Figure Caption 中说明
    sns.heatmap(similarity_matrix, cmap='viridis', ax=ax1, cbar_ax=cbar_ax) 
    # sns.heatmap(similarity_matrix, cmap='viridis', ax=ax1, cbar_ax=cbar_ax, vmin=0.0, vmax=1.0) 
    
    # 旋转 Colorbar 标签以节省空间
    # cbar_ax.yaxis.label.set_size(14)
   
    # 优化 Colorbar 样式：
    # 1. 保留数值：这是必要的，为了显示数据范围（0~1 或 -1~1），让图表有定量的意义。
    # 2. 隐藏刻度线（Tick Marks）：为了与主图风格统一，保持整洁。
    cbar_ax.tick_params(which='both', length=0) 
    # 3. 设置刻度字体大小 (可选，确保和主图一致)
    cbar_ax.tick_params(labelsize=14)


    # --- Fix: Use SECONDS for axis labels ---
    import matplotlib.ticker as ticker
    
    # 定义一个格式化函数：将 index 转换为 seconds
    def index_to_seconds(x, pos):
        return f"{x * seconds_per_step:.1f}s"

    # 使用 MaxNLocator 自动选择约 10 个漂亮的整数刻度 (基于 index)
    locator = ticker.MaxNLocator(nbins=10, integer=True)
    
    # 设置 X 轴 (ax1 和 ax2 是共享的，设置 ax1 即可)
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(index_to_seconds)) # 显示为秒
    
    # 设置 Y 轴
    ax1.yaxis.set_major_locator(ticker.MaxNLocator(nbins=10, integer=True))
    # ax1.yaxis.set_major_formatter(ticker.FuncFormatter(index_to_seconds)) # 用户要求不再显示纵轴数值

    # 2. 隐藏刻度线 (Ticks)，但保留刻度标签 (Labels)
    ax1.tick_params(axis='both', which='both', length=0)
    ax2.tick_params(axis='both', which='both', length=0)

    # 3. 处理标签显示
    # 隐藏 ax1 的 X 轴标签
    plt.setp(ax1.get_xticklabels(), visible=False)
    
    # 隐藏 ax1 的 Y 轴标签 (只保留标题)
    plt.setp(ax1.get_yticklabels(), visible=False)

    # 强制显示 ax2 的 X 轴标签
    plt.setp(ax2.get_xticklabels(), visible=True)
    
    ax1.set_ylabel('Time (s)', fontweight='bold') # 加粗标签
    # model_type = "Input Features" if args.use_input else "Encoder Output Features"
    # ax1.set_title(f'{model_type} Similarity: {video_name}') # 论文中通常不需要图内标题，移除
    
    # 叠加 GT 框 (使用 Index 坐标绘制，因为底图坐标系是 Index)
    for start, end in gt_intervals_indices:
        # 确保不超出特征长度
        start = max(0, start)
        end = min(T, end)
        if end > start:
            # 增加线宽 linewidth=3，颜色加深，使其在热力图上更清晰
            rect = Rectangle((start, start), end - start, end - start,
                             linewidth=3, edgecolor='#FF3333', facecolor='none', linestyle='-') # 实线可能比虚线更清晰
            ax1.add_patch(rect)
        
    # (b) Timeline Bar
    ax2.set_xlim(0, T)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Time (s)', fontweight='bold') # 加粗标签
    ax2.set_ylabel('GT', rotation=0, labelpad=20, fontweight='bold', va='center') # 优化 GT 标签位置: 水平放置
    ax2.set_yticks([])
    
    for start, end in gt_intervals_indices:
        start = max(0, start)
        end = min(T, end)
        # 使用稍微深一点的绿色，在打印时对比度更好
        ax2.fill_between([start, end], 0, 1, color='#32CD32', alpha=0.8)
        
    # plt.tight_layout() # GridSpec 布局下通常不需要 tight_layout，且可能破坏对齐
    output_path = (Path(__file__).resolve().parent.parent.parent / "output" / "figures" / "heatmap.png")
    
    print(f"Saving figure to: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    print(f"Done. Saved visualization to {output_path}")

if __name__ == "__main__":
    main()