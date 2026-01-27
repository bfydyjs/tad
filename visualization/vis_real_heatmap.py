import argparse
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Rectangle

# 将项目根目录添加到路径，以便导入 opentad
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from opentad.models import build_detector
from opentad.datasets import build_dataset
from opentad.utils import Config

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
    # Index = Seconds * FPS / Stride
    scale_factor = 1.0
    if fps:
        scale_factor = fps / feature_stride
    else:
        print("Warning: FPS not found. Assuming 1:1 mapping.")

    gt_intervals = gt_segments * scale_factor

    # 8. 绘图
    print("Plotting heatmap...")
    plt.figure(figsize=(12, 10))
    
    # (a) Similarity Heatmap
    ax1 = plt.subplot(2, 1, 1)
    sns.heatmap(similarity_matrix, cmap='YlGnBu', cbar=True, ax=ax1)
    ax1.set_xlabel('Time (index)')
    ax1.set_ylabel('Time (index)')
    model_type = "Input Features" if args.use_input else "Encoder Output Features"
    ax1.set_title(f'{model_type} Similarity: {video_name}')
    
    # 叠加 GT 框
    for start, end in gt_intervals:
        # 确保不超出特征长度
        start = max(0, start)
        end = min(T, end)
        if end > start:
            rect = Rectangle((start, start), end - start, end - start,
                             linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
            ax1.add_patch(rect)
        
    # (b) Timeline Bar
    ax2 = plt.subplot(2, 1, 2)
    ax2.set_xlim(0, T)
    ax2.set_ylim(0, 1)
    ax2.set_xlabel('Time (index)')
    ax2.set_ylabel('Action Existence')
    ax2.set_yticks([])
    ax2.set_title('Ground Truth Timeline')
    
    for start, end in gt_intervals:
        start = max(0, start)
        end = min(T, end)
        ax2.fill_between([start, end], 0, 1, color='lightgreen', alpha=0.5)
        
    plt.tight_layout()
    plt.savefig(args.output, dpi=300)
    print(f"Done. Saved visualization to {args.output}")

if __name__ == "__main__":
    main()