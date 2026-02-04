import argparse
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from pathlib import Path
from tqdm import tqdm

# 添加项目根目录到路径，确保能导入 tad 模块
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from tad.models import build_detector
from tad.datasets import build_dataset
from tad.utils import Config

def parse_args():
    parser = argparse.ArgumentParser(description="Calculate and Plot Average Cosine Similarity across Layers")
    parser.add_argument("config", help="Path to config file (e.g., configs/anet_tsp.yaml)")
    parser.add_argument("checkpoint", help="Path to checkpoint file (e.g., work_dirs/best.pt)")
    parser.add_argument("--index", type=int, default=0, help="Index of the video sample to analyze (only used if --samples is not specified)")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples to average over. If not set, use all samples.")
    parser.add_argument("--output", default="average_cosine_similarity.png", help="Output image filename")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    return parser.parse_args()

def compute_avg_cosine_similarity(feature_tensor):
    """
    计算给定特征Tensor的平均余弦相似度。
    feature_tensor: [C, T] or [1, C, T]
    """
    # 移除 batch 维度 [1, C, T] -> [C, T]
    if feature_tensor.dim() == 3: 
        feature_tensor = feature_tensor[0]
    
    # 转置为 [T, C] 用于计算
    features = feature_tensor.transpose(0, 1).detach().cpu().numpy() 
    
    # 归一化 (L2 Norm)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features_norm = features / (norms + 1e-8)
    
    # 计算余弦相似度矩阵 [T, T]
    similarity_matrix = np.dot(features_norm, features_norm.T)
    
    # 计算平均值 matrix mean
    # 注意：通常包含对角线(1.0)能反映整体分布，也可以选择去除对角线只看互相似度
    # 这里直接采用整体平均值，反映了特征在时间维度上的整体一致性/平滑度
    avg_sim = np.mean(similarity_matrix)
        
    return avg_sim

def main():
    args = parse_args()
    
    # 1. 加载配置
    print(f"Loading config from {args.config}...")
    cfg = Config.fromfile(args.config)
    cfg.dataset.val.test_mode = False 
    
    # 2. 构建数据集
    print("Building dataset...")
    try:
        dataset = build_dataset(cfg.dataset.val)
    except Exception as e:
        print(f"Error building dataset: {e}")
        return
    print(f"Dataset loaded with {len(dataset)} samples.")

    # 3. 构建模型
    print(f"Building model and loading checkpoint from {args.checkpoint}...")
    model = build_detector(cfg.model)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
        
    model.to(args.device)
    model.eval()

    # Determine range of samples
    if args.samples is None:
        num_samples = len(dataset)
        indices = range(num_samples)
        title_suffix = f"(All {num_samples} samples)"
    elif args.samples == 1:
        num_samples = 1
        indices = [args.index]
        metas = dataset[args.index]['metas'].data if hasattr(dataset[args.index]['metas'], 'data') else dataset[args.index]['metas']
        video_name = metas.get('video_name', f'sample_{args.index}')
        title_suffix = f": {video_name}"
    else:
        num_samples = min(args.samples, len(dataset))
        indices = range(num_samples)
        title_suffix = f"({num_samples} samples)"

    print(f"Processing {len(indices)} samples...")

    global_raw_sim = []
    global_layer_sims = {}

    for i in tqdm(indices):
        data_sample = dataset[i]
        
        inputs = data_sample['inputs'].to(args.device).unsqueeze(0) 
        masks = data_sample['masks'].to(args.device).unsqueeze(0)   
        
        # (a) Raw Features
        raw_sim = compute_avg_cosine_similarity(inputs[0])
        global_raw_sim.append(raw_sim)

        # (b) Model Output Features
        with torch.no_grad():
            feats, _ = model.extract_feat(inputs, masks)
            
        if isinstance(feats, (list, tuple)):
            for lvl, f in enumerate(feats):
                if lvl not in global_layer_sims: global_layer_sims[lvl] = []
                avg_sim = compute_avg_cosine_similarity(f)
                global_layer_sims[lvl].append(avg_sim)
        else:
             if 0 not in global_layer_sims: global_layer_sims[0] = []
             avg_sim = compute_avg_cosine_similarity(feats)
             global_layer_sims[0].append(avg_sim)

    # Calculate global averages
    avg_raw_sim = np.mean(global_raw_sim)
    
    sorted_levels = sorted(global_layer_sims.keys())
    avg_layer_sims = [np.mean(global_layer_sims[lvl]) for lvl in sorted_levels]

    print(f"Final Results:")
    print(f" >> Avg Raw Sim: {avg_raw_sim:.4f}")
    for i, sim in zip(sorted_levels, avg_layer_sims):
        print(f" >> Avg Layer {i} Sim: {sim:.4f}")

    print("-" * 30)
    print(f"Data for {args.config}:")
    # Format: [Raw, Level 0, Level 1, ...]
    data_list = [avg_raw_sim] + avg_layer_sims
    print(f"[{', '.join(f'{x:.4f}' for x in data_list)}]")
    print("-" * 30)

if __name__ == "__main__":
    main()
