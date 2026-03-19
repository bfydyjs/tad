# Installation
```bash
git clone https://github.com/bfydyjs/tad.git
cd tad
pip install -e . --no-build-isolation
```
# Prepare the Annotation and Data

# Usage

## Training
```bash
python tools/train.py configs/ddiou/thumos_videomaev2_g.yaml
```
```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/train.py configs/ddiou/thumos_videomaev2_g.yaml
```

## Inference
```bash
python tools/eval.py configs/ddiou/thumos_videomaev2_g.yaml --checkpoint exps/thumos/videomaev2_g/gpu1_id0/checkpoint/best.pt
```
```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    tools/eval.py configs/ddiou/thumos_videomaev2_g.yaml --checkpoint exps/thumos/videomaev2_g/gpu1_id0/checkpoint/best.pt
```
## 改进
- 尽量使用稳定的社区版本包，避免重复造轮子。
- 重构了checkpoint的保存和加载代码，使其更简洁易用,并且占用资源更少。
- 保存模型从依赖验证损失改为依赖验证mAP，更符合TAD任务的需求。
- 配置文件不再支持.py结尾，可以直接使用yaml结尾的配置文件。
- 未来我会将这些改进push到原仓库中。
- 减少了对废弃包的依赖
- 支持wandb日志记录
- 支持非分布式训练
- 支持学习率范围测试（LR Range Test）
- 检测器优化，速度加快

# 未来改进计划
- 将warmup_epoch改为按百分比（warmup_ratio）设置

- torch.compiles实现失败，如果未来 PyTorch 3.0+等高版本让用 torch.compile 成为可能，可以尝试重新启用。
- 代码不再改动


I3D :1024
Slowfast:2304
R(2+1D):512
Videomaev2_g:1408
egovlp:256

--lr-range-test不好用，不如直接看损失好
先用exp模式选出0.01x Min Loss LR，然后使用linear模型在一个范围内选出更精确的值


注释掉print(f"feature length {feat_len} is larger than padding length. Will be resized to {self.length}.")，训练速度会加快


data|training subset|Size of a single .npy file (per sample)|num_workers
-|-|-|-
anet_tsp|9987|~652.5K|1
ego4d_egovlp|1486|~900K|1
ego4d_slowfast|1486|~8.0M|2
epic_kitchens_slowfast_noun|495|~28M|2
epic_kitchens_slowfast_verb|495|~28M|2
fineaction|8436|~344K|1
hacs_slowfast|37605|~1.2M|1
thumos_i3d|200|~2.0M|2
thumos_videomaev2_g|200|~1.4M|1


特征相关性热力图、特征重要性排序图、特征累计贡献度
模型对比图、多模型性能对比雷达图，包括准确率、精确率、召回率、F1分数对比图、AUC值对比、运行时间对比
数据 t-SNE 三维降维与状态标定可视化

AUC(Area Under the Curve)
最常见的是ROC-AUC，用于评估二分类模型的性能。
## recall.py

2026-01-29 20:04:01  INFO: Fixed threshold for tiou score: [0.3, 0.4, 0.5, 0.6, 0.7]
2026-01-29 20:04:01  INFO: AUC: 73.46 (%)
2026-01-29 20:04:01  INFO: AR@  1 is 3.12%
2026-01-29 20:04:01  INFO: AR@  5 is 22.45%
2026-01-29 20:04:01  INFO: AR@ 10 is 39.62%
2026-01-29 20:04:01  INFO: AR@100 is 90.68%

recall.py中的AUC并不是ROC-AUC，而是AR-AUC。
方式 1：基于所有 tIoU 平均后的召回率曲线（当前代码实现）
- 计算过程 ：1. 对每个 proposal 数量，计算所有 tIoU 阈值的平均召回率 2. 基于这条平均召回率曲线计算 AUC
- 横坐标：平均每个视频的提议数（Average Number of Proposals per Video）
- 纵坐标：平均召回率（Average Recall）
- AUC₁ = ∫[meanₜ(recall(t, p))]dp / P_max
方式 2：所有 tIoU 各自 AUC 的平均值
- 计算过程 ：1. 对每个 tIoU 阈值，计算其单独的召回率曲线和 AUC 2. 对所有 tIoU 阈值的 AUC 求平均
- 横坐标：平均每个视频的提议数（Average Number of Proposals per Video）
- 纵坐标：召回率（Recall）
- AUC₂ = meanₜ[∫recall(t, p)dp / P_max]

在目标检测和时序动作检测领域 ：当提到 "recall" 作为评估指标时，默认指的是 数据集级别的平均召回率 （所有样本的平均）
单个样本的召回率 ：通常会明确标注为 "per-video recall" 或类似名称

方式 1 强调的是"平均召回率"的曲线下面积；方式 2 强调的是"每个 tIoU 的 AUC"的平均值，但是两种方式计算的结果是相同的。
Accuracy = (TP + TN) / (TP + TN + FP + FN)正负样本不平衡 ：视频中大部分区域是背景（负样本），动作片段（正样本）占比很小，直接计算准确率会被背景预测主导（如模型全预测为背景，即TN=1，也能达到高准确率，但无实际检测价值）。

## mAP.py
mAP.py中使用查准率（Precision）来计算AP，而不是查全率（Recall）。使用的AUC是PR-AUC。
PR-AUC=AP（Average Precision）≠P(Precision)
AR-AUC: AUC 是 平均召回率与平均每个视频的 proposal 数量曲线下的面积
mAP所有类别 AP 的算术平均值

[0.1:0.1:0.5] 是 MATLAB 的冒号运算符表示法

在“AR@AN”这一评估指标中：

AN 代表 平均建议数量（Average Number of proposals）。

具体含义：它指的是在评估过程中，为每个视频（或整个数据集）平均生成的时序动作建议（proposal）的数量。这个数量通常是算法的一个控制参数或输出结果。

AR@AN 的完整解释：

AR 是 平均召回率（Average Recall），它衡量的是算法生成的建议在多个IoU（交并比）阈值下，能够覆盖数据集中所有真实动作实例的能力。

AN 是 平均建议数量，它反映了生成建议的“密度”或“效率”。

因此，AR@AN 表示：在平均每个视频生成AN个建议的条件下，算法能达到的平均召回率。