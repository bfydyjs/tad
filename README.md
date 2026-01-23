# Installation
```
git clone https://github.com/bfydyjs/tad.git
cd tad
pip install . --no-build-isolation
pip install ./opentad/models/utils/post_processing/nms --no-build-isolation
```
# Prepare the Annotation and Data

# Usage

## Training
```bash
python train.py configs/ddiou/thumos_videomaev2_g.yaml
```
```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    train.py configs/ddiou/thumos_videomaev2_g.yaml
```

## Inference
```bash
python eval.py configs/ddiou/thumos_videomaev2_g.yaml --checkpoint exps/thumos/actionformer_i3d/gpu1_id0/checkpoint/epoch_34.pth
```
```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    eval.py configs/ddiou/thumos_videomaev2_g.yaml --checkpoint exps/thumos/actionformer_i3d/gpu1_id0/checkpoint/epoch_34.pth
```
## 改进
- 尽量使用稳定的社区版本包，避免重复造轮子。
- 重构了checkpoint的保存和加载代码，使其更简洁易用,并且占用资源更少。
- 保存模型从依赖验证损失改为依赖验证mAP，更符合TAD任务的需求。
- 配置文件不再支持.py结尾，可以直接使用yaml结尾的配置文件。
- 未来我会将这些改进push到原仓库中。
- 减少了对mmaction2的依赖，方便安装和使用。
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