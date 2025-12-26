# Installation
```
git clone https://gitcode.com/vxz/tad.git
cd tad
pip install . --no-build-isolation
pip install ./opentad/models/utils/post_processing/nms --no-build-isolation
```
# Prepare the Annotation and Data

# Usage

## Training
```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    train.py configs/ddiou/thumos_videomaev2_g.yaml
```
## Inference and Evaluation
```bash
torchrun \
    --nnodes=1 \
    --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    eval.py \
    configs/ddiou/thumos_videomaev2_g.yaml \
    --checkpoint exps/thumos/actionformer_i3d/gpu1_id0/checkpoint/epoch_34.pth
```
## 改进
- 尽量使用稳定的社区版本包，避免重复造轮子。
- 重构了checkpoint的保存和加载代码，使其更简洁易用,并且占用资源更少。
- 保存模型从依赖验证损失改为依赖验证mAP，更符合TAD任务的需求。
- 配置文件不再支持.py结尾，可以直接使用yaml结尾的配置文件。
- 未来我会将这些改进push到原仓库中。
- 减少了对mmaction2的依赖，方便安装和使用。
- 减少了对废弃包的依赖


I3D :1024
Slowfast:2304
R(2+1D):512
Videomaev2_g:1408
egovlp:256