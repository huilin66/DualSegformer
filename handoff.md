# DualSegFormer 实验交接与工作计划

更新时间：2026-10-03
当前阶段：数据映射已完成，尚未开始正式复现实验
核心顺序：**先复现比赛结果，再冻结基线，最后进行新模型尝试**

## 1. 工作目标

本阶段不直接追求新的网络结构，而是先回答两个问题：

1. 当前代码、数据预处理和训练配置能否稳定复现旧版比赛方案；
2. 在旧版结果可复核之后，新模型是否带来可重复、可解释的提升。

实验结论必须区分以下三类结果：

~~~text
历史比赛结果：官方/线上榜单结果，作为历史参考，不重新包装成本地验证结果
复现结果：使用固定代码、固定数据和固定协议重新训练得到的结果
新模型结果：在复现基线冻结后，使用相同协议进行的新增实验
~~~

## 2. 当前环境与路径

### 2.1 本地仓库

~~~text
仓库：E:\repository\DualSegformer
本地比赛原始数据：Z:\huilin\bdd\cp_data\mars_seg
本地 MMLSv2：Z:\huilin\bdd\cp_data\mmlsv2
本地映射数据：Z:\huilin\bdd\cp_data\mmlsv2_mapped_mars_ls
~~~

映射脚本：

~~~text
scripts/map_mmlsv2_to_mars.py
~~~

映射报告：

~~~text
Z:\huilin\bdd\cp_data\mmlsv2_mapped_mars_ls\mapping_manifest.json
~~~

### 2.2 远端训练环境

~~~text
SSH Host：rtx6000-2
代码：/localnvme/project/DualSegformer
远端映射数据：/scrinvme/huilin/bdd/cp_data/mmlsv2_mapped_mars_ls
训练入口：/localnvme/project/DualSegformer/train.py
~~~

远端数据规模：

~~~text
train：465 images / 465 masks
val：   66 images /  66 masks
test： 133 images / 133 masks
~~~

远端 train.py 已修改为读取环境变量：

~~~bash
export MARS_DATA_ROOT=/scrinvme/huilin/bdd/cp_data/mmlsv2_mapped_mars_ls
~~~

优先级为：

~~~text
MARS_DATA_ROOT > DATA_ROOT > 原始比赛数据默认路径
~~~

训练启动时会在日志中打印最终使用的数据根目录。原文件备份为：

~~~text
/localnvme/project/DualSegformer/train.py.bak
~~~

当前 SSH 非交互环境的系统 Python 只有 Python 3.10，尚未发现可直接使用的 torch、tifffile 和 Conda 环境。正式训练前必须先激活或配置包含项目依赖的环境。

### 2.3 Shell 实验入口

顶层启动脚本为 `exp_train.sh`，默认只启动比赛复现组：

~~~bash
bash exp_train.sh reproduce
bash exp_train.sh fusion
bash exp_train.sh capacity
bash exp_train.sh new_model
bash exp_train.sh all
~~~

实验组的默认顺序是：比赛复现、旧模型融合消融、模型容量对比，最后才是 SKG 新模型。路径、seed、设备和 epoch 通过环境变量覆盖，具体配置见各组脚本。

## 3. 数据映射结论

映射后的内部通道顺序为旧比赛顺序：

~~~text
[Thermal, Slope, DEM, Grayscale, Red, Green, Blue]
~~~

MMLSv2 原始顺序为：

~~~text
[Red, Green, Blue, DEM, Slope, Thermal, Grayscale]
~~~

映射同时完成通道重排和数值反归一化，使数据可以被旧版 MarsSegDataset 按比赛数据方式读取。

### 3.1 适用范围

映射数据可以用于：

- 复用旧版比赛代码；
- 训练和验证旧版 DualSegFormer；
- 在公开 test mask 上进行冻结后的本地评估；
- 对比旧模型和新模型。

但论文中必须称为：

~~~text
MMLSv2 converted to the original Mars-LS channel ordering and radiometric representation
~~~

不能把它描述为新的独立数据集，也不能把公开 MMLSv2 test 结果直接称为官方隐藏测试集成绩。

### 3.2 已知差异

- train mask 有一个像素与本地比赛版本不同；
- Slope 通道存在少量边界零值差异；
- MMLSv2 的 test mask 是公开标签，比赛原始 test 没有本地公开 mask；
- 映射数据用于复现输入格式，不代表可以重新获得官方线上 leaderboard 评价。

## 4. 阶段 A：复现比赛结果

这一阶段完成前，不开始新模型结构搜索。

### A0. 环境和数据冒烟测试

目标：确认训练环境、数据读取、归一化和 GPU 都正确。

检查项：

- train.py 能通过语法检查；
- MARS_DATA_ROOT 生效，日志打印映射数据路径；
- train/val 数据数量分别为 465/66；
- 图像读取后形状为 7 x 128 x 128；
- mask 只包含 0/1；
- 经过比赛 mean/std 后不存在 NaN 或 Inf；
- cuda:1 能正常初始化；
- 只运行一个 batch，确认 loss 可以反向传播。

建议命令模板：

~~~bash
cd /localnvme/project/DualSegformer
export MARS_DATA_ROOT=/scrinvme/huilin/bdd/cp_data/mmlsv2_mapped_mars_ls
python3 -m py_compile train.py scripts/train_competition_reproduction.py
~~~

正式运行前需要先激活正确的 Python 环境。train.py 当前主程序会顺序运行多个模型；正式复现应使用本地单模型入口 scripts/train_competition_reproduction.py，避免一次提交完整模型列表。

### A1. 单模型复现

首先只复现历史核心模型：

~~~text
dual_segformer_convnexttiny_chv1_add
dual_segformer_convnextsmall_chv1_add
~~~

固定配置：

~~~text
数据：mmlsv2_mapped_mars_ls
训练 split：train
验证 split：val
输入尺寸：128 x 128
Epoch：100
Dual 模型 batch size：16
优化器：AdamW
学习率：1e-4
weight decay：5e-4
Scheduler：CosineAnnealingLR
随机种子：42
设备：cuda:1
通道划分：chv1，即 0,1,2,3 / 4,5,6
增强：沿用旧版代码，包括 MosaicCastDataset
~~~

注意：比赛 test 不参与 checkpoint 选择，验证必须使用 val。

单模型启动命令：

~~~bash
python scripts/train_competition_reproduction.py \
  --model-name dual_segformer_convnexttiny_chv1_add \
  --seed 42 \
  --device cuda:1
~~~

### A2. 历史模型矩阵复现

单模型链路正常后，再依次运行：

~~~text
dual_segformer_convnexttiny_chv1_add
dual_segformer_convnextsmall_chv1_add
dual_segformer_convnextbase_chv1_add
dual_segformer_convnextlarge_chv1_add
~~~

每个模型必须单独保存：

~~~text
训练配置
Git commit
随机种子
数据路径和 mapping_manifest
训练日志
best checkpoint
last checkpoint
验证指标
~~~

### A3. 复现判定标准

历史线上结果只作为参考，不要求本地 val 数值完全相等。复现通过应满足：

- 数据读取和归一化链路一致；
- 模型结构、通道划分、增强、loss 和优化器一致；
- 同一 seed 重复运行结果接近；
- 训练曲线和前景预测没有明显坍缩；
- best checkpoint 和验证指标可以从日志中追溯；
- 至少 Tiny 和 Small 两个核心模型成功完成。

历史结果参考：

~~~text
比赛 dataA 的历史 chv1_add online score：0.8665
旧记录中的 MMLSv2 train -> test chv1_cat：mIoU 0.8619，IoU_fg 0.8210，F1 0.9017
~~~

以上数值均为历史记录，不能替代本轮按 train -> val 协议产生的复现结果。

### A4. 复现阶段报告

复现阶段至少生成一张表：

| Model | Channels | Fusion | Seed | Best mIoU | Best IoU_fg | Best F1 | Best epoch | Checkpoint |
|---|---|---|---:|---:|---:|---:|---:|---|
| Tiny | chv1 | add | 42 |  |  |  |  |  |
| Small | chv1 | add | 42 |  |  |  |  |  |
| Base | chv1 | add | 42 |  |  |  |  |  |
| Large | chv1 | add | 42 |  |  |  |  |  |

只有这张表和对应 checkpoint 都齐全后，才进入阶段 B。

## 5. 阶段 B：冻结基线后的新模型尝试

### B0. 冻结事项

阶段 A 完成后冻结以下内容：

- 映射数据版本和 mapping_manifest.json；
- train/val/test split；
- 输入尺寸和基础增强；
- 评价指标；
- baseline checkpoint 选择规则；
- 训练 seed 集合；
- test 只评估一次的流程。

任何新模型都必须与冻结后的 baseline 使用同一数据和评价协议。

### B1. 新模型主线

当前仓库的新模型主线为 SKG-DualSegFormer，包括：

~~~text
single-stream / dual-stream baseline
spectral descriptor
train-only prototype bank
knowledge-guided fusion
knowledge consistency loss
~~~

建议顺序：

1. 先运行与旧模型同规模的 single-stream 和 dual-stream baseline；
2. 加入 spectral descriptor；
3. 加入 prototype prior；
4. 加入 knowledge-guided fusion；
5. 最后加入 consistency loss；
6. 每次只改变一个主要因素。

### B2. 新模型实验矩阵

最小矩阵：

| 编号 | 模型 | 目的 |
|---|---|---|
| B-01 | Single SegFormer | 单流基线 |
| B-02 | Dual SegFormer | 双流基线 |
| B-03 | Dual + descriptor | 检查显式光谱描述子 |
| B-04 | Dual + prototype | 检查类别原型先验 |
| B-05 | Dual + knowledge fusion | 检查知识引导融合 |
| B-06 | Full SKG-DualSegFormer | 完整方法 |
| B-07 | Full - descriptor | 消融 |
| B-08 | Full - prototype | 消融 |
| B-09 | Full - consistency | 消融 |

### B3. 新模型训练协议

正式结果至少使用三个 seed：

~~~text
42, 123, 7
~~~

报告：

~~~text
mean ± std
mIoU
IoU_fg
F1
precision
recall
参数量
FLOPs 或推理时间
显存占用
失败案例
~~~

归一化、光谱统计量、prototype 和聚类中心只能由 train split 计算。val 用于 checkpoint 选择和模型决策，test 必须在所有设计冻结后进行一次最终评估。

如果新模型使用映射后的 raw-like 数据，不能使用 SKG 代码默认的 normalization=auto/none 直接输入，建议使用 train-only z-score，并在实验记录中明确说明。

## 6. 数据与评价边界

必须遵守：

- MMLSv2 是公开的比赛数据发布版本，不是独立外部数据集；
- 转换后的数据可作为比赛格式复现数据，但不能制造新的 leaderboard 结果；
- 不使用 test mask 选择模型、调参或确定消融项；
- 不将历史 top-2 线上成绩和本地 val/test 指标混在同一张表中；
- 论文中单独标注“历史比赛结果”“本地复现结果”“新模型结果”；
- 所有结果必须记录 seed、代码 commit、配置、数据版本和 checkpoint 路径。

## 7. 当前待办清单

### 阶段 A：复现比赛结果

- [x] 完成 MMLSv2 到比赛格式的映射
- [x] 导出映射数据并完成数量检查
- [x] 修改远端 train.py 使用环境变量
- [x] 远端 train.py 语法检查通过
- [x] 编写按实验目的分组的 exp_train.sh 启动入口
- [ ] 激活远端正确 Conda/虚拟环境
- [ ] 完成单 batch 数据和 GPU smoke test
- [ ] 复现 Tiny chv1_add
- [ ] 复现 Small chv1_add
- [ ] 复现 Base/Large chv1_add
- [ ] 固定复现 baseline checkpoint
- [ ] 整理 train -> val 结果表

### 阶段 B：新模型

- [ ] 固定 baseline 配置和 checkpoint
- [ ] 运行 single/dual baseline
- [ ] 运行 descriptor ablation
- [ ] 运行 prototype ablation
- [ ] 运行 knowledge fusion ablation
- [ ] 运行 consistency loss ablation
- [ ] 使用 42/123/7 三个 seed
- [ ] 对冻结模型进行一次 test 评估
- [ ] 统计 mean ± std、参数量、速度和显存
- [ ] 保存可视化和失败案例
- [ ] 更新论文实验表和方法限制

## 8. 结果记录模板

每个 run 至少记录：

~~~text
run_id:
date:
remote_host:
git_commit:
data_root:
mapping_manifest:
train_split:
val_split:
test_split:
model:
encoder:
channels1:
channels2:
fusion:
loss:
augmentation:
normalization:
seed:
epochs:
batch_size:
learning_rate:
weight_decay:
best_checkpoint:
best_miou:
best_iou_fg:
best_f1:
test_evaluated: yes/no
notes:
~~~

## 9. 下一步

下一步只做阶段 A：先在远端激活正确环境，确认 MARS_DATA_ROOT 生效，然后执行 `bash exp_train.sh reproduce` 完成比赛基线训练。未完成该复现前，不开始 SKG 或其他新模型搜索。
