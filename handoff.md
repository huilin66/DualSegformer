# DualSegformer 实验交接文档

更新时间：2026-07-22  
仓库：`E:\repository\DualSegformer`

## 目标

本仓库用于“火星滑坡分割”实验复盘、补充实验和可复现训练流程整理。仓库只组织、运行和分析实验，不进行论文正文编写。

当前后续实验主线围绕表现最强的 `chv3` 通道划分继续验证：

```text
channels1 = 0,1,2
channels2 = 3,4,5,6
```

## 数据路径

数据根目录通过 `.env` 管理，便于不同服务器切换：

```env
MMLSV2_DATA_ROOT=\\158.132.186.40\isds\huilin\bdd\cp_data\mmlsv2
```

运行时也可以临时覆盖：

```sh
DATA_ROOT=/path/to/mmlsv2 sh scripts/train_chv3_experiments.sh
```

已确认 released 数据结构包含：

```text
train/images, train/masks
val/images, val/masks
test/images, test/masks
```

当前训练脚本默认：

```text
train_split = train
val_split   = test
```

也就是用 released test labels 做实验评估。

## 关键训练入口

主训练入口：

```text
train_ablation.py
```

重要能力：

```text
--summary-csv
--max-train-samples
--max-val-samples
--primary-metric
--early-stopping-patience
--min-delta
```

其中：

- `--summary-csv`：每个 run 自动追加一行结果。
- `--max-train-samples / --max-val-samples`：用于 smoke test 或小样本 dry-run。
- `--primary-metric`：控制 `best.pth` 的选择标准，可选 `miou / iou_fg / f1 / val_loss`。
- `--early-stopping-patience`：基于 `primary_metric` 的早停。

每个 run 会保存：

```text
checkpoints/best.pth
checkpoints/best_miou.pth
checkpoints/best_iou_fg.pth
checkpoints/best_f1.pth
checkpoints/best_val_loss.pth
checkpoints/last.pth
```

summary CSV 中会记录：

```text
best_iou_fg_epoch
best_iou_fg_value
best_f1_epoch
best_f1_value
best_val_loss_epoch
best_val_loss_value
best_miou_checkpoint
best_iou_fg_checkpoint
best_f1_checkpoint
best_val_loss_checkpoint
```

## 脚本说明

### 1. 对比模型训练

```sh
sh scripts/train_comparison_models.sh
```

默认输出：

```text
outputs_experiments/comparison
outputs_experiments/comparison/comparison_summary.csv
```

### 2. 原始消融实验

```sh
sh scripts/train_ablation_experiments.sh
```

默认输出：

```text
outputs_experiments/ablation
outputs_experiments/ablation/ablation_summary.csv
```

### 3. 后续 chv3 实验

```sh
sh scripts/train_chv3_experiments.sh
```

默认输出：

```text
outputs_experiments/chv3
outputs_experiments/chv3/chv3_summary.csv
```

默认实验矩阵：

```text
exp_01_chv3_add_unetformer
exp_02_chv3_add_dice
exp_03_chv3_add_ce
exp_04_chv3_add_no_mosaic
exp_05_chv3_cat_unetformer
exp_06_chv3_att_unetformer
exp_07_chv3_moe_unetformer
exp_08_chv3_moev2_unetformer
```

常用覆盖：

```sh
DEVICE=cuda:0 sh scripts/train_chv3_experiments.sh
SEEDS="42 2024 3407" DEVICE=cuda:0 sh scripts/train_chv3_experiments.sh
PRIMARY_METRIC=iou_fg DEVICE=cuda:0 sh scripts/train_chv3_experiments.sh
RUN_FUSION=0 DEVICE=cuda:0 sh scripts/train_chv3_experiments.sh
RUN_SIZE=1 SIZE_INPUT=256 SIZE_BATCH_SIZE=8 DEVICE=cuda:0 sh scripts/train_chv3_experiments.sh
```

### 4. Smoke test

数据链路测试：

```sh
sh scripts/smoke_test_data.sh
```

最小训练链路测试：

```sh
sh scripts/smoke_test_train.sh
```

smoke test 会记录脚本状态：

```text
outputs_experiments/smoke/data/smoke_status.csv
outputs_experiments/smoke/train/smoke_status.csv
```

## 已分析结果结论

已有结果显示：

- `dual_segformer_convnexttiny_chv3_add` 是目前最有效的 DualSegFormer 配置。
- `chv1_add` 表现弱，不能代表 DualSegFormer 的真实潜力。
- OCRNet HRNet-W48 是当前强 baseline。
- UPerNet ConvNeXt-Tiny 次之。
- 很多 run 后期出现前景坍缩，不能只看 `last.pth` 或 `final_miou`。

最强记录：

```text
abl_04_dual_tiny_chv3_add
best mIoU = 0.553624
best epoch = 7
IoU_fg at best mIoU = 0.393572
F1 at best mIoU = 0.564839
```

需要注意：旧 summary 中的 `best_iou_fg` 是“best mIoU 对应 epoch 的前景 IoU”，不是全训练过程最高前景 IoU。新的训练逻辑已经单独记录 `best_iou_fg_epoch / best_iou_fg_value`。

## 推荐下一步

优先运行：

```sh
DEVICE=cuda:0 sh scripts/train_chv3_experiments.sh
```

若资源紧张，先跑核心矩阵：

```sh
RUN_FUSION=0 DEVICE=cuda:0 sh scripts/train_chv3_experiments.sh
```

若重点关注前景分割：

```sh
PRIMARY_METRIC=iou_fg DEVICE=cuda:0 sh scripts/train_chv3_experiments.sh
```

后续分析时优先比较：

```text
best_miou
best_iou_fg_value
best_f1_value
final_iou_fg
best_iou_fg_epoch
best_f1_epoch
```

不要只看 `final_miou` 或 `last.pth`。

## 当前未提交改动

截至本 handoff 创建时，工作区主要改动包括：

```text
M  .gitignore
M  scripts/smoke_test_data.sh
M  scripts/smoke_test_train.sh
M  scripts/train_ablation_experiments.sh
M  scripts/train_comparison_models.sh
M  train_ablation.py
?? handoff.md
?? scripts/train_chv3_experiments.sh
```

已做静态检查：

```text
python -m py_compile train_ablation.py env_utils.py
python train_ablation.py --help
```

均通过。

## 依赖注意

实际训练环境需要完整依赖，包括：

```text
torch
tifffile
segmentation_models_pytorch
timm
python-dotenv
tqdm
numpy
```

Windows PowerShell 环境通常没有 `sh`，`.sh` 脚本应在 Linux、服务器 shell、WSL 或 Git Bash 中运行。
