# DualSegformer 实验交接文档

更新时间：2026-07-22  
仓库：`/localnvme/project/DualSegformer`

## 目标

本仓库用于"火星滑坡分割"实验复盘、补充实验和可复现训练流程整理。仓库只组织、运行和分析实验，不进行论文正文编写。

当前实验主线围绕 **chv1 通道划分 + 融合方式探索**：

```text
channels1 = 0,1,2,3   (VIS/NIR 波段)
channels2 = 4,5,6     (SWIR 波段)
```

依据：
- dataA（比赛数据）：chv1_add 最佳，online score = 0.8665
- dataB（mmlsv2 发布数据）：chv1_cat 最佳，best IoU_fg = 0.8210

## 数据集说明

### 两套数据

| 数据集 | 路径 | 像素值域 | dataset_type |
|--------|------|----------|:---:|
| 比赛原始数据 (dataA) | `/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase` | 原始 DN [0, ~5629] | `mars_ls` |
| mmlsv2 发布数据 (dataB) | `/scrinvme/huilin/bdd/cp_data/mmlsv2` | 已归一化 [0, 1] | `mmlsv2` |

**重要**：两组数据文件名和 mask 完全一致，但像素值无相关性（不是同一图像的不同归一化版本，而是不同处理流程的产物）。

### 数据路径配置

通过 `.env` 管理：

```env
MMLSV2_DATA_ROOT=/scrinvme/huilin/bdd/cp_data/mmlsv2
# MMLSV2_DATA_ROOT=/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase
```

运行时也可临时覆盖：

```sh
DATA_ROOT=/path/to/data sh scripts/train_chv1_fusion_experiments.sh
```

### 数据结构差异

```text
mmlsv2:       train(465) / val(66) / test(133 images + 133 masks)  ← test 有标签
比赛数据:     train(465) / val(66) / test(133 images, 无 masks)    ← test 无标签
```

**比赛数据必须用 `VAL_SPLIT=val`**，否则因 test/masks 不存在而报错。

### 默认训练配置

```text
train_split = train
val_split   = test     (mmlsv2 可用 test；比赛数据须改为 val)
```

## 关键训练入口

主训练入口：`train_ablation.py`

### 核心参数

| 参数 | 说明 |
|------|------|
| `--dataset-type {mars_ls, mmlsv2}` | 数据类型。mars_ls 应用 mean/std 归一化；mmlsv2 跳过归一化 |
| `--summary-csv` | 每个 run 自动追加一行结果到 CSV |
| `--primary-metric` | best.pth 选择标准：miou / iou_fg / f1 / val_loss |
| `--early-stopping-patience` | 基于 primary_metric 的早停（0=禁用） |
| `--max-train-samples / --max-val-samples` | smoke test 或小样本 dry-run |
| `--model-name auto --arch dual_segformer --encoder ... --channels1 ... --channels2 ... --fusion ...` | 灵活组装模型 |

### 归一化逻辑

```text
dataset_type=mars_ls  → 使用 MARS_MEAN_TRAIN/STD_TRAIN 做标准化（适合原始 DN 数据）
dataset_type=mmlsv2   → 跳过归一化，直接使用 [0,1] 值（MMLSV2Dataset 类）
```

`dataset.py` 中的 `MMLSV2Dataset` 继承 `MarsSegDataset`，禁用 mean/std 归一化。

### 每个 run 保存

```text
checkpoints/best.pth
checkpoints/best_miou.pth
checkpoints/best_iou_fg.pth
checkpoints/best_f1.pth
checkpoints/best_val_loss.pth
checkpoints/last.pth
```

## 脚本说明

### 1. 消融实验 (16 个)

```sh
sh scripts/train_ablation_experiments.sh
```

输出：`outputs_experiments/ablation/ablation_summary.csv`

涵盖：架构贡献、通道划分、融合策略、损失函数、增强/Mosaic、Backbone 规模。

### 2. 对比模型训练 (10 个)

```sh
sh scripts/train_comparison_models.sh
```

输出：`outputs_experiments/comparison/comparison_summary.csv`

涵盖：M3LSNet、OCRNet、UPerNet、SegFormer、DualSegFormer 各尺寸。

### 3. Chv1 融合探索 (默认 12 个)

```sh
sh scripts/train_chv1_fusion_experiments.sh
```

输出：`outputs_experiments/chv1_fusion/chv1_fusion_summary.csv`

实验矩阵：

```text
# 核心融合对比 (5)
exp_01_chv1_add / exp_02_chv1_cat / exp_03_chv1_att / exp_04_chv1_moe / exp_05_chv1_moev2

# Loss 探索 (6, RUN_LOSS=1)
exp_06~08: cat × {ce, dice, combined}
exp_09~11: add × {ce, dice, combined}

# Mosaic 消融 (1)
exp_12_chv1_cat_no_mosaic

# 可选：无增强 (RUN_NOAUG=1)、输入尺寸 (RUN_SIZE=1)、Backbone (RUN_BACKBONE=1)
```

常用覆盖：

```sh
# mmlsv2 数据
DATASET_TYPE=mmlsv2 sh scripts/train_chv1_fusion_experiments.sh

# 比赛数据
DATA_ROOT=/scrinvme/.../Mars_LSc_2025_dataset_1st_phase VAL_SPLIT=val \
  sh scripts/train_chv1_fusion_experiments.sh

# 多种子
SEEDS="42 123 7" sh scripts/train_chv1_fusion_experiments.sh

# 全开
SEEDS="42 123 7" RUN_SIZE=1 RUN_BACKBONE=1 RUN_NOAUG=1 \
  sh scripts/train_chv1_fusion_experiments.sh
```

### 4. Smoke test

```sh
sh scripts/smoke_test_data.sh    # 数据链路
sh scripts/smoke_test_train.sh   # 最小训练链路
```

### 5. 结果汇总

```sh
python summarize_results.py outputs_experiments/ablation outputs_experiments/comparison
python summarize_results.py outputs_experiments --recursive --sort-by best_iou_fg
python summarize_results.py outputs_experiments -r --output results_summary.csv
```

## 已确认的实验结论

### 比赛数据 (dataA) — online score

| 通道 | add | cat | att | moe |
|:---:|:---:|:---:|:---:|:---:|
| chv1 | **0.8665** | 0.8391 | 0.8372 | 0.7667 |
| chv2 | 0.8628 | 0.8592 | 0.8562 | 0.8522 |
| chv3 | 0.7646 | — | — | — |

### mmlsv2 数据 (dataB) — 修复归一化后

#### 消融实验完整结果 (`outputs_experiments/ablation/ablation_summary.csv`)

数据：mmlsv2, train→test, 100 epochs, seed=42

| # | 实验 | 模型/配置 | Loss | best_miou | best_iou_fg | best_f1 | best_epoch | final_iou_fg |
|:---:|------|------|:---:|:---:|:---:|:---:|:---:|:---:|
| 01 | single_segformer_tiny | 单流 baseline | unetformer | 0.8519 | 0.8085 | 0.8941 | 63 | 0.8014 |
| 02 | dual_tiny_chv1_add | chv1 + add | unetformer | 0.8543 | 0.8109 | 0.8956 | 86 | 0.8103 |
| 03 | dual_tiny_chv2_add | chv2 + add | unetformer | 0.8545 | 0.8116 | 0.8960 | 88 | 0.8112 |
| 04 | dual_tiny_chv3_add | chv3 + add | unetformer | 0.8574 | 0.8152 | 0.8982 | 68 | 0.8097 |
| **05** | **dual_tiny_chv1_cat** | **chv1 + cat** | unetformer | **0.8619** | **0.8210** | **0.9017** | 70 | 0.8204 |
| 06 | dual_tiny_chv1_att | chv1 + att | unetformer | 0.8569 | 0.8148 | 0.8979 | 62 | 0.8107 |
| 07 | dual_tiny_chv1_moe | chv1 + moe | unetformer | 0.8478 | 0.8030 | 0.8907 | 88 | 0.8023 |
| 08 | dual_tiny_chv1_moev2 | chv1 + moev2 | unetformer | 0.8543 | 0.8109 | 0.8956 | 80 | 0.8085 |
| 09 | dual_tiny_loss_combined | chv1 + add | combined | 0.8545 | 0.8108 | 0.8955 | 67 | 0.8078 |
| 10 | dual_tiny_loss_ce | chv1 + add | ce | 0.8594 | 0.8179 | 0.8998 | 65 | 0.8110 |
| 11 | dual_tiny_loss_dice | chv1 + add | dice | 0.8461 | 0.8006 | 0.8892 | 74 | 0.7910 |
| 12 | dual_tiny_no_mosaic | chv1 + add, mosaic=0 | unetformer | 0.8557 | 0.8127 | 0.8967 | 70 | 0.8045 |
| 13 | dual_tiny_no_aug_no_mosaic | chv1 + add, 无增强 | unetformer | 0.8402 | 0.7935 | 0.8849 | 42 | 0.7884 |
| 14 | dual_small_chv1_add | chv1 + add (small) | unetformer | 0.8568 | 0.8142 | 0.8976 | 74 | 0.8136 |
| 15 | dual_base_chv1_add | chv1 + add (base) | unetformer | 0.8583 | 0.8160 | 0.8987 | 84 | 0.8144 |

#### Chv1 融合探索完整结果 (`outputs_experiments/chv1_fusion/chv1_fusion_summary.csv`)

数据：mmlsv2, train→test, 100 epochs, seed=42, chv1 (0,1,2,3 / 4,5,6)

| # | 实验 | 融合 | Loss | Mosaic | best_miou | best_iou_fg | best_f1 | best_epoch | final_iou_fg |
|:---:|------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **02** | **chv1_cat** | **cat** | unetformer | 0.5 | **0.8619** | **0.8210** | **0.9017** | 70 | 0.8204 |
| 09 | chv1_add_ce | add | ce | 0.5 | 0.8594 | 0.8179 | 0.8998 | 65 | 0.8110 |
| 08 | chv1_cat_combined | cat | combined | 0.5 | 0.8570 | 0.8148 | 0.8980 | 71 | 0.8127 |
| 03 | chv1_att | att | unetformer | 0.5 | 0.8569 | 0.8148 | 0.8979 | 62 | 0.8107 |
| 06 | chv1_cat_ce | cat | ce | 0.5 | 0.8558 | 0.8122 | 0.8964 | 67 | 0.8108 |
| 11 | chv1_add_combined | add | combined | 0.5 | 0.8545 | 0.8108 | 0.8955 | 67 | 0.8078 |
| 05 | chv1_moev2 | moev2 | unetformer | 0.5 | 0.8543 | 0.8109 | 0.8956 | 80 | 0.8085 |
| 01 | chv1_add | add | unetformer | 0.5 | 0.8543 | 0.8109 | 0.8956 | 86 | 0.8103 |
| 07 | chv1_cat_dice | cat | dice | 0.5 | 0.8522 | 0.8092 | 0.8946 | 69 | 0.8025 |
| 12 | chv1_cat_no_mosaic | cat | unetformer | 0.0 | 0.8507 | 0.8070 | 0.8932 | 79 | 0.8018 |
| 10 | chv1_add_dice | add | dice | 0.5 | 0.8461 | 0.8006 | 0.8892 | 74 | 0.7910 |
| 04 | chv1_moe | moe | unetformer | 0.5 | 0.8478 | 0.8030 | 0.8907 | 88 | 0.8023 |

### 关键发现

- chv1 在两组数据上均为最优通道划分，但最佳融合方式不同（dataA: add, dataB: cat）
- chv3 在 dataA 上远差于 chv1（0.76 vs 0.87），在 dataB 上仅排第 4
- 许多 run 后期出现前景坍缩（final IoU_fg = 0），不能只看 last.pth
- 比赛数据与 mmlsv2 不是同一图像的归一化版本（像素相关性 ≈ 0）

## 历史问题与修复

| 问题 | 原因 | 修复 |
|------|------|------|
| mmlsv2 精度极低 (~0.55) | 对 [0,1] 数据错误应用 raw DN 的 mean/std | 新增 `--dataset-type mmlsv2` 跳过归一化 |
| 比赛数据 test 验证报错 | test/masks 不存在，MarsSegDataset 返回字符串 | 使用 `VAL_SPLIT=val` |
| 旧 outputs_jstar 结果不可靠 | 归一化 bug 未修复时跑的 | 已用修复后代码重跑 |

## 推荐下一步

1. **在 mmlsv2 上运行 chv1 融合探索**：
   ```sh
   DATASET_TYPE=mmlsv2 SEEDS="42 123 7" sh scripts/train_chv1_fusion_experiments.sh
   ```

2. **在比赛数据上验证**（需 VAL_SPLIT=val）：
   ```sh
   DATA_ROOT=/scrinvme/huilin/bdd/cp_data/mars_seg/Mars_LSc_2025_dataset_1st_phase \
   VAL_SPLIT=val sh scripts/train_chv1_fusion_experiments.sh
   ```

3. **分析时优先比较**：
   ```text
   best_miou, best_iou_fg_value, best_f1_value
   final_iou_fg（检测前景坍缩）
   best_iou_fg_epoch（判断收敛速度）
   ```

## 依赖

```text
torch==2.5.1+cu121
tifffile
segmentation_models_pytorch
timm
python-dotenv
tqdm
numpy
scipy (数据分析用)
```

Conda 环境：`M3LSNet`（路径 `/home/23039356r/.conda/envs/M3LSNet/bin/python`）

## 文件结构

```text
train_ablation.py          # 主训练入口
dataset.py                 # MarsSegDataset + MMLSV2Dataset
env_utils.py               # .env 路径解析
summarize_results.py       # 结果汇总表格工具
scripts/
  train_ablation_experiments.sh      # 消融实验 (16)
  train_comparison_models.sh         # 对比模型 (10)
  train_chv1_fusion_experiments.sh   # chv1 融合探索 (12+)
  smoke_test_*.sh                    # 冒烟测试
outputs_jstar/             # 历史实验结果（旧，有归一化 bug）
outputs_experiments/       # 新实验输出目录
training_metrics.csv       # 比赛提交历史 (242 条)
```
