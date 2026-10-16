# scripts/ — 分组实验启动脚本

本目录包含 MMLSv2 论文协议（`configs/mmlsv2_paper_protocol.json`，`pmdfnet_mmlsv2_remote_sensing_v4_ce_only_extended_baselines`）的分组启动入口。
所有脚本都必须在仓库根目录之外的任意位置以 `bash scripts/<脚本>.sh` 方式调用；它们会自行定位仓库根目录。

## 分组

| 脚本 | 分组 | 实验 | runs |
|---|---|---|---|
| `run_core.sh` | core | M1–M4 代表性基线 | 12 |
| `run_ablations.sh` | ablations | A1、A2、A5、B2、B3、C2、L1、L2、L3 | 27 |
| `run_extended.sh` | extended | M6 UPerNet、M7 FPN、M8 MANet、M9 U-Net++、M10 OCRNet+HRNet18、M11 M3LSNet | 18 |
| `run_all.sh` | all | 上述三组顺序执行 | 57 |

三个分组**互斥且并集完整**（`core` = M1–M4、`extended` = M6–M11、`ablations` = 其余），
分组是协议里每个实验显式存储的 `group` 字段。因此 `run_all.sh` 会让每个实验**恰好训练一次**，共 19 × 3 = 57 个 run。

分组选择逻辑只有一处实现（`train_ablation.experiment_in_group`），训练器、显存探针和启动脚本共用，三者不会对"某个组包含哪些实验"产生分歧。

`run_all.sh` 按组顺序执行；若想用单进程跑完所有组，用 `GROUP=all bash scripts/train_paper_experiments.sh`。

## 校验与评估

| 脚本 | 作用 |
|---|---|
| `run_data_smoke.sh` | 只读数据冒烟：CLI 解析、data-root 解析、dataset 导入、读取一个样本（不建模型、不训练） |
| `run_vram_smoke.sh` | 有界合成显存探针：逐个协议条目建模型并跑一个训练步，报告峰值显存（不读数据、不训练、不存 checkpoint） |
| `run_test_eval.sh` | 冻结后的批量 Test 评价，写入各 run 的 `test_metrics.json` |

## 环境变量

`scripts/_env.sh` 被各启动脚本 source，统一解析以下变量（均可 export 覆盖）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `DATA_ROOT` | `$MARS_DATA_ROOT` / `$MAPPED_DATA_ROOT` / `$MMLSV2_DATA_ROOT` / `.env` | mapped MMLSv2 数据根目录 |
| `SEEDS` | `41 42 43` | 随机种子 |
| `DEVICE` | `cuda:0` | torch 设备 |
| `BATCH_SIZE` | 协议默认（32） | 训练 batch |
| `NUM_WORKERS` | `4` | DataLoader workers |
| `EXPERIMENTS` | 空（整组） | 组内子集，逗号分隔，如 `M4` 或 `L1,L2,L3` |
| `CONFIG` | `configs/mmlsv2_paper_protocol.json` | 协议文件 |
| `PYTHON_BIN` | `python` | Python 解释器 |
| `RUN_ROOT` | `outputs_experiments/pmdfnet_mmlsv2` | 仅评价脚本使用 |
| `EXPECTED_RUNS` | `57` | 仅评价脚本使用；部分矩阵评估时下调 |
| `DRY_RUN` / `SKIP_MODEL_INIT` | `0` | 数据/构建检查开关 |

`GROUP` 可覆盖单组脚本的分组（例如用 `run_core.sh` 这个入口去跑 `GROUP=ablations`），但可读性差，建议直接用对应脚本。

## 常用示例

```bash
# 补齐 E1 主表：先跑最关键的对照 M4，再跑新增基线
DATA_ROOT=/data/.../mmlsv2_mapped_mars_ls EXPERIMENTS=M4 bash scripts/run_core.sh
DATA_ROOT=/data/.../mmlsv2_mapped_mars_ls bash scripts/run_extended.sh

# 显存不够时先探针，再决定 batch
GROUP=extended BATCH_SIZE=32 bash scripts/run_vram_smoke.sh
GROUP=extended BATCH_SIZE=16 bash scripts/run_extended.sh

# 只跑损失消融
EXPERIMENTS=L1,L2,L3 bash scripts/run_ablations.sh

# 单 seed 快速核查
SEEDS=41 DRY_RUN=1 bash scripts/run_core.sh

# 全矩阵（57 runs）
DATA_ROOT=/data/.../mmlsv2_mapped_mars_ls bash scripts/run_all.sh

# 评价
DATA_ROOT=/data/.../mmlsv2_mapped_mars_ls bash scripts/run_test_eval.sh
```

## 注意

- `L1` 同时是论文主模型 M5，以及 A4/B1/C1 的来源，**只训练一次**，在 `run_ablations.sh` 里。
- `EXPERIMENTS` 只在所选组内过滤；给某个组传别组的 id 会被训练器拒绝并报错。
- 正式运行集为 **57** 个 run；`run_test_eval.sh` 会校验该数量（可用 `EXPECTED_RUNS` 下调）。
  该脚本按 `RUN_ROOT` 锚定匹配，`_backup_*`、`_archive*`、`_old*`、隐藏目录以及形如 `<run>_s41_..._DEBUG` 的手工副本都会被排除，不会混入统计。
- CE-only 重跑的实验名带 `_ce` 后缀，与旧 combined-loss 的历史 run 目录区分，两代不得混用。
- `DATA_ROOT` 必须显式提供：训练与评估脚本都不再内置远程主机路径（此前训练默认 `/data/huilin/...`、评估默认 `/scrinvme/...`，两者不一致）。可用 `DATA_ROOT` / `MARS_DATA_ROOT` / `MAPPED_DATA_ROOT` / `MMLSV2_DATA_ROOT` 或 `.env` 提供。
