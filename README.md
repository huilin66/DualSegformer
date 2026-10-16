# PMDFNet: Physically Motivated Dual-Stream Fusion for Multimodal Landslide Segmentation

本仓库当前以 **PMDFNet** 在 **converted MMLSv2** 上的论文级训练与复现实验为主线。比赛提交、旧版训练、SKG 分支和历史分析代码已归档到 `archive/legacy/`，不参与当前论文训练主流程。

项目显示名和仓库目录均为 PMDFNet；Python 模块和历史结果中保留的 `DualSegFormer`/`dual_segformer_*` 仅作为兼容标识，不改变旧 checkpoint 和复现入口。

## 目录结构

```text
PMDFNet/
├── train_ablation.py                 # 配置驱动的论文实验训练器
├── exp_train.sh                      # 当前论文实验总入口
├── scripts/                          # 训练启动脚本
│   ├── train_paper_experiments.sh    # 论文协议入口
│   ├── train_paper_experiments.py
│   └── ...                           # 当前论文启动辅助脚本
├── configs/                          # 训练配置和描述子配置
├── networks/                         # 模型结构
├── losses/                           # 损失函数
├── dataset.py                        # 数据集定义
├── augmentations.py                  # 训练增强
├── reproducibility.py                # 随机种子和确定性设置
├── doc/                              # 当前实验计划
├── tools/                            # 当前数据、评估和验证工具
│   ├── data/                         # MMLSv2 映射
│   ├── evaluation/                   # 冻结 checkpoint 的 val/test 评估
│   └── validation/                   # 数据与显存冒烟检查
└── archive/legacy/                   # 旧版训练、SKG、提交和分析代码
```

## 安装

```bash
conda create -n pmdfnet python=3.10
conda activate pmdfnet
pip install -r requirements.txt
```

## 论文实验

正式协议位于 `configs/mmlsv2_paper_protocol.json`，包含 M1–M4 基线与关键消融（A1/A2/A5/B2/B3/C2、L1/L2/L3），固定 seed 为 `41 42 43`。全部配置统一使用 CE-only 训练目标（`common.loss=ce`），仅损失消融 L2/L3 显式覆盖。

论文主模型 M5 即协议中的 **L1**（PMDFNet-Small + CE-only），只训练一次；A3/A4/B1/C1 是对 M4/L1 的展示别名，不单独训练。协议内 `paper_id_map` 记录了论文 ID 与运行 ID 的对应关系。

```bash
cd /data/huilin/projects/PMDFNet
DATA_ROOT=/data/huilin/scrinvme/huilin/bdd/cp_data/mmlsv2_mapped_mars_ls \
  GROUP=core BATCH_SIZE=32 sh scripts/train_paper_experiments.sh
```

关键消融：

```bash
DATA_ROOT=/data/huilin/scrinvme/huilin/bdd/cp_data/mmlsv2_mapped_mars_ls \
  GROUP=ablations sh scripts/train_paper_experiments.sh
```

运行期间的暂存结果位于 `outputs_experiments/pmdfnet_mmlsv2`；完成后复制到数据集根目录的 `outputs/`。正式配置默认只保存 `best.pth` 和 `last.pth`，其中 `best.pth` 按验证集 global mIoU 选择。

也可以使用总入口：

```bash
bash exp_train.sh paper_core
bash exp_train.sh paper_ablations
bash exp_train.sh paper_all
```

启动显存 smoke test（默认 batch=8、PyTorch allocator 上限 38GB、总安全上限 40GB）：

```bash
bash exp_train.sh vram_smoke
```

结果写入 `outputs_experiments/smoke/vram/vram_smoke_results.csv` 和
`vram_smoke_results.json`；该测试使用随机输入，不读取数据集、不训练和不保存 checkpoint。

## 数据映射

映射脚本位于 `tools/data/map_mmlsv2_to_mars.py`。它将 MMLSv2 的通道顺序转换为旧比赛格式，保留原始数据并写出新的 mapped 数据集。

```bash
python tools/data/map_mmlsv2_to_mars.py \
  --source-root /path/to/mmlsv2 \
  --output-root /path/to/mmlsv2_mapped_mars_ls
```

## 冻结模型评估

测试评估只能在模型、超参数和 checkpoint 选择规则冻结后执行：

```bash
CHECKPOINT=/path/to/best.pth \
DATA_ROOT=/path/to/mmlsv2_mapped_mars_ls \
  bash tools/evaluation/evaluate_reproduction_test.sh
```

历史结果汇总和比赛提交工具位于 `archive/legacy/tools/`，不属于当前论文实验协议。



