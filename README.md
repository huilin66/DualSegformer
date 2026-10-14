# DualSegFormer for Mars Landslide Segmentation

本仓库当前以 **converted MMLSv2** 上的论文级训练与复现实验为主线。比赛提交、数据转换、冻结测试评估和历史结果分析均已移到 `tools/`，不参与论文训练主流程。

## 目录结构

```text
DualSegformer/
├── train_ablation.py                 # 配置驱动的论文实验训练器
├── train_skg.py                      # SKG-DualSegFormer 训练器
├── train.py                          # 历史比赛训练入口
├── exp_train.sh                      # 历史实验和论文实验总入口
├── scripts/                          # 训练启动脚本
│   ├── train_paper_experiments.sh    # 论文协议入口
│   ├── train_paper_experiments.py
│   ├── train_competition_reproduction.py
│   └── ...                           # legacy/SKG 训练与 smoke train
├── configs/                          # 训练配置和描述子配置
├── networks/                         # 模型结构
├── losses/                           # 损失函数
├── knowledge/                        # SKG 训练所需的知识模块
├── dataset.py                        # 数据集定义
├── augmentations.py                  # 训练增强
├── reproducibility.py                # 随机种子和确定性设置
├── doc/                              # 实验计划和交接文档
└── tools/                            # 与训练主流程无关的工具
    ├── data/                         # 数据准备和 MMLSv2 映射
    ├── evaluation/                   # 冻结 checkpoint 的 val/test 评估
    ├── analysis/                     # 结果汇总、分析和日志解析
    ├── validation/                   # 数据冒烟检查
    ├── knowledge/                    # train-only prototype 构建入口
    └── submission/                   # 历史比赛提交生成代码，不用于论文
```

## 安装

```bash
conda create -n dualsegformer python=3.10
conda activate dualsegformer
pip install -r requirements.txt
```

## 论文实验

正式协议位于 `configs/mmlsv2_paper_protocol.json`，包含 M1–M5 和关键消融，固定 seed 为 `41 42 43`。

```bash
cd /data/huilin/projects/DualSegformer
DATA_ROOT=/data/huilin/scrinvme/huilin/bdd/cp_data/mmlsv2_mapped_mars_ls \
  GROUP=core sh scripts/train_paper_experiments.sh
```

关键消融：

```bash
DATA_ROOT=/data/huilin/scrinvme/huilin/bdd/cp_data/mmlsv2_mapped_mars_ls \
  GROUP=ablations sh scripts/train_paper_experiments.sh
```

运行期间的暂存结果位于 `outputs_experiments/paper_mmlsv2`；完成后复制到数据集根目录的 `outputs/`。正式配置默认只保存 `best.pth` 和 `last.pth`，其中 `best.pth` 按验证集 global mIoU 选择。

也可以使用总入口：

```bash
bash exp_train.sh paper_core
bash exp_train.sh paper_ablations
bash exp_train.sh paper_all
```

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

结果汇总工具位于 `tools/analysis/`。`tools/submission/` 下的脚本只保留作历史比赛提交参考，不属于当前论文实验协议。



