# PMDFNet：MMLSv2 实验结果记录

> 对应计划：[`doc/PMDFNet_MMLSv2_Experiment_Plan_v2.md`](doc/PMDFNet_MMLSv2_Experiment_Plan_v2.md)  
> 记录日期：2026-10-15  
> 文档性质：阶段性实验台账，不等同于最终论文中的 Results/Discussion。

## 结果摘要

当前已经完成首轮核心比较 M1–M5。每个配置均使用 seed `41/42/43`，每个 seed 训练 100 epochs，并使用验证集 mIoU 最优 checkpoint 在 converted MMLSv2 的公开标注 Test split 上评价。表中的 Test 指标是全测试集像素汇总后的 global-pixel 指标，不是逐图像指标的简单平均。

| 配置 | Val 最优 mIoU | Test mIoU | Test Foreground IoU | Test F1 | 当前状态 |
|---|---:|---:|---:|---:|---|
| M1 U-Net | 0.8077 ± 0.0030 | 0.8195 ± 0.0037 | 0.7670 ± 0.0077 | 0.8681 ± 0.0049 | 已完成 |
| M2 DeepLabV3+ | 0.8083 ± 0.0039 | 0.8321 ± 0.0025 | 0.7823 ± 0.0039 | 0.8778 ± 0.0025 | 已完成 |
| M3 SegFormer-MiT-B2 | 0.8226 ± 0.0012 | 0.8331 ± 0.0055 | 0.7834 ± 0.0073 | 0.8785 ± 0.0046 | 已完成 |
| M4 单流 ConvNeXt-Small + MLP Decoder | **0.8352 ± 0.0051** | 0.8507 ± 0.0021 | 0.8054 ± 0.0043 | 0.8922 ± 0.0026 | 已完成 |
| M5 PMDFNet-Small | 0.8208 ± 0.0146 | **0.8534 ± 0.0143** | **0.8089 ± 0.0188** | **0.8943 ± 0.0116** | 已完成 |

初步观察如下：

- M5 的 Test mIoU 均值比直接对照 M4 高 `0.00264`（约 `0.26` 个百分点），但 M5 的跨 seed 标准差为 `0.0143`，明显高于 M4 的 `0.0021`。
- M5 的 Val 最优 mIoU 均值反而低于 M4（`0.8208` vs. `0.8352`），因此目前只能说 M5 在本次 Test 汇总中取得了最高均值，不能直接写成“稳定显著优于单流模型”。
- M5 相比 M1/M2/M3 的 Test mIoU 均值差分别约为 `+3.39/+2.13/+2.03` 个百分点；相对于公平的 M4 单流对照，增益较小，且尚未完成容量匹配和通道分组消融。

## 0. 研究问题、边界与待验证主张

### RQ1：统一协议下是否有稳定性能改善

**阶段性答案：有初步信号，但证据尚不足以称为稳定改善。** M5 在 M1–M5 的 Test mIoU 均值中最高，但 M4 的结果更加稳定，且 M5 还没有通过 A5 近似容量对照、B2/B3 通道分组和 C2 融合消融进行机制验证。

### RQ2：性能是否超出双编码器带来的容量影响

**尚未回答。** Smoke test 中 M4 约为 `50.09M` 参数，M5 约为 `99.54M` 参数，M5 接近 M4 的两倍。A5“近似容量对照”尚未正式训练，因此目前不能把 M5 的 Test 增益单独归因于双流解耦或模态互补。

### RQ3：原始 3+4 通道分组是否更优

**尚未回答。** B2、B3 目前只有显存 smoke test，没有正式训练和 Test 结果。

### RQ4：Add Fusion 与 CE + Dice 的设计是否合理

**尚未回答。** M5 使用逐尺度 Add Fusion 和 combined loss，但 C2、L1、L2 尚未完成正式对照。

### RQ5：性能差异是否可靠并对扰动/缺失模态稳定

**部分回答。** M1–M5 已完成三个 seed 的重复；M5 的 Test 方差偏大。E5 受控扰动和模态缺失实验尚未执行。

### 结论边界

当前结果最多支持以下谨慎表述：在 converted MMLSv2 的当前划分和训练协议下，PMDFNet-Small 在三 seed Test 汇总中取得最高平均 mIoU，但相对于参数约半数的单流 ConvNeXt-Small，其增益很小且稳定性较弱。当前不能声称严格的 physics-informed learning，也不能声称跨区域泛化或优于原比赛隐藏 Test。

### 0.1 主模型选择

本轮在 Test 评价前按计划冻结 `PMDFNet-Small` 为主模型，并以单流 ConvNeXt-Small + MLP Decoder 作为直接对照。当前没有根据 Test 结果在 Small/Tiny 之间后验切换主模型。

原 Workshop 的 Tiny/Small 表格、Test1/Test2 定义与本轮 converted MMLSv2 结果尚未在本文件中混排；后续论文需要单独说明 Workshop 结果与本轮无 TTA 的公平比较之间的关系。

### 0.2 固定网络定义

当前 M5 配置如下：

| 项目 | 当前配置 |
|---|---|
| Stream 1 | 输出通道 `0,1,2`：Thermal + Slope + DEM |
| Stream 2 | 输出通道 `3,4,5,6`：Grayscale + R + G + B |
| Backbone | 两条独立 ConvNeXt-Small |
| Fusion | 四尺度逐元素 Add |
| Decoder | SegFormer All-MLP Decoder |
| Loss | CE + Dice（代码中的 `combined`） |

## E0. 数据版本与转换映射协议审计

### E0.1 转换与通道审计

**状态：已建立基础记录，但尚未闭环。** 数据根目录的 `mapping_manifest.json` 已记录来源目录、参考比赛数据目录、输入/输出通道顺序、scale/offset 和 split 数量。当前清单给出的输出通道顺序为：

`Thermal → Slope → DEM → Grayscale → R → G → B`。

划分数量为 Train `465`、Val `66`、Test `133`。映射验证记录显示多数通道的误差比例很低，但 Slope/DEM 仍出现较大的最大绝对误差记录，Train mask 验证还记录了 `1` 个差异像素。因此，映射清单可以作为当前运行的来源记录，但在投稿前仍需解释这些异常值并完成独立复核，不能直接写成“逐像素完全一致”。

尚未在仓库中形成完整、版本化的 `channel_mapping.yaml`、通道统计表、直方图和转换脚本交付物；当前部分证据仍位于远程数据目录。

### E0.2 划分与潜在泄漏审计

**状态：划分固定，场景级泄漏审计未完成。** 本轮训练和评价使用固定的 `train/val/test` 文件列表与 465/66/133 划分；但相邻切片、原始场景 ID、跨 split 哈希重复和 scene-level disjoint 尚未形成正式审计报告。论文暂时不能声称场景级独立或跨区域泛化。

### E0.3 最低交付物

| 交付物 | 当前状态 |
|---|---|
| `dataset_manifest.csv` | 未在仓库中形成最终版本；数据侧有 split/映射清单信息 |
| `channel_mapping.yaml` | 未完成 |
| `dataset_audit.md` | 未完成 |
| 固定文件列表、转换脚本与 SHA-256 | 未完成统一归档 |

## 1. 全实验通用协议

### 1.1 数据与模型选择

本轮已完成的核心 Test 评价使用固定的 converted MMLSv2 Test split，并采用每个 seed 的验证集最优 mIoU checkpoint。没有使用 Test 分数回头选择 seed、阈值或主模型。需要注意：这里的 Test 是有标注的公开 Test split，不能与原比赛的隐藏测试榜单等同。

### 1.2 基础训练设置

正式核心运行的实际 `config.json` 和启动命令记录为：

- 输入尺寸 `128×128`，训练 `100 epochs`，**batch size=32**；
- AdamW，学习率 `1e-4`，weight decay `5e-4`，cosine scheduler；
- `augmentation=mars`，`aug_prob=0.5`，`mosaic_prob=0.5`；
- ImageNet/代码配置的预训练开关为 `pretrain=true`；
- `mixed_precision=false`，`num_workers=4`，设备为 A100 `cuda:0`；
- `deterministic=true`，严格确定性设置已记录；
- 每个 epoch 验证并按 `primary_metric=miou` 选择 best checkpoint，另保存 last checkpoint。

配置模板 `configs/mmlsv2_paper_protocol.json` 的默认 batch size 仍是 `16`，正式核心运行通过命令行显式覆盖为 `32`。论文和公开复现配置应在后续统一这一处，避免模板与运行记录不一致。

### 1.3 统一评价口径

Test 结果使用全数据集像素累计混淆矩阵计算 mIoU、Foreground IoU、Background IoU、Precision、Recall 和 F1。每个运行的 `test_metrics.json` 同时保留 global-pixel 与 batch-average 两套数值；本文档和 `paper_aggregate.csv` 使用前者作为主口径。

Test 评价共覆盖 `133` 张图像，每个运行使用 `9` 个 batch。阈值采用当前二分类评估实现的默认规则；独立的 Val 阈值冻结实验尚未补齐，投稿前需在协议中明确写出。

## E1. 主实验：公平基线对比

### E1.1 M1–M5 三 seed 汇总

下表为跨 seed 的均值 ± 样本标准差（`n=3`）。Test 使用 best validation mIoU checkpoint，Test 指标为 global-pixel 汇总。

| ID | 方法 | Val best mIoU | Test mIoU | Test IoU-FG | Test IoU-BG | Test F1 |
|---|---|---:|---:|---:|---:|---:|
| M1 | U-Net | 0.8077 ± 0.0030 | 0.8195 ± 0.0037 | 0.7670 ± 0.0077 | 0.8720 ± 0.0008 | 0.8681 ± 0.0049 |
| M2 | DeepLabV3+ ResNet50 | 0.8083 ± 0.0039 | 0.8321 ± 0.0025 | 0.7823 ± 0.0039 | 0.8819 ± 0.0011 | 0.8778 ± 0.0025 |
| M3 | SegFormer MiT-B2 | 0.8226 ± 0.0012 | 0.8331 ± 0.0055 | 0.7834 ± 0.0073 | 0.8828 ± 0.0040 | 0.8785 ± 0.0046 |
| M4 | 单流 ConvNeXt-Small + MLP | **0.8352 ± 0.0051** | 0.8507 ± 0.0021 | 0.8054 ± 0.0043 | 0.8961 ± 0.0007 | 0.8922 ± 0.0026 |
| M5 | PMDFNet-Small | 0.8208 ± 0.0146 | **0.8534 ± 0.0143** | **0.8089 ± 0.0188** | **0.8979 ± 0.0097** | **0.8943 ± 0.0116** |

Test Precision/Recall 汇总为：

| ID | Precision | Recall |
|---|---:|---:|
| M1 | 0.8603 ± 0.0177 | 0.8769 ± 0.0282 |
| M2 | 0.8748 ± 0.0041 | 0.8810 ± 0.0089 |
| M3 | 0.8769 ± 0.0091 | 0.8803 ± 0.0119 |
| M4 | 0.8953 ± 0.0130 | 0.8895 ± 0.0176 |
| M5 | 0.8964 ± 0.0069 | 0.8921 ± 0.0166 |

### E1.2 分 seed Test mIoU

| ID | Seed 41 | Seed 42 | Seed 43 |
|---|---:|---:|---:|
| M1 | 0.8236 | 0.8164 | 0.8184 |
| M2 | 0.8299 | 0.8348 | 0.8315 |
| M3 | 0.8351 | 0.8373 | 0.8268 |
| M4 | 0.8504 | 0.8488 | 0.8530 |
| M5 | 0.8635 | 0.8596 | 0.8371 |

### E1.3 当前解释

M5 的最高均值主要由 seed 41/42 支撑，seed 43 明显下降；M4 的三个 seed 更集中。基于当前 `n=3` 结果，不进行显著性检验结论、不报告“显著提升”，也不把 M5 与 M4 的小幅均值差写成确定的双流因果收益。E1 的 Table 2 可以作为**阶段性表格**，但在 E0、A5、B2/B3 和统计置信区间完成前不应作为投稿终稿。

## E2. 单流/双流与模态贡献消融

**状态：正式消融尚未开始。** 目前 M5 可以复用为 A4，M4 可以复用为 A3，但 A1、A2、A5 尚未正式训练。A5 已按计划命名为“近似容量对照”，其 smoke test 使用单流 ConvNeXt-Base，但尚未核对正式配置下的 Params/GFLOPs 偏差。

| ID | 当前状态 | 备注 |
|---|---|---|
| A1 光学单流 | 待训练 | Smoke test 已通过，未产生正式指标 |
| A2 物理单流 | 待训练 | Smoke test 已通过，未产生正式指标 |
| A3 7 通道单流 | 复用 M4 | 不是独立运行 |
| A4 原始双流 | 复用 M5 | 不是独立运行 |
| A5 **近似容量对照** | 待训练 | Smoke test 已通过，不能用 smoke 结果代替精度结果 |

## E3. 通道分组消融

**状态：正式消融尚未开始。** B1 复用 M5；B2、B3 已完成 batch32 smoke test，但没有训练和 Test 指标。B4 仍未执行。

| ID | Stream 1 | Stream 2 | 当前状态 |
|---|---|---|---|
| B1 | Thermal + Slope + DEM | Grayscale + RGB | 复用 M5，已完成 |
| B2 | RGB | Thermal + Slope + DEM + Grayscale | 待正式训练 |
| B3 | Thermal + DEM + Grayscale | Slope + RGB | 待正式训练 |
| B4 | Slope + DEM + Grayscale | Thermal + RGB | 未执行 |

因此当前不能回答“原始物理分组在多个 seed 上是否占优”。

## E4. 融合与损失消融

**状态：正式消融尚未开始。** C1/L3 复用 M5，C2 只完成 smoke test；CE-only 和 Dice-only 尚未训练。

| ID | 当前状态 |
|---|---|
| C1 Add Fusion | 复用 M5 |
| C2 Concat + 1×1 Conv | 待正式训练；smoke test 参数约 `101.11M` |
| L1 CE only | 未执行 |
| L2 Dice only | 未执行 |
| L3 CE + Dice | 复用 M5 |

## E5. 受控扰动及模态缺失

**状态：未执行。** 当前没有冻结的 `robustness_protocol.yaml`，也没有 Val 确定后的扰动等级、Test 结果或鲁棒性曲线。不得根据现有 clean Test 结果推断抗噪声、抗缺失模态或跨区域泛化能力。

## E6. 重复实验与复杂度

### E6.1 统计可靠性

M1–M5 均完成 seed `41/42/43`，并保留所有 seed，没有剔除低分运行。当前报告的是跨 seed 的 mean ± std；图像级 bootstrap 置信区间和配对统计检验尚未完成。

需要特别记录：M1 seed 41 的正式运行元数据显示 `git dirty=true`，其余本轮核心正式运行记录为 clean working tree。该运行不应在投稿终稿中默认视为完全可复现；建议在最终冻结代码后至少重跑 M1 seed 41，或核对 dirty 状态对应的具体差异并保存说明。

### E6.2 参数量与显存 smoke test

以下为 A100 80GB、输入 `32×7×128×128` 的 batch32 smoke test 结果。参数量来自模型实例，显存为本次 smoke test 的峰值 reserved；所有配置均未超过 40GB 限制。

| ID | Params (M) | Peak allocated (GB) | Peak reserved (GB) | Smoke 状态 |
|---|---:|---:|---:|---|
| M1 | 32.53 | 2.93 | 3.01 | 通过 |
| M2 | 26.69 | 1.40 | 1.83 | 通过 |
| M3 | 24.74 | 1.73 | 2.14 | 通过 |
| M4 | 50.09 | 2.40 | 2.82 | 通过 |
| M5 | 99.54 | 4.32 | 4.83 | 通过 |

A1/A2/A5/B2/B3/C2 的 smoke test 也已通过；其中 A5 约 `88.33M`，C2 约 `101.11M`。当前尚未完成统一工具下的 GFLOPs、FPS、latency、warm-up 和多次测速，因此 Table 6 仍不完整。

### E6.3 日志与 checkpoint 完整性

核心正式运行的每个 run 目录均保留：`config.json`、`command.txt`、`metrics.csv`、`logs/train.log`、`results.json`、`runtime.json`、`reproducibility.json`、`checkpoints/best.pth`、`checkpoints/last.pth`，以及 Test 阶段生成的 `test_metrics.json`。best 的选择依据是验证集 `mIoU`，不是 Test 指标；当前没有额外保存辅助 checkpoint。

旧的 batch16 试运行目录与正式 batch32 结果同时存在于远程输出目录中，但本文件的 M1–M5 汇总只使用后者的 15 个正式 run，不混入旧试运行。

## E7. 定性与误差分析

**状态：未执行。** 当前已有各模型 best checkpoint 和 Test 指标，但尚未按预注册规则固定 Test 图像 ID，也没有生成 `RGB | GT | M4 | M5 | FP/FN map` 图和总体 FP/FN 分布。因此暂不从个别案例推导模型优势。

## 2. 首轮最小实验矩阵执行状态

| 类别 | 计划配置 | 当前状态 |
|---|---|---|
| 主方法及直接对照 | M4、M5 | 已完成，均为 3 seeds |
| 常见代表性基线 | M1、M2、M3 | 已完成，均为 3 seeds |
| 模态贡献 | A1、A2 | 待正式训练；仅 smoke |
| 容量对照 | A5 近似容量对照 | 待正式训练；仅 smoke |
| 物理分组 | B1、B2、B3 | B1 复用 M5；B2/B3 待正式训练；仅 smoke |
| Fusion | C1、C2 | C1 复用 M5；C2 待正式训练；仅 smoke |
| 统计 | M1–M5 使用 41/42/43 | 已完成 |
| 效率/可视化 | Params、GFLOPs、FPS、固定案例图 | Params/VRAM 部分完成；其余待做 |

当前分析关口结论：

1. M5 对 M4 的均值优势较小且方差较大，不能仅凭当前结果确认双流设计的稳定收益。
2. M5 与 M4 参数量差异接近 2 倍，A5 是解释结果所必需的控制实验。
3. B1/B2/B3 尚无正式指标，不能确认物理分组优越性。
4. 现阶段应优先完成 E0、A1/A2/A5、B2/B3 和 C2，再决定是否扩展 E4/E5，而不是根据现有 Test 分数继续挑选结构。

## 3. 推荐执行顺序的实际进度

| 阶段 | 计划内容 | 当前进度 |
|---|---|---|
| Phase 0 | E0 数据审计与协议冻结 | 部分完成：有 mapping manifest 和固定 split；完整泄漏/统计审计未完成 |
| Phase 1 | M1–M5、A5、A1/A2 | M1–M5 完成；A1/A2/A5 未完成 |
| Phase 2 | B2/B3、C2 | 未开始正式训练 |
| Phase 3 | 最终 Test、扰动、可视化、效率与写作 | 核心 M1–M5 Test 已完成；其余部分待做 |

## 4. 期刊图表清单状态

| 编号 | 图表 | 当前状态 |
|---|---|---|
| Table 1 | Converted MMLSv2 划分、通道统计、映射协议 | 部分完成，统计与泄漏审计待补 |
| Table 2 | M1–M5 主基线 mean±std | 已有阶段性版本，见 E1 |
| Table 3 | A1–A5 模态/容量消融 | 待完成 |
| Table 4 | B1–B3/B4 通道分组 | 待完成 |
| Table 5 | Add vs. Concat、CE/Dice | 待完成 |
| Table 6 | Params/GFLOPs/FPS | Params/VRAM 部分完成，GFLOPs/FPS 待补 |
| Table 7 | 受控扰动与模态缺失 | 待完成 |
| Figure 1 | PMDFNet-Small 结构图 | 未在本结果记录中更新 |
| Figure 2 | 固定规则的定性比较 | 待完成 |
| Figure 3 | 鲁棒性曲线 | 待完成 |
| Figure 4 | 错误统计与失败案例 | 待完成 |
| Supplementary | manifest、实现细节、单 seed 结果 | 单 seed 核心结果已在远程 JSON；manifest/审计尚未统一归档 |

## 5. 复现目录与记录

### 已存在的代码与远程产物

- 训练协议：`configs/mmlsv2_paper_protocol.json`。
- 训练入口：`scripts/train_paper_experiments.py`。
- Test 批量评价日志：远程 `outputs_experiments/launch_logs/test_metrics_all.log`，末尾状态为 `TEST_BATCH_COMPLETED`。
- 汇总脚本：`tools/evaluation/summarize_paper_results.py`。
- 汇总 CSV/JSON：远程 `outputs_experiments/pmdfnet_mmlsv2/aggregate/paper_aggregate.csv` 和 `paper_aggregate.json`。
- 正式运行目录：远程 `outputs_experiments/pmdfnet_mmlsv2/`；15 个核心 run 已同步到数据集 `outputs/` 目录。
- 数据映射记录：远程数据根目录下的 `mapping_manifest.json`。

### 复现注意事项

1. 运行数据路径、远程主机和输出目录应通过环境变量提供，不应写死到论文命令中。
2. 要复现本文件的正式核心结果，应显式使用 batch32；不能直接使用协议模板当前的 batch16 默认值。
3. 应在 clean working tree 和固定 commit 下重跑 M1 seed 41，并把最终 commit/hash 写入结果归档。
4. 当前结果依赖不能公开再分发的映射数据；论文需要提供合法数据来源和确定性转换步骤，而不能把 converted MMLSv2 表述为新的独立数据集。

## 6. 投稿前检查清单

- [x] 已完成 M1–M5 的 `41/42/43` 三 seed 核心训练与 Test 汇总。
- [x] 已记录 best checkpoint 按验证集 mIoU 选择，并保存 best/last 与完整训练日志。
- [x] 已完成 batch32、A100 80GB、40GB 显存上限下的核心 smoke test。
- [ ] 完成 E0 通道、单位、归一化、映射、哈希和场景/相邻切片泄漏审计。
- [ ] 完成 A1/A2/A5 正式训练；A5 必须保留“近似容量对照”表述。
- [ ] 完成 B2/B3（必要时 B4）严格 3+4 通道分组比较。
- [ ] 完成 C2、L1、L2 融合/损失消融。
- [ ] 完成统一的 GFLOPs、FPS、latency 与统计置信区间。
- [ ] 完成预注册 Test 可视化、FP/FN 误差分析和 E5 鲁棒性实验。
- [ ] 清理/重跑 dirty run，并冻结最终 commit、配置和数据版本。
- [ ] 论文中明确 converted MMLSv2、公开标注 Test 与原比赛隐藏榜单的边界。

## 结论（当前阶段）

首轮结果表明，PMDFNet-Small 在 converted MMLSv2 的三 seed Test 平均 mIoU 上略高于 M1–M4，其中相对于单流 ConvNeXt-Small 的优势约为 `0.26` 个百分点；但该优势伴随更大的随机波动，而且双流模型参数量接近单流对照的两倍。因此，这批结果可以作为论文实验的起点，尚不能单独支撑“PMDFNet 稳定优于单流方法”或“性能提升来自物理动机分组”的结论。

下一步应按 E0 → A1/A2/A5 → B2/B3 → C2 的顺序补齐控制证据，并在完成 clean rerun、数据审计和效率统计后，再决定是否进入 Remote Sensing 投稿稿件的最终 Results/Discussion。
