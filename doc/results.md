# PMDFNet：MMLSv2 实验结果记录

> 对应计划：[`PMDFNet_MMLSv2_Experiment_Plan_v2.md`](PMDFNet_MMLSv2_Experiment_Plan_v2.md)<br>
> 记录日期：2026-10-16<br>
> 文档性质：阶段性实验台账，不等同于最终论文中的 Results/Discussion。

## 结果摘要

当前文档已将论文主模型口径切换为 M5-CE（沿用已完成的 L1 CE-only 三 seed 结果作为临时结果）。原 M5 combined-loss 结果保留为历史对照；M5-CE 正在远程重新训练。A1、A2、A5、B2、B3、C2 的当前表格结果已清空，等待统一使用 CE-only 重跑。表中的 Test 指标是全测试集像素汇总后的 global-pixel 指标，不是逐图像指标的简单平均。

| 配置 | Val 最优 mIoU | Test mIoU | Test Foreground IoU | Test F1 | 当前状态 |
|---|---:|---:|---:|---:|---|
| M1 U-Net | 0.8077 ± 0.0030 | 0.8195 ± 0.0037 | 0.7670 ± 0.0077 | 0.8681 ± 0.0049 | 已完成 |
| M2 DeepLabV3+ | 0.8083 ± 0.0039 | 0.8321 ± 0.0025 | 0.7823 ± 0.0039 | 0.8778 ± 0.0025 | 已完成 |
| M3 SegFormer-MiT-B2 | 0.8226 ± 0.0012 | 0.8331 ± 0.0055 | 0.7834 ± 0.0073 | 0.8785 ± 0.0046 | 已完成 |
| M4 单流 ConvNeXt-Small + MLP Decoder | **0.8352 ± 0.0051** | 0.8507 ± 0.0021 | 0.8054 ± 0.0043 | 0.8922 ± 0.0026 | 已完成 |
| M5 PMDFNet-Small（CE-only） | 0.8209 ± 0.0096 | **0.8594 ± 0.0034** | **0.8166 ± 0.0041** | **0.8990 ± 0.0025** | 临时沿用 L1；M5-CE 重跑中 |

初步观察如下：

- 当前 M5-CE 的临时数值来自已完成的 L1 CE-only 三 seed 结果，Test mIoU 为 `0.8594 ± 0.0034`；待 M5-CE 重跑结束后再确认是否一致。
- A1、A2、A5、B2、B3、C2 的结果已从当前有效结果表中清空，不能继续使用旧的 combined-loss 数值。
- 原 M5 combined-loss 数值 `0.8534 ± 0.0143` 仅作为历史对照保留，不再作为当前主模型结果。

## 0. 研究问题、边界与待验证主张

### RQ1：统一协议下是否有稳定性能改善

**当前答案待 M5-CE 重跑确认。** 按当前主模型口径，M5 使用 CE-only；临时结果高于旧 combined-loss M5，但 M5-CE 的独立重跑尚未完成。

### RQ2：性能是否超出双编码器带来的容量影响

**待 CE-only 消融重跑后更新。** M4 约为 `50.09M` 参数，M5 约为 `99.54M` 参数；A5 当前结果已清空，暂不比较容量因素。

### RQ3：原始 3+4 通道分组是否更优

**待 CE-only 分组消融重跑后更新。** B1 当前沿用 M5-CE 的临时结果 `0.8594 ± 0.0034`；B2、B3 的当前结果已清空。

### RQ4：Add Fusion 与 CE + Dice 的设计是否合理

**待 CE-only 融合消融重跑后更新。** C1/Add 当前沿用 M5-CE 的临时结果 `0.8594 ± 0.0034`；C2 的当前结果已清空。L1/L2 仍保留为已完成的损失函数参考结果。

### RQ5：性能差异是否可靠并对扰动/缺失模态稳定

**尚未完成。** M5-CE 与 A1/A2/A5/B2/B3/C2 的新一轮结果尚未全部完成；E5 受控扰动和模态缺失实验尚未执行。

### 结论边界

当前结果暂支持以下表述：论文主模型改为 CE-only 的 PMDFNet-Small，当前临时结果沿用 L1 的 `0.8594 ± 0.0034`，但需要等待 M5-CE 重新训练确认。A1/A2/A5/B2/B3/C2 的 CE-only 结果尚未形成，因此暂不对双流、容量或通道分组作最终结论。当前不能声称严格的 physics-informed learning，也不能声称跨区域泛化或优于原比赛隐藏 Test。

### 0.1 主模型选择

当前主模型定义为 `PMDFNet-Small + CE-only`，即原 L1 配置；M5-CE 的独立三 seed 重跑用于确认该定义。单流 ConvNeXt-Small + MLP Decoder 仍作为直接对照。

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
| Loss | CE-only（代码中的 `ce`） |

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

Test 评价共覆盖 `133` 张图像，每个运行使用 batch size `32`、共 `5` 个 batch。阈值采用当前二分类评估实现的默认规则；独立的 Val 阈值冻结实验尚未补齐，投稿前需在协议中明确写出。

## E1. 主实验：公平基线对比

### E1.1 M1–M5 三 seed 汇总

下表为跨 seed 的均值 ± 样本标准差（`n=3`）。Test 使用 best validation mIoU checkpoint，Test 指标为 global-pixel 汇总。

| ID | 方法 | Val best mIoU | Test mIoU | Test IoU-FG | Test IoU-BG | Test F1 |
|---|---|---:|---:|---:|---:|---:|
| M1 | U-Net | 0.8077 ± 0.0030 | 0.8195 ± 0.0037 | 0.7670 ± 0.0077 | 0.8720 ± 0.0008 | 0.8681 ± 0.0049 |
| M2 | DeepLabV3+ ResNet50 | 0.8083 ± 0.0039 | 0.8321 ± 0.0025 | 0.7823 ± 0.0039 | 0.8819 ± 0.0011 | 0.8778 ± 0.0025 |
| M3 | SegFormer MiT-B2 | 0.8226 ± 0.0012 | 0.8331 ± 0.0055 | 0.7834 ± 0.0073 | 0.8828 ± 0.0040 | 0.8785 ± 0.0046 |
| M4 | 单流 ConvNeXt-Small + MLP | **0.8352 ± 0.0051** | 0.8507 ± 0.0021 | 0.8054 ± 0.0043 | 0.8961 ± 0.0007 | 0.8922 ± 0.0026 |
| M5 | PMDFNet-Small + CE-only | 0.8209 ± 0.0096 | **0.8594 ± 0.0034** | **0.8166 ± 0.0041** | **0.9022 ± 0.0028** | **0.8990 ± 0.0025** |

Test Precision/Recall 汇总为：

| ID | Precision | Recall |
|---|---:|---:|
| M1 | 0.8603 ± 0.0177 | 0.8769 ± 0.0282 |
| M2 | 0.8748 ± 0.0041 | 0.8810 ± 0.0089 |
| M3 | 0.8769 ± 0.0091 | 0.8803 ± 0.0119 |
| M4 | 0.8953 ± 0.0130 | 0.8895 ± 0.0176 |
| M5 | 0.9014 ± 0.0075 | 0.8967 ± 0.0054 |

### E1.2 分 seed Test mIoU

| ID | Seed 41 | Seed 42 | Seed 43 |
|---|---:|---:|---:|
| M1 | 0.8236 | 0.8164 | 0.8184 |
| M2 | 0.8299 | 0.8348 | 0.8315 |
| M3 | 0.8351 | 0.8373 | 0.8268 |
| M4 | 0.8504 | 0.8488 | 0.8530 |
| M5 | 0.8634 | 0.8576 | 0.8573 |

### E1.3 当前解释

E1 中 M5 的当前临时行沿用 L1 CE-only 的三 seed Test 结果，待 M5-CE 重跑完成后替换为实际 M5-CE 数值。当前不进行显著性检验结论，也不把该临时结果写成确定的双流因果收益。

## E2. 单流/双流与模态贡献消融

**状态：重置中。** A1、A2、A5 的旧结果已从当前表格清空，等待 CE-only 重跑；A3 复用 M4，A4 临时复用 M5-CE。下表为 global-pixel Test 指标的跨 seed 均值 ± 样本标准差。

| ID | 方法 | Val best mIoU | Test mIoU | Test IoU-FG | Test F1 | 当前状态 |
|---|---|---:|---:|---:|---:|---|
| A1 | 光学单流（4 通道） | — | — | — | — | 待 CE-only 重跑 |
| A2 | 物理单流（3 通道） | — | — | — | — | 待 CE-only 重跑 |
| A3 | 7 通道单流（复用 M4） | 0.8352 ± 0.0051 | 0.8507 ± 0.0021 | 0.8054 ± 0.0043 | 0.8922 ± 0.0026 | 复用 M4 |
| A4 | 原始双流（临时复用 M5-CE） | 0.8209 ± 0.0096 | 0.8594 ± 0.0034 | 0.8166 ± 0.0041 | 0.8990 ± 0.0025 | 临时复用 M5-CE |
| A5 | **近似容量对照：单流 ConvNeXt-Base** | — | — | — | — | 待 CE-only 重跑 |

A4/A5 的最终比较等待 A5 的 CE-only 重跑完成；当前不对双流增益或容量因素作结论。

## E3. 通道分组消融

**状态：重置中。** B1 临时复用 M5-CE；B2/B3 的旧结果已清空，等待 CE-only 重跑；B4 未纳入本轮计划矩阵。

| ID | Stream 1 | Stream 2 | Test mIoU | Test IoU-FG | Test F1 | 当前状态 |
|---|---|---|---:|---:|---:|---|
| B1 | Thermal + Slope + DEM | Grayscale + RGB | 0.8594 ± 0.0034 | 0.8166 ± 0.0041 | 0.8990 ± 0.0025 | 临时复用 M5-CE |
| B2 | RGB | Thermal + Slope + DEM + Grayscale | — | — | — | 待 CE-only 重跑 |
| B3 | Thermal + DEM + Grayscale | Slope + RGB | — | — | — | 待 CE-only 重跑 |

B1/B2/B3 的最终分组结论等待 B2/B3 的 CE-only 重跑完成。

## E4. 融合与损失消融

**状态：部分重置。** C1 临时复用 M5-CE；C2 的旧结果已清空，等待 CE-only 重跑。L1/L2 保留为已有损失函数参考结果；L3 不再复用当前 M5。

| ID | 变量 | Test mIoU | Test IoU-FG | Test F1 | 当前状态 |
|---|---|---:|---:|---:|---|
| C1 | Add Fusion（临时复用 M5-CE） | 0.8594 ± 0.0034 | 0.8166 ± 0.0041 | 0.8990 ± 0.0025 | 临时复用 M5-CE |
| C2 | Concat + 1×1 Conv | — | — | — | 待 CE-only 重跑 |
| L1 | CE only | **0.8594 ± 0.0034** | **0.8166 ± 0.0041** | **0.8990 ± 0.0025** | 已完成 |
| L2 | Dice only | 0.8323 ± 0.0054 | 0.7812 ± 0.0099 | 0.8771 ± 0.0062 | 已完成 |
| L3 | CE + Dice（历史 M5-combined） | 0.8534 ± 0.0143 | 0.8089 ± 0.0188 | 0.8943 ± 0.0116 | 历史对照，不再复用当前 M5 |

Add Fusion 与 Concat 的最终比较等待 C2 的 CE-only 重跑完成；L3 的历史 combined-loss 结果仅作为旧协议对照，不与当前 M5-CE 混称。

## E5. 受控扰动及模态缺失

**状态：未执行。** 当前没有冻结的 `robustness_protocol.yaml`，也没有 Val 确定后的扰动等级、Test 结果或鲁棒性曲线。不得根据现有 clean Test 结果推断抗噪声、抗缺失模态或跨区域泛化能力。

## E6. 重复实验与复杂度

### E6.1 统计可靠性

历史版本保留 M1–M5 的 15 个基线 run，以及 A1/A2/A5/B2/B3/C2/L1/L2 的 24 个新增消融 run；本次结果口径已将 A1/A2/A5/B2/B3/C2 置为空，等待 CE-only 重跑。当前 M5-CE 数值暂沿用 L1 的三 seed 结果；图像级 bootstrap 置信区间和配对统计检验尚未完成。

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

历史正式运行目录保留：`config.json`、`command.txt`、`metrics.csv`、`logs/train.log`、`results.json`、`runtime.json`、`reproducibility.json`、`checkpoints/best.pth`、`checkpoints/last.pth`，以及 Test 阶段生成的 `test_metrics.json`。best 的选择依据是验证集 `mIoU`，不是 Test 指标；当前没有额外保存辅助 checkpoint。A1/A2/A5/B2/B3/C2 的旧结果不再作为当前表格结果；M5-CE 正在重新训练。

旧的 batch16 试运行目录与正式 batch32 结果同时存在于远程输出目录中；本文件只使用 24 个正式 batch32 run，不混入旧试运行。Test 结果当前保存在远程项目目录，尚未统一同步到数据集 `outputs/`。

## E7. 定性与误差分析

**状态：未执行。** 当前已有各模型 best checkpoint 和 Test 指标，但尚未按预注册规则固定 Test 图像 ID，也没有生成 `RGB | GT | M4 | M5 | FP/FN map` 图和总体 FP/FN 分布。因此暂不从个别案例推导模型优势。

## 2. 首轮最小实验矩阵执行状态

| 类别 | 计划配置 | 当前状态 |
|---|---|---|
| 主方法及直接对照 | M4、M5-CE | M4 已完成；M5-CE 重跑中 |
| 常见代表性基线 | M1、M2、M3 | 已完成，均为 3 seeds |
| 模态贡献 | A1、A2 | CE-only 待重跑 |
| 容量对照 | A5 近似容量对照 | CE-only 待重跑 |
| 物理分组 | B1、B2、B3 | B1 临时复用 M5-CE；B2/B3 待重跑 |
| Fusion/损失 | C1、C2、L1、L2 | C1 临时复用 M5-CE；C2 待重跑；L1/L2 保留历史参考 |
| 统计 | M5-CE、A1/A2/A5、B2/B3、C2 使用 41/42/43 | 待重跑完成 |
| 效率/可视化 | Params、GFLOPs、FPS、固定案例图 | Params/VRAM 部分完成；其余待做 |

当前分析关口结论：

1. M5-CE 当前临时沿用 L1 结果，待独立重跑确认。
2. A5、B2、B3、C2 已清空，不能使用旧 combined-loss 数值作当前结论。
3. L1 的 CE-only 结果作为当前 M5-CE 的临时参考；L3 保留为历史 M5-combined 对照。

## 3. 推荐执行顺序的实际进度

| 阶段 | 计划内容 | 当前进度 |
|---|---|---|
| Phase 0 | E0 数据审计与协议冻结 | 部分完成：有 mapping manifest 和固定 split；完整泄漏/统计审计未完成 |
| Phase 1 | M1–M5-CE、A5、A1/A2 | M5-CE、A1/A2/A5 待完成 |
| Phase 2 | B2/B3、C2、L1/L2 | B2/B3/C2 待完成；L1/L2 保留历史参考 |
| Phase 3 | 最终 Test、扰动、可视化、效率与写作 | 待 CE-only 矩阵完成；E5、可视化、效率与统计区间待做 |

## 4. 期刊图表清单状态

| 编号 | 图表 | 当前状态 |
|---|---|---|
| Table 1 | Converted MMLSv2 划分、通道统计、映射协议 | 部分完成，统计与泄漏审计待补 |
| Table 2 | M1–M5 主基线 mean±std | 已有阶段性版本，见 E1 |
| Table 3 | A1–A5 模态/容量消融 | 已有阶段性版本，见 E2 |
| Table 4 | B1–B3 通道分组 | 已有阶段性版本，见 E3；B4 未执行 |
| Table 5 | Add vs. Concat、CE/Dice | 已有阶段性版本，见 E4 |
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
- Test 批量评价日志：远程 `outputs_experiments/launch_logs/test_metrics_all_20261016.log`，末尾状态为 `[paper-test] all formal runs evaluated`。
- 汇总脚本：`tools/evaluation/summarize_paper_results.py`。
- 汇总 CSV/JSON：远程 `outputs_experiments/pmdfnet_mmlsv2/aggregate/paper_aggregate.csv` 和 `paper_aggregate.json`。
- 正式运行目录：远程 `outputs_experiments/pmdfnet_mmlsv2/`；历史 run 仍保留，当前有效表格已将 A1/A2/A5/B2/B3/C2 置空。M5-CE 正在重新训练，完成后再刷新正式汇总；当前尚未统一同步到数据集 `outputs/` 目录。
- 数据映射记录：远程数据根目录下的 `mapping_manifest.json`。

### 复现注意事项

1. 运行数据路径、远程主机和输出目录应通过环境变量提供，不应写死到论文命令中。
2. 要复现本文件的正式核心结果，应显式使用 batch32；不能直接使用协议模板当前的 batch16 默认值。
3. 应在 clean working tree 和固定 commit 下重跑 M1 seed 41，并把最终 commit/hash 写入结果归档。
4. 当前结果依赖不能公开再分发的映射数据；论文需要提供合法数据来源和确定性转换步骤，而不能把 converted MMLSv2 表述为新的独立数据集。

## 6. 投稿前检查清单

- [ ] 完成 M5-CE 及 A1/A2/A5、B2/B3、C2 的 `41/42/43` 三 seed 训练与 Test 汇总；L1/L2 保留为历史参考。
- [x] 已记录 best checkpoint 按验证集 mIoU 选择，并保存 best/last 与完整训练日志。
- [x] 已完成 batch32、A100 80GB、40GB 显存上限下的核心 smoke test。
- [ ] 完成 E0 通道、单位、归一化、映射、哈希和场景/相邻切片泄漏审计。
- [ ] 完成 A1/A2/A5 的 CE-only 正式训练；A5 保留“近似容量对照”表述。
- [ ] 完成 B2/B3 的 CE-only 三 seed 通道分组比较；B4 未执行。
- [ ] 完成 C2 的 CE-only 融合消融；L1/L2 保留为历史参考。
- [ ] 完成统一的 GFLOPs、FPS、latency 与统计置信区间。
- [ ] 完成预注册 Test 可视化、FP/FN 误差分析和 E5 鲁棒性实验。
- [ ] 清理/重跑 dirty run，并冻结最终 commit、配置和数据版本。
- [ ] 论文中明确 converted MMLSv2、公开标注 Test 与原比赛隐藏榜单的边界。

## 结论（当前阶段）

当前论文主模型暂定义为 PMDFNet-Small + CE-only。M5-CE 的临时结果沿用 L1 的 `0.8594 ± 0.0034`，但独立重跑尚未完成；A1/A2/A5/B2/B3/C2 已清空等待统一重跑。因此目前不对双流结构、容量或通道分组作最终强结论。

下一步应完成 E0 数据与泄漏审计、严格的 Params/GFLOPs 对齐、图像级 bootstrap/配对统计、固定规则定性分析和 E5 鲁棒性实验，并将 Test 结果统一归档后，再决定是否进入 Remote Sensing 投稿稿件的最终 Results/Discussion。
