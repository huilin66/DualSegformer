# PMDFNet：MMLSv2 实验结果记录

> 对应计划：[`PMDFNet_MMLSv2_Experiment_Plan_v2.md`](PMDFNet_MMLSv2_Experiment_Plan_v2.md)<br>
> 记录日期：2026-10-16（本轮：结果表按"仅保留 L1–L3"重置）<br>
> 文档性质：阶段性实验台账，不等同于最终论文中的 Results/Discussion。

## 结果摘要

当前论文主模型定义为 **PMDFNet-Small + CE-only**，其配置为原 L1；论文结果表中记为 M5。

**本次重置口径**：除损失函数消融 L1/L2/L3 外，其余全部实验（M1–M4、M5 的独立 run、A1/A2/A5、B2/B3、C2）的数值均已从本文档清空。这些配置的旧结果来自此前的 combined-loss（CE + Dice）协议，与当前主模型的 CE-only 口径不一致，不得继续引用，需按统一口径重跑后重新填入。

重置后本文档保留的唯一数值是损失函数消融参考结果（CE / Dice / CE+Dice，同一 PMDFNet-Small 架构、同一训练协议）：

| ID | 变量 | Val 最优 mIoU | Test mIoU | Test IoU-FG | Test F1 | 当前状态 |
|---|---|---:|---:|---:|---:|---|
| L1 | CE only | 0.8209 ± 0.0096 | **0.8594 ± 0.0034** | **0.8166 ± 0.0041** | **0.8990 ± 0.0025** | 保留（主模型即此配置） |
| L2 | Dice only | — | 0.8323 ± 0.0054 | 0.7812 ± 0.0099 | 0.8771 ± 0.0062 | 保留（历史损失对照） |
| L3 | CE + Dice | — | 0.8534 ± 0.0143 | 0.8089 ± 0.0188 | 0.8943 ± 0.0116 | 保留（历史 combined-loss run；非当前主模型） |

> 说明：L2/L3 的 Val 最优 mIoU 与逐 seed 数值当前未归档在本文档内，仅在远程运行目录中，故此处留空。

初步观察如下：

- 当前主模型为 L1 的 CE-only 配置，Test mIoU 为 `0.8594 ± 0.0034`；L1 已作为 M5 登记，不再单独训练 M5。
- 除 L1/L2/L3 外的表格结果已全部清空；M1–M4 使用 combined loss，A1/A2/A5/B2/B3/C2 的旧结果亦为 combined-loss 协议，均不能继续使用。
- L3 的 `0.8534 ± 0.0143` 来自旧 combined-loss 协议下的 PMDFNet run，仅作为损失对照保留，不与当前主模型混称。
- 本文档当前不包含任何基线间比较结论；M1–M4 及全部消融需按 CE-only 统一口径重跑。

## 0. 研究问题、边界与待验证主张

### RQ1：统一协议下是否有稳定性能改善

**暂无有效答案。** 主模型已固定为 CE-only 的 PMDFNet-Small（L1），但 M1–M4 基线当前使用 combined loss，损失口径不一致，不能与主模型直接对比。M1–M4 需按 CE-only 重跑；在重跑完成前不做任何基线间比较结论。

### RQ2：性能是否超出双编码器带来的容量影响

**暂无有效答案。** M4 约为 `50.09M` 参数，M5/L1 约为 `99.54M` 参数；A5（近似容量对照，约 `88.33M`）当前结果已清空，需按 CE-only 重跑后再比较容量因素。

### RQ3：原始 3+4 通道分组是否更优

**暂无有效答案。** B1 不再复用任何已完成结果；B2、B3 当前结果已清空，需按 CE-only 重跑后比较分组。

### RQ4：Add Fusion 与 CE + Dice 的设计是否合理

**暂无有效答案（损失部分除外）。** C1/Add 不再复用已完成结果，C2 当前已清空，二者需按 CE-only 重跑后比较融合算子。损失函数方面，同一架构下的 L1（CE-only）/L2（Dice-only）/L3（CE+Dice）已完成，是本轮保留的参考结果。

### RQ5：性能差异是否可靠并对扰动/缺失模态稳定

**尚未完成。** 除 L1–L3 外无有效结果；E5 受控扰动和模态缺失实验尚未执行。

### 结论边界

当前文档不包含任何方法比较结论。可确认的仅有：论文主模型为 CE-only 的 PMDFNet-Small（L1）；同一架构下的损失函数消融中 CE-only 优于 Dice-only 与 CE + Dice。基线对比、双流增益、容量解释、通道分组优势均待重跑后判断。当前不能声称严格的 physics-informed learning，也不能声称跨区域泛化或优于原比赛隐藏 Test。

### 0.1 主模型选择

当前主模型定义为 `PMDFNet-Small + CE-only`，即原 L1 配置；论文结果表中将其记为 M5。单流 ConvNeXt-Small + MLP Decoder（M4）仍作为直接对照，但需按 CE-only 重跑后才可用于比较。

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

上述配置与协议中的 L1 逐字段一致（仅 `experiment_name` 与 `seed` 不同），因此 M5 不再单独训练。

## E0. 数据版本与转换映射协议审计

### E0.1 转换与通道审计

**状态：已建立基础记录，但尚未闭环。** 数据根目录的 `mapping_manifest.json` 已记录来源目录、参考比赛数据目录、输入/输出通道顺序、scale/offset 和 split 数量。当前清单给出的输出通道顺序为：

`Thermal → Slope → DEM → Grayscale → R → G → B`。

划分数量为 Train `465`、Val `66`、Test `133`。映射验证记录显示多数通道的误差比例很低，但 Slope/DEM 仍出现较大的最大绝对误差记录，Train mask 验证还记录了 `1` 个差异像素。因此，映射清单可以作为当前运行的来源记录，但在投稿前仍需解释这些异常值并完成独立复核，不能直接写成"逐像素完全一致"。

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

本轮评价使用固定的 converted MMLSv2 Test split，并采用每个 seed 的验证集最优 mIoU checkpoint；不得使用 Test 分数回头选择 seed、阈值或主模型。需要注意：这里的 Test 是有标注的公开 Test split，不能与原比赛的隐藏测试榜单等同。

### 1.2 基础训练设置

正式核心运行的实际 `config.json` 和启动命令记录为：

- 输入尺寸 `128×128`，训练 `100 epochs`，**batch size=32**；
- AdamW，学习率 `1e-4`，weight decay `5e-4`，cosine scheduler；
- `augmentation=mars`，`aug_prob=0.5`，`mosaic_prob=0.5`；
- ImageNet/代码配置的预训练开关为 `pretrain=true`；
- `mixed_precision=false`，`num_workers=4`，设备为 A100 `cuda:0`；
- `deterministic=true`，严格确定性设置已记录；
- 每个 epoch 验证并按 `primary_metric=miou` 选择 best checkpoint，另保存 last checkpoint。

配置模板 `configs/mmlsv2_paper_protocol.json` 的 `common` 段当前 batch size 已为 `32`，与正式运行一致；但 `common.loss` 仍为 `"combined"`，与当前主模型的 `ce` 口径相反，需在重跑前统一，避免新增实验静默使用错误损失。

### 1.3 统一评价口径

Test 结果使用全数据集像素累计混淆矩阵计算 mIoU、Foreground IoU、Background IoU、Precision、Recall 和 F1。每个运行的 `test_metrics.json` 同时保留 global-pixel 与 batch-average 两套数值；本文档和 `paper_aggregate.csv` 使用前者作为主口径。

Test 评价共覆盖 `133` 张图像，每个运行使用 batch size `32`、共 `5` 个 batch。阈值采用当前二分类评估实现的默认规则；独立的 Val 阈值冻结实验尚未补齐，投稿前需在协议中明确写出。

## E1. 主实验：公平基线对比

**状态：已重置。** 主模型固定为 CE-only 的 PMDFNet-Small（L1）。M1–M4 基线的旧结果使用 combined loss，与主模型口径不一致，已从本文档清空。

为使 E1 成立，M1–M4 必须按 CE-only 重跑（3 seeds）；否则只能把主表降级为"各方法使用各自常规损失"的说明性对比，并在正文逐行标注损失口径。

| ID | 方法 | Val best mIoU | Test mIoU | Test IoU-FG | Test IoU-BG | Test F1 | 当前状态 |
|---|---|---:|---:|---:|---:|---:|---|
| M1 | U-Net | — | — | — | — | — | 待 CE-only 重跑 |
| M2 | DeepLabV3+ ResNet50 | — | — | — | — | — | 待 CE-only 重跑 |
| M3 | SegFormer MiT-B2 | — | — | — | — | — | 待 CE-only 重跑 |
| M4 | 单流 ConvNeXt-Small + MLP | — | — | — | — | — | 待 CE-only 重跑 |
| M5 | PMDFNet-Small + CE-only | — | — | — | — | — | 采用 L1（见损失消融表） |

### E1.1 损失函数消融（本轮保留的唯一数值结果）

同一 PMDFNet-Small 架构、同一训练协议，仅改变损失函数：

| ID | 损失 | Val best mIoU | Test mIoU | Test IoU-FG | Test IoU-BG | Test F1 |
|---|---|---:|---:|---:|---:|---:|
| L1 | CE only | 0.8209 ± 0.0096 | **0.8594 ± 0.0034** | **0.8166 ± 0.0041** | **0.9022 ± 0.0028** | **0.8990 ± 0.0025** |
| L2 | Dice only | — | 0.8323 ± 0.0054 | 0.7812 ± 0.0099 | — | 0.8771 ± 0.0062 |
| L3 | CE + Dice | — | 0.8534 ± 0.0143 | 0.8089 ± 0.0188 | — | 0.8943 ± 0.0116 |

L3 为旧 combined-loss 协议下的 PMDFNet run，作为损失对照保留；本文档不将其等同于当前主模型的独立 run。L2/L3 的 Val 与 IoU-BG 数值未归档在本文件（见"复现目录与记录"）。

### E1.2 分 seed Test mIoU

**待重跑后填写。** M1–M4 的逐 seed 数值已重置；L1/L2/L3 的逐 seed 数值当前仅在远程运行目录中，尚未统一归档。

### E1.3 当前解释

本轮不给出基线对比结论。已完成的损失函数消融显示 CE-only（L1）高于 CE + Dice（L3）与 Dice-only（L2）；该结论仅在同一架构内成立，不外推到其他骨干。基线间差异需在 M1–M4 按 CE-only 重跑后，结合配对统计检验再作判断。

## E2. 单流/双流与模态贡献消融

**状态：已重置。** A1、A2、A5 的旧结果已从当前表格清空，需按 CE-only 重跑；A3 复用 M4，A4 复用 M5/L1（二者均需等待对应重跑/登记确认）。下表为 global-pixel Test 指标的跨 seed 均值 ± 样本标准差。

| ID | 方法 | Val best mIoU | Test mIoU | Test IoU-FG | Test F1 | 当前状态 |
|---|---|---:|---:|---:|---:|---|
| A1 | 光学单流（4 通道） | — | — | — | — | 已重置，待 CE-only 重跑 |
| A2 | 物理单流（3 通道） | — | — | — | — | 已重置，待 CE-only 重跑 |
| A3 | 7 通道单流（复用 M4） | — | — | — | — | 等待 M4 CE-only 重跑 |
| A4 | 原始双流（复用 M5/L1） | — | — | — | — | 复用 L1，数值见 E1.1 |
| A5 | 近似容量对照：单流 ConvNeXt-Base | — | — | — | — | 已重置，待 CE-only 重跑 |

A5 是当前最关键的单点风险：若 A5（约 88.33M）在 CE-only 下达到或超过 L1（约 99.54M），则"双流收益来自容量"的解释成立，主结论需要收缩。在 A5 完成前不对双流增益或容量因素作任何结论。

## E3. 通道分组消融

**状态：已重置。** B1 不再复用 M5 的旧结果；B2/B3 的旧结果已清空；B4 未纳入本轮计划矩阵。

| ID | Stream 1 | Stream 2 | Test mIoU | Test IoU-FG | Test F1 | 当前状态 |
|---|---|---|---:|---:|---:|---|
| B1 | Thermal + Slope + DEM | Grayscale + RGB | — | — | — | 复用 L1，数值见 E1.1 |
| B2 | RGB | Thermal + Slope + DEM + Grayscale | — | — | — | 已重置，待 CE-only 重跑 |
| B3 | Thermal + DEM + Grayscale | Slope + RGB | — | — | — | 已重置，待 CE-only 重跑 |

分组结论待 B2/B3 按 CE-only 重跑后判断。

## E4. 融合与损失消融

**状态：已重置（损失部分除外）。** C1 不再复用旧结果；C2 的旧结果已清空；L1/L2/L3 保留为损失函数参考结果。

| ID | 变量 | Test mIoU | Test IoU-FG | Test F1 | 当前状态 |
|---|---|---:|---:|---:|---|
| C1 | Add Fusion（复用 L1） | 0.8594 ± 0.0034 | 0.8166 ± 0.0041 | 0.8990 ± 0.0025 | 复用 L1 |
| C2 | Concat + 1×1 Conv | — | — | — | 已重置，待 CE-only 重跑 |
| L1 | CE only | **0.8594 ± 0.0034** | **0.8166 ± 0.0041** | **0.8990 ± 0.0025** | 保留 |
| L2 | Dice only | 0.8323 ± 0.0054 | 0.7812 ± 0.0099 | 0.8771 ± 0.0062 | 保留 |
| L3 | CE + Dice | 0.8534 ± 0.0143 | 0.8089 ± 0.0188 | 0.8943 ± 0.0116 | 保留（历史 combined-loss run） |

损失函数结论：同一架构下 CE-only > CE + Dice > Dice-only。融合算子（Add vs. Concat）的最终比较待 C2 按 CE-only 重跑完成。

## E5. 受控扰动及模态缺失

**状态：未执行。** 当前没有冻结的 `robustness_protocol.yaml`，也没有 Val 确定后的扰动等级、Test 结果或鲁棒性曲线。不得根据现有 clean Test 结果推断抗噪声、抗缺失模态或跨区域泛化能力。

## E6. 重复实验与复杂度

### E6.1 统计可靠性

除保留的 L1–L3 外，本轮结果表已重置。历史版本曾完成 M1–M5 的 15 个基线 run，以及 A1/A2/A5/B2/B3/C2/L1/L2 的 24 个新增消融 run；本次重置后，这些历史数值不再作为表格结果使用（L1/L2/L3 除外）。图像级 bootstrap 置信区间和配对统计检验尚未完成。

需要特别记录：历史 M1 seed 41 的运行元数据显示 `git dirty=true`，其余核心正式运行记录为 clean working tree。该运行在重跑时应在冻结 commit 下重新执行，并把最终 commit/hash 写入结果归档。

### E6.2 参数量与显存 smoke test

以下为 A100 80GB、输入 `32×7×128×128` 的 batch32 smoke test 结果。参数量来自模型实例，显存为本次 smoke test 的峰值 reserved；所有配置均未超过 40GB 限制。该表是模型实例的结构性属性快照，与损失函数口径无关，故本轮保留。

| ID | Params (M) | Peak allocated (GB) | Peak reserved (GB) | Smoke 状态 |
|---|---:|---:|---:|---|
| M1 | 32.53 | 2.93 | 3.01 | 通过 |
| M2 | 26.69 | 1.40 | 1.83 | 通过 |
| M3 | 24.74 | 1.73 | 2.14 | 通过 |
| M4 | 50.09 | 2.40 | 2.82 | 通过 |
| M5 | 99.54 | 4.32 | 4.83 | 通过 |

A1/A2/A5/B2/B3/C2 的 smoke test 也已通过；其中 A5 约 `88.33M`，C2 约 `101.11M`。当前尚未完成统一工具下的 GFLOPs、FPS、latency、warm-up 和多次测速，因此 Table 6 仍不完整。

### E6.3 日志与 checkpoint 完整性

历史正式运行目录保留：`config.json`、`command.txt`、`metrics.csv`、`logs/train.log`、`results.json`、`runtime.json`、`reproducibility.json`、`checkpoints/best.pth`、`checkpoints/last.pth`，以及 Test 阶段生成的 `test_metrics.json`。best 的选择依据是验证集 `mIoU`，不是 Test 指标；当前没有额外保存辅助 checkpoint。历史结果不再作为当前表格结果；M5 不单独训练，直接采用 L1。

旧的 batch16 试运行目录与正式 batch32 结果同时存在于远程输出目录中；本文件只使用正式 batch32 run，不混入旧试运行。Test 结果当前保存在远程项目目录，尚未统一同步到数据集 `outputs/`。

## E7. 定性与误差分析

**状态：未执行。** 当前已有各模型 best checkpoint 和 Test 指标，但尚未按预注册规则固定 Test 图像 ID，也没有生成 `RGB | GT | M4 | M5 | FP/FN map` 图和总体 FP/FN 分布。因此暂不从个别案例推导模型优势。

## 2. 首轮最小实验矩阵执行状态

| 类别 | 计划配置 | 当前状态 |
|---|---|---|
| 主方法及直接对照 | M4、M5（采用 L1） | M4 待 CE-only 重跑；M5 采用 L1 |
| 常见代表性基线 | M1、M2、M3 | 已重置，待 CE-only 重跑 |
| 模态贡献 | A1、A2 | 已重置，待 CE-only 重跑 |
| 容量对照 | A5 近似容量对照 | 已重置，待 CE-only 重跑 |
| 物理分组 | B1、B2、B3 | B1 复用 L1；B2/B3 已重置，待 CE-only 重跑 |
| Fusion/损失 | C1、C2、L1、L2、L3 | C1 复用 L1；C2 已重置，待 CE-only 重跑；L1/L2/L3 保留 |
| 统计 | M1–M5、A1/A2/A5、B2/B3、C2、L1/L2 | 仅 L1/L2/L3 有价值；其余待重跑 |
| 效率/可视化 | Params、GFLOPs、FPS、固定案例图 | Params/VRAM 部分完成；其余待做 |

当前分析关口结论：

1. 主模型为 CE-only 的 PMDFNet-Small，直接采用 L1 结果，不再单独训练 M5。
2. 除 L1/L2/L3 外，全部结果表已重置；M1–M4 与 A1/A2/A5/B2/B3/C2 需按 CE-only 统一口径重跑。
3. 在重跑完成前，本文档不支撑任何双流增益、容量解释、通道分组或融合算子结论。

## 3. 推荐执行顺序的实际进度

| 阶段 | 计划内容 | 当前进度 |
|---|---|---|
| Phase 0 | E0 数据审计与协议冻结 | 部分完成：有 mapping manifest 和固定 split；完整泄漏/统计审计未完成 |
| Phase 1 | M1–M4（CE-only 重跑）、A5、A1/A2 | 待执行；M5 采用 L1 |
| Phase 2 | B2/B3、C2（CE-only 重跑）、L1/L2/L3 | B2/B3/C2 待执行；L1/L2/L3 已保留 |
| Phase 3 | 最终 Test、扰动、可视化、效率与写作 | E5、可视化、效率与统计区间待做 |

建议重跑优先级：M4（唯一公平直接对照）→ A5（容量对照，决定主结论存废）→ M1/M2/M3（统一协议）→ A1/A2/B2/B3/C2。

## 4. 期刊图表清单状态

| 编号 | 图表 | 当前状态 |
|---|---|---|
| Table 1 | Converted MMLSv2 划分、通道统计、映射协议 | 部分完成，统计与泄漏审计待补 |
| Table 2 | M1–M5 主基线 mean±std | 已重置，待 CE-only 重跑 |
| Table 3 | A1–A5 模态/容量消融 | 已重置，待 CE-only 重跑 |
| Table 4 | B1–B3 通道分组 | 已重置，待 CE-only 重跑；B4 未执行 |
| Table 5 | Add vs. Concat、CE/Dice | 损失部分已有（L1/L2/L3）；融合部分待 C2 重跑 |
| Table 6 | Params/GFLOPs/FPS | Params/VRAM 部分完成，GFLOPs/FPS 待补 |
| Table 7 | 受控扰动与模态缺失 | 待完成 |
| Figure 1 | PMDFNet-Small 结构图 | 未在本结果记录中更新 |
| Figure 2 | 固定规则的定性比较 | 待完成 |
| Figure 3 | 鲁棒性曲线 | 待完成 |
| Figure 4 | 错误统计与失败案例 | 待完成 |
| Supplementary | manifest、实现细节、单 seed 结果 | 待统一归档 |

## 5. 复现目录与记录

### 已存在的代码与远程产物

- 训练协议：`configs/mmlsv2_paper_protocol.json`。
- 训练入口：`scripts/train_paper_experiments.py`。
- Test 批量评价日志：远程 `outputs_experiments/launch_logs/test_metrics_all_20261016.log`。
- 汇总脚本：`tools/evaluation/summarize_paper_results.py`。
- 汇总 CSV/JSON：远程 `outputs_experiments/pmdfnet_mmlsv2/aggregate/paper_aggregate.csv` 和 `paper_aggregate.json`。
- 正式运行目录：远程 `outputs_experiments/pmdfnet_mmlsv2/`；历史 run 仍保留，当前有效表格仅保留 L1/L2/L3。M5 采用 L1，不单独训练；当前尚未统一同步到数据集 `outputs/` 目录。
- 数据映射记录：远程数据根目录下的 `mapping_manifest.json`。

### 复现注意事项

1. 运行数据路径、远程主机和输出目录应通过环境变量提供，不应写死到论文命令中。
2. 重跑必须显式使用 batch32，并确认 `loss` 口径为 `ce`（当前协议 `common.loss` 仍为 `combined`，需先统一）。
3. 应在 clean working tree 和固定 commit 下重跑 M1（含此前 dirty 的 seed 41），并把最终 commit/hash 写入结果归档。
4. 当前结果依赖不能公开再分发的映射数据；论文需要提供合法数据来源和确定性转换步骤，而不能把 converted MMLSv2 表述为新的独立数据集。
5. L2/L3 的 Val 最优 mIoU、IoU-BG 与逐 seed 数值未归档在本文件，重跑或补录前不要在论文中引用这些空缺项。

## 6. 投稿前检查清单

- [x] 主模型固定为 CE-only 的 PMDFNet-Small，直接采用 L1 结果；M5 不再单独训练。
- [ ] 完成 M1–M4 的 CE-only 三 seed 重跑（统一损失口径后才可写"统一协议公平对比"）。
- [ ] 完成 A1/A2/A5、B2/B3、C2 的 CE-only 三 seed 重跑。
- [x] 已记录 best checkpoint 按验证集 mIoU 选择，并保存 best/last 与完整训练日志。
- [x] 已完成 batch32、A100 80GB、40GB 显存上限下的核心 smoke test。
- [ ] 统一 `configs/mmlsv2_paper_protocol.json` 的 `common.loss` 口径，并清理 M5/L1 的重复配置项。
- [ ] 完成 E0 通道、单位、归一化、映射、哈希和场景/相邻切片泄漏审计。
- [ ] A5 保留"近似容量对照"表述，并在重跑后明确容量解释。
- [ ] B4 未执行，需在论文中说明。
- [ ] 完成统一的 GFLOPs、FPS、latency 与统计置信区间。
- [ ] 完成预注册 Test 可视化、FP/FN 误差分析和 E5 鲁棒性实验。
- [ ] 清理/重跑 dirty run，并冻结最终 commit、配置和数据版本。
- [ ] 论文中明确 converted MMLSv2、公开标注 Test 与原比赛隐藏榜单的边界。

## 结论（当前阶段）

当前论文主模型定义为 PMDFNet-Small + CE-only（L1），已完成的损失函数消融显示 CE-only 优于 Dice-only 与 CE + Dice。除 L1/L2/L3 外，本文档全部结果表已重置：M1–M4 基线使用 combined loss，A1/A2/A5/B2/B3/C2 的旧结果同属旧协议，均不能作为当前结果。因此目前不对双流结构、容量或通道分组作任何结论。

下一步应先统一协议损失口径并完成 M1–M4（尤其是 M4）与 A5 的 CE-only 重跑，再补 A1/A2/B2/B3/C2；同时完成 E0 数据与泄漏审计、严格的 Params/GFLOPs 对齐、图像级 bootstrap/配对统计和 E5 鲁棒性实验，并将 Test 结果统一归档后，再决定是否进入 Remote Sensing 投稿稿件的最终 Results/Discussion。
