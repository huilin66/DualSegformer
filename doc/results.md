# PMDFNet：MMLSv2 实验结果记录

> 对应计划：[`PMDFNet_MMLSv2_Experiment_Plan_v2.md`](PMDFNet_MMLSv2_Experiment_Plan_v2.md)<br>
> 记录日期：2026-10-16<br>
> 文档性质：阶段性实验台账，不等同于最终论文中的 Results/Discussion。

## 结果摘要

当前已经完成 M1–M5 主基线以及 E2–E4 的 24 个新增正式 run。每个配置均使用 seed `41/42/43`，每个 seed 训练 100 epochs，并使用验证集 mIoU 最优 checkpoint 在 converted MMLSv2 的公开标注 Test split 上评价。表中的 Test 指标是全测试集像素汇总后的 global-pixel 指标，不是逐图像指标的简单平均。

| 配置 | Val 最优 mIoU | Test mIoU | Test Foreground IoU | Test F1 | 当前状态 |
|---|---:|---:|---:|---:|---|
| M1 U-Net | 0.8077 ± 0.0030 | 0.8195 ± 0.0037 | 0.7670 ± 0.0077 | 0.8681 ± 0.0049 | 已完成 |
| M2 DeepLabV3+ | 0.8083 ± 0.0039 | 0.8321 ± 0.0025 | 0.7823 ± 0.0039 | 0.8778 ± 0.0025 | 已完成 |
| M3 SegFormer-MiT-B2 | 0.8226 ± 0.0012 | 0.8331 ± 0.0055 | 0.7834 ± 0.0073 | 0.8785 ± 0.0046 | 已完成 |
| M4 单流 ConvNeXt-Small + MLP Decoder | **0.8352 ± 0.0051** | 0.8507 ± 0.0021 | 0.8054 ± 0.0043 | 0.8922 ± 0.0026 | 已完成 |
| M5 PMDFNet-Small | 0.8208 ± 0.0146 | **0.8534 ± 0.0143** | **0.8089 ± 0.0188** | **0.8943 ± 0.0116** | 已完成 |

初步观察如下：

- M5 的 Test mIoU 均值比直接对照 M4 高 `0.00264`（约 `0.26` 个百分点），但 M5 的跨 seed 标准差为 `0.0143`，明显高于 M4 的 `0.0021`。
- M5 的 Val 最优 mIoU 均值反而低于 M4（`0.8208` vs. `0.8352`），因此只能说 M5 在 M1–M5 汇总中取得了最高均值，不能直接写成“稳定显著优于单流模型”。
- E2–E4 的 Test 结果已经补齐：A5（近似容量对照）为 `0.8310 ± 0.0116`，B3 为 `0.8498 ± 0.0064`，C2 为 `0.8470 ± 0.0113`，L1（CE-only）为当前最高的 `0.8594 ± 0.0034`。
- M5 相比 M1/M2/M3 的 Test mIoU 均值差分别约为 `+3.39/+2.13/+2.03` 个百分点；相对于公平的 M4 单流对照，增益较小，且 M5 的方差更大。

## 0. 研究问题、边界与待验证主张

### RQ1：统一协议下是否有稳定性能改善

**阶段性答案：有初步信号，但证据不足以称为稳定改善。** M5 在 M1–M5 的 Test mIoU 均值中最高，但 M4 的结果更加稳定；在完整消融中，L1（CE-only）均值高于 M5，说明当前不能把 combined loss 视为已验证的最优选择。

### RQ2：性能是否超出双编码器带来的容量影响

**得到初步回答，但仍需谨慎。** M4 约为 `50.09M` 参数，M5 约为 `99.54M` 参数，M5 接近 M4 的两倍。A5“近似容量对照”Test mIoU 为 `0.8310 ± 0.0116`，低于 M5 的 `0.8534 ± 0.0143`；这不支持将 M5 的结果完全解释为容量差异，但 A5 不是严格参数匹配，不能据此建立因果结论。

### RQ3：原始 3+4 通道分组是否更优

**得到初步回答。** B1（复用 M5）为 `0.8534 ± 0.0143`，B2 为 `0.8371 ± 0.0060`，B3 为 `0.8498 ± 0.0064`。原始 B1 分组均值最高，但与 B3 的差距很小，不能写成强优势。

### RQ4：Add Fusion 与 CE + Dice 的设计是否合理

**得到初步回答。** C1/Add（复用 M5）为 `0.8534 ± 0.0143`，C2/Concat 为 `0.8470 ± 0.0113`；L1/CE-only 为 `0.8594 ± 0.0034`，L2/Dice-only 为 `0.8323 ± 0.0054`。当前结果支持 Add 优于 Concat、CE-only 高于 Dice-only，但不支持 combined loss 优于 CE-only。

### RQ5：性能差异是否可靠并对扰动/缺失模态稳定

**部分回答。** M1–M5 及 A1/A2/A5/B2/B3/C2/L1/L2 均完成三个 seed 的重复；M5 的 Test 方差偏大，L1 的方差较小。E5 受控扰动和模态缺失实验尚未执行。

### 结论边界

当前结果最多支持以下谨慎表述：在 converted MMLSv2 的当前划分和训练协议下，PMDFNet-Small 在 M1–M5 基线中取得最高平均 mIoU，但在完整 E2–E4 矩阵中低于 L1 的 CE-only 结果；相对于参数约半数的单流 ConvNeXt-Small，其均值增益很小且稳定性较弱。当前不能声称严格的 physics-informed learning，也不能声称跨区域泛化或优于原比赛隐藏 Test。

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

M5 的最高均值主要由 seed 41/42 支撑，seed 43 明显下降；M4 的三个 seed 更集中。基于当前 `n=3` 结果，不进行显著性检验结论、不报告“显著提升”，也不把 M5 与 M4 的小幅均值差写成确定的双流因果收益。E1 的 Table 2 已具备三 seed Test 汇总，但 E0 审计、严格容量匹配、置信区间和配对统计检验仍未完成。

## E2. 单流/双流与模态贡献消融

**状态：已完成。** A1、A2、A5 均使用 seed `41/42/43` 完成正式训练和 Test；A3 复用 M4，A4 复用 M5。下表为 global-pixel Test 指标的跨 seed 均值 ± 样本标准差。

| ID | 方法 | Val best mIoU | Test mIoU | Test IoU-FG | Test F1 | 当前状态 |
|---|---|---:|---:|---:|---:|---|
| A1 | 光学单流（4 通道） | 0.7955 ± 0.0051 | 0.8266 ± 0.0178 | 0.7736 ± 0.0234 | 0.8722 ± 0.0149 | 已完成 |
| A2 | 物理单流（3 通道） | 0.8258 ± 0.0025 | 0.8460 ± 0.0096 | 0.7999 ± 0.0127 | 0.8888 ± 0.0079 | 已完成 |
| A3 | 7 通道单流（复用 M4） | 0.8352 ± 0.0051 | 0.8507 ± 0.0021 | 0.8054 ± 0.0043 | 0.8922 ± 0.0026 | 复用 M4 |
| A4 | 原始双流（复用 M5） | 0.8208 ± 0.0146 | 0.8534 ± 0.0143 | 0.8089 ± 0.0188 | 0.8943 ± 0.0116 | 复用 M5 |
| A5 | **近似容量对照：单流 ConvNeXt-Base** | 0.8308 ± 0.0026 | 0.8310 ± 0.0116 | 0.7822 ± 0.0152 | 0.8777 ± 0.0096 | 已完成 |

A4 相比 A3 的 Test mIoU 均值仅高 `0.00264`，而 A4 的标准差更大；A4 相比 A5 高 `0.02237`，但 A5 是近似容量对照而非严格参数匹配。因此，当前结果不支持把双流增益写成稳定或完全由模态解耦导致的提升。

## E3. 通道分组消融

**状态：已完成首轮规定的 B1–B3。** B1 复用 M5，B2/B3 使用相同的训练协议和三个 seed 完成正式训练与 Test；B4 未纳入本轮计划矩阵。

| ID | Stream 1 | Stream 2 | Test mIoU | Test IoU-FG | Test F1 | 当前状态 |
|---|---|---|---:|---:|---:|---|
| B1 | Thermal + Slope + DEM | Grayscale + RGB | 0.8534 ± 0.0143 | 0.8089 ± 0.0188 | 0.8943 ± 0.0116 | 复用 M5 |
| B2 | RGB | Thermal + Slope + DEM + Grayscale | 0.8371 ± 0.0060 | 0.7870 ± 0.0087 | 0.8808 ± 0.0055 | 已完成 |
| B3 | Thermal + DEM + Grayscale | Slope + RGB | 0.8498 ± 0.0064 | 0.8048 ± 0.0071 | 0.8918 ± 0.0043 | 已完成 |

B1 的均值在 B1–B3 中最高，B3 接近 B1（差 `0.00361`），B2 明显较低。该结果对原始分组提供了初步支持，但尚不足以宣称强的分组因果优势。

## E4. 融合与损失消融

**状态：已完成。** C2、L1、L2 均完成三个 seed 的正式训练和 Test，C1/L3 复用 M5。

| ID | 变量 | Test mIoU | Test IoU-FG | Test F1 | 当前状态 |
|---|---|---:|---:|---:|---|
| C1 | Add Fusion（复用 M5） | 0.8534 ± 0.0143 | 0.8089 ± 0.0188 | 0.8943 ± 0.0116 | 复用 M5 |
| C2 | Concat + 1×1 Conv | 0.8470 ± 0.0113 | 0.8023 ± 0.0132 | 0.8903 ± 0.0081 | 已完成 |
| L1 | CE only | **0.8594 ± 0.0034** | **0.8166 ± 0.0041** | **0.8990 ± 0.0025** | 已完成 |
| L2 | Dice only | 0.8323 ± 0.0054 | 0.7812 ± 0.0099 | 0.8771 ± 0.0062 | 已完成 |
| L3 | CE + Dice（复用 M5） | 0.8534 ± 0.0143 | 0.8089 ± 0.0188 | 0.8943 ± 0.0116 | 复用 M5 |

在当前协议下，Add Fusion 比 Concat 高 `0.00637` mIoU；CE-only 比 combined loss（M5/L3）高 `0.00603`，Dice-only 则低 `0.02111`。因此后续论文应把 loss 结果作为真实消融结论，而不能预先假设 CE + Dice 最优。

## E5. 受控扰动及模态缺失

**状态：未执行。** 当前没有冻结的 `robustness_protocol.yaml`，也没有 Val 确定后的扰动等级、Test 结果或鲁棒性曲线。不得根据现有 clean Test 结果推断抗噪声、抗缺失模态或跨区域泛化能力。

## E6. 重复实验与复杂度

### E6.1 统计可靠性

M1–M5 的 15 个基线 run，以及 A1/A2/A5/B2/B3/C2/L1/L2 的 24 个新增消融 run，均完成 seed `41/42/43`，共 39 个正式 run，并保留所有 seed，没有剔除低分运行。当前报告的是跨 seed 的 mean ± std；图像级 bootstrap 置信区间和配对统计检验尚未完成。

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

核心正式运行的每个 run 目录均保留：`config.json`、`command.txt`、`metrics.csv`、`logs/train.log`、`results.json`、`runtime.json`、`reproducibility.json`、`checkpoints/best.pth`、`checkpoints/last.pth`，以及 Test 阶段生成的 `test_metrics.json`。best 的选择依据是验证集 `mIoU`，不是 Test 指标；当前没有额外保存辅助 checkpoint。24 个新增消融 run 均已生成 Test 指标，M1–M5 基线也已有 Test 指标。

旧的 batch16 试运行目录与正式 batch32 结果同时存在于远程输出目录中；本文件只使用 24 个正式 batch32 run，不混入旧试运行。Test 结果当前保存在远程项目目录，尚未统一同步到数据集 `outputs/`。

## E7. 定性与误差分析

**状态：未执行。** 当前已有各模型 best checkpoint 和 Test 指标，但尚未按预注册规则固定 Test 图像 ID，也没有生成 `RGB | GT | M4 | M5 | FP/FN map` 图和总体 FP/FN 分布。因此暂不从个别案例推导模型优势。

## 2. 首轮最小实验矩阵执行状态

| 类别 | 计划配置 | 当前状态 |
|---|---|---|
| 主方法及直接对照 | M4、M5 | 已完成，均为 3 seeds |
| 常见代表性基线 | M1、M2、M3 | 已完成，均为 3 seeds |
| 模态贡献 | A1、A2 | 已完成，均为 3 seeds |
| 容量对照 | A5 近似容量对照 | 已完成，均为 3 seeds |
| 物理分组 | B1、B2、B3 | 已完成，B1 复用 M5 |
| Fusion/损失 | C1、C2、L1、L2 | 已完成，C1 复用 M5 |
| 统计 | M1–M5、A1/A2/A5、B2/B3、C2、L1/L2 使用 41/42/43 | 已完成 |
| 效率/可视化 | Params、GFLOPs、FPS、固定案例图 | Params/VRAM 部分完成；其余待做 |

当前分析关口结论：

1. M5 对 M4 的均值优势较小且方差较大，不能仅凭当前结果确认双流设计的稳定收益。
2. M5 高于 A5，但 A5 是近似容量对照；仍需补齐严格 Params/GFLOPs 对齐后再讨论容量解释。
3. B1 在 B1–B3 中均值最高，B3 接近 B1；物理分组优势目前是初步而非决定性证据。
4. L1 CE-only 在当前 E4 结果中最高，说明 combined loss 不是已验证的最优配置；应按实际结果修改论文主张，而不是回避该结果。

## 3. 推荐执行顺序的实际进度

| 阶段 | 计划内容 | 当前进度 |
|---|---|---|
| Phase 0 | E0 数据审计与协议冻结 | 部分完成：有 mapping manifest 和固定 split；完整泄漏/统计审计未完成 |
| Phase 1 | M1–M5、A5、A1/A2 | 已完成训练与 Test |
| Phase 2 | B2/B3、C2、L1/L2 | 已完成训练与 Test |
| Phase 3 | 最终 Test、扰动、可视化、效率与写作 | M1–M5 与 24 个新增消融 run 的 clean Test 已完成；E5、可视化、效率与统计区间待做 |

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
- 正式运行目录：远程 `outputs_experiments/pmdfnet_mmlsv2/`；24 个新增消融 run 均已生成 `test_metrics.json`，M1–M5 基线已有对应 Test 结果，汇总文件已刷新。当前尚未统一同步到数据集 `outputs/` 目录。
- 数据映射记录：远程数据根目录下的 `mapping_manifest.json`。

### 复现注意事项

1. 运行数据路径、远程主机和输出目录应通过环境变量提供，不应写死到论文命令中。
2. 要复现本文件的正式核心结果，应显式使用 batch32；不能直接使用协议模板当前的 batch16 默认值。
3. 应在 clean working tree 和固定 commit 下重跑 M1 seed 41，并把最终 commit/hash 写入结果归档。
4. 当前结果依赖不能公开再分发的映射数据；论文需要提供合法数据来源和确定性转换步骤，而不能把 converted MMLSv2 表述为新的独立数据集。

## 6. 投稿前检查清单

- [x] 已完成 M1–M5 基线及 A1/A2/A5、B2/B3、C2、L1/L2 的 `41/42/43` 三 seed 训练与 Test 汇总（15 个基线 run + 24 个新增消融 run）。
- [x] 已记录 best checkpoint 按验证集 mIoU 选择，并保存 best/last 与完整训练日志。
- [x] 已完成 batch32、A100 80GB、40GB 显存上限下的核心 smoke test。
- [ ] 完成 E0 通道、单位、归一化、映射、哈希和场景/相邻切片泄漏审计。
- [x] 完成 A1/A2/A5 正式训练；A5 保留“近似容量对照”表述。
- [x] 完成 B2/B3 三 seed 通道分组比较；B4 未执行。
- [x] 完成 C2、L1、L2 融合/损失消融。
- [ ] 完成统一的 GFLOPs、FPS、latency 与统计置信区间。
- [ ] 完成预注册 Test 可视化、FP/FN 误差分析和 E5 鲁棒性实验。
- [ ] 清理/重跑 dirty run，并冻结最终 commit、配置和数据版本。
- [ ] 论文中明确 converted MMLSv2、公开标注 Test 与原比赛隐藏榜单的边界。

## 结论（当前阶段）

本轮结果表明，PMDFNet-Small 在 converted MMLSv2 的 M1–M5 基线中 Test 平均 mIoU 最高（`0.8534 ± 0.0143`），相对于单流 ConvNeXt-Small M4 的优势约为 `0.26` 个百分点，但稳定性较弱。A5 近似容量对照为 `0.8310 ± 0.0116`，B1 原始分组高于 B2/B3，C1/Add 略高于 C2/Concat；同时 L1 CE-only 达到全矩阵最高均值（`0.8594 ± 0.0034`），因此当前证据不支持“PMDFNet 的 combined loss 或双流结构在所有设置下最优”的强结论。

下一步应完成 E0 数据与泄漏审计、严格的 Params/GFLOPs 对齐、图像级 bootstrap/配对统计、固定规则定性分析和 E5 鲁棒性实验，并将 Test 结果统一归档后，再决定是否进入 Remote Sensing 投稿稿件的最终 Results/Discussion。
