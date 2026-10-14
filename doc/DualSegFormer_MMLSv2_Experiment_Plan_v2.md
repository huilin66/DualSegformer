# DualSegFormer：MMLSv2 期刊实验计划（修订版 v2）

> **目标期刊**：MDPI *Remote Sensing*  
> **研究定位**：保留既有 DualSegFormer，不增加新的主网络模块；通过公平基线、模态分组消融、统计重复和可复现的数据转换协议，检验双流设计的实际价值。  
> **主数据集**：`mmlsv2_mapped_mars_ls`（由 MMLSv2 转换/映射获得；论文中称为 *converted MMLSv2*，**不是新的独立数据集**）。  
> **主模型**：`DualSegFormer-Small`，两条独立 ConvNeXt-Small 编码器 + 四尺度 Add Fusion + SegFormer All-MLP Decoder。**在开始正式测试集比较前冻结此选择**。  
> **对照模型**：单流 ConvNeXt-Small + 同类 MLP Decoder；其他经典分割基线。  
> **明确排除**：TRB-Net（缺少可重复执行的代码/权重/协议，本计划不将其列为必做对比）。  
> **实验范围**：仅包含 MMLSv2；Landslide4Sense 另立计划。  
> **状态**：执行蓝本；参数、样本映射与预训练来源须经 E0/实现核验后最终冻结。

## 0. 研究问题、边界与待验证主张

文章拟验证的中心假设（**不是已被实验结果证实的结论**）：

> A physically motivated dual-stream decomposition can improve segmentation from heterogeneous optical and terrain-related Martian remote-sensing inputs without requiring complex cross-modal interaction modules.

对应研究问题：

- **RQ1：效果**——在统一 MMLSv2 协议下，DualSegFormer-Small 相较代表性分割方法能否获得稳定的性能改善？
- **RQ2：容量控制**——双流优点是否超出“两个编码器导致参数量增加”的影响？
- **RQ3：输入分组**——原始物理动机的 `3+4` 通道分组是否优于相同通道宽度的其他分组？
- **RQ4：设计合理性**——简单 Add Fusion 和 CE + Dice 是否能提供有竞争力的准确率—复杂度权衡？
- **RQ5：可靠性**——性能差异是否跨随机种子稳定，在预设的受控扰动和模态缺失下表现如何？

**结论边界**：即便取得性能优势，也只可声称“物理含义驱动的通道分组（physically motivated modality grouping）”，不应将它描述为通过物理方程约束训练的严格 *physics-informed learning*。如果容量匹配对照或替代分组更好，须如实收缩方法主张。

### 0.1 主模型为何选择 Small

本轮以**当前已验证的 `DualSegFormer-Small`**为主模型，配套使用 ConvNeXt-Small 单流直接对照；**Tiny 仅用于容量/效率补充**，不允许依据 Test 表现事后切换论文主模型。

应在实验记录中分别交代：

1. **原 Workshop 稿**：Table 2 同时包含 Tiny 和 Small，Tiny 具有完整 Test1/Test2 报告，而 Small 在该表仅报告 Test2；不能笼统宣称“原稿以 Small 为唯一主模型”。
2. **本轮期刊研究**：研究开始前选择 Small 作为主配置的原因、已验证的训练配置及代码版本；保存主模型冻结记录。
3. **Tiny 扩展**：按预先规定的辅助实验展示，不作为后验挑选的主结果。

### 0.2 固定网络定义

原始 7 通道（以 **E0 最终核实的索引映射**为准）：

| 分支 | 物理含义 | 通道数 |
|---|---|---:|
| Stream 1 | Thermal Inertia + Slope + DEM | 3 |
| Stream 2 | Grayscale + R + G + B | 4 |

两条权重独立的 ConvNeXt-Small 在相应尺度上产生特征，逐尺度相加后交给 SegFormer All-MLP Decoder；默认损失为 CE + Dice。主体网络不修改。对照实验允许更改**基线**骨干规模、首层输入通道数、融合消融算子或损失项，但须声明这些是实验对照而非最终方法升级。

---

## E0. 数据版本与转换映射协议审计【正式训练前必须完成】

**目的**：建立 `mmlsv2_mapped_mars_ls` 可追溯、可复现的来源链，排除输入顺序、数据版本和数据泄漏带来的伪提升。

### E0.1 转换与通道审计

- [ ] 明确原始 MMLSv2 下载来源、发布日期/commit/数据发布版本；记录本地转换脚本的 Git commit。
- [ ] 列出转换前后的 **7 个通道顺序、名称、物理单位、数据类型、NoData、取值范围**。
- [ ] 记录几何对齐、重采样、裁剪、插值、异常值处理、归一化及映射规则。
- [ ] 计算 Train / Val / Test 各通道的 min/max、均值、标准差、分位数和直方图；**Test 统计量只允许用于最终描述/分布差异分析，不得用于参数拟合或方案选择**。
- [ ] 所有归一化参数、裁剪阈值或校正系数必须明确来源；若来自 Train，只在 Train 拟合后应用到 Val/Test。
- [ ] 校验每张图像与 GT 的配对、mask 值域（前景/背景/ignore）、空间尺寸、波段次序。

### E0.2 划分与潜在泄漏审计

- [ ] 固定文件级的 Train / Val / Test manifest（相对路径、样本 ID、原场景 ID/地理来源，若有）。
- [ ] 检查跨划分重复文件/哈希一致样本，以及高度相似的相邻切片。
- [ ] 如果有场景/区域元数据，检查是否 **scene-level disjoint**；否则明确“无法证实场景级独立”，不能默认宣称跨区域泛化。
- [ ] 如果发现相邻切片跨集合，保留原划分用于对标时必须在论文中披露其影响；资源与元数据允许时，另设**预先冻结的场景独立划分**作为补充，不可依据 Test 结果重划分。
- [ ] 把全部图像/标签 manifest、转换配置、评估协议与文件 SHA-256/hash 存档。

### E0.3 最低交付物

- `dataset_manifest.csv`：文件名、split、scene/group、来源 ID、哈希、对应标注；
- `channel_mapping.yaml`：各通道来源索引、单位与映射方法；
- `dataset_audit.md`：数据分布与泄漏检查记录；
- 固定 `train/val/test` 文件列表和完整转换脚本。

**论文写法**：*Converted MMLSv2*（转换后的 MMLSv2）。只有证据支持时才能说明“与竞赛数据基本一致”；不得据此称其为新数据集。

---

## 1. 全实验通用协议【必须先冻结】

### 1.1 数据与模型选择

1. **Train**：仅用于优化模型参数、拟合归一化统计和训练增强。
2. **Validation**：用于 checkpoint、阈值和预定义超参数选择；受控扰动的**类型/强度/生成规则也先在这里确定并冻结**。
3. **Test**：仅在全部协议、模型及阈值冻结后执行最终评价；不得依据 Test 分数重新选择主模型、调整分组、阈值、扰动强度或模型配置。
4. 数据或映射后续发生变化，须生成新的版本号并在实验日志中说明；此前 Test 评估不能当作未使用。

### 1.2 基础训练设置

以现有代码可复现设置为优先，初始参考原稿：`128×128`、`100 epochs`、`batch size=16`、`AdamW`、初始学习率 `1e-4`、weight decay `5e-4`、cosine schedule。增强包括水平/垂直翻转、90° 倍数旋转、裁剪、Mosaic 等，需由同一实现控制。

- 主方法 **Small 与单流 Small** 的优化配置和训练预算尽可能一致。
- 单独记录每个模型的初始化（ImageNet/其他预训练）、首层通道适配方法和可训练参数。
- 主表仅用**单模型、无 TTA、无 Ensemble、无额外 Test 分布校正**的结果；TTA 单独补充。
- 先定义阈值策略：例如各模型在 Val 上按同一规则选择阈值，冻结后应用于 Test；同时可补充固定 0.5 的敏感性对照。
- 正式实验随机种子固定为 `41, 42, 43`，在首轮正式运行后不再依据结果更改。

### 1.3 统一评价口径

| 指标 | 定义/用途 |
|---|---|
| mIoU ↑ | 由全测试集累计混淆矩阵分别计算前景/背景 IoU 后取均值；**主要指标** |
| Foreground IoU ↑ | 滑坡类别 IoU；**主要指标** |
| Precision / Recall / F1 ↑ | 均以 Landslide 为正类 |
| Params (M) ↓ | 推理模型总参数数 |
| GFLOPs ↓ | 同一 `128×128` 输入、相同计量工具 |
| FPS / latency ↓ | 同一 GPU、batch、precision、warm-up、计时策略（FPS ↑，latency ↓） |

**不要混淆**：全数据集混淆矩阵汇总的 mIoU 与逐图像 mIoU 的均值不同，论文和统计检验必须标明所用定义。对每个模型保存 TP/FP/FN/TN、预测概率和最终阈值，便于重算。

---

## E1. 主实验：公平基线对比【必须】

| ID | 方法 | Backbone / 架构 | 输入 | 阶段 |
|---|---|---|---|---|
| M1 | U-Net | 经典 U-Net | 全部 7 通道 | 首轮 |
| M2 | DeepLabV3+ | ResNet50 | 全部 7 通道 | 首轮 |
| M3 | SegFormer | MiT-B2 | 全部 7 通道 | 首轮 |
| M4 | **Single-stream ConvNeXt-Small + MLP Decoder** | ConvNeXt-Small | 全部 7 通道 | **首轮，直接对照** |
| M5 | **DualSegFormer-Small** | ConvNeXt-Small ×2 | 原始 3+4 通道 | **首轮，主方法** |
| M6 | OCRNet | HRNet | 全部 7 通道 | 第二轮可选 |
| M7 | UPerNet | ConvNeXt | 全部 7 通道 | 第二轮可选 |
| M8 | DualSegFormer-Tiny | ConvNeXt-Tiny ×2 | 原始 3+4 通道 | 容量/效率补充 |

**命名提醒**：M4 是 ConvNeXt 编码器配 All-MLP Decoder 的单流对照，不能把它写成原始 MiT 架构的官方 SegFormer。

主结果至少报告 mIoU、Foreground IoU、F1、Params、GFLOPs；**M1–M5 全部使用 `41/42/43` 三个 seed，并报告 mean±std**，测试集仅在冻结后评估。论文正文不把原 Workshop 的带 TTA 比赛分数与本地无 TTA 结果混到同一公平比较列。

**交付**：`Table 2. Comparison with segmentation baselines on converted MMLSv2`。

---

## E2. 单流/双流与模态贡献消融【必须】

| ID | 网络 | 输入 | 对照目的 | 复用 |
|---|---|---|---|---|
| A1 | 单流 ConvNeXt-Small + MLP Decoder | RGB + Grayscale（4） | 仅光学信息 | 新训练 |
| A2 | 单流 ConvNeXt-Small + MLP Decoder | Thermal + Slope + DEM（3） | 仅地形/热物理信息 | 新训练 |
| A3 | 单流 ConvNeXt-Small + MLP Decoder | 7 通道 | Early Fusion | **M4** |
| A4 | DualSegFormer-Small | 原始 3+4 | 原始双流 | **M5** |
| A5 | **近似容量对照：单流 ConvNeXt + MLP Decoder** | 7 通道 | 控制容量/参数量混杂 | 新训练 |

A5 的骨干型号按实际 Params 选择最接近 M5 的已有标准 ConvNeXt 规格，并在**不看 Test 结果**的情况下冻结（例如考虑 ConvNeXt-Base，但必须实测总参数量和 GFLOPs）。如无法严格匹配，应报告百分比偏差，并将比较称为“近似容量对照”，不写成“参数完全相同”。

**主要比较**：`A4 vs A3`（双流增益）、`A4 vs A5`（排除纯容量解释）、`A1/A2 vs A3/A4`（模态互补性）。

**交付**：`Table 3. Effect of dual-stream encoding and input modalities`。

---

## E3. 通道分组消融【必须】

**严格控制**：所有方案都使用同一 `DualSegFormer-Small`、相同融合和 Decoder，**Stream 1 固定 3 通道，Stream 2 固定 4 通道**。除首层输入映射外，不调整网络配置。

| ID | Stream 1：3 通道 | Stream 2：4 通道 | 理由 | 阶段 |
|---|---|---|---|---|
| B1 | Thermal + Slope + DEM | Grayscale + RGB | 原始物理分组；**复用 M5** | 必须 |
| B2 | RGB | Thermal + Slope + DEM + Grayscale | 将 RGB 单独成组；检验灰度归属 | 必须 |
| B3 | Thermal + DEM + Grayscale | Slope + RGB | 非原始混合分组 | 必须 |
| B4 | Slope + DEM + Grayscale | Thermal + RGB | 另一种严格 3+4 混合分组 | 可选 |

说明：旧版 B4（`Slope + Grayscale + RGB` vs `Thermal + DEM`）实际是 **5+2**，不能用于严格 3+4 对比；已删除。这里 B4 采用审查意见提供的 `Thermal + RGB` 与 `Slope + DEM + Grayscale` 组合，并将其排列为固定 **3+4** 的分支顺序。

- 比较 B1/B2/B3，观察 B1 是否在多个 seed 上稳定占优；B4 用于补充验证，不允许依据 Test 结果决定是否增添更多有利分组。
- 若 B1 不占优，论文主张应从“物理分组最优”收缩为“双流分组对输入配置敏感，原始分组具有实用性”。

**交付**：`Table 4. Ablation of channel grouping strategies`。

---

## E4. 融合与损失消融【第二轮，建议】

| ID | 只改变的变量 | 配置 | 复用 |
|---|---|---|---|
| C1 | Fusion | 四尺度逐元素 Add | **M5** |
| C2 | Fusion | Concat + 1×1 Conv，统一回原通道数 | 新训练 |
| L1 | Loss | CE only | 新训练 |
| L2 | Loss | Dice only | 新训练 |
| L3 | Loss | CE + Dice | **M5** |

- C1/C2 的参数量可能不同，必须报告额外 Params 和 GFLOPs。
- Loss 对照固定 Backbone、输入、融合与训练协议。
- 这些实验验证**原设计选择**，不意味着将 Concat 写成论文方法升级。

**交付**：`Table 5. Fusion and loss ablation`。

---

## E5. 受控扰动及模态缺失【第二轮，建议】

**先 Val 确定协议，再冻结，最后 Test 一次性评价。** 不从 Test 结果决定扰动类别、等级、随机种子、选择哪张图或采用哪个模型。

| 扰动 | 对象 | 验证目标 |
|---|---|---|
| DEM offset / Gaussian noise | DEM | 地形测量偏移与噪声敏感性 |
| 光照/对比度变化 | RGB + Grayscale | 光学信号变化敏感性 |
| 分通道 Gaussian noise | 预先指定通道集合 | 传感器噪声敏感性 |
| Optical modality missing | RGB + Grayscale | 光学模态依赖程度 |
| Physical modality missing | Thermal + Slope + DEM | 地形与热物理模态依赖程度 |

统一比较 **M4（单流 Small） vs M5（双流 Small）**，保持 clean/perturbed 模型权重不变。建议在 Val 确定三个固定扰动强度（轻/中/重），并以归一化尺度或物理单位清楚定义；如将缺失通道替换为训练均值，标准化后等价于替换为 0。记录扰动 RNG seed，并在 Test 评估前保存 `robustness_protocol.yaml`。

报告 clean 指标、各扰动等级指标、相对/绝对性能下降。此实验只能支持**受控扰动鲁棒性**，不能替代真实的跨区域测试。

**交付**：`Table 7. Robustness under controlled perturbations`、`Figure 3. Robustness curves`。

---

## E6. 重复实验与复杂度【必须】

### E6.1 统计可靠性

- **M1–M5 全部必须使用 `41/42/43` 三个随机种子**；A1/A2/A5 和 B1–B3 优先采用同样的三 seed 标准。资源不足时可以推迟非主实验，但不能把单次最优值冒充均值。
- 每次运行固定并记录训练种子、数据读取顺序、预训练权重、checkpoint 选择策略。
- 主要结果报告 mean±std；差值的区间可用按测试**图像**配对重采样的 bootstrap。应区分两类统计对象：跨 seed 的训练随机性、测试图像采样的置信区间，二者不可混为一谈。
- 即便某个 seed 表现不佳，也不得从报告中删除；不以 Test 选择“最优 seed”。

### E6.2 推理效率

| 指标 | 统一设置 |
|---|---|
| Params | 完整推理网络（编码器 + Fusion + Decoder） |
| GFLOPs | 同一 128×128 输入，双流分别按真实通道计入 |
| FPS / latency | 同一 GPU、batch、精度、warm-up 和测速次数 |
| Peak VRAM（可选） | 同一 batch size，报告测量范围 |

主表默认不含 TTA；如另作 TTA，单独报告预测质量和额外耗时。不同分辨率/硬件的速度不能直接比较。

**交付**：`Table 6. Model complexity and inference speed`、`Supplementary Table S1. Multi-seed results`。

---

## E7. 定性与误差分析【必须】

- **Validation 例图**：可以用于开发阶段分析和论文的解释性展示，但选择标准必须透明。
- **Test 例图**：在查看模型预测前，按预先确定的随机种子/样本 ID 哈希选择固定测试图像；不得事后只挑出 M5 优势最大的案例。
- 推荐对每张例图展示：`RGB | GT | Single-stream Small | DualSegFormer-Small | FP/FN error map`。
- 分析类型包括：明显滑坡主体、易误判背景、边界复杂、漏检、误检、两种模型都失败的情形。
- 建议使用测试集的总体 FP/FN 统计或按图像分布分位数支撑定性讨论，不应从几个案例推导总体性能。

**交付**：`Figure 2. Qualitative comparison`、`Figure 4. Failure analysis`。

---

## 2. 首轮最小实验矩阵（先验证关键结论）

> 原则：首轮**不做** OCRNet、UPerNet、大规模扰动、TTA/Ensemble 或额外结构升级；先确认 M4 vs M5 与容量/分组控制下的结论是否成立。

| 类别 | 必做配置 | 对应编号 |
|---|---|---|
| 主方法及直接对照 | 单流 Small；双流 Small | M4、M5 |
| 常见代表性基线 | U-Net、DeepLabV3+、SegFormer MiT-B2 | M1、M2、M3 |
| 模态贡献 | 仅光学单流；仅物理单流 | A1、A2 |
| 容量对照 | 近似容量对照的单流 | A5 |
| 物理分组 | 原始；RGB 独立；混合替代 | B1（复用 M5）、B2、B3 |
| Fusion | Add；Concat+1×1 Conv | C1（复用 M5）、C2 |
| 统计 | M1–M5 均使用 `41/42/43`；A5 和关键分组优先扩展 | E6 |
| 效率/可视化 | Params、GFLOPs、FPS；固定规则的案例图 | E6、E7 |

**训练去重**：`M4=A3`；`M5=A4=B1=C1=L3`。同配置、同 seed、同数据版本可以直接共享模型与预测。A1/A2、A5、B2/B3、C2 需要独立训练。

第一轮结束设立**分析关口**：

1. 若 M5 在 3 seeds 上对 M4 的优势不稳定，先排查数据、统计口径和训练差异，不要根据 Test 回头调模型。
2. 若 M5 不优于 A5，不能把性能提升单方面归因为模态解耦；如实讨论参数与容量影响。
3. 若 B1 不优于 B2/B3，修改**论文解释与结论范围**，而不是通过 Test 导向的分组搜索寻找支持原主张的结果。
4. 若核心证据不足，可以决定暂缓期刊投稿或另启**新的实验版本**；不得把同一 Test 的多轮反馈伪装成独立最终测试。

---

## 3. 推荐执行顺序

### Phase 0：E0 数据审计与协议冻结

1. 完成映射索引、单位、直方图、配对、场景独立性/近重复检查。
2. 生成 manifest、hash、转换脚本和 `experiment_protocol.yaml`。
3. 冻结主模型 Small、随机种子、训练/验证流程、指标、阈值策略和 Test 的使用规则。

### Phase 1：基础比较和 3-seed 稳定性

4. 训练 M1–M5（包括单流/双流 Small 及三种代表性基线），全部使用 `41/42/43` 三个 seeds。
5. 训练 A5（近似容量对照的单流模型），使用 `41/42/43` 三个 seeds。
6. 完成 M1/M2/M3 基线、A1/A2 模态基线；先在 Validation 确认实现正确。

### Phase 2：验证核心机制

7. 训练 B2/B3（必要时预先设定 B4），按规定重复。
8. 训练 C2，与 C1/原模型做单变量对照。
9. 核实双流、模态划分和参数效率方面的结论是否成立。

### Phase 3：冻结后最终测试与写作

10. 完成全部模型的最终 Test 评估；输出标准化预测、混淆矩阵和指标。
11. 执行已在 Val 冻结的扰动协议（若做 E5）。
12. 使用预注册固定 ID 选择 Test 可视化；完成效率测试和多 seed 汇总。
13. 根据实际证据撰写 Discussion，补充必要的 E4 Loss 消融或附录数据。

> **注意**：理想流程是先在 Validation 完成各配置、checkpoint、阈值和实验矩阵的冻结，然后一次性评价 Test。不能完成一次 Test 后再决定是否增加某个更可能获胜的分组或改主模型。

---

## 4. 期刊图表清单

| 编号 | 图表 | 优先级 |
|---|---|---|
| Table 1 | Converted MMLSv2：数据划分、各通道统计和映射协议摘要 | 必须 |
| Table 2 | M1–M5 主基线（含均值±标准差） | 必须 |
| Table 3 | A1–A5 单流/双流、模态贡献及容量控制 | 必须 |
| Table 4 | B1–B3/B4：严格 3+4 分组比较 | 必须 |
| Table 5 | Add vs Concat；CE/Dice Loss | 建议 |
| Table 6 | Params / GFLOPs / FPS | 必须 |
| Table 7 | 预设受控扰动及模态缺失 | 建议 |
| Figure 1 | 原始 DualSegFormer-Small 结构图（更新输入和维度标注） | 必须 |
| Figure 2 | 固定选择规则下的可视化比较 | 必须 |
| Figure 3 | 扰动等级—性能变化 | 建议 |
| Figure 4 | 错误统计和失败案例 | 建议 |
| Supplementary | 映射 manifest、详细实现、单 seed 结果 | 建议 |

---

## 5. 复现目录与记录建议

```text
experiments/
  protocol/experiment_protocol.yaml
  protocol/robustness_protocol.yaml
  dataset/channel_mapping.yaml
  dataset/dataset_manifest.csv
  dataset/dataset_audit.md
  configs/{model_name}.yaml
  runs/{model_name}/seed_{41,42,43}/
    config.yaml
    checkpoints/
    validation_metrics.json
    test_metrics.json
    confusion_matrix.json
    runtime.json
  summary/main_results.csv
  summary/ablation_results.csv
  summary/efficiency.csv
```

如用于公开复现，至少提供数据处理脚本、训练/评估脚本、主模型配置和运行命令。对于不能公开再分发的数据，提供合法下载途径和确定性转换步骤。

## 6. 投稿前检查清单

- [ ] 已明确主模型 Small 的版本及其与原 Workshop Tiny/Small 结果的关系。
- [ ] 已完成 E0：通道、单位、归一化、映射、哈希、场景/相邻切片泄漏检查。
- [ ] 标注来源、数据划分、Test 使用边界已完整披露。
- [ ] 所有主基线在相同评估协议下比较；参数近似匹配对照已报告差异。
- [ ] B1–B3（以及可选 B4）都满足固定 3+4 的分支输入宽度。
- [ ] M1–M5 均使用 `41/42/43` 三个 seeds，报告 mean±std，不依据 Test 选 seed。
- [ ] 扰动类型和强度在 Validation 确定，最终 Test 不参与修改方案。
- [ ] Test 定性样本选择不依据模型表现。
- [ ] 不将 TTA、Ensemble、测试集校正和模型主体效果混为一谈。
- [ ] 物理分组、效率和鲁棒性方面的论文表述有对应的实证支持。
- [ ] 不把 converted MMLSv2 说成独立新数据集，不在缺乏证据时声称真实跨区域泛化。

**最后原则**：上述计划提升的是论文方法论质量与可重复性，并不保证 *Remote Sensing* 接收。是否能够围绕不升级的 DualSegFormer 形成足够的期刊贡献，取决于上述对照实验实际揭示了什么。
