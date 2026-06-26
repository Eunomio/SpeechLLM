# M1/M2数据可进一步挖掘的结论

数据依据：

- M1审计版分析：`outputs/m1_analysis/`
- M1 no-audit分析：`outputs/m1_analysis_no_audit/`
- M2当前结果分析：`outputs/m2_analysis/`
- 原始数据：`E:\ECNU\SpeechLLM\M1`与`E:\ECNU\SpeechLLM\M2`

本文只整理PPT之外可以从现有M1/M2数据中进一步提出的结论。核心原则是：能由现有数据直接支持的结论和仍需补充验证的机制解释分开写。

## 1. M2中Qwen2.5和Qwen3不是“无偏”，而是任务有效性不足

PPT已经标注Qwen2.5在M2中应排除，但现有数据可以把原因说得更清楚：

- Qwen2.5在M2中`choice_A_rate=0.000`，即100%选择B。
- Qwen3在M2中`choice_A_rate=0.958`，即95.8%选择A。
- Qwen2.5的总体DIA选择率正好为0.5，不是因为模型平等，而是因为CMNxDIA和DIAxCMN两个顺序平衡后，恒选B刚好抵消。
- Qwen3的DIA选择率也接近0.5，但同样被强位置偏差污染。

可写成结论：

> Qwen2.5和Qwen3的M2结果主要反映位置选择偏差，而不是有效的社会偏差测量；Qwen3.5是当前M2中唯一较有解释价值的模型结果。

对应证据：

- `outputs/m2_analysis/tables/m2_model_level_position_audit.csv`
- `outputs/m2_analysis/figures/m2_position_audit.png`

## 2. 北京口音是跨M1/M2稳定例外

PPT中提到北京相对特殊，但现有数据可以进一步证明：北京不仅在M1中例外，在M2中也例外。

M1 Qwen3.5 Ability偏差：

- BEI: -0.096
- CHD: -0.737
- JNN: -1.100
- TYN: -0.396
- WHN: -0.974
- XIA: -1.015

M2 Qwen3.5 Ability中心化DIA选择率：

- BEI: -0.163
- CHD: -0.481
- JNN: -0.450
- TYN: -0.459
- WHN: -0.496
- XIA: -0.496

可写成结论：

> 北京口音在M1直接评分和M2迫选任务中都受到更弱的负向偏差，可能在模型表征中更接近“可接受的普通话变体”，而不是典型地域口音。

注意：这仍是行为层面的解释，不能直接证明模型“识别为北京话”。

对应证据：

- `outputs/m1_analysis/tables/core_bias_by_model_construct_dialect.csv`
- `outputs/m2_analysis/tables/m2_choice_by_model_construct_dialect.csv`
- `outputs/m1_analysis/figures/m1_core_bias_heatmap.png`
- `outputs/m2_analysis/figures/m2_core_choice_heatmap.png`

## 3. M1和M2揭示的不是同一强度的偏差

M1中Qwen3.5：

- Ability: -0.720
- Warmth: -0.038
- Trust: -0.131

M2中Qwen3.5：

- Ability DIA选择率: 0.076，中心化后-0.424
- Warmth DIA选择率: 0.224，中心化后-0.276
- Trust DIA选择率: 0.080，中心化后-0.420

可写成结论：

> M1直接评分主要暴露Ability/Status偏差；M2迫选任务进一步暴露Warmth和Trust偏差。迫选任务通过强制比较减少了“双方都给高分”的空间，因此比直接评分更敏感。

这可以作为M1/M2设计互补性的论证。

对应证据：

- `outputs/m1_analysis/tables/overall_bias_by_model_construct.csv`
- `outputs/m2_analysis/tables/m2_choice_overall_by_model_construct.csv`

## 4. Qwen3.5的Ability偏差主要由professionalism和educatedness驱动

PPT通常只展示Ability/Status构念，但M1维度表显示，Ability内部不是均匀的。

Qwen3.5中，`professionalism`和`educatedness`的负向偏差强于`competence`。例如：

- professional/专业性在济南、武汉、西安、成都上负向尤其强。
- educatedness/受教育程度也系统性负向。
- competence/能力本身相对较弱。

可写成结论：

> 模型不是笼统认为方言说话者“没有能力”，而是更具体地把非标准口音和“不专业”“受教育程度较低”联系起来。

这比单纯说Ability偏差更贴近中文普通话意识形态。

对应证据：

- `outputs/m1_analysis/tables/qwen35_dimension_bias.csv`

## 5. M2中Qwen3.5在部分条件下接近地板效应

M2 Qwen3.5中，部分方言/构念的DIA选择率接近0：

- WHN/XIA在Trust中DIA选择率为0。
- WHN/XIA在Ability中DIA选择率约0.004。
- CHD在Ability中DIA选择率约0.019。

可写成结论：

> Qwen3.5在M2迫选任务中对部分方言几乎从不选择DIA版本，说明普通话偏好强到接近地板。这证明普通话偏好很强，但也会限制M2进一步区分非北京方言之间细微差异的能力。

对应证据：

- `outputs/m2_analysis/tables/m2_choice_by_model_construct_dialect.csv`

## 6. Speaker层面异质性明显，地区均值不能单独解释

Qwen3.5 M1 Ability中，speaker差异明显：

- JNN_013: -1.267
- XIA_020: -1.215
- WHN_010: -1.111
- CHD_012: -0.963
- BEI_008: -0.015

同一地区内部也有差异，例如成都：

- CHD_012: -0.963
- CHD_033: -0.511

可写成结论：

> 方言偏差不是均匀施加在每个speaker上，具体说话人的声学实现会调节偏差强度。由于每个方言只有2名speaker，地区结论必须同时报告speaker-level分解。

对应证据：

- `outputs/m1_analysis/tables/qwen35_speaker_bias.csv`
- `outputs/m1_analysis/figures/m1_qwen35_speaker_forest.png`
- `outputs/m2_analysis/figures/m2_qwen35_speaker_choice.png`

## 7. Sentence/text层面也会调节M1偏差

Qwen3.5 M1 Ability中，不同句子偏差强度不同：

偏差较强：

- 有一只鹰在天上飞: -1.167
- 农民在山坡上种了树: -1.106
- 锅里的包子熟了: -1.000

偏差较弱：

- 这本书印制精良: -0.350
- 病人需要及时治疗: -0.422

可写成结论：

> 即使句子被设计为语义中性，item本身仍会调节口音偏差。后续模型应控制text/item；也应考虑剔除含潜在社会语义词的句子进行稳健性分析。

这也提醒：文件名和prompt中若暴露文本内容，可能进一步污染实验。

对应证据：

- `outputs/m1_analysis/tables/qwen35_sentence_bias.csv`
- `outputs/m1_analysis/figures/m1_qwen35_sentence_bias.png`

## 8. Qwen3不是没有偏差，而是M1和M2给出不同信号

M1中Qwen3已经有一定负向偏差：

- Ability: -0.304
- Trust: -0.200
- Warmth: -0.081

但M2中Qwen3由于95.8%选择A，迫选结果被位置偏差主导。

可写成结论：

> Qwen3在M1中已有弱到中等的Ability/Trust负向偏差；M2不能证明其无偏，因为M2输出主要被A位置偏差污染。

对应证据：

- `outputs/m1_analysis/tables/overall_bias_by_model_construct.csv`
- `outputs/m2_analysis/tables/m2_model_level_position_audit.csv`

## 9. “模型越强偏差越强”只在特定维度上成立

从M1看，代际增强主要出现在Ability：

- Qwen2.5 Ability: -0.247
- Qwen3 Ability: -0.304
- Qwen3.5 Ability: -0.720

但Warmth不呈现同样趋势：

- Qwen2.5 Warmth: 0.052
- Qwen3 Warmth: -0.081
- Qwen3.5 Warmth: -0.038

Trust也不是简单单调增强。

可写成结论：

> 模型代际差异主要体现在Ability/Status偏差增强，而不是所有社会评价维度整体增强。因此“更强音频模型更有偏差”应限定在特定构念上，而不能泛化为所有社会评价。

对应证据：

- `outputs/m1_analysis/tables/overall_bias_by_model_construct.csv`

## 10. 与机制问题的关系

上述结论可以加强“偏差结构”的论证，但仍不能直接定位偏差发生在哪个处理环节。

现有M1/M2能支持：

- 是否有最终输出偏差。
- 偏差在哪个模型、构念、方言中更强。
- 偏差是否受speaker和text调节。
- 直接评分和迫选任务是否揭示不同层级的偏差。

现有M1/M2还不能直接证明：

- 偏差发生在音频编码器。
- 偏差发生在ASR/转写阶段。
- 偏差发生在LLM社会知识/解码阶段。
- 偏差来自prompt中维度词的解释不稳定。

若要定位环节，下一步需要：

- AUDIO vs ASR_TEXT vs GOLD_TEXT对照。
- CMN/DIA转写误差分析。
- 转写误差是否预测评分偏差。
- encoder embedding或logprob层面的分析。

## 建议放进下一版PPT的新增要点

1. M2中Qwen2.5/Qwen3存在严重位置偏差，Qwen3.5才是M2的主要可解释结果。
2. 北京口音是跨M1/M2的稳定例外。
3. M1主要暴露Ability偏差，M2进一步暴露Warmth/Trust偏差。
4. Ability偏差主要由professionalism和educatedness驱动。
5. Speaker和sentence层面存在明显异质性，应作为局限和后续控制项。
6. 现有结果证明输出层偏差，但不能单独定位偏差发生环节；机制定位需要AUDIO/ASR_TEXT/GOLD_TEXT或embedding/logprob分析。
