# M1 no-audit分析报告

数据来源：`E:\ECNU\SpeechLLM\M1`  
分析口径：不做run审计折叠；qwen2.5的5个run按独立run保留；qwen3/qwen3.5不跨run取均值。  
核心偏差定义：`Bias = DIA评分 - CMN评分`。

## 1. 口径说明

这个版本是为了和PPT/原始跑法保持一致的“无审计版”。它会低估重复run不独立带来的不确定性，尤其是qwen2.5，因为qwen2.5五个run完全相同。更严谨的推断仍应优先使用`outputs/m1_analysis`中的审计版。

## 2. 总体构念偏差

| model_name        | construct | bias_ci                 | n     |
| ----------------- | --------- | ----------------------- | ----- |
| Qwen2.5-Omni-7B   | Ability   | -0.247 [-0.303, -0.191] | 540.0 |
| Qwen2.5-Omni-7B   | Trust     | 0.074 [0.024, 0.124]    | 540.0 |
| Qwen2.5-Omni-7B   | Warmth    | 0.052 [0.010, 0.095]    | 540.0 |
| Qwen3-Omni-Flash  | Ability   | -0.304 [-0.388, -0.221] | 540.0 |
| Qwen3-Omni-Flash  | Trust     | -0.200 [-0.264, -0.136] | 540.0 |
| Qwen3-Omni-Flash  | Warmth    | -0.081 [-0.135, -0.026] | 540.0 |
| Qwen3.5-Omni-Plus | Ability   | -0.720 [-0.784, -0.655] | 540.0 |
| Qwen3.5-Omni-Plus | Trust     | -0.131 [-0.165, -0.098] | 540.0 |
| Qwen3.5-Omni-Plus | Warmth    | -0.038 [-0.078, 0.002]  | 540.0 |

## 3. 按模型、构念、方言的平均偏差

| model_name        | construct | BEI    | CHD    | JNN    | TYN    | WHN    | XIA    |
| ----------------- | --------- | ------ | ------ | ------ | ------ | ------ | ------ |
| Qwen2.5-Omni-7B   | Ability   | 0.167  | -0.130 | -0.259 | -0.556 | -0.296 | -0.407 |
| Qwen2.5-Omni-7B   | Trust     | 0.333  | 0.111  | -0.056 | -0.278 | 0.056  | 0.278  |
| Qwen2.5-Omni-7B   | Warmth    | 0.204  | 0.019  | 0.111  | -0.185 | -0.056 | 0.222  |
| Qwen3-Omni-Flash  | Ability   | -0.248 | -0.059 | -0.111 | -0.526 | -0.230 | -0.652 |
| Qwen3-Omni-Flash  | Trust     | -0.156 | -0.133 | -0.211 | -0.044 | -0.167 | -0.489 |
| Qwen3-Omni-Flash  | Warmth    | -0.144 | 0.104  | 0.159  | -0.352 | 0.052  | -0.304 |
| Qwen3.5-Omni-Plus | Ability   | -0.096 | -0.737 | -1.100 | -0.396 | -0.974 | -1.015 |
| Qwen3.5-Omni-Plus | Trust     | -0.044 | -0.067 | -0.078 | -0.033 | -0.211 | -0.356 |
| Qwen3.5-Omni-Plus | Warmth    | -0.000 | 0.030  | -0.078 | -0.074 | 0.048  | -0.156 |

![no-audit core heatmap](D:/Users/Documents/SpeechLLM/outputs/m1_analysis_no_audit/figures/m1_no_audit_core_bias_heatmap.png)

解释：

- no-audit版的均值模式和审计版一致：最强偏差仍集中在Qwen3.5-Omni-Plus的`Ability`构念。
- Qwen3.5的`Ability`总体偏差为`-0.720`，表示地域口音版本相对同speaker普通话版本平均低约0.72分。
- 方言分布上，北京仍是稳定例外；Qwen3.5中北京`Ability`偏差约`-0.096`，而济南、西安、武汉分别约为`-1.100`、`-1.015`、`-0.974`。
- `Warmth`在Qwen3.5中接近0，说明M1直接评分里模型不是把所有社会维度都统一打低，而是更集中地把非标准口音和较低Ability/Status联系起来。
- `Trust`有小幅负向，尤其西安、武汉更明显，但强度远低于Ability。

需要注意：由于此版本把重复run都当作独立观测，置信区间比审计版更窄，显著性外观更强；这不代表证据真的增加了同等倍数。

## 4. 原始CMN/DIA评分

| model_name        | construct | guise | rating_ci            | n     |
| ----------------- | --------- | ----- | -------------------- | ----- |
| Qwen2.5-Omni-7B   | Ability   | CMN   | 5.238 [5.196, 5.279] | 540.0 |
| Qwen2.5-Omni-7B   | Ability   | DIA   | 4.991 [4.937, 5.044] | 540.0 |
| Qwen2.5-Omni-7B   | Trust     | CMN   | 5.731 [5.692, 5.771] | 540.0 |
| Qwen2.5-Omni-7B   | Trust     | DIA   | 5.806 [5.767, 5.845] | 540.0 |
| Qwen2.5-Omni-7B   | Warmth    | CMN   | 5.802 [5.769, 5.836] | 540.0 |
| Qwen2.5-Omni-7B   | Warmth    | DIA   | 5.855 [5.826, 5.884] | 540.0 |
| Qwen3-Omni-Flash  | Ability   | CMN   | 4.913 [4.824, 5.002] | 540.0 |
| Qwen3-Omni-Flash  | Ability   | DIA   | 4.609 [4.518, 4.699] | 540.0 |
| Qwen3-Omni-Flash  | Trust     | CMN   | 5.369 [5.310, 5.427] | 540.0 |
| Qwen3-Omni-Flash  | Trust     | DIA   | 5.169 [5.113, 5.224] | 540.0 |
| Qwen3-Omni-Flash  | Warmth    | CMN   | 5.062 [5.019, 5.106] | 540.0 |
| Qwen3-Omni-Flash  | Warmth    | DIA   | 4.981 [4.934, 5.029] | 540.0 |
| Qwen3.5-Omni-Plus | Ability   | CMN   | 5.397 [5.334, 5.460] | 540.0 |
| Qwen3.5-Omni-Plus | Ability   | DIA   | 4.677 [4.598, 4.757] | 540.0 |
| Qwen3.5-Omni-Plus | Trust     | CMN   | 5.991 [5.983, 5.999] | 540.0 |
| Qwen3.5-Omni-Plus | Trust     | DIA   | 5.859 [5.827, 5.891] | 540.0 |
| Qwen3.5-Omni-Plus | Warmth    | CMN   | 5.196 [5.158, 5.235] | 540.0 |
| Qwen3.5-Omni-Plus | Warmth    | DIA   | 5.158 [5.120, 5.196] | 540.0 |

![no-audit raw ratings](D:/Users/Documents/SpeechLLM/outputs/m1_analysis_no_audit/figures/m1_no_audit_raw_cmn_vs_dia.png)

解释：

- 这张图不是按具体方言分开，而是把6个地区合在一起，比较普通话版`CMN`和地域口音版`DIA`的原始评分。
- Qwen3.5的Ability差异来自`CMN`评分更高、`DIA`评分更低；也就是说，差值不是统计构造出来的，而是在原始评分层面可见。
- Qwen3.5的Warmth中`CMN`和`DIA`几乎重叠，因此Warmth没有明显口音效应。
- Qwen3.5的Trust整体分数较高，差异较小；这可能存在天花板效应，即模型普遍给可信度较高分，导致可观察差异空间受限。

## 5. Qwen3.5 speaker层面

![no-audit speaker forest](D:/Users/Documents/SpeechLLM/outputs/m1_analysis_no_audit/figures/m1_no_audit_qwen35_speaker_forest.png)

Qwen3.5的Ability偏差最强speaker：

| speaker_id | dialect | bias_ci                 | n    |
| ---------- | ------- | ----------------------- | ---- |
| JNN_013    | JNN     | -1.267 [-1.486, -1.047] | 45.0 |
| XIA_020    | XIA     | -1.215 [-1.464, -0.966] | 45.0 |
| WHN_010    | WHN     | -1.111 [-1.281, -0.942] | 45.0 |
| CHD_012    | CHD     | -0.963 [-1.107, -0.819] | 45.0 |
| JNN_019    | JNN     | -0.933 [-1.121, -0.745] | 45.0 |
| WHN_003    | WHN     | -0.837 [-1.025, -0.649] | 45.0 |

解释：

- speaker层面显示，Qwen3.5的Ability偏差不是所有speaker完全同等强度。
- `JNN_013`、`XIA_020`、`WHN_010`等speaker驱动了较强的负向Ability偏差。
- 北京两个speaker都接近0，说明“北京例外”不是单个speaker偶然造成的。
- 因此，报告地区均值时应同时展示speaker层面分解，避免把个别speaker的强效应误读成整个地区的均匀效应。

## 6. Qwen3.5 sentence层面补充

![no-audit sentence bias](D:/Users/Documents/SpeechLLM/outputs/m1_analysis_no_audit/figures/m1_no_audit_qwen35_sentence_bias.png)

Qwen3.5的Ability偏差最强句子：

| text      | bias_ci                 | n    |
| --------- | ----------------------- | ---- |
| 有一只鹰在天上飞  | -1.167 [-1.355, -0.978] | 60.0 |
| 农民在山坡上种了树 | -1.106 [-1.344, -0.867] | 60.0 |
| 锅里的包子熟了   | -1.000 [-1.176, -0.824] | 60.0 |
| 面条在锅里煮着   | -0.756 [-0.918, -0.593] | 60.0 |
| 儿童需要悉心抚养  | -0.628 [-0.752, -0.504] | 60.0 |

解释：

- 句子之间的偏差强度不同，说明文本内容、发音难度、切分质量或具体声学实现可能影响模型评分。
- 例如`有一只鹰在天上飞`、`农民在山坡上种了树`、`锅里的包子熟了`在Ability上偏差更强。
- 后续正式建模仍应控制`text`，或者做剔除潜在社会语义词句后的稳健性分析。

## 7. 混合效应模型

    no-audit版保留每个run作为观测，但混合效应模型结构与审计版保持一致：

`bias ~ model * construct * dialect + (1 | speaker) + (1 | text)`

也就是说，这里只改变数据口径，不改变模型公式。这样no-audit版和审计版的差异主要来自是否把重复run保留为观测，而不是来自模型结构变化。

模型收敛状态：

| model_spec           | nobs | aic     | bic     | converged_text |
| -------------------- | ---- | ------- | ------- | -------------- |
| full                 | 4860 | 8262.02 | 8631.88 | yes            |
| no_three_way         | 4860 | 8354.72 | 8594.80 | yes            |
| no_model_construct   | 4860 | 8504.32 | 8718.45 | yes            |
| no_model_dialect     | 4860 | 8569.22 | 8744.41 | yes            |
| no_construct_dialect | 4860 | 8448.20 | 8623.40 | no             |
| main_effects         | 4860 | 8771.49 | 8855.84 | no             |

似然比检验：

| comparison                   | lr_chisq | df | p     |
| ---------------------------- | -------- | -- | ----- |
| full vs no_three_way         | 132.70   | 20 | <.001 |
| full vs no_model_construct   | 290.29   | 24 | <.001 |
| full vs no_model_dialect     | 367.19   | 30 | <.001 |
| full vs no_construct_dialect | 246.18   | 30 | <.001 |
| full vs main_effects         | 597.46   | 44 | <.001 |

解释：

- 该模型用于检验在控制speaker和text重复结构后，`model × construct × dialect`结构是否仍能解释偏差。
- 如果完整三重交互模型未收敛，涉及完整模型的LRT应视为探索性，而不是最终显著性证据。
- no-audit模型的主要用途是和no-audit描述性结果配套；正式推断仍建议以审计版和稳健性分析为主。

## 8. 与审计版的关系

均值模式与审计版基本一致，因为两者的核心差别不在均值计算，而在是否把重复run当作独立证据。主要变化是：

- no-audit版每个模型-构念的`n=540`，审计版为`n=108`。
- no-audit版置信区间明显变窄。
- qwen2.5受影响最大，因为它5个run完全重复；把它们当作独立run会人为放大证据强度。
- 因此，no-audit版适合复现PPT展示口径；如果要写论文/报告中的推断性结论，建议优先采用审计版。
