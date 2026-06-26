# M2现有结果报告

数据来源：`E:\ECNU\SpeechLLM\M2`  
核心指标：`DIA choice probability - 0.5`。负值表示模型在迫选中更偏向普通话版本。

## 1. 数据质量与任务有效性

M2每个run理论完整trial数同样为`1512`。每个模型的run1文件包含5行缺API key失败/重试残留，但均有1512个成功trial；run2-5均为1512个成功trial。

### 位置偏差和DIA选择率

| model_name        | ok_rows | choice_A_rate | choice_B_rate | dia_choice_rate | cmnxdia_dia_rate | diaxcmn_dia_rate | mean_confidence |
| ----------------- | ------- | ------------- | ------------- | --------------- | ---------------- | ---------------- | --------------- |
| Qwen2.5-Omni-7B   | 7560    | 0.000         | 1.000         | 0.500           | 1.000            | 0.000            | 2.968           |
| Qwen3-Omni-Flash  | 7560    | 0.958         | 0.042         | 0.484           | 0.026            | 0.943            | 2.718           |
| Qwen3.5-Omni-Plus | 7560    | 0.568         | 0.432         | 0.140           | 0.072            | 0.208            | 3.183           |

![M2 position audit](D:/Users/Documents/SpeechLLM/outputs/m2_analysis/figures/m2_position_audit.png)

解释：

- Qwen2.5在M2中完全恒选`B`，因此总体DIA选择率正好为0.5，但这是位置偏差造成的，不是没有口音偏差。该模型的M2结果不应作为社会偏差证据。
- Qwen3极强偏向`A`位置，`choice_A_rate=0.958`；由于两个order平衡，DIA总体选择率仍接近0.5。这说明Qwen3的M2结果也主要受位置偏差污染。
- Qwen3.5的位置偏差较小于前两者，但DIA选择率只有`0.140`，说明其在迫选任务中强烈偏向普通话版本。

## 2. M2核心结果

### 总体构念DIA选择率

| model_name        | construct | dia_rate_ci          | centered | n     |
| ----------------- | --------- | -------------------- | -------- | ----- |
| Qwen2.5-Omni-7B   | Ability   | 0.500 [0.433, 0.567] | 0.000    | 216.0 |
| Qwen2.5-Omni-7B   | Trust     | 0.500 [0.433, 0.567] | 0.000    | 216.0 |
| Qwen2.5-Omni-7B   | Warmth    | 0.500 [0.433, 0.567] | 0.000    | 216.0 |
| Qwen3-Omni-Flash  | Ability   | 0.462 [0.400, 0.524] | -0.038   | 216.0 |
| Qwen3-Omni-Flash  | Trust     | 0.500 [0.433, 0.567] | 0.000    | 216.0 |
| Qwen3-Omni-Flash  | Warmth    | 0.502 [0.436, 0.567] | 0.002    | 216.0 |
| Qwen3.5-Omni-Plus | Ability   | 0.076 [0.047, 0.104] | -0.424   | 216.0 |
| Qwen3.5-Omni-Plus | Trust     | 0.080 [0.048, 0.112] | -0.420   | 216.0 |
| Qwen3.5-Omni-Plus | Warmth    | 0.224 [0.185, 0.264] | -0.276   | 216.0 |

### 按模型、构念、方言的中心化DIA选择率

| model_name        | construct | BEI    | CHD    | JNN    | TYN    | WHN    | XIA    |
| ----------------- | --------- | ------ | ------ | ------ | ------ | ------ | ------ |
| Qwen2.5-Omni-7B   | Ability   | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  |
| Qwen2.5-Omni-7B   | Trust     | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  |
| Qwen2.5-Omni-7B   | Warmth    | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  |
| Qwen3-Omni-Flash  | Ability   | 0.000  | -0.044 | -0.022 | -0.022 | -0.083 | -0.056 |
| Qwen3-Omni-Flash  | Trust     | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  | 0.000  |
| Qwen3-Omni-Flash  | Warmth    | 0.007  | 0.013  | 0.019  | -0.050 | -0.007 | 0.028  |
| Qwen3.5-Omni-Plus | Ability   | -0.163 | -0.481 | -0.450 | -0.459 | -0.496 | -0.496 |
| Qwen3.5-Omni-Plus | Trust     | -0.172 | -0.472 | -0.411 | -0.467 | -0.500 | -0.500 |
| Qwen3.5-Omni-Plus | Warmth    | -0.069 | -0.252 | -0.217 | -0.361 | -0.348 | -0.407 |

![M2 core heatmap](D:/Users/Documents/SpeechLLM/outputs/m2_analysis/figures/m2_core_choice_heatmap.png)

主要结果：

- Qwen3.5在Ability、Warmth、Trust三个构念上都明显低于0.5，说明迫选时更常选择普通话版本。
- Qwen3.5的Ability和Trust最强，DIA选择率约为0.076和0.080；Warmth也明显低于0.5，约为0.224。
- 北京在Qwen3.5中仍是相对例外：虽然也偏向普通话，但负向程度弱于成都、济南、武汉、西安等方言。
- Qwen2.5和Qwen3的M2解释必须以位置偏差审计为前提；不能只看DIA选择率。

## 3. Qwen3.5 speaker层面

![Qwen3.5 speaker-level M2 choice](D:/Users/Documents/SpeechLLM/outputs/m2_analysis/figures/m2_qwen35_speaker_choice.png)

解释：

- Qwen3.5的M2普通话偏好不是由单个speaker驱动，几乎所有speaker在Ability和Trust上都低于0。
- Warmth也整体偏负，但北京speaker的负向程度较弱，和PPT中“北京是例外”的描述一致。

## 4. 和PPT结论的对应

PPT中对M2的表述基本成立，但需要更精确：

1. Qwen2.5应明确标记为任务失败/位置偏差：它恒选B，不是有效社会判断。
2. Qwen3也有严重A位置偏差，因此其“无稳定方言差异”应解释为位置偏差主导下的弱证据，而不是可靠的无偏结果。
3. Qwen3.5是目前M2中唯一较有解释价值的结果：它在三类构念上都强烈偏普通话，且Ability/Trust最强，Warmth次之。

## 5. 输出文件

- 清理后的dimension单位：`outputs/m2_analysis/tables/m2_dimension_units_clean.csv`
- 清理后的construct单位：`outputs/m2_analysis/tables/m2_construct_units_clean.csv`
- 模型位置偏差审计：`outputs/m2_analysis/tables/m2_model_level_position_audit.csv`
- 核心结果表：`outputs/m2_analysis/tables/m2_choice_by_model_construct_dialect.csv`
- 图表目录：`outputs/m2_analysis/figures/`
