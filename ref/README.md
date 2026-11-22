# EP 参考论文

正文、主线与动态分支见[研究方案](../docs/loop_distillation_research_proposal_zh.md)。

核心与相关论文的固定版本、来源、哈希、阅读范围与证据定位见[核心清单](../papers/manifest.json)和[参考清单](manifest.json)。本目录的原文快照见 [sources/](sources/)。

Cross-Loop Query Handoff 的查询冻结和移植实验是阶段蒸馏的动机证据之一。

## 监督价值与候选修正

| 工作 | 本地 PDF | 解决什么问题 | 对 EP 的意义 |
|---|---|---|---|
| AOPD | [PDF](2610.31563v1_aopd.pdf) | 教师动作能否提高学生续写的成功率 | §3；输出动作价值的直接近邻 |
| Reward-Aligned Distillation | [PDF](2610.14572v1_rad.pdf) | 蒸馏更新能否提高训练后奖励 | §4；收益选择已有方法，固定映射 + RAD 是重要对照 |
| Token Teachability / TA-OPD | [PDF](2605.26844v1_token_teachability.pdf) | 大分歧中哪些信号可学 | 固定上下文和支持集相容性；不是终局价值真值 |
| TIP | [PDF](2604.14084v4_tip.pdf) | 哪些生成位置值得接受监督 | 熵与师生分歧的位置选择基线 |
| Demystifying OPD | [PDF](2610.19145v1_teacher_advantage.pdf) | 教师纠正能恢复多少学生错误 | token flip 和学生续写诊断原型 |
| RouteOPD | [PDF](2609.08337v1_routeopd.pdf) | 概率应从哪些候选移向哪些候选 | 概率修正并非新主题；需说明循环轴的新增问题 |

## 自动层次匹配与计算压缩

| 工作 | 本地 PDF | 解决什么问题 | 对 EP 的意义 |
|---|---|---|---|
| BERT-EMD | [PDF](2010.06133v1_bert_emd.pdf) | 深浅网络如何多对多匹配 | 已有自适应层映射与权重 |
| ALP-KD | [PDF](2012.14022v1_alp_kd.pdf) | 固定跳层会漏掉教师信息 | attention 融合教师层的近邻 |
| RAIL-KD | [PDF](2109.10164v2_rail_kd.pdf) | 层匹配工程复杂、计算昂贵 | 随机映射是必要对照 |
| Revisiting Intermediate-Layer Matching | [PDF](2502.04499v2_revisit_layer_matching.pdf) | 层选择策略是否真的重要 | 不预设精巧映射能带来独立收益 |
| 扩散语言 OPTD | [PDF](2608.02942v1_diffusion_optd.pdf) | 多步压缩怎样保持中间依赖 | consistency-guided adaptive compression 的强近邻 |

## 原生循环结构与计算功能

| 工作 | 本地 PDF | 解决什么问题 | 对 EP 的意义 |
|---|---|---|---|
| Huginn 原论文 | [PDF](2502.05171v2_huginn.pdf) | 通过 recurrent core 扩大测试时内部计算 | 架构、平均训练 recurrence、初态与 coda 接口 |
| Shared Weights, Selected Computations | [PDF](2609.39892v1_selected_computations.pdf) | 同权循环如何选择下一轮执行的操作 | 状态选择计算；不能把一轮等同固定推理步骤 |
| Cross-Loop Query Handoff | [PDF](2610.34639v1_query_handoff.pdf) | 后轮怎样纠正两跳检索 | 主要动机证据；Ouro query 冻结、移植与 donor 控制；未测试 LoopOPD |
| Jacobian Lens | [PDF](2609.01924v1_jacobian_lens.pdf) | 信息怎样跨轮维持、被读取和写入 | 不同模型干预窗口与状态坐标不同 |
| LatentHarness | [PDF](2609.39740v1_latent_harness.pdf) | 何时 Think、Recall、Exit | 已有循环状态上的单步反事实动作蒸馏 |
| FlexLoop | [PDF](2609.34488v1_flexloop.pdf) | 保留深层策略能力并改善浅层执行 | 相邻深度蒸馏与自适应停止；主要是控制任务 |

部分条目只核对了摘要和特定方法段落，阅读深度以清单为准。
