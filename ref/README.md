# EP 参考论文

当前主方案是[JS阈值阶段数与动态学生退出](../docs/loop_distillation_research_proposal_zh.md)。本表保留已有背景与近邻，价值选择、概率修正等不自动等于当前主损失；新增引用见[计划引用索引](../experiments/proposal_references_20261113.json)，仅登记的引用不计作已下载全文。

六篇[核心论文](../docs/03_paper_notes.md)存放于 papers；本目录另有十七篇参考论文。固定版本、来源、哈希、阅读范围与证据定位见 [manifest.json](manifest.json)，原文快照在 [sources/](sources/)。

其中 **Cross-Loop Query Handoff 已选为 EP 的主要动机证据论文**，用于支持“部分中间循环更新有任务价值”。其存储位置仍在本目录；详细干预结果及与 LoopOPD 改进的关系见[证据说明](../docs/06_loopopd_motivation_evidence.md)。

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

正文区别见[相关工作](../docs/05_related_work.md)。这些论文提供背景和对照，不代表全部方法已独立实现或复现。部分条目只核对摘要和特定方法段落，具体范围以清单为准。

Huginn 原生代码接入和当前计划逐步对应见[实现文档](../docs/08_implementation_and_reproduction.md)。`huginn-0125` 是本项目选用的官方基模；Best-of-Depth 没有公开足以确认相同 checkpoint 的加载配置，不能由参数量相近直接认定。

当前默认方法已加入学生继续／停止学习；旧固定预算Top-S为F对照。阶段数标签S_t与实测学生轮数分别报告，不能由标签均值推出速度。
