# 核心论文阅读笔记与主要动机证据

六篇核心论文的版本、哈希与原文快照见[清单](../papers/manifest.json)；第 7 节的主要动机论文收录于[参考清单](../ref/manifest.json)。以下结论来自原作者，未独立复现。下载全文不等于逐字精读；阅读范围单列。

2026-11-08 已另做 Ouro-Thinking 上的小规模干预适配与短训，见[本地结果](07_pilot_results_20261108.md)。它与下文原论文的模型、数据和协议不同，结果分别报告。

## 1. LoopOPD / D-LoopOPD：idea 来源、基础方法与首要基线

- [PDF](../papers/2610.10623v2_loopopd.pdf)，[官方版本](https://arxiv.org/abs/2610.10623v2)。
- 完整标题：Recurrent Self-Improvement: Dynamic Cross-Loop On-Policy Distillation for Looped Language Models。
- 核对：§4.1–4.2、§5.3–5.4、附录 C–D。

**问题。**循环模型后训练需要密集监督，同时面临奖励稀疏、内存与更强教师不足。

**方案。**两轮学生生成前缀，冻结四轮教师在相同前缀给终点分布，使用 reverse KL。D-LoopOPD 每步使用当前共享模型四轮的停止梯度目标。

**已有证据。**Ouro-Thinking 上深浅轮均可提高；动态版本的学生深度消融表明，较大深浅差距不一定更可学。Ouro-2.6B-Thinking 第一轮学生训练后 AIME 2024 Avg@16 仍为 0.63，第二轮从 39.16 到 47.50。这不证明 T3 是 S1 的好教师。

**与 EP 的关系。**EP 直接基于它提出改进：沿用同初始化深浅轮策略、学生生成前缀及终点 reverse KL，在原目标上增加按任务价值选择的候选保留和关键修正。它未单独识别、保留任务相关的中间候选过程，但这不意味着该过程已被证明受损。“dynamic”是教师随优化刷新，不能误读为自动选择轮次对应。

**边界。**终点分布包含最终修正结果；没有内部损失不等于必然丢失有用信息。原文主要是后训练，也不是仅追求推理减半。部分 RLTT 比较使用 2,048 token 训练 rollout，LoopOPD 为 31,744，预算不相同。论文正文声明代码／模型待接收后公开，本包不假设已有可运行复现。

## 2. On-Policy Transition Distillation：已有预测修正监督

- [PDF](../papers/2610.28048v1_transition_distillation.pdf)，[官方版本](https://arxiv.org/abs/2610.28048v1)。
- 核对：§2–3，实际 transition update 和实验设置。

**问题。**普通 OPD 仅匹配最终预测，是否错过了深度中逐步修正预测的监督？

**方案。**在学生生成前缀上，对选定 anchor layers 之间的 log-probability 增量加监督，同时保留终点 OPD。诊断会归一化增量来比较 transition KL；实际训练采用 top-K 权重与停止梯度信号构造增量更新。

**证据。**同题正确回答与教师内部预测变化更一致，随后过程监督在多个普通 Transformer 设置有收益。相关性不能证明每次修正都有因果价值。

**与 EP 的区别。**EP 需证明有用探索被压缩，以及选择性保留的价值。仅将 layer 改为 loop，首先是 OPTD 迁移基线。

## 3. LoopGR：已有循环轨迹压缩

- [PDF](../papers/2610.34157v1_loopgr.pdf)，[官方版本](https://arxiv.org/abs/2610.34157v1)。
- 完整标题：Parameter-Efficient Scaling for Generative Recommendation via Recurrent Depth。
- 核对：§4.2 的预测目标、固定映射与有限差分。

**问题。**生成式推荐希望用循环提高表达能力，并降低部署循环预算。

**方案。**高预算教师训练后初始化低预算学生；预测分布与 log-probability 变化方向共同蒸馏。映射是 `a(s)=ceil(s*T_teacher/T_student)`，四到二轮对应 T2→S1、T4→S2。方向项比较教师 T2→T4 与学生 S1→S2。

**与 EP 的区别。**任务是 SID 推荐，采用固定预算比例；没有根据候选探索的任务价值决定保留哪些过程。它已覆盖“少轮学习多轮演化”，是必须对照的方法。

## 4. Learning to Compute Is Not Learning to Retain or Reuse

- [PDF](../papers/2610.19768v1_compute_retain_reuse.pdf)，[官方版本](https://arxiv.org/abs/2610.19768v1)。
- 核对：摘要及计算、保持、重用的干预实验段落；§5、Figure 4 与 PDF 第 8 页。

**问题。**循环继续后答错，究竟是没算出、没保住，还是已有状态无法被后续计算重用？

**方案与证据。**用有中间真值和模拟器的 Mealy-machine 执行任务分离上述现象。作者观察到答案首次可读出时的状态尚可能不适合固定后供后续使用；再经过一些循环后才可安全固定。受控任务中的精确反事实支持这种区分。

**关键结果。**sampled-prefix 设置中，答案首次可读出时冻结状态，最终位置准确率从 100% 降到 4.1%；延迟两轮恢复至 99.8%。matched random exit 需要延迟四轮。缓存答案并保持状态更新则不产生同样损伤。这支持状态继续计算的价值，不意味着当前预测分布一定明显变化。

**与 EP 的关系。**它支持“可读答案不等于计算阶段已经完成”，提示不能只按熵、正确 token 概率或稳定性决定压缩。

**边界。**不等于已在 Ouro/Huginn 证明同一现象；也没有证明 EP 的分布辅助能够恢复准备性状态。若关键功能完全不体现在预测读出中，EP 需要另外设计机制监督。

## 5. Entropy-Aware OPD：探索／多样性的强基线

- [PDF](../papers/2603.07079v3_entropy_aware_opd.pdf)，[官方版本](https://arxiv.org/abs/2603.07079v3)。
- 核对：摘要、按教师熵补充 forward KL 的方法及多样性讨论。

**问题。**高熵教师下，reverse KL 的 mode-seeking 会影响生成多样性和学习信号。

**方案。**在高教师熵位置补充 forward KL，兼顾高置信模仿与多模式覆盖；作者报告多样性与数学 Pass@8 改善。

**动机证据。**§3.1 报告 AIME24/25 上教师 Qwen3-8B 的高熵位置比例为 18.5%，OPD 学生 Qwen3-1.7B-Base 为 6.8%（阈值为熵 ≥1.0）。模型容量与生成轨迹也不同，单凭该比较不能隔离散度因果；该论文也未测循环内探索。

**与 EP 的关系。**如果 EP 的收益主要来自“保持熵”，该方法可能已足够。EP 要建立的是任务相关候选演化或内部计算的额外价值，不能只展示更高熵。

## 6. Divergence controls entropy in distillation

- [PDF](../papers/2610.03529v1_divergence_entropy.pdf)，[官方版本](https://arxiv.org/abs/2610.03529v1)。
- 核对：摘要和理论适用范围；没有逐项复核其证明。

**问题。**学生熵变化究竟由数据、采样方式还是目标散度造成？

**证据。**论文从熵角度分析不同散度的性质，并区分 token-level reverse KL 与 on-policy 采样。其结论带有模型和优化设置的条件，不能概括为任何一步 reverse-KL 更新都会降低熵。

**与 EP 的关系。**观测到学生更确定，需要把散度、教师自身分布和计算压缩分离。输出熵变化不是内部探索丢失的直接证明。

## 7. Cross-Loop Query Handoff：主要动机证据

- [PDF](../ref/2610.34639v1_query_handoff.pdf)，[官方固定版本](https://arxiv.org/abs/2610.34639v1)。
- 完整标题：Cross-Loop Query Handoff: How Later Loops Redirect Two-Hop Retrieval in the Ouro Looped Language Models。
- 核对：§2–7、Tables 2/5/6；关键数据与 PDF 第 5、7、8 页交叉核对。

**问题。**后续循环怎样帮助 Ouro 把“人→盒子”和“盒子→颜色”组合起来？部分循环更新是否有具体计算作用？

**方案与证据。**保持循环次数，冻结回答位置 query 的跨轮更新会消除原有两跳增益；无关位置的匹配冻结基本不影响增益。对“第一轮错、第三轮对”的合成题，将同题第三轮各层／head 的 query 移入第一轮，可修复 89.8%（1.4B，n=245）和 96.9%（2.6B，n=223），打乱 donor 仅为 23.7%／20.6%。

**与 EP 的关系。**它是目前本包支持“部分中间更新具有任务价值”的主要因果证据，给 LoopOPD 改进提供动机与干预原型。详细论证见[说明](06_loopopd_motivation_evidence.md)。

**边界。**未测试 LoopOPD 或多候选探索；原生 Ouro 与 Ouro-Thinking 需区分。1.4B 的合成修复局限于候选排序，未转为生成修复；自然问答修复较弱，可靠性还依赖模型规模。query 移植需要额外教师前向，不能当作零成本加速。冻结特定组件也可能引入状态兼容性问题，不足以证明原路径唯一或每一轮都不可压缩。

## 8. 这些论文怎样组成实验问题

LoopOPD 是 EP 的方法起点和首要基线；Query Handoff 是中间更新具有任务作用的主要证据；Compute/Retain/Reuse 补充“可读出不等于可复用”；Entropy-Aware OPD 与 Divergence/Entropy 为候选覆盖风险及散度归因提供背景；OPTD 与 LoopGR 是新增过程项必须超过的近邻。

EP 尚需补足：LoopOPD 上有用过程迁移不足或受损的证据、有用性标签、学生可学性，以及等预算超过上述解释的效果。现有论文没有单独证明整条“LoopOPD 压缩必要探索”的因果链。相关价值选择和动态匹配方法见[相关工作](05_related_work.md)。
