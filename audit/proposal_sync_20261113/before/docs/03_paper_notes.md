# 论文笔记：当前阶段蒸馏的来源与边界

更新：2026-11-13。当前主方案见[研究计划](loop_distillation_research_proposal_zh.md)。原文数字为作者报告；本地早期实验和本次代码核验均单独记录。

## 1. LoopOPD / D-LoopOPD：直接基础

[固定版本](https://arxiv.org/abs/2610.10623v2)，[本地 PDF](../papers/2610.10623v2_loopopd.pdf)。核对 §4.1–4.3、Appendix B、Tables 4/7/8/9。

LoopOPD 让两轮学生生成前缀，冻结四轮教师在同一前缀给最终分布，使用 reverse KL。D-LoopOPD 用当前共享模型的四轮分布并停止目标侧梯度，每步更新后刷新教师。其 dynamic 指参数刷新，不是自动退出或 JS 选点。

本项目 A 分别实现这两种参数策略；B 加原生退出，C/D 再加阶段末轮监督。原文 LR 1e-6、weight decay 0.01、32×8=256 回答／更新、训练 top-p 1.0 与评测 0.7 均进入配置。D-LoopOPD warmup 10 步，冻结版无 warmup。原代码当前仍未发布，精确复现结果未完成，见[代码与参数对应](08_implementation_and_reproduction.md)。

原文共享参数分析的改善条件涉及梯度对齐；终点教师较强本身不保证深浅策略同时改善。原文也报告动态教师具有较高熵，不能笼统声称所有 LoopOPD 都压缩必要探索。

## 2. 原生退出与预测动态

Ouro 的官方源码提供 gate 的累计退出概率；所核对发布实现先执行配置中的全部循环，再选输出状态。我们复用这项出口语义，不把它命名为真实提前停算。

Huginn 的预测需经过 coda；其原生 KL 自适应退出、随机初态和 KV 语义不能用 Ouro 接口替代。本次 Ouro 代码明确拒绝未实现的模型 family。Ouro v5、Huginn、PABEE/F-PABEE 和其他退出研究的具体依据及限制见主计划参考文献；未在本地归档的条目单独登记，不虚构全文核验。

## 3. OPTD 与 LoopGR：过程监督近邻

[OPTD](../papers/2610.28048v1_transition_distillation.pdf) 在普通 Transformer 深度上监督预测变化；[LoopGR](../papers/2610.34157v1_loopgr.pdf) 使用固定比例循环轨迹映射和 log-probability 方向项。这些工作已覆盖“少轮学习多轮过程”的重要部分。

当前方法与其比较的具体对象是原生退出限定监督范围、教师相邻分布转折、阶段末轮与固定学生检查点映射。第一版不使用旧计划中的 log-probability 修正 loss；仅换成 loop 轴或加中间 loss 不能单独宣布创新。

## 4. 计算／保持／重用与查询干预

[Compute/Retain/Reuse](../papers/2610.19768v1_compute_retain_reuse.pdf) §5 区分可读答案和可供后续计算使用的状态。在其受控 sampled-prefix 任务中，过早冻结状态使最终位置准确率从 100% 降至 4.1%，延迟两轮恢复 99.8%。它没有证明本项目的 JS 或输出分布监督能恢复同样功能。

[Cross-Loop Query Handoff](../ref/2610.34639v1_query_handoff.pdf) 的 Ouro 查询冻结和移植支持部分跨轮更新具有任务作用，但未测试 LoopOPD 的探索损伤或 JS 选点。自然问题上的转移效果更弱，模型、排序与生成的边界须保留。

本项目 11月8日做过 Thinking 模型上的小规模适配，见[历史结果](07_pilot_results_20261108.md)。这些机制实验不直接给当前训练提供标签。

## 5. 熵、散度与自动映射

[Entropy-Aware OPD](../papers/2603.07079v3_entropy_aware_opd.pdf) 和 [Divergence/Entropy](../papers/2610.03529v1_divergence_entropy.pdf) 提醒我们：不同散度、模型容量和访问前缀都影响熵，熵变化不能直接证明内部探索损伤。

BERT-EMD、ALP-KD、RAIL-KD 与选层研究已覆盖自动／随机／软映射。C 的数量匹配均匀段末是必要的简单对照；D 需要实验证明转折位置有额外价值。

本地六篇核心与十七篇参考的版本、哈希及阅读范围见[核心清单](../papers/manifest.json)和[参考入口](../ref/README.md)。新增引用登记见[计划引用索引](../experiments/proposal_references_20261113.json)。
