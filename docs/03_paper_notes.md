# 论文笔记：当前动态学生方案的来源

权威方案见[主计划](loop_distillation_research_proposal_zh.md)。原论文数字、当前实现语义和本项目效果分开记录。

## LoopOPD / D-LoopOPD

[LoopOPD v2](https://arxiv.org/abs/2610.10623v2)，[本地PDF](../papers/2610.10623v2_loopopd.pdf)。核对Eq.9/11、Appendix B、Tables4/7/8/9。LoopOPD在学生生成前缀上用冻结深轮终点reverse KL监督浅轮学生；D-LoopOPD使用当前共享模型的停止梯度深轮目标。dynamic在原方法中指教师参数刷新。

B保留其固定轮次目标与主要超参。E增加教师原生监督终点、全部JS阈值边界、阶段数目标和学生停止BCE。当前代码是论文对齐参考实现，没有取得作者未公开的完整训练栈，也未复现分数。共享参数的改善条件涉及梯度对齐，不能把更深、更高熵或中间监督直接等同能力收益。

## 原生模型、停止与缓存

Ouro已有gate提供累计退出分布，但发布实现先执行全部K轮。教师保留该语义；学生训练复用同一类gate参数，以本地P(stop)≥0.5执行真正逐轮停止。当前默认使用精确按需补齐的深度KV缓存，避免给未执行的深层伪造KV；第一版完整前缀重算保留为诊断。这是本项目的学生执行实现选择，不冒称官方已提供该优化。

Huginn原生KL比较方向为 `KL(q_previous||q_current)`，首轮参考均匀分布，代码auto阈值1e-3；当前显式用5e-4。预测经过完整coda。动态学生增加独立线性退出头，原生教师仍用自身KL与latest-m4，不共享师生缓存。

Think-at-Hard、CALM和冻结轨迹gate研究支持用标签学习轻量决策器的组件思想；它们没有验证“JS阶段数就是学生正确预算”。本项目不直接挪用其模型或正确性保证。

## 过程与因果证据

[OPTD](../papers/2610.28048v1_transition_distillation.pdf)与[LoopGR](../papers/2610.34157v1_loopgr.pdf)已研究预测演化或循环轨迹监督。当前不使用旧的log-probability增量项。

[Compute/Retain/Reuse](../papers/2610.19768v1_compute_retain_reuse.pdf)区分可读答案与供后续计算使用的状态；[Query Handoff](../ref/2610.34639v1_query_handoff.pdf)提供跨轮查询冻结与移植范式。它们支持某些更新具有具体作用，未证明本项目的JS分段或退出标签有效。历史本地干预见[结果](07_pilot_results_20261108.md)。

RAIL-KD和选层研究要求保留均匀、随机与固定映射等简单对照。C/D/E同目标阶段数的比较，用来判断JS位置是否有额外价值。

## Best-of-Depth的模型核对边界

[Best-of-Depth v1](https://arxiv.org/html/2611.04663v1) §4.4、附录A.3和TeX源码写Huginn 3.565B，训练深度32，评测4/8/12/16/24/32各自完整生成。它依据参考答案整体损失选一个出口，和当前逐token JS阶段数及on-policy reverse KL不同。

材料没有指定 `huginn-0125` 仓库、预训练步数或revision。本项目自选官方基模 `tomg-group-umd/huginn-0125`、revision `bb6621b65e90b6a4b9b29ef88dc83866d450470c`，不能称为已确认与BoD同checkpoint。BoD未加入本地23篇归档。

本地固定版本、来源哈希与阅读范围见[核心清单](../papers/manifest.json)、[参考目录](../ref/README.md)。引用登记见[索引](../experiments/proposal_references_20261113.json)，登记不等于全文已核验。
