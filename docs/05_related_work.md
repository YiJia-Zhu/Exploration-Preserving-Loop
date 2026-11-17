# 相关工作与当前定位

当前方案直接基于LoopOPD；完整定义见[主方案](loop_distillation_research_proposal_zh.md)。

| 方向 | 已覆盖什么 | 当前需要验证什么 |
|---|---|---|
| LoopOPD / D-LoopOPD | 学生前缀、跨轮终点reverse KL、教师参数冻结／刷新 | 阶段目标和动态学生是否改善精度—成本曲线 |
| Ouro / Huginn原生退出 | gate累计概率或KL收敛 | 原生教师出口与新增学生停止任务分别评价 |
| Think-at-Hard / CALM / Adaptive Depth | 参考标签、轻量退出决策与校准 | 用JS阶段数做停止标签是否合适 |
| OPTD / LoopGR / PKD | 中间状态或预测过程监督 | 阈值分段和阶段数的独立价值 |
| RAIL-KD / ALP-KD / BERT-EMD | 随机、软、有序或多层匹配 | D的均匀位置与F固定预算是否已经足够 |
| Best-of-Depth | 用最低参考损失训练深度候选集合 | 当前是逐token阈值阶段与停止学习；checkpoint一致性未确认 |
| Query Handoff / Compute-Retain-Reuse | 一些跨轮更新具有计算功能 | JS选中的阶段是否承载这些功能 |
| Outcome-guided distillation | 用结果筛选监督 | 可选正确性筛选是已有组件，不能单独当创新 |
| Language Models (Mostly) Know What They Know | P(True)评估候选答案、P(IK)预测是否知道 | 仅核对[论文摘要](https://arxiv.org/abs/2207.05221)；不能据此认定Ouro/Huginn的自报信心已校准 |

完整组合是“原生教师终点→JS阈值阶段数→段末蒸馏→学生停止学习”。组件已有依据不等于组合已有效；动态退出也不自动等于关键探索迁移。论文固定版本与适用模型见[参考入口](../ref/README.md)。

[新增消融](12_confidence_and_correctness_ablations.md)分别比较原始分布置信度与输入改变后的prompt探针，保持原始前缀上的蒸馏目标。正确性消融仅筛教师答对的训练题目，学生当前回答全部参与蒸馏；不能把自报信心当作正确性验证器。
