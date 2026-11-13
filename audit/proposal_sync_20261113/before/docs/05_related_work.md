# 当前方法的相关工作与区别

当前方案直接基于 LoopOPD / D-LoopOPD，权威定义见[研究计划](loop_distillation_research_proposal_zh.md)。以下区别是待验证的设计定位，不表示已确认新颖性或有效性。

| 方向 | 已覆盖什么 | 当前要检验的部分 |
|---|---|---|
| LoopOPD / D-LoopOPD | 学生前缀、深轮终点监督、冻结／参数刷新教师 | 保留原目标和主要超参，增加原生监督终点及阶段端点 |
| Ouro / Huginn 原生退出 | gate 累计概率或预测收敛选择出口 | 复用基模规则确定 n_t，不新训选择网络 |
| OPTD / LoopGR | 预测演化、固定比例循环轨迹与变化方向 | 当前不用增量 loss，检验 JS 转折和段末选择 |
| BERT-EMD / ALP-KD / RAIL-KD | 自动、软、多对多或随机层匹配 | 均匀、随机等简单对照不可省略 |
| PABEE / F-PABEE / CALM 等 | 预测稳定性、patience 或校准 | 标准 JS 只作阶段边界，不挪用其有条件保证 |
| Outcome-guided distillation | 结果与分歧／不确定性筛选 | 正确性筛选是可选已有组件，不单独当创新 |
| Query Handoff / Compute-Retain-Reuse | 部分状态变化的因果作用 | 验证阶段是否承载功能，尚不直接给 JS 标签 |

两个需要固定的区别：教师参数每步刷新，不等于每个 token 动态退出；选段末监督，不等于让教师跳过形成这些状态所需的循环。使用 Ouro/Huginn 时还需分别核对原生状态与 KV。

当前独立价值至少需要 A/B/C/D 连接起证据：原生出口的作用、阶段监督的作用、转折位置的额外作用，以及固定浅轮学生的等成本收益。只有“中间 loss 能训练”仍不足以支撑方法有效性。

具体论文编号、版本和适用模型见[主计划参考文献](loop_distillation_research_proposal_zh.md)；本地版本与阅读范围见[参考目录](../ref/README.md)。
