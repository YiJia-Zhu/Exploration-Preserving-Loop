# 动态阶段蒸馏单次运行记录

权威定义：[主方案](../docs/loop_distillation_research_proposal_zh.md)。当前默认E，旧固定Top-S为F。

## 配置与来源

- run ID、日期、A–F组别、模型family、配置及解析哈希：
- teacher_update=frozen/current；teacher_endpoint=fixed/native：
- student_policy=fixed/learned/native；固定S或最大上限K：
- 基模revision、原生源码、环境版本、实现差异：
- 状态：随机CPU语义检查／真实权重短前向／正式训练／基准评测：

## 数据与采样

- 数据发布、revision、哈希、题数、保留原消息与ID：
- train/dev/test去重、阈值校准集合：
- prompt/response上限、thinking、模板修正、停止token、温度、top-p：
- 全局prompt数×每题回答数、world size、微批与optimizer更新频率：
- 可选正确性筛选是否共用接受池、每题一次生成、成本与verifier版本：

## 阶段与停止

- 教师n_t、目标S_t、实际S_hat_t分别统计：
- JS自然对数、温度1、τ；E是否保留所有超阈值边界：
- 段末轮次、第j段→学生第j轮；C/D/E标签数量是否可比：
- 无边界一轮、全边界不压缩、n_t=1：
- 学生继续／停止标签、停止阈值、β、每token BCE归一化：
- teacher/selection stop-grad；Ouro gate复用或Huginn新增头初始化：
- 提前退出后补算量、回放max(S_t)及实际前向总成本：
- 退出预测误差、目标—实际混淆矩阵、到K比例与S_t=n_t比例：

## 成本、评测和保存

- 缓存／完整前缀重算；师生缓存是否独立：
- 是否加入相同重算条件下的LoopOPD对照：
- 阶段KL、停止BCE、实际token-loops、coda读出、重计算与反向范围：
- 生成时间、端到端延迟、显存、生成长度、截断与停止分布：
- 测试精度、Avg@N／Pass@N、轮数／耗时曲线、多seed：
- 退出头、参数、optimizer、各rank RNG、恢复一致性证据：
- 原始逐题输出、metrics、checkpoint与核验记录：
- 支持的结论与尚未验证的假设：
