# 阶段蒸馏单次运行记录

当前主定义：[研究计划](../docs/loop_distillation_research_proposal_zh.md)。填写真实执行情况；配置、预期与观测分开。

## 标识与论文对应

- run ID、日期、配置文件与解析后配置哈希：
- A/B/C/D；teacher_update=frozen/current；teacher_endpoint=fixed/native：
- 原文版本、对齐的参数、明确的实现／数据／硬件差异：
- 状态：计划／CPU语义验证／真实权重训练／基准评测／失败：
- 模型 family、代码、原生模型源码、模型 revision 与环境版本：
- Huginn-0125 是自选官方基模，还是有证据与被比较论文 checkpoint 一致：

## 数据与采样

- 源数据发布／revision／源哈希、原始题数、过滤后题数：
- 训练／阈值校准／dev／最终测试 ID 与去重证据：
- 原始消息保留、thinking 开关、chat template 修正与 prompt 上限：
- 全局 prompt 数、每题回答数、回答总数、rank 数、每 rank 微批：
- 当前 S 轮学生生成、温度、top-p、最大长度、EOS／截断规则：
- 一批生成与反向完成前是否保持参数不变；optimizer 更新次数：

## 教师与阶段目标

- K、固定 S、原生退出规则与阈值、每个 token 的 n_t：
- 教师与学生独立 KV；训练前向是否重算相同完整文本：
- Huginn 的 whole-prompt prefill、latest-m4、原生随机初态和完整 coda；学生反传范围：
- JS 定义、自然对数、τ、最大边界数、并列处理：
- 每个 token 的段末、学生映射、有效阶段数与权重：
- 无转折／n_t<=S 的仅终点回退比例：
- C/D 是否在同一前缀上匹配阶段数量：
- teacher 与分段是否 stop-grad；共享教师的参数版本：
- 段末监督是否误称为教师跳过计算；实际执行轮数：

## 可选结果筛选

- 是否启用；冻结初始教师 revision、动态退出和解码策略：
- 每题一次完整作答、验证器版本、内容哈希与接受率：
- 所有比较组是否共享接受题目池：
- 是否只复用正确性标签，当前学生前缀重新选 n_t：
- 筛选生成、验证和数据准备成本：

## 优化、评测与成本

- reverse KL 完整词表／近似范围；阶段、token、回答归一化：
- 完整参数／其它适配；AdamW LR、betas、eps、WD、warmup、clip：
- 参数／前向精度、checkpointing、读出 chunk、DDP 或其它后端：
- 检查点、optimizer／各 rank RNG、恢复一致性与原日志边界：
- 固定 S 的无额外干预评测；Avg@N／Pass@N、题数、verifier、解码：
- 原生出口直方图、阶段数量、学生轮→教师深度、JS 统计：
- 实际 teacher/student token-loops、teacher coda token-positions、读出、生成长度、时间、显存、保存体积：
- 动态 Huginn 的 teacher_max_loops 与实际执行量是否分开；不可把 teacher_executed_loops 写成统一 K：
- 多 seed、配对区间、一般能力与深轮副作用：
- 原始逐题输出、metrics、checkpoint、核验脚本与结果：
- 当前结论支持什么、尚不能支持什么：
