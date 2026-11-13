# EP 单次实验／诊断记录

状态：待填写。复制后记录真实运行；预期、设计与观察分开写，不预填效果。

## 标识与目的

- 实验 ID / run ID / 日期 / 执行者：
- 阶段：工程检查 / 诊断 / 训练 / 评测：
- 检验哪条假设 / 对照是谁：
- 方法基础：LoopOPD 或 D-LoopOPD；与原文模型、训练配方的差异：
- 判断过程迁移不足还是训练损伤；原始同预算学生对照 run ID：
- 状态：计划 / 运行中 / 完成 / 失败：

## 模型与环境

- 完整 checkpoint / revision / Base或Thinking / tokenizer / native template：
- 教师与学生是否同初始化；教师快照与冻结方式：
- 层栈与循环定义 / 教师轮数 / 学生轮数 / early-exit：
- prelude/core/coda 或 per-loop normalization / readout 位置：
- GPU / dtype / 量化 / 参数更新方式：
- 训练仓库 commit / 软件版本 / optimizer 与初始状态：
- seed / 随机初态 / dropout / 分支共同随机数规则：

## 数据与前缀

- 训练、诊断、开发、测试 ID 文件及哈希：
- 题目级去重 / verifier / 答案提取 / 失败处理：
- 前缀来源：固定库还是当前两轮学生：
- prefix hash / token 位置 / 问题 ID / 已生成长度：
- cache 策略 / 完整前缀重算与缓存恒等结果：
- teacher rescore 与独立预算策略是否等价；差异：

## 干预与有用性标签

- 候选初筛规则 / top-K / 各轮原始概率 / OTHER 质量：
- 候选／修正选择规则、阈值和发现集：
- 干预对象：token / 分布 / log-probability 修正子集 / hidden state：
- 分支策略 / 后续轮数 / 长度 / 每分支样本数：
- 恒等、随机、同尺度与 donor 对照：
- 若干预 query：每层／head 范围、donor 前向成本、排序与生成评测区别：
- 每题分支结果、区间、标签覆盖与选择成本：
- 单位置与多修正交互是否分别检查：

## 训练监督

- 终点目标 / 候选保留 / 修正过程 / λ / μ：
- 中间教师点位：固定或动态；选择结果路径：
- S1 辅助的显式分布接口；采样来源与梯度是否一致：
- 候选空间、归一化、tail、裁剪、stop-gradient：
- 门控密度 / 梯度尺度 / 与随机对照的匹配：
- teacher / 选择器 / 临时 optimizer step 的停止梯度与隔离：

## 成本与结果

- 更新数 / student rollout tokens / teacher rescore tokens：
- 额外 output-head / coda / 分支续写 / 搜索与临时更新：
- 各阶段 GPU 时间 / FLOPs估计 / 峰值显存：
- 评测 checkpoint / 普通固定轮数 / 温度 / 长度 / 每题采样数：
- 正确率 / Avg@K / Pass@K / 长度 / 重复 / 截断：
- 候选保留 / 修正学会程度 / 成功解法覆盖 / 熵：
- 四轮与一般能力副作用 / 实际部署延迟：
- 问题级配对统计与置信区间：
- 原始日志、逐题结果、checkpoint 与配置文件：
- 观察 / 与解释之间仍缺什么 / 下一步或停止理由：
