> 历史协议：本文件是11月8日的 LoRA／候选辅助实验。当前主线改为[原生动态退出与 JS 阶段蒸馏](../docs/loop_distillation_research_proposal_zh.md)，复现配置和代码见[实现说明](../docs/08_implementation_and_reproduction.md)。

> 当前核心代码与 Ouro/Huginn 接口核验见[实现说明](../docs/08_implementation_and_reproduction.md)；本文件的历史结果不变。

# 2026-11-08 首轮可行性协议

本轮基于官方 ByteDance/Ouro-1.4B-Thinking，revision `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`。LoopOPD 项目页目前标注代码／模型 coming soon；本项目按其冻结教师目标自行实现小规模适配，不称官方复现。

使用原始 HF 模型源码和 cql-handoff commit `75b52dbfdb160eb5da59b7e7897bbeb0bb16ec91` 的读出／干预接口。模型卡所列 Ouro GitHub 地址本次返回 404，HF 官方源码可用。仅使用空闲 GPU 4；预留可调整的短程运行，不启动完整 benchmark。

工程修正：首次使用模型卡建议的 Transformers 4.54.1，D0 在创建缓存时因 `key_cache` 只读属性报错，未进入任务评测。改用 4.57.1／tokenizers 0.22.2，保留失败日志并在新目录重跑；数据、目标和成功门槛不变。训练和诊断主计算仍使用完整前缀重算。

## 诊断

- 新生成 64 个两跳场景，随机种子 20261108；每场景有两跳题及两个单跳控制。
- 使用论文的原始 completion 模板，评测四个候选的排序。该设定与 Thinking 原生长推理生成不同，需单列结果。
- D0：各轮读出与独立停止前向精确一致，query 恒等写回精确一致；两／四轮 cache 与完整前缀比较，预设 KL <0.01 且 full-vocab top-1 相同。
- D1：保留四轮候选 logits、原始概率质量、熵、正确候选排名，报告每轮准确率。
- D3：保持计算量，冻结回答位置 query 为第一轮值；对照同规模的无关事实位置冻结。所有三种问题都测。
- 把第三轮每层／head 的末位置 query 移入第一轮，比较同题和同长度其他题 donor。先按长度分组轮换确定 donor，再观察修复结果。报告早错晚对子集及全体修复后准确率。
- 5000 次按场景配对 bootstrap；区间为探索性、未做多重比较调整。没有显著收益也保留完整输出。

## 训练实现检查

冻结底座，只训练 q/v 的 rank-8、alpha-16 LoRA；adapter 关闭且 no_grad 时为固定四轮教师。FP32 adapter 与 BF16 底座分开，教师底座不更新。

实现完整词表 reverse KL，以及候选 + OTHER 聚合的中间 forward KL。第一轮比较 LoopOPD 目标与固定 T2→S1 辅助：辅助使用四个颜色 token 及 OTHER、权重 0.2，暂不声称已实现按因果价值选择的完整 EP。每次更新前重新用当前两轮学生采样，最多 16 token；无 top-k/top-p 截断。损失只作用于生成 token 对应的前缀，按回答长度归一化。

训练题与诊断题按完整场景指纹及忽略事实排列的语义场景去重；短训初始计划每臂 12 步、同 seed 与 optimizer 设置，AdamW 学习率 1e-4、weight decay 0、梯度范数裁剪 1。报告每臂真实 rollout 数、tokens、损失、梯度、时间与训练前后留出题表现。短训主要验证工程和学习信号，不能据单 seed、单合成任务宣布 EP 有效或探索已被保留。
