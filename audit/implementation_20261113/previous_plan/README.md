# Exploration-Preserving Loop（EP）

更新日期：2026-11-08。工作名 **Exploration-Preserving Loop，简称 EP**。

**已运行首轮可行性实验。**Ouro-1.4B-Thinking 上完成 64 个新场景的读出／查询干预和两组各 12 步短训。中间更新的任务作用得到局部支持；目标两轮准确率两组均为 75%，尚未显示中间辅助优于 LoopOPD。详见[实验结果与复跑方法](docs/07_pilot_results_20261108.md)。

**Idea 的直接来源是 LoopOPD。EP 基于 LoopOPD 的跨轮 on-policy 蒸馏，保留其“浅轮学生生成前缀、深轮教师提供最终分布”的训练框架，研究如何增加有用中间过程的监督。LoopOPD 同时是方法起点和首要对照。**

**研究问题：LoopOPD 将深轮最终预测教给浅轮学生时，是否有部分有价值的候选演化未被学会？在 LoopOPD 上保留这些探索与修正，能否进一步提高固定低预算学生的能力？**

这里的探索首先指候选保留、比较和概率修正的过程。它与输出采样多样性分别测量。已有局部内部更新证据；LoopOPD 是否丢失必要的多候选探索仍待验证，Huginn 尚未运行。

当前候选方案：**LoopOPD + 有用候选保留 + 关键修正监督。**第一版沿用冻结教师的 LoopOPD；D-LoopOPD 上的同类改进和动态轮次对应留作后续验证。这里的 4→2 指深轮教浅轮，不代表参数减半，也不预设训练总成本减半。

主要动机证据选用 [Cross-Loop Query Handoff](ref/2610.34639v1_query_handoff.pdf)：它在 Ouro 上通过查询冻结和移植，证明特定任务中的部分跨轮更新具有因果作用。该论文没有测试 LoopOPD；“LoopOPD 未保留必要探索”仍是我们要验证的假设。完整论证见[LoopOPD 改进的动机与证据](docs/06_loopopd_motivation_evidence.md)。

## 从这里开始

| 顺序 | 文件 | 用途 |
|---|---|---|
| 0 | [首轮可行性结果](docs/07_pilot_results_20261108.md) | 已运行内容、观察、代码、检查与局限 |
| 1 | [Idea 与方法](docs/01_idea.md) | EP 的主问题、探索定义、候选损失与研究边界 |
| 2 | [LoopOPD 改进的动机与证据](docs/06_loopopd_motivation_evidence.md) | 为什么分布变化可能有价值；主证据论文具体证明什么 |
| 3 | [实验方案](docs/02_experiment_plan.md) | 以 LoopOPD 为起点的诊断、训练对照与模型注意事项 |
| 4 | [论文笔记与主要动机证据](docs/03_paper_notes.md) | 方法来源、过程蒸馏、探索与熵、内部计算证据 |
| 5 | [证据、insights 与待验证假设](docs/04_insights.md) | 哪些已知、哪些待证，什么结果应收窄或停止当前方案 |
| 6 | [相关工作与主要区别](docs/05_related_work.md) | 与轨迹压缩、价值加权、动态层匹配、计算路由的区别 |
| 7 | [实验矩阵](experiments/experiment_matrix.csv) | 诊断和训练组别、阶段依赖、预算与结果状态 |
| 8 | [运行记录模板](experiments/run_record_template.md) | 模型、数据、缓存、分支、监督与成本记录 |

## 论文材料

六篇核心论文：

1. [LoopOPD / D-LoopOPD](papers/2610.10623v2_loopopd.pdf)：EP 的直接方法来源与首要基线。
2. [On-Policy Transition Distillation](papers/2610.28048v1_transition_distillation.pdf)：预测增量监督的直接近邻。
3. [LoopGR](papers/2610.34157v1_loopgr.pdf)：固定比例循环轨迹压缩与变化方向监督。
4. [Learning to Compute Is Not Learning to Retain or Reuse](papers/2610.19768v1_compute_retain_reuse.pdf)：可读答案与可复用状态的区别。
5. [Entropy-Aware OPD](papers/2603.07079v3_entropy_aware_opd.pdf)：输出多样性与散度混合基线。
6. [Divergence controls entropy in distillation](papers/2610.03529v1_divergence_entropy.pdf)：区分散度作用、采样方式和学生熵。

[版本与哈希清单](papers/manifest.json)；[十七篇参考论文与阅读入口](ref/README.md)；[材料检查结果](audit/package_validation.json)。论文按固定版本存放，附官方正文与摘要快照。全部结果均为作者报告，未独立复现。

## 实验状态与下一步

已使用固定 Ouro-Thinking checkpoint 跑通各轮读出、缓存检查、查询冻结／移植，以及 LoRA LoopOPD 目标和固定中间辅助的短训。完整方案仍需原生生成任务上的候选价值诊断、更充分训练和多个 seed 对照。

**暂不默认有用探索已经被压缩；暂不直接训练复杂动态选择器。**Huginn 优先用 32→16 或 16→8 单独验证，不能把其早期轮次与 Ouro 等同。

本轮 revision、环境、数据、GPU 与结果已记录于[执行协议](experiments/pilot_protocol_20261108.md)和[环境记录](audit/pilot_environment.json)。目录已包含模型下载、代码、adapter 和原始结果；完整 EP 的任务价值门控、关键修正训练和数学／代码评测尚未实现。

模型统一存放于 `/home/zhenning/data/huggingface_models/Ouro-1.4B-Thinking`，项目内的 `models/Ouro-1.4B-Thinking` 为指向该目录的软链接，可供其他项目复用。下载脚本也使用共享目录，现有运行命令继续通过项目内路径加载。

本目录参考 `steer_opd` 的材料组织方式，内容围绕 EP 独立整理。论文材料不依赖原讨论目录；运行环境和外部源码依赖另在结果文档中记录。

重跑材料检查：

```bash
python scripts/validate_package.py
```

该检查验证材料、链接、实验记录结构和已有运行报告的哈希，不重新运行模型。实际模型输出的独立复核入口为 `scripts/verify_pilot.py --replay-checkpoints`。
