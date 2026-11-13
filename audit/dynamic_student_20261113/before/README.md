# 原生动态退出与 JS 阶段蒸馏

更新：2026-11-13。项目目录名称保留 Exploration-Preserving-Loop；**当前方法以[研究计划](docs/loop_distillation_research_proposal_zh.md)为准**，直接基于 LoopOPD / D-LoopOPD。

**主流程：固定浅轮学生生成 → 教师在相同学生前缀上按原生规则选择退出轮次 n_t → 在前 n_t 轮内按相邻 JS 分段，取段末 → 以 reverse KL 训练固定 S 轮学生。**

Ouro 第一版为教师上限 K=4、学生固定 S=2。教师参数冻结／每步刷新，与教师固定／原生动态出口是两种独立设置。JS 选择监督点；所用 Ouro 发布实现仍实际执行 K 轮，不能把较早读出当成已经节省教师循环计算。

当前代码已包含完整参数训练、LoopOPD 和 D-LoopOPD 终点基线、A/B/C/D 对照、按 token 的原生退出与 JS 分段、可选一次教师作答的正确性缓存、检查点与恢复、固定学生深度生成以及整数数学评测。实现说明和已执行核验见[代码与复现对应](docs/08_implementation_and_reproduction.md)。已补齐 Huginn 原生 coda、KL 退出与独立自适应缓存；Ouro 与 Huginn 共用阶段选点和损失。完整论文分数及方法收益尚未复现。

## 阅读入口

| 文件 | 用途 |
|---|---|
| [当前研究计划](docs/loop_distillation_research_proposal_zh.md) | 主方案的权威定义、动机与参考文献 |
| [代码与复现对应](docs/08_implementation_and_reproduction.md) | 原文参数、实现差异、命令与测试证据 |
| [Idea 摘要](docs/01_idea.md) | K / n_t / S、段末映射与损失 |
| [实验方案](docs/02_experiment_plan.md) | A/B/C/D、冻结与参数刷新、阈值和成本控制 |
| [论文笔记](docs/03_paper_notes.md) | 方法来源与近邻，不把文献证据当作本项目结果 |
| [证据与假设](docs/04_insights.md) | 哪些可验证、哪些尚未成立 |
| [相关工作](docs/05_related_work.md) / [动机证据](docs/06_loopopd_motivation_evidence.md) | 原生退出、阶段监督与因果证据的边界 |
| [2026-11-08 历史实验](docs/07_pilot_results_20261108.md) | 旧候选保留方案的工程／查询干预结果，不代表本次方案效果 |
| [当前矩阵](experiments/experiment_matrix.csv) / [记录模板](experiments/run_record_template.md) | Ouro 八种配置及运行需要记录的量 |

## 起步

```bash
# 只解析论文对齐配置，不加载模型或启动训练。
.venv/bin/python scripts/train_stage_distillation.py --config configs/a_d_loopopd.json --show-config

# 官方 Ouro 的随机微型架构，CPU 上检查完整训练过程；不加载 1.4B 权重。
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/train_stage_distillation.py \
  --config configs/d_current_native_js.json --smoke --device cpu \
  --output experiments/runs/my_stage_smoke

CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest discover -s tests -v
python scripts/validate_package.py
```

正式数据入口为 `data/dapo_math17k/train.jsonl`，论文表 8 报告 17,398 题。当前没有把其他项目的处理版或不同 DAPO 发布默认为同一个子集；准备命令及完整训练／评测命令见实现文档。JSONL 缺失或题数不符会明确报错。

正式模型集中存放于 `/home/zhenning/data/huggingface_models/`；项目 `models/Ouro-1.4B-Thinking`、`models/huginn-0125` 均为同名软链接。Huginn 的 K=32、S=8/16 是本计划的验证设置；`huginn-0125` 是明确选择的官方基模，尚未确认与 Best-of-Depth 使用同一 checkpoint。Huginn 配置和命令见[实现文档第8节](docs/08_implementation_and_reproduction.md#8-huginn-原生后端与验证配置)，八组对照见[Huginn矩阵](experiments/huginn_experiment_matrix.csv)。原始模型源码保持原样，新增逻辑在 `ep_loop/`；旧玩具训练脚本与原始运行记录继续保留为历史证据。

## 论文材料与状态

本地保留六篇核心论文、十七篇参考论文及固定版本原文快照，见[核心清单](papers/manifest.json)和[参考入口](ref/README.md)。研究计划新增引用单独登记于[引用索引](experiments/proposal_references_20261113.json)；只有条目被登记不等于其全文已下载或本次独立核验。

2026-11-08 的 64 场景干预支持局部跨轮更新具有任务作用，两组 12 步 LoRA 短训没有显示目标两轮优势。本次 CPU 随机模型检查提供代码语义证据，不能代替真实权重训练、AIME 分数或必要探索的证明。
