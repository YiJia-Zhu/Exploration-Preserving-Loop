# JS 阈值驱动的跨轮阶段蒸馏

当前方法以[主方案](docs/loop_distillation_research_proposal_zh.md)为准，直接基于 LoopOPD。**默认流程：动态学生生成 → 教师在同一学生前缀上原生退出 → 保留全部 JS 超阈值边界、取段末 → 学生按阶段顺序学习预测，并学习何时停止。**

学生目标轮数 `S_t=1+边界数`，实际轮数由自己的退出头决定，记为 `S_hat_t`。主方法不固定二轮，也不做 Top-S；旧固定预算方案保留为 F 对照。

## 默认运行

```bash
# 不传 --config 即使用 configs/default.json → e_dynamic_js.json。
.venv/bin/python scripts/train_stage_distillation.py --show-config
.venv/bin/python scripts/train_stage_distillation.py --check-assets

# 无需正式数据或GPU：随机小模型，检查训练、停止学习与保存。
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/train_stage_distillation.py \
  --smoke --device cpu --output experiments/runs/my_dynamic_smoke

# 原始LoopOPD：冻结四轮教师，固定二轮学生，只学终点。
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/train_stage_distillation.py \
  --config configs/a_loopopd.json --smoke --device cpu \
  --output experiments/runs/my_loopopd_smoke
```

正式四卡示例，GPU编号按实际空闲卡修改，输出目录每次使用新名称：

```bash
# 我们的方法，使用默认配置。
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=4 scripts/train_stage_distillation.py \
  --device cuda --output experiments/runs/my_dynamic_student

# 原始LoopOPD baseline。
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=4 scripts/train_stage_distillation.py \
  --config configs/a_loopopd.json --device cuda \
  --output experiments/runs/my_loopopd
```

当前默认 Ouro，教师和学生最大上限均为4，学生本地停止概率达到0.5时退出。JS阈值0.05 nats、stop BCE系数1是明确的开发起点，尚未校准。Huginn 使用 `configs/huginn_e_dynamic_js.json`，上限32，复用其完整coda；新增线性学生退出头。教师参数冻结／刷新与学生动态轮数分别配置。

## 阅读与实验入口

| 文件 | 用途 |
|---|---|
| [主方案](docs/loop_distillation_research_proposal_zh.md) | 权威定义与研究范围 |
| [实现与命令](docs/08_implementation_and_reproduction.md) | 参数、缓存约定、训练／评测、已验证范围 |
| [Idea](docs/01_idea.md) / [实验计划](docs/02_experiment_plan.md) | 阈值阶段数、停止标签、A–F对照 |
| [论文笔记](docs/03_paper_notes.md) / [相关工作](docs/05_related_work.md) | LoopOPD来源与证据边界 |
| [认识与假设](docs/04_insights.md) / [动机证据](docs/06_loopopd_motivation_evidence.md) | 待验证机制与局限 |
| [Ouro矩阵](experiments/experiment_matrix.csv) / [Huginn矩阵](experiments/huginn_experiment_matrix.csv) | 当前A–F；旧固定预算矩阵已归档 |
| [历史实验](docs/07_pilot_results_20261108.md) | 查询干预和早期短训，不证明当前方案有效 |

## 当前边界

正式数据 `data/dapo_math17k/train.jsonl` 尚未准备，期望17,398题；不会自动替换为其他项目的子集。两种基模集中存放于 `/home/zhenning/data/huggingface_models/`，项目使用同名软链接。

第一版动态学生通过前缀重算实现真正的逐轮停止，避免近似补齐缺失KV；训练回放补足阶段监督需要的循环。前缀重算也可能增加总开销，正式速度比较需包括统一缓存条件的LoopOPD对照，不能由平均轮数下降直接宣称延迟改善。Ouro原生教师仍先算K轮再选读出。

最新工程核验见[报告](audit/dynamic_student_20261113/validation.json)。这些检查支持实现语义和工程可行性，尚未建立正式任务收益。论文材料仍保留六篇核心与十七篇参考，见[核心清单](papers/manifest.json)和[参考入口](ref/README.md)。
