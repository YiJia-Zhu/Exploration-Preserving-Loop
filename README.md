# JS 阈值驱动的跨轮阶段蒸馏

当前方法以[主方案](docs/loop_distillation_research_proposal_zh.md)为准，直接基于 LoopOPD。**默认流程：动态学生生成 → 教师在同一学生前缀上原生退出 → 保留全部 JS 超阈值边界、取段末 → 学生按阶段顺序学习预测，并学习何时停止。**

学生目标轮数 `S_t=1+边界数`，实际轮数由自己的退出头决定，记为 `S_hat_t`。主方法不固定二轮，也不做 Top-S；旧固定预算方案保留为 F 对照。

现在三条研究线分别实现，统一入口为[scripts/run_three_route_ablation.sh](scripts/run_three_route_ablation.sh)：`ROUTE=hf`为教师／学生都真实动态停止，`ROUTE=vllm`为固定K教师／固定N学生，`ROUTE=hybrid`为HF动态训练／vLLM多固定预算部署。新入口默认交叉线；完整定义与首轮4→2消融见[三线协议](docs/15_three_route_protocol.md)。三线均用完整词表监督，词表top-20近似仍单独显式选择。

所有vLLM入口均省略`max_num_seqs`，使用框架默认值。训练按当前更新、评测按每卡完整分片一次提交请求，由vLLM连续调度。

当前vLLM线默认由vLLM教师固定计算K轮，再按原生gate累计CDF确定监督终点、在终点前按JS选阶段；学生始终固定N轮。HF学生／optimizer与两个vLLM引擎按阶段切换显存，HF侧只保留冻结教师的读出头。默认消融组为`endpoint,top_n`，暂不运行均匀阶段。2026-11-17已按用户要求清理旧运行输出和checkpoint，保留轻量工程记录与当前GPU6、7运行；下一步正式baseline仅LoopOPD 4→2。

```bash
# 本轮先做vLLM固定预算线，终点与gate＋JS两组消融顺序运行。
ROUTE=vllm GPUS=2,3 bash scripts/run_three_route_ablation.sh --dry-run
```

## 当前阶段：Ablation确定方案

当前方案与各项对照都作为方法消融。先在开发集确定阶段监督、边界信号、教师更新与教师筛题策略，再做参数实验，最后做固定方案的正式baseline比较。

```bash
# 主消融入口。默认EXECUTION_MODE=hf_dynamic、METHOD=ours。
bash scripts/run_ablation.sh --dry-run

# 选择GPU后运行；超参数都在脚本开头。
GPUS=2,3 bash scripts/run_ablation.sh

# C/D/E阶段监督消融：三者均由学生退出头动态决定实际轮数。
STAGE_SELECTION=threshold_endpoint bash scripts/run_ablation.sh --dry-run
STAGE_SELECTION=threshold_uniform bash scripts/run_ablation.sh --dry-run
STAGE_SELECTION=threshold bash scripts/run_ablation.sh --dry-run

# 显式加入LoopOPD消融对照，不表示已进入正式baseline实验。
METHOD=both bash scripts/run_ablation.sh --dry-run

# 显式可选方案：固定K轮计算、动态读出，并使用top-20教师近似。
EXECUTION_MODE=vllm_top20 bash scripts/run_ablation.sh --dry-run
```

结果默认保存到`experiments/ablations/`，配置、计划与训练／评测摘要均记录`stage=ablation`。三个启动脚本统一默认每更新32题×8回答＝256条，目标100次更新，即25,600条回答和3,200次题目抽取。每批无放回抽32道不同题，跨更新可重复；这不是一个epoch，论文未明确公开跨批采样细节。默认`TRAIN_EXAMPLES=all`，排除AIME原文重合题后，当前Ouro为17,309道训练题和64道DAPO备用开发题；新划分独立保存。

**默认先评AIME 2024，30题、每题16次，共480条回答／模型预算；2025／2026保留为可选扩展，分别报告Avg@16与Pass@16。** 题目与答案来源固定revision和SHA256，提示不含解答；目前从训练源排除1道原文重合题。详见[AIME评测协议](docs/17_aime_evaluation.md)。默认GPU 2、3、CPU_THREADS=16，训练／评测上限仍为16,384 tokens；JS、不按正确性筛题、不筛学生回答、seed42。统一入口保留10小时上限，包含评测，不保证100次更新及全部评测能完成。

**默认`EXECUTION_MODE=hf_dynamic`：保留原HF逐token真实提前停止、精确深度KV缓存、完整词表JS和reverse KL，不使用top-20教师近似。** `vllm_top20`作为独立可选消融；`hf_top20`只启用教师分布近似，生成与评测继续真实提前停止。两种近似方案保存至`experiments/ablations_fastpath/`，不与主方法混报。兼容入口`run_fastpath_ablation.sh`不加模式参数也会转入默认HF主方法。

底层`train_stage_distillation.py`与`evaluate_stage_distillation.py`是共享执行脚本；不同实验阶段可以复用。无需正式数据或GPU的随机小模型检查：

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/train_stage_distillation.py \
  --smoke --device cpu --output experiments/runs/my_dynamic_smoke
```

当前默认 Ouro，教师和学生最大上限均为4，学生本地停止概率达到0.5时退出。JS阈值0.05 nats、stop BCE系数1是明确的开发起点，尚未校准。Huginn 使用 `configs/huginn_e_dynamic_js.json`，上限32，复用其完整coda；新增线性学生退出头。教师参数冻结／刷新与学生动态轮数分别配置。

## 阅读与实验入口

| 文件 | 用途 |
|---|---|
| [主方案](docs/loop_distillation_research_proposal_zh.md) | 权威定义与研究范围 |
| [实验阶段与命名](docs/13_experiment_workflow.md) | 先方法消融，再参数实验，最后正式baseline比较 |
| [实现与命令](docs/08_implementation_and_reproduction.md) | 参数、缓存约定、训练／评测、已验证范围 |
| [Ablation启动脚本](scripts/run_ablation.sh) / [种子与复现说明](docs/09_reproducibility.md) | 默认种子42、GPU与超参数、环境记录；保留高效算子路径 |
| [动态KV缓存与验证](docs/10_dynamic_kv_cache.md) | 精确按需补齐深层KV，避免每个token重算完整前缀；测试与测速命令 |
| [日志与进度](docs/11_logging_and_progress.md) | tqdm、每rank完整日志、阶段起止时间、耗时与异常记录 |
| [置信度与正确性消融](docs/12_confidence_and_correctness_ablations.md) | 置信度／prompt探针替代JS，教师正确题目筛选；默认关闭 |
| [执行模式与可选近似](docs/14_vllm_top20_fastpath.md) | HF完整动态默认；vLLM固定K计算／top-20近似显式选择 |
| [三条研究路线](docs/15_three_route_protocol.md) | 真动态HF、固定预算vLLM、动态训练／固定部署交叉线；新统一入口默认交叉线 |
| [verl OPD兼容性调查](docs/16_verl_opd_loop_compatibility.md) | Ouro／Huginn接入判断、完整词表与逐轮监督差异、版本和双卡资源要求 |
| [AIME评测协议](docs/17_aime_evaluation.md) | 默认AIME24的30题×16次、可选三年扩展、Avg@16／Pass@16及训练重合排除 |
| [下一步正式实验](docs/18_formal_experiment_plan.md) | 唯一baseline为LoopOPD 4→2，与gate＋JS 4→2在同一步数比较；100步目标与续训预算 |
| [Idea](docs/01_idea.md) / [实验计划](docs/02_experiment_plan.md) | 阈值阶段数、停止标签、A–F对照 |
| [论文笔记](docs/03_paper_notes.md) / [相关工作](docs/05_related_work.md) | LoopOPD来源与证据边界 |
| [认识与假设](docs/04_insights.md) / [动机证据](docs/06_loopopd_motivation_evidence.md) | 待验证机制与局限 |
| [Ouro矩阵](experiments/experiment_matrix.csv) / [Huginn矩阵](experiments/huginn_experiment_matrix.csv) | 当前A–F；旧固定预算矩阵已归档 |
| [历史实验](docs/07_pilot_results_20261108.md) | 查询干预和早期短训，不证明当前方案有效 |

## 当前边界

已准备公开 DAPO-Math-17k 固定版本 `65877096c24ffa7abc4e4fa5edb95cf3413a5674`：去重后17,398道题，排除7道标签冲突题，得到 `data/dapo_math17k/train.jsonl` 的17,391题。保留原始prompt；处理记录见[本轮启动记录](audit/dapo_launch_20261114/README.md)。这份公开数据的处理记录不能独立证明与论文内部训练子集完全一致。两种基模集中存放于 `/home/zhenning/data/huggingface_models/`，项目使用同名软链接。

动态学生默认使用精确的按深度KV与状态缓存。后续token需要更深循环时，仅补算历史前缀中尚未运行的对应深度，不替换成浅层KV；训练回放仍补足阶段监督需要的循环。补算成本纳入统计，不能由平均轮数下降直接宣称延迟改善。`rollout.use_cache=false`可恢复旧前缀重算对照；Ouro原生教师仍先算K轮再选读出。主路径使用PyTorch/HF，vLLM＋top-20入口为显式可选消融；未接入SGLang。

最新工程核验见[报告](audit/dynamic_student_20261113/validation.json)。这些检查支持实现语义和工程可行性，尚未建立正式任务收益。论文材料仍保留六篇核心与十七篇参考，见[核心清单](papers/manifest.json)和[参考入口](ref/README.md)。
