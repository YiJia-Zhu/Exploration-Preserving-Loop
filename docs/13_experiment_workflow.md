# 实验阶段与入口命名

下一步正式比较已限定为LoopOPD 4→2与我们的gate＋JS 4→2两组，见[正式实验计划](18_formal_experiment_plan.md)和[正式矩阵](../experiments/formal_experiment_matrix.csv)。其他深度、HF／交叉线及均匀选段仍属于独立消融。当前GPU6、7运行继续，启动时的ablation元数据保留；旧运行输出与旧checkpoint已按用户要求清理，当前运行、代码与输入数据保留。

当前候选方案和各项对照统一作为 **ablation方法消融**。先确定方法，再研究参数，最后进行正式baseline比较。开发阶段依据开发集结果做选择，最终测试集用于固定方案后的评测。

| 阶段 | 要确定什么 | 当前状态 |
|---|---|---|
| 1. Ablation | 阶段监督、边界信号、教师更新、是否筛教师答对的题目 | 当前执行阶段，入口`run_ablation.sh` |
| 2. 参数实验 | 在已选定的方法上选择JS／置信度阈值、BCE系数、停止阈值等 | 方法确定后再开展，当前不自动扫描 |
| 3. 正式baseline比较 | 用固定方法和参数，在完整协议下与LoopOPD等方法比较 | 参数确定后再开展，当前不自动启动 |

LoopOPD在方法上仍是论文来源与直接baseline，但目前显式运行它只是消融阶段的参考组。A–F、置信度替代和教师筛题均按当前ablation阶段记录；现有历史测试与测速保留原报告名称和实际执行范围。

## 当前入口

三条研究线新增统一入口`scripts/run_three_route_ablation.sh`，默认`ROUTE=hybrid`；先做vLLM线时指定`ROUTE=vllm`，运行K=4／N=2的endpoint、top_n两组，uniform_n仅显式选择。HF线指定`ROUTE=hf`，教师也是真实CDF提前停止。详见[三线协议](15_three_route_protocol.md)。这套研究路线与旧的“HF／vLLM＋top-20执行选项”分开记录，三线均用完整词表监督。

原`scripts/run_pilot.sh`改名为[scripts/run_ablation.sh](../scripts/run_ablation.sh)。底层[train_stage_distillation.py](../scripts/train_stage_distillation.py)和[evaluate_stage_distillation.py](../scripts/evaluate_stage_distillation.py)保留通用名称，三个实验阶段均可复用，避免复制训练实现。

默认`METHOD=ours`，运行当前动态候选方法；不自动连跑LoopOPD。默认输出目录为`experiments/ablations/`，数据划分为`data/dapo_ablation/`。每次运行的配置记录`experiment.stage=ablation`，launcher计划、训练provenance／summary、评测summary也记录阶段与用途。2026-11-17已清理旧运行结果，当前运行及轻量工程记录保留；清理明细见audit/storage_cleanup_20261117/。

默认执行模式为`EXECUTION_MODE=hf_dynamic`，完整词表JS／KL、逐token真实提前停止和精确深度KV缓存全部保留。两个sh入口不传模式参数都进入该主路径。`vllm_top20`和`hf_top20`属于明确标记的可选方案，输出到独立目录；它们不是默认方法的等价替代。

## 脚本开头的主要选择

| 参数 | 作用 |
|---|---|
| `EXECUTION_MODE=hf_dynamic` | 默认HF真实提前停止＋完整词表监督；显式可选vllm_top20／hf_top20 |
| `STAGE_SELECTION=threshold` | E：所有超阈值边界的真实段末监督 |
| `STAGE_SELECTION=threshold_uniform` | D：相同阶段数，均匀选择教师目标 |
| `STAGE_SELECTION=threshold_endpoint` | C：相同阶段数，只学教师终点 |
| `BOUNDARY_SIGNAL` | JS／最大概率／归一化熵／prompt信心探针 |
| `QUESTION_FILTER=all` | 默认不筛题；可选教师正确题目池或同数量随机题目 |
| `TEACHER_UPDATE=frozen` | 默认冻结教师；可选current教师参数刷新 |
| `METHOD=both`或`baseline` | 显式加入／单独运行LoopOPD消融对照 |

C/D/E都使用学生自己的退出头，K仅为上限；学生回答无论对错均参与蒸馏。先单独改变一个设置以确定方法。当前三个启动脚本默认32题×8回答＝256条／更新、目标100更新。两卡各处理16题×8＝128条，累积完256条后只执行一次optimizer.step；微批仍为每卡1条回答。每批均匀无放回抽32题，跨批可重复，100步为3,200次题目抽取、25,600条新on-policy回答，不表示扫完整数据集。跨批采样是本项目的明确实现选择，尚未验证与作者内部采样器一致。

默认`TRAIN_EXAMPLES=all`，从17,391题公开源中排除超长prompt和AIME原文重合题，seed42打乱后留出末尾64题，其余全部用于训练；当前Ouro为17,309训练题／64道DAPO备用开发题。默认先评AIME2024的30题×16次，报告Avg@16和Pass@16；每个模型预算480条回答，2025／2026可选扩展。旧256题与未清理AIME重合的划分不覆盖、不与新结果混报。`EVAL_SUITE=dapo_dev TRAIN_EXAMPLES=256`可恢复旧题池切分及DAPO评测逻辑。GPU 2、3、CPU_THREADS=16、训练／评测上限16,384、JS、不按正确性筛题、冻结教师继续保留；统一入口10小时上限包含评测，不保证100步全部完成。详见[AIME协议](17_aime_evaluation.md)。

```bash
# 只读查看当前消融配置和命令。
bash scripts/run_ablation.sh --dry-run

# 方法消融：均匀阶段目标。
GPUS=2,3 STAGE_SELECTION=threshold_uniform bash scripts/run_ablation.sh

# 只读查看置信度或教师正确题目池消融。
BOUNDARY_SIGNAL=max_probability bash scripts/run_ablation.sh --dry-run
QUESTION_FILTER=teacher_correct bash scripts/run_ablation.sh --dry-run

# 根据需要显式加入LoopOPD参考组。
METHOD=both bash scripts/run_ablation.sh --dry-run

# 显式可选消融：固定K计算、动态读出、top-20教师近似。
EXECUTION_MODE=vllm_top20 bash scripts/run_ablation.sh --dry-run

# 单独检查top-20近似，生成／评测均保留HF真实提前停止。
EXECUTION_MODE=hf_top20 bash scripts/run_ablation.sh --dry-run
```

运行标签默认包含模型、阶段监督、边界信号与题目筛选设置；其他变量可用`RUN_TAG`额外标注。启动前仍会检查模型与数据资源，`--dry-run`不写文件，`--prepare-only`不启动训练。正式训练数据尚缺时明确停止，不自动使用其他来源。

运行标签也包含执行模式；解析配置和计划明确记录`execution_mode`、生成／评测后端、完整词表或top-20教师分布，以及真实停算或固定K读出。模式与后端、分布元数据不一致时直接报错，防止HF名义下静默启用近似。

本轮默认源文件为去重并排除7道标签冲突题后的公开DAPO数据（17,391题），详细来源与启动参数见[运行记录](../audit/dapo_launch_20261114/README.md)。`DISTRIBUTED_TIMEOUT_SECONDS=21600`仅延长多卡集合通信等待时间，以容纳长回答造成的rank进度差异，不改变训练目标。

长序列训练默认`KEEP_LATEST_CHECKPOINTS=1`、`SAVE_EVERY_UPDATES=$UPDATES`，每次更新保存完整恢复状态并轮换一个checkpoint，永久保留最终快照；默认仅评最终更新。旧实验快照已按用户要求清理。vLLM教师临时状态默认使用`TEACHER_TRACE_DIR=/dev/shm`内存盘，每条读入GPU后删除，使用前检查空间；可改成有足够空间的临时卷。2026-11-15的显存修复与评测复用仍保留轻量[重启记录](../audit/long_context_recovery_20261115/README.md)。
