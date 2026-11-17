# 三条研究路线与vLLM分阶段执行

统一入口为`scripts/run_three_route_ablation.sh`，新入口默认`ROUTE=hybrid`。历史`run_ablation.sh`与`configs/default.json`的HF学生动态默认保留。三条研究线均使用完整词表JS与reverse KL；词表top-20近似另列为可选实验。

| ROUTE | 教师实际执行 | 学生训练回答 | 阶段监督 | 部署评测 |
|---|---|---|---|---|
| `hf` | HF原生CDF达到阈值即停止，必要时精确补齐历史KV | HF逐token动态停止 | 实际终点前全部JS超阈值边界＋终点 | HF真实动态停止 |
| `vllm` | vLLM固定K轮，按CDF选择逻辑终点n_t | vLLM固定N轮 | 终点对照与gate＋JS top-N两组 | vLLM固定N轮 |
| `hybrid` | HF固定K轮，再按CDF选择逻辑终点 | HF逐token动态停止 | 逻辑终点前全部JS超阈值边界＋终点 | 同checkpoint经vLLM评多个固定N |

默认先研究Ouro-1.4B-Thinking，教师保持冻结的初始权重，学生全参数训练。师生前向独立；HF／交叉线保留停止BCE，固定N学生不使用停止BCE。

## 当前vLLM方案：固定计算，按gate与JS选择监督

教师实际计算K=4轮，取得每轮规范化状态和gate。原生Ouro出口依照gate形成的stick-breaking累计CDF，在阈值`TEACHER_EXIT_THRESHOLD=0.9`首次达到时选择逻辑终点n_t，第K轮强制接收剩余退出概率。这与学生单轮sigmoid停止阈值不同；逻辑终点变浅不代表vLLM教师少算了循环。

`top_n`仅在n_t之前比较相邻JS，选择最大的至多N−1个边界，保留时间顺序，追加逻辑终点n_t，共min(N,n_t)个不同教师目标。不以JS阈值筛边界；平局优先较早边界。终点之后的变化不参加选择。

学生生成、可微回放和部署均实际执行N轮。浅终点默认`SHALLOW_TARGET_PLACEMENT=prefix`：当n_t<N时，只监督学生前n_t轮，例如教师终点1、固定学生2轮，只给学生第1轮教师第1轮目标。部署仍读学生第N轮，因此这一设置存在训练与部署读出差异，应在结果中说明。可显式选`last`，将有效目标均匀映射至学生轮次且包含最后一轮。n_t>=N时两种设置一致。

| 默认组别 | 教师逻辑终点 | 学生第1轮目标（N=2） | 学生第2轮目标 |
|---|---|---|---|
| `endpoint` | 固定K=4 | 无直接阶段KL | 教师第4轮；LoopOPD式终点对照 |
| `top_n` | 原生CDF选择n_t | JS所选阶段末；n_t=1时为教师第1轮 | n_t>=2时为教师终点n_t；n_t=1时默认无直接KL |

均匀阶段`uniform_n`保留为可手动选择的旧消融，已从默认`ABLATION_GROUPS=endpoint,top_n`中移除。两组同时改变了中间监督与教师逻辑终点；现阶段比较整体方案，不据此单独归因JS的效果。

## 一个更新的框架与显存生命周期

默认`TEACHER_BACKEND=vllm`、`PHASE_OFFLOAD=1`：

1. 清空上一更新梯度，将HF学生参数、AdamW状态和冻结读出头移至CPU；释放上一更新的DDP梯度桶。
2. 唤醒固定N的vLLM学生，载入当前学生权重并核对版本，一次提交本rank全部请求，生成本更新的on-policy回答；完成后sleep。
3. 唤醒固定K的vLLM教师，把题目与学生回答的已知前缀整体prefill。额外产生的一个采样token仅用于驱动标准引擎接口，不作为训练文本或监督答案；保存的是此前已知位置的各轮状态与gate。教师完成后sleep。
4. 把HF学生与optimizer移回GPU，按需读取教师状态，分块重建完整词表分布、计算原生终点／JS和KL；逐回答反向、累积梯度，DDP同步后执行一次AdamW更新。
5. 保存完整恢复点，删除本更新的临时教师读出；下一更新同步新学生权重重新生成。

vLLM教师在repo内的适配模型中输出所有轮次的规范化状态，worker对当前引擎实例的prefill bookkeeping挂接读出；不修改安装环境或共享模型文件。这条路径固定依赖已验证的vLLM 0.11接口。分块前缀及预占重算按请求位置拼回，真实教师token-loops包含重算。

跨进程只传输状态与gate，通过临时safetensors逐回答读取和删除；不保存所有位置的密集词表logits，不做top-k概率近似。启动脚本默认`TEACHER_TRACE_DIR=/dev/shm`，使用内存盘，运行前检查余量；临时文件不是永久样本缓存。16k位置、4轮、2048隐藏维度、BF16的状态约256 MiB／回答；两卡合计256条、每条均达到prompt＋回答上限时，临时状态峰值约68 GiB，主要占主机RAM。可以将该参数改为其他有足够空间的临时卷。HF进程只保留初始教师的冻结LM head，用它重建完整分布，不再分配完整HF教师骨干。

sleep释放vLLM权重和KV的GPU占用，保留引擎、编译图和少量运行时显存。HF状态在CPU保留，optimizer参数对象与动量不会重建；每次训练阶段仅重建DDP梯度桶。两个vLLM引擎不会同时处于活动推理阶段。没有每轮从磁盘重新初始化模型或重编译引擎。

所有vLLM入口均省略max_num_seqs。训练一次提交本更新的全部rank请求，评测一次提交每卡完整分片，由框架连续调度。训练不提前生成下一更新的回答。HF的expandable_segments设置仅在vLLM actor子进程隔离。

## 参数与启动

- 模型revision：`3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`；K4、N2、seed42。
- `CPU_THREADS=16`：每个HF／vLLM进程的CPU算子线程数，可在脚本开头调整；GPU rank数量仍由GPUS决定。JS／KL保持GPU计算，未增加模型副本或数据worker；旧的4线程计时不代表新设置的速度。
- 公开DAPO固定数据版本；默认`TRAIN_EXAMPLES=all`，当前17,391题源排除17道超长题和1道AIME原文重合题，得到17,309训练／64题DAPO备用开发，互斥。seed42打乱后，训练取前面全部题、开发取末尾64题；清理后的新划分独立保存，旧划分保留。
- 对齐LoopOPD Table7：每更新32题×8回答＝全局256；目标100次更新。两卡各128条，一次提交本rank全部vLLM请求；每卡微批1条回放，DDP累积到全局256条才更新一次。100步合计25,600条回答、3,200次题目抽取，不是完整epoch。
- 本项目每批均匀无放回抽32题，批间独立、允许重复题，数据RNG可恢复。作者跨批采样细节未公开，不宣称采样器完全一致。不按正确性过滤教师题目或学生回答。
- LR1e-6、WD0.01、AdamW betas=(0.9,0.999)、eps=1e-8；warmup0，无梯度裁剪；FP32主参数、BF16计算、梯度检查点、readout chunk128。
- prompt最多1024，训练／评测回答上限16384；训练T1/top-p1，评测T1/top-p0.7。默认先评AIME2024的30题，每题16次，每模型预算480条回答；2025／2026可选扩展；分年份报告Avg@16和Pass@16。来源、采样和判分见[评测协议](17_aime_evaluation.md)。
- 每更新保存模型、optimizer及各rank RNG，完整后原子写checkpoint.complete.json；滚动保留1份及最终快照。
- MAX_RUNTIME_HOURS=10限制整个panel，包括编译、生成、训练、保存及评测；到时停止本次进程树，不擅自降低长度上限。100更新是目标，不能保证在10小时内完成；旧batch16计时不用于直接推断新batch256的完成时间。

对齐范围是批量32×8、全局256、100更新目标，以及三年AIME各30题×16次的评测规模、T1／top-p0.7。仍保留用户指定的16,384长度上限；论文为31,744上限、4张H200，本地为2张RTX PRO 5000。当前使用boxed整数答案判分，尚未接入论文所述的通用符号等价验证；公开数据artifact身份未与作者内部逐项核对，因此当前配置仍不是完整论文复现。冻结教师LoopOPD控制用warmup0，D-LoopOPD/current教师参考用warmup10。

```bash
# 当前两组计划，不加载模型或启动实验。
ROUTE=vllm GPUS=0,1 bash scripts/run_three_route_ablation.sh --dry-run

# 目标：两组各100更新、每步32×8，最后评AIME2024的30题×16次；总上限10小时。
ROUTE=vllm GPUS=0,1 UPDATES=100 PROMPTS_PER_BATCH=32 RESPONSES_PER_PROMPT=8 \
  EVAL_INITIAL=0 MAX_RUNTIME_HOURS=10 bash scripts/run_three_route_ablation.sh

# HF教师回退；仍可按阶段卸载。
ROUTE=vllm TEACHER_BACKEND=hf PHASE_OFFLOAD=1 bash scripts/run_three_route_ablation.sh --dry-run

# 复现旧的固定教师终点选点定义。
ROUTE=vllm TEACHER_ENDPOINT=fixed SHALLOW_TARGET_PLACEMENT=last \
  bash scripts/run_three_route_ablation.sh --dry-run

# 均匀阶段仅显式加入。
ROUTE=vllm ABLATION_GROUPS=uniform_n bash scripts/run_three_route_ablation.sh --dry-run

# 另外两条线。
ROUTE=hf bash scripts/run_three_route_ablation.sh --dry-run
ROUTE=hybrid EVAL_STUDENT_LOOPS=1,2,3,4 bash scripts/run_three_route_ablation.sh --dry-run
```

OUTPUT_ROOT可选择有足够checkpoint空间的卷。现有输出不会覆盖，--check为只读预检，--prepare-only只准备数据和配置。

默认`EVAL_SUITE=aime`、`EVAL_DATA_PATH=data/aime/aime2024.jsonl`、`EVAL_EXAMPLES=30`、`EVAL_SAMPLES_PER_PROBLEM=16`。旧DAPO留出评测可显式选`EVAL_SUITE=dapo_dev`，默认恢复64题×1次。AIME24评测按每卡完整分片一次提交；480条请求对应两卡各240条，由vLLM连续调度，不设置max_num_seqs。初始参考、最终快照与交叉线多个预算的回答数分别累加，均计入10小时上限。

## 历史结果与验证边界

2026-11-17存储清理已删除旧实验原始输出与checkpoint，保留当前GPU6、7运行及轻量工程记录。下述旧分数是历史文字与审计摘要，不再提供已删除的原始输出，不作为当前方案的正式结果。正式baseline范围限定为LoopOPD 4→2，见[正式计划](18_formal_experiment_plan.md)。

2026-11-17已完成的首轮属于旧版本：教师固定K终点、HF教师常驻、每卡2条vLLM并发及评测外部分批；三组各10更新，64题成绩为endpoint30／64、uniform_n27／64、top_n30／64，总墙钟约8小时58分。这些仅为旧版历史摘要，不当作当前gate＋vLLM教师方案的成绩。

工程检查位于audit/three_routes_20261117/。默认调度检查包含8／32条实际请求和分块前缀；教师新路径完成真实双卡两次短更新，验证教师vLLM4轮、学生vLLM2轮、完整词表、有限梯度、版本同步与CPU卸载／恢复。CPU测试验证缓存状态读出与直接HF读出得到同样的损失／梯度、gate浅终点遮罩、分块拼接及重算计数。

HF与vLLM的BF16算子、参数存储与舍入不同，不宣称分布逐项相同。真实教师后端数值与选段一致性另行记录；短诊断不证明任务效果或长回答速度提升。HF机制线可能因精确历史补算而慢；交叉线需要报告同checkpoint的多个固定N预算曲线。
