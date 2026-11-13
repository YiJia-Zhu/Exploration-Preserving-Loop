# 代码实现与 LoopOPD 复现对应

更新：2026-11-13。权威方案为[原生动态退出与 JS 阶段蒸馏](loop_distillation_research_proposal_zh.md)。当前已包含 Ouro 主实现与 Huginn 原生验证后端；完整论文成绩和当前方法的有效性尚未验证。

## 1. 当前实现对应什么

公开入口 [LoopOPD-page](https://github.com/wanglne/LoopOPD-page) 的 Code 仍标注 Coming soon。本项目根据固定版本 [2610.10623v2](https://arxiv.org/abs/2610.10623v2) 的 Eq.9/11、Appendix B Algorithms 1/2 和 Appendix D Tables 7–9 实现，不称为已取得作者训练代码或已复现其分数。

| 部分 | 实现入口 | 行为 |
|---|---|---|
| 原生模型接口 | [ouro.py](../ep_loop/ouro.py) | 读取共享目录中的官方源码，保留循环末 norm、gate 和独立 KV；源码不改写 |
| Huginn 原生接口 | [huginn.py](../ep_loop/huginn.py) / [后端选择](../ep_loop/backends.py) | 完整 coda、原生 KL evaluator、随机初态、独立 latest-m4 与实际提前退出 |
| 出口和分段 | [selection.py](../ep_loop/selection.py) | 官方累计退出概率、标准 JS、最大边界、段末与有序映射 |
| 损失 | [objective.py](../ep_loop/objective.py) | 完整词表 reverse KL；分阶段／回答长度归一化；teacher/selection stop-grad |
| 训练 | [training.py](../ep_loop/training.py) | 全参数 AdamW、冻结／当前教师、fresh on-policy rollout、全局 batch、DDP、保存与恢复 |
| 生成／验证 | [inference.py](../ep_loop/inference.py) | 固定深度学生或原生出口教师生成、独立 cache、boxed 整数验证 |
| 数据 | [data.py](../ep_loop/data.py) / [准备脚本](../scripts/prepare_stage_data.py) | 保留源消息、题目哈希、长度检查和可选结果筛选 |

旧 `runtime.py`、`run_training_pilot.py` 继续支持11月8日历史实验；新主入口为 `train_stage_distillation.py`，主损失不再使用旧的候选 forward KL 或 log-probability 增量项。

## 2. 原文参数与明确的实现选择

| 项目 | 本次正式配置 | 原文依据／状态 |
|---|---|---|
| 模型 | Ouro-1.4B-Thinking，revision `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d` | 原文用 1.4B / 2.6B Thinking；本地已有 1.4B |
| 学生／教师深度 | S=2，K=4 | §4、Table 7 |
| 数据 | DAPO-Math-17k，期望 17,398 题 | Table 8；精确发布／子集仍需落实 |
| prompt 上限 | 1,024 token | Table 7；过长题丢弃是本实现选择 |
| 每轮 prompt／回答 | 32 × 8 = 256 | Algorithms 1/2、Table 7 |
| 参数更新 | 完整模型，AdamW | Table 7；新主入口不使用 LoRA |
| LR／weight decay | 1e-6／0.01 | Table 7 |
| warmup | 冻结版 0，当前版 10 步 | Table 7；10 步线性后恒定是实现选择 |
| 更新次数 | 100 | 参考 Table 4 报告至第100步的验证曲线 |
| 训练解码 | T=1.0，top-p=1.0，最多31,744 token | Table 7 |
| AIME 验证 | T=1.0，top-p=0.7，每题16次，最多31,744 token | Tables 7/9 |
| 原文硬件 | 4×H200 | Appendix D；本次未运行这一规模 |
| 种子／betas／eps | 42／(0.9,0.999)／1e-8 | 原文未充分指定，明确采用的选择 |
| 精度与微批 | FP32 参数/optimizer，BF16 CUDA 前向；每个微批一条回答 | 实现选择，不能宣称与作者精度／并行完全相同 |
| 原生阈值／JS阈值 | 0.9／0.01 nats | 新方法开发起点，尚未校准，不是 LoopOPD 的报告参数 |

配置在[公共基配置](../configs/paper_ouro_base.json)与[实验矩阵](../experiments/experiment_matrix.csv)。四卡时每 rank 8 个 prompt、每题8条回答，逐条反向累积，完成256条后才更新一次；微批数不等于多次改变策略。所有当前教师目标在这一全局更新之前使用同一套参数。

本实现为清晰可审查的 DDP 参考引擎，未声称具备作者未公开训练栈的吞吐、FSDP内存或vLLM调度。31K 长序列的实际显存／速度需要在目标硬件验证。全参数 checkpoint 和 AdamW 状态也有明显磁盘成本。

## 3. A/B/C/D 配置

| 组别 | 冻结教师配置 | 当前共享教师配置 |
|---|---|---|
| A：固定四轮终点 | [a_loopopd.json](../configs/a_loopopd.json) | [a_d_loopopd.json](../configs/a_d_loopopd.json) |
| B：原生出口终点 | [b_native_endpoint.json](../configs/b_native_endpoint.json) | [b_current_native_endpoint.json](../configs/b_current_native_endpoint.json) |
| C：原生出口＋数量匹配均匀段末 | [c_native_uniform.json](../configs/c_native_uniform.json) | [c_current_native_uniform.json](../configs/c_current_native_uniform.json) |
| D：原生出口＋JS段末 | [d_native_js.json](../configs/d_native_js.json) | [d_current_native_js.json](../configs/d_current_native_js.json) |

原生退出按每个 token 计算，标准 JS 仅在前 n_t 轮内筛边界，最多 S−1 个。无转折或 n_t<=S 仅终点。M 个阶段的第 j 个目标映射到学生 `ceil(j*S/M)`，各阶段等权；随后按回答长度平均，再按回答平均。C 在同一前缀上使用与 D 相同的阶段数量，位置采用均匀段末。

Ouro 教师在 native 模式仍执行 K 轮；日志分别记录所选 n_t 与实际前向 token-loops。Huginn 按原生 KL 停止，动态模式的 `teacher_executed_loops` 为 null，上限另记为 `teacher_max_loops`，真实前向量单独累计。阶段监督不会偷用学生 KV，也不让教师跳过形成段末状态的计算。Ouro A/B 只读监督终点，不为日志强制增加整条词表 JS 读出。Huginn 原生 KL 本来就要运行 coda；D 复用这些读出得到相邻 JS，不重复运行 coda。

## 4. 本次已执行的验证

核验记录见[实现验证报告](../audit/implementation_20261113/validation.json)，单元测试日志见[unit_tests.log](../audit/implementation_20261113/unit_tests.log)。

- JS 的定义、范围、对称性、最大边界、退出之后边界屏蔽、n_t<=S 回退、目标数匹配及非整除学生映射。
- 累计退出选出的读出与官方 Ouro `forward` 一致；固定深度 cache 解码与独立完整前缀重算一致。
- stop-grad、全词表 reverse KL、chunk checkpoint 与未分块损失／梯度一致，原生梯度检查点兼容。
- 冻结教师保持初始状态，当前教师随参数更新改变。原有 Ouro 核验及本次 Huginn 检查合计32项语义测试通过，见[最新日志](../audit/proposal_sync_20261113/unit_tests.log)。
- 官方随机微型架构上的 A/B/C/D 冻结／当前教师完整训练检查；另有双进程 CPU DDP 和 checkpoint/RNG/optimizer 恢复检查。

随机微型模型只有2个物理层、32维隐藏状态和260词表，CPU 检查将 batch、长度、LR 等显式改小，并把 JS 阈值设为0以覆盖阶段路径；`summary.json` 标记 `smoke=true`。这些结果不构成1.4B真实权重的性能证据。

确定性 Ouro 的当前共享教师还需关注同深度自对齐：同一前缀上 T1→S1 或 T2→S2 理论上没有额外教师信息，不能把这些项算作有效过程迁移。Huginn 教师和学生采用独立的原生随机初态，同深度分布也可能不同；这类差异不能直接归为深度提供的新信息。现有 `teacher_depth_by_student_slot` 日志用于统计这种映射；当前按计划保留其映射及归一化，解释 C/D 差异时要检查其覆盖和梯度尺度。

## 5. 数据准备与资产预检

```bash
# 以下仅检查本地模型 revision 元数据和数据文件，不加载权重或申请GPU。
.venv/bin/python scripts/train_stage_distillation.py \
  --config configs/a_d_loopopd.json --check-assets

# SOURCE 应是已确认的 DAPO 发布文件；默认保留原始消息。
.venv/bin/python scripts/prepare_stage_data.py \
  --input /path/to/confirmed-dapo.parquet \
  --output data/dapo_math17k/train.jsonl --expected-examples 17398 \
  --source-repo BytedTsinghua-SIA/DAPO-Math-17k --source-revision EXACT_REVISION
```

`--deduplicate` 必须显式开启，重复问题答案冲突会报错。题数、源文件哈希和题目ID会保存；题数相同仍不能单独证明完全等于作者子集。真实训练会在加载模型前检查资产，不把缺数据的命令静默转成玩具运行。

当前期望的训练 JSONL 尚未准备，资产检查报告见[assets_preflight.json](../audit/implementation_20261113/assets_preflight.json)。其他项目处理后的数据不能直接替代17,398题的论文口径。开发阈值和最终测试还需独立划分，不使用测试题校准 0.9／0.01。

## 6. 运行与恢复

```bash
# CPU完整过程检查：使用新的输出目录，已有逐题日志拒绝覆盖。
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/train_stage_distillation.py \
  --config configs/d_current_native_js.json --smoke --device cpu \
  --output experiments/runs/my_js_smoke

# 正式四卡示例；先用nvidia-smi选择实际可用的GPU，且先通过资产检查。
STAGE_GPUS=0,1,2,3
CUDA_VISIBLE_DEVICES="$STAGE_GPUS" .venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=4 scripts/train_stage_distillation.py \
  --config configs/a_d_loopopd.json --device cuda \
  --output experiments/runs/my_d_loopopd

# 同一参数刷新策略的完整方案；其它设置继承相同配置。
CUDA_VISIBLE_DEVICES="$STAGE_GPUS" .venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=4 scripts/train_stage_distillation.py \
  --config configs/d_current_native_js.json --device cuda \
  --output experiments/runs/my_current_js
```

checkpoint 默认每20步及最后一步保存。恢复时同一配置与 world size，加 `--resume <checkpoint目录>`；保存完整参数、optimizer、每个rank采样与数据RNG。若原日志已有更新比所选 checkpoint 更新，必须使用新输出目录，避免混合两条训练历史。`--stop-after-updates` 可有意提前结束当前执行而保留原计划总步数。

## 7. 可选结果筛选与评测

```bash
# 冻结初始教师按原生动态策略每题完整作答一次；不搜索逐深度解答。
.venv/bin/python scripts/build_teacher_filter.py \
  --config configs/d_native_js.json --device cuda \
  --output data/dapo_math17k/teacher_outcomes.jsonl

# 固定深度数学评测，默认沿用配置中的2/4轮和16次采样。
.venv/bin/python scripts/evaluate_stage_distillation.py \
  --config configs/a_d_loopopd.json \
  --checkpoint experiments/runs/my_d_loopopd/checkpoints/update_000100 \
  --data /path/to/aime2024.jsonl --device cuda \
  --output experiments/runs/my_d_loopopd/aime2024.jsonl
```

使用筛选时，把所有组的 `data.correctness_cache` 指向同一文件。缓存核对固定初始模型 revision、原生退出／解码设置及内容哈希，只读取接受标签；OPD 的退出与JS仍在当前学生前缀重算。当前简单验证器仅接受最终 boxed 整数，复杂表达式不等价于论文的符号验证，需替换正式验证器并另记版本。

评测输出 Avg@N 与 Pass@N 两个量。OlympiadBench/BeyondAIME 按原文使用 `--samples-per-problem 1`；非整数 ground truth、MMLU 与 EvalPlus 需外部正式评测器。没有执行这些基准，就不填写论文分数。

## 8. Huginn 原生后端与验证配置

选择官方 `tomg-group-umd/huginn-0125`，固定 revision `bb6621b65e90b6a4b9b29ef88dc83866d450470c`。共享目录为 `/home/zhenning/data/huggingface_models/huginn-0125`，项目使用同名软链接，官方源码与权重保持原样。BoD 只写明 Huginn 3.565B，没有指定仓库或预训练 checkpoint；不能把本选择说成已确认与其权重相同，详见[论文笔记](03_paper_notes.md#6-best-of-depth-与-huginn-checkpoint-的核对范围)。

Huginn 固定教师 K=32、学生 S=8，另有 S=16 变体。数学数据、AdamW、batch 和参数刷新沿用 Ouro 对照；这是本项目适配配置，不能声称来自 LoopOPD 的 Huginn 实验。基模原生上下文上限4096，因此 prompt最多1024、生成最多3072。参考[基配置](../configs/huginn_math_base.json)与[八组矩阵](../experiments/huginn_experiment_matrix.csv)。

| 组别 | 冻结教师 | 当前共享教师 |
|---|---|---|
| A：固定终点 | [配置](../configs/huginn_a_loopopd.json) | [配置](../configs/huginn_a_d_loopopd.json) |
| B：原生退出终点 | [配置](../configs/huginn_b_native_endpoint.json) | [配置](../configs/huginn_b_current_native_endpoint.json) |
| C：数量匹配均匀段末 | [配置](../configs/huginn_c_native_uniform.json) | [配置](../configs/huginn_c_current_native_uniform.json) |
| D：JS段末 | [配置](../configs/huginn_d_native_js.json) | [配置](../configs/huginn_d_current_native_js.json) |

[S=16 的 D/current 配置](../configs/huginn_d_current_native_js_s16.json)单独比较学生预算；不与 S=8 混作同预算对照。

原生执行细节：

1. 教师整体预填 prompt，再按顺序读取学生已经生成的 token；batch size=1，使用独立 `HuginnDynamicCache(lookup_strategy="latest-m4")`。这与官方 `generate_with_adaptive_compute` 的整体 prompt 预填方式一致，不把教师先前未执行的深轮 KV 补算出来。
2. 每轮末端执行 norm→完整 coda→norm→LM head。直接对 recurrent latent 接 LM head 会遗漏 coda 的计算，当前后端没有这样简化。
3. 生成同时沿用官方 end_text/end_turn 等停止 token；不只检查 config 中单个 EOS。直接使用官方 `KLExitEvaluator`：首轮与均匀分布比较，后续为 `KL(q_previous || q_current)`；严格小于阈值退出，未满足则到 K。显式使用5e-4，不依赖 `auto=1e-3`。
4. JS来自退出检查时已有的词表分布，复用到共用阶段选择器。退出后的填充值不参与边界选择或监督；记录真实循环量及 `teacher_coda_token_positions`。这些量只计前向，不含 checkpoint 重算、反向和输出 head 成本，不直接等价于 FLOPs 或延迟。
5. 学生始终完整执行固定 S 轮，保留原生随机初态；本实现使用全 S 轮反传与非重入检查点，没有照搬 BoD 的最后至多八轮截断策略。每条回答的 teacher/student 初态独立抽样，损失对应抽样后的分布，未对初态随机性做精确边缘化。冻结教师固定的是参数，初态仍按原生规则采样。
6. HF保存会省去 tied embedding 的重复张量。恢复同时支持单文件和分片，只允许已加载共享参数的合法别名缺失，其余缺失／多余权重仍报错。

```bash
# 先解析或检查资产，不加载模型。
.venv/bin/python scripts/train_stage_distillation.py \
  --config configs/huginn_d_current_native_js.json --check-assets

# 随机小模型训练检查：smoke明确将K/S改为4/2，并缩小batch、长度。
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/train_stage_distillation.py \
  --config configs/huginn_d_current_native_js.json --smoke --device cpu \
  --output experiments/runs/my_huginn_js_smoke
```

新核验见[本次报告](../audit/proposal_sync_20261113/validation.json)：Ouro/Huginn各八组、每组两次完整CPU更新；Huginn冻结／当前教师保存恢复后76个权重张量、采样与指标完全一致；双进程CPU训练及backbone/readout同时检查点的梯度归约通过。另以真实预训练 Huginn 权重在CPU做了两轮短前向，输出与官方固定深度完全一致，原生退出前向也通过，见[真实权重检查](../audit/proposal_sync_20261113/pretrained_huginn.json)。这些检查没有训练预训练基模，也没有执行代码生成或数学基准，不能作为方法有效性结果。

