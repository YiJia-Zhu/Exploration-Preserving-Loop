# 当前实现、默认配置与LoopOPD对照

以[主方案](loop_distillation_research_proposal_zh.md)为准。当前默认是JS阈值驱动的动态学生，旧固定轮数Top-S保留为F。研究方案、工程检查和任务效果分别记录。

## 1. 默认配置与关键约定

`configs/default.json` 继承 [e_dynamic_js.json](../configs/e_dynamic_js.json)，默认Ouro、冻结教师。训练、评测和正确性筛选入口不传 `--config` 时都读取它。未准备数据时，正式训练明确报错，不会静默进入小模型模式。

| 参数 | 默认Ouro | Huginn验证 |
|---|---|---|
| 教师最大上限K | 4 | 32 |
| 学生固定轮数 | 无，`student_loops=null` | 无 |
| 学生最大上限 | K | K |
| 教师出口 | 原生gate累计概率，阈值0.9 | 原生KL，显式5e-4 |
| JS阶段阈值 | 0.05 nats，温度1 | 同为开发起点，需分别校准 |
| 学生停止 | 本地P(stop)≥0.5 | 同左 |
| 停止BCE系数β | 1 | 1 |
| 学生退出头 | 复用已有gate，不重置 | 新增线性头，weight=0、bias=−2 |
| 学生rollout缓存 | 关闭，逐前缀重算 | 同左 |
| 教师参数更新 | 默认frozen；可选current | 同左 |

JS阈值和β是明确的未校准选择，不是LoopOPD原文参数。第一批也直接使用学生停止策略，没有额外原生策略warmup；主方案允许原生warm start，但不要求它。

## 2. 代码怎样对应主方案

| 步骤 | 入口 | 行为 |
|---|---|---|
| 阈值阶段数 | [selection.py](../ep_loop/selection.py) `make_dynamic_stage_plan` | 保留所有JS>τ连接，不Top-S，不用旧n_t≤S回退；第j段末监督学生第j轮 |
| 继续／停止标签 | 同文件 `stage_stop_bce` | j<S_t继续、j=S_t停止；其后轮次不进入BCE |
| 联合目标 | [objective.py](../ep_loop/objective.py) | 完整词表reverse KL＋βBCE；教师、分段、S_t停止梯度 |
| 真正逐轮停止 | [ouro.py](../ep_loop/ouro.py) / [huginn.py](../ep_loop/huginn.py) | 根据学生头本地概率提前结束，无教师标签参与部署 |
| 生成与成本 | [inference.py](../ep_loop/inference.py) | 保存所选退出、真实执行深度、实际前向token-loops与生成耗时 |
| 训练与恢复 | [training.py](../ep_loop/training.py) | 全参数AdamW、fresh on-policy生成、全局batch后更新、DDP、optimizer/RNG和完整退出头保存 |

每个token先对有效阶段KL平均；BCE对1至S_t平均；随后对响应token平均，再对回答平均。阶段多的token不因目标数更多而自动获得更大总权重。

C/D/E都用同一阈值规则生成S_t。C只在学生第S_t轮匹配教师终点，BCE仍覆盖1至S_t；D把相同阶段数均匀映射到教师末轮；E保留真实JS段末。教师在各自当前学生前缀上重新计算，不能跨训练组缓存统一S_t标签。

训练先生成教师标签，再将完整学生前缀重放到该响应的 `max_t S_t`。因此即使rollout过早退出，监督所需轮次仍会补足；部分位置也会因整段向量化重放而多算。这些前向开销与 `max(S_t−S_hat_t,0)` 的必要补算量分别记账，不把后者当作完整训练成本。

## 3. 缓存、读出与真实计算

Ouro教师保留官方先计算全部K轮、再按CDF选出口的实现。教师早读出不代表真正少执行。学生复用gate参数，但停止规则是本地概率≥0.5，与教师CDF不同。

第一版动态学生不近似填补未运行循环的KV，而是对当前完整文本前缀逐轮重算并停止。这给出明确、可核对的条件分布；代价是每次重复处理既有前缀。`executed_token_loops`统计输入位置数×真实执行轮数，不只统计生成token的退出深度。它不含反向、checkpoint重算和LM head；教师coda另记token-positions，不能直接当作完整FLOPs。

原始LoopOPD保留缓存。除原始B外，使用 [loopopd_recompute.json](../configs/loopopd_recompute.json) 或 [huginn_loopopd_recompute.json](../configs/huginn_loopopd_recompute.json) 在相同重算条件下比较。精度—轮数与精度—耗时均需报告；降低平均轮数不自动建立加速。

Huginn教师沿用batch size=1、原生whole-prompt prefill和独立latest-m4 KV。KL首轮参考均匀分布，后续为KL(q_previous||q_current)。每轮预测经过norm→完整coda→norm→LM head，退出检查产生的分布同时用于JS。学生线性头读取归一化循环状态，动态rollout采用完整前缀重算；不与教师共享缓存。

Huginn保留原生随机初态，teacher/student独立抽样。本实现全学生回放轮次反传，未声称复现BoD的最后至多八轮截断。选择官方huginn-0125作为本项目基模，仍未确认与BoD使用相同checkpoint，见[论文笔记](03_paper_notes.md)。

## 4. LoopOPD对齐部分与A–F配置

原始B的入口仍为 [a_loopopd.json](../configs/a_loopopd.json)，这是文件旧名称；当前实验编号是B。D-LoopOPD为 [a_d_loopopd.json](../configs/a_d_loopopd.json)。两者分别为冻结／current教师，固定K=4、S=2，只监督终点。

保留的参考参数来自LoopOPD v2 Appendix B、Appendix C Table4、Appendix D Tables7–9：Ouro-1.4B-Thinking；DAPO17,398题；prompt上限1024；32×8=256条回答／更新；AdamW LR1e-6、WD0.01；冻结warmup0、current warmup10；100次更新；训练T=1、top-p=1、回答最多31,744；AIME评测T=1、top-p=0.7、每题16次。

seed42、betas(0.9,0.999)、eps1e-8、FP32参数／BF16前向、线性warmup后恒定LR、顺序微批和DDP是明确的实现选择。作者完整训练代码尚未取得；本项目是参考实现，未复现其表格分数。

| 组别 | Ouro配置 | Huginn配置 |
|---|---|---|
| A：原生基模推理 | [native_base.json](../configs/native_base.json) | [配置](../configs/huginn_native_base.json) |
| B：LoopOPD | [a_loopopd.json](../configs/a_loopopd.json) | [配置](../configs/huginn_a_loopopd.json) |
| C：动态终点 | [c_dynamic_endpoint.json](../configs/c_dynamic_endpoint.json) | [配置](../configs/huginn_c_dynamic_endpoint.json) |
| D：动态均匀 | [d_dynamic_uniform.json](../configs/d_dynamic_uniform.json) | [配置](../configs/huginn_d_dynamic_uniform.json) |
| E：主方法 | [e_dynamic_js.json](../configs/e_dynamic_js.json) | [配置](../configs/huginn_e_dynamic_js.json) |
| F：固定Top-(S−1) | [f_fixed_top_s.json](../configs/f_fixed_top_s.json) | [配置](../configs/huginn_f_fixed_top_s.json) |

C/D/E的current教师版使用同名 `_current.json`。正式Huginn数学验证继承优化设置，但prompt+生成受其4096上下文限制，为1024+3072，不冒称原论文Huginn复现。两套[当前矩阵](../experiments/experiment_matrix.csv)、[Huginn矩阵](../experiments/huginn_experiment_matrix.csv)按新编号记录，旧矩阵在experiments/archive保留。

## 5. 启动与恢复

```bash
# 默认主方法，先只看配置和资产。
.venv/bin/python scripts/train_stage_distillation.py --show-config
.venv/bin/python scripts/train_stage_distillation.py --check-assets

# 随机小模型完整过程检查。
CUDA_VISIBLE_DEVICES='' .venv/bin/python scripts/train_stage_distillation.py \
  --smoke --device cpu --output experiments/runs/my_dynamic_smoke

# 正式四卡主方法。GPU编号按可用卡修改。
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=4 scripts/train_stage_distillation.py \
  --device cuda --output experiments/runs/my_dynamic_student

# 正式LoopOPD baseline。
CUDA_VISIBLE_DEVICES=0,1,2,3 .venv/bin/python -m torch.distributed.run \
  --standalone --nproc-per-node=4 scripts/train_stage_distillation.py \
  --config configs/a_loopopd.json --device cuda \
  --output experiments/runs/my_loopopd
```

恢复加 `--resume <checkpoint目录>`，必须使用同一配置和world size。保存参数、退出头、optimizer、各rank的数据／采样／Torch RNG。输出已有更晚更新时拒绝从旧检查点混写历史，需新输出目录。Huginn新增头和原有tied embedding都支持单文件／分片；动态checkpoint应通过本项目loader加载，官方原始类不知道新增学生头。

## 6. 数据、筛选与评测

正式 `data/dapo_math17k/train.jsonl` 尚缺。用已确认来源运行 `prepare_stage_data.py`，默认期望17,398题；保留原消息、ID和内容哈希，源发布／revision也记录。不自动使用其他项目的17,162题子集。

可选 `build_teacher_filter.py` 使用冻结初始教师原生动态作答一次，只缓存正确性接受标签；当前前缀上的n_t与阶段必须重算。现有boxed整数验证器不能替代复杂符号或EvalPlus官方验证。

```bash
# 动态学生部署评测：不传 --loops，使用自己学到的停止头。
.venv/bin/python scripts/evaluate_stage_distillation.py \
  --checkpoint experiments/runs/my_dynamic_student/checkpoints/update_000100 \
  --data /path/to/aime2024.jsonl --device cuda \
  --output experiments/runs/my_dynamic_student/aime2024.jsonl

# 原生基模A，不需要训练checkpoint。
.venv/bin/python scripts/evaluate_stage_distillation.py \
  --config configs/native_base.json --data /path/to/aime2024.jsonl \
  --device cuda --output experiments/runs/native_aime2024.jsonl

# LoopOPD B：显式原始配置，默认固定2/4轮评测。
.venv/bin/python scripts/evaluate_stage_distillation.py \
  --config configs/a_loopopd.json \
  --checkpoint experiments/runs/my_loopopd/checkpoints/update_000100 \
  --data /path/to/aime2024.jsonl --device cuda \
  --output experiments/runs/my_loopopd/aime2024.jsonl
```

`--loops N`明确切换为固定轮数评测，用于预算曲线；不会误把动态学生默认评成固定K。输出Avg@N、Pass@N、所选出口和真实执行深度直方图、实际平均深度／分位数、真实前向量和生成耗时。Ouro原生A所选出口可能浅于实际K，两套量分开保存。`--smoke --device cpu`可检查已保存随机小模型的评测流程，不作为基准结果。

## 7. 工程证据与尚未完成的事

当前核验记录在[动态学生报告](../audit/dynamic_student_20261113/validation.json)。覆盖阈值全部边界、无旧浅轮回退、C/D/E同S_t标签、停止BCE掩码、真实逐轮停止、监督补算、教师梯度隔离、chunk／backbone checkpoint梯度、保存／恢复及CPU训练。旧[固定预算核验](../audit/proposal_sync_20261113/validation.json)作为历史保留。

测试和CPU随机训练只能证明执行语义。真实基模的短CPU前向用于核对接口，不训练预训练参数。正式预训练模型训练、阈值校准、多seed完整测试、精度—实际成本曲线尚未完成，不宣称关键探索或性能收益已被证明。
