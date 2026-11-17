# verl OPD与循环模型兼容性调查

调查日期：2026-11-17。结论：Ouro固定深度OPD具备改造接入的基础；当前gate＋JS跨轮监督需要自定义教师接口和学生训练路径，不能只替换模型名称；Huginn需要更多架构适配。verl可作为后续规模化底座，目前没有证据表明整体迁移会自动加快我们这套双卡实验。

本次检查当前上游源码并做CPU小模型接口探测，尚未安装或运行verl端到端训练。调查没有修改当前训练环境、共享模型或已有实验。

## 版本范围

- verl主仓库：`07490cb5ce1fa1818e0d98540abc9bccd02534ac`。
- verl-recipe独立仓库HEAD：`acc239fc978bf001e1957b7148b36e69df63cc19`；另查看了旧GKD异步蒸馏说明，不能把它与主仓库当前OPD混为一套实现。
- 当前项目环境：Torch2.8.0、Transformers4.57.1、vLLM0.11.0；未安装verl、tensordict或flash-attn独立包。
- 当前verl的uv推荐组合为Torch2.13.0、vLLM0.32.0、Transformers5.16.1。setup.py的约束较宽，但Transformers>=5.5.3、vLLM>=0.18也高于当前项目版本。迁移应使用独立环境并固定版本。

## 接入判断

| 目标 | 判断 | 主要工作 |
|---|---|---|
| Ouro固定N学生、固定K教师，使用verl原有OPD目标 | 有接入基础，非现成验证支持 | HF模型装载、固定轮次、vLLM适配、权重同步、FSDP共享循环反传检查 |
| 精确复现我们的LoopOPD式完整词表reverse KL终点对照 | 需要扩展损失与教师返回协议 | 原有采样式KL／top-k前向KL不能直接替代 |
| Ouro固定计算、gate确定逻辑终点、JS选中间阶段 | 需要模型与训练引擎扩展 | 逐轮状态／gate传输、每token阶段映射、学生中间轮次可微读出、完整分布损失 |
| HF逐token真实动态停止路线 | 需要自定义rollout执行器 | 真实停算、独立深度KV与历史补算不能由标准vLLM rollout自动实现 |
| Huginn | 改造量更大 | 原生循环梯度语义、prelude/core/coda、随机初始化、专用KV及batch/packing支持 |

循环结构本身并不排斥FSDP。Ouro有标准HF CausalLM接口和`_no_split_modules=[OuroDecoderLayer]`；Huginn有HF封装与`SandwichBlock`声明。FSDP/FSDP2比从Megatron模型转换路径起步更合适，但共享模块跨轮重复调用、检查点重算和导出权重仍须真实分布式验证。

## 已有OPD与我们目标的区别

主仓库已实现学生rollout→教师条件logprob→学生更新，支持冻结教师、CPU卸载、vLLM sleep/wake、内存内权重同步、动态microbatch与负载均衡；不应再说verl没有OPD。

当前loss registry分两类：

- `forward_kl_topk`／`forward_kl_topk_tail`：教师top-k前向KL或top-k＋聚合尾桶的前向KL。
- `k1`／`k2`／`k3`等：使用已采样token的师生logprob构成估计量，部分模式结合policy gradient。

我们当前目标是在固定学生文本前缀上，直接对完整词表计算`KL(p_student || q_teacher)`，并可同时监督学生多个循环读出。方向、概率支持集合、梯度路径和阶段维度都必须保留。把配置换成k1、k3或top-k前向KL，会改变方法，不能当作原方案的纯工程加速。LossSettings还限制top-k与estimator恰选其一，因此增加full-vocab路径需要连同接口和引擎一起扩展，不能只注册一个函数名。

标准教师接口调用prompt_logprobs，对已知学生序列prefill，并生成一个丢弃的dummy token；返回每位置一个采样token或top-k集合的概率。它不提供所有循环的状态／gate或阶段维度。我们的vLLM教师读出和压缩状态传输可以迁入自定义worker，但不是标准OPD API直接支持。

## Ouro的具体注意点

1. HF训练可通过trust_remote_code和AutoModel加载。固定N必须实际修改`total_ut_steps=N`；若走原始CausalLM输出接口，还应显式固定`early_exit_step=N-1`（零基索引），保证读第N轮。只改出口读出而仍计算K轮不符合固定N预算。
2. HF训练配置与vLLM引擎配置分别核验。当前verl服务使用model_path启动，不能假定HF的override_config自动成为推理引擎的hf_overrides；需保证HF回放和vLLM rollout都实际计算N轮，教师实际K轮。
3. 首个接入检查应关闭通用fused forward、remove-padding／packing及Ulysses SP，先验证原始HF前向。通用fused forward假定骨干返回普通hidden tensor；Ouro骨干实际返回`(output, loop_states, gates)`，不能直接套用。padding/packing优化随后逐项核验，避免跨样本注意力泄漏。
4. 教师model_config在当前管理器内仅由model_path构造，未显式透传actor的trust_remote_code、external_lib或轮次override。不能假定学生端设置会自动应用到自定义教师。
5. vLLM0.32的registry将Ouro列入PREVIOUSLY_SUPPORTED，最后原生支持版为0.26；0.26还有ouro.py，0.27/0.32路径已不存在。Transformers后端可能提供另一条途径，但有后端标志不等于循环KV已验证，仍需确认每个round/layer的缓存隔离。当前项目的repo内vLLM0.11适配也不等于在新版本中原生支持。

## Huginn额外的语义问题

本地huginn-0125的普通forward默认在训练态采样循环次数并截断反传。显式`num_steps=N`会进入N轮no_grad、0轮有梯度；要表示固定N轮全部反传，需要类似`num_steps=[0,N]`的明确语义。CPU小模型已验证这个区别：整数2时循环核心无梯度，[0,2]时核心有非零梯度。

其forward还使用独立coda与随机初始状态，原始路径没有真正应用传入attention_mask，position_ids逻辑也须验证多样本batch。其原生block_size是4096，不能默认沿用Ouro的16k设置。检索本次verl主仓库和相关recipe，未发现现成Ouro/Huginn专用集成或测试；这不代表泛用HF接口原则上无法接入。

## 双卡资源与on-policy边界

当前主仓库的标准OPD为教师单独创建teacher_pool，actor／rollout属于global_pool。两卡可设计为1卡学生＋1卡教师，但不是我们当前“两卡共同训练，且每卡分阶段复用师生”的布局。若要保留两卡训练又在同两卡运行教师，需要扩展资源调度和阶段sleep／offload，不能只复制默认配置。

旧`gkd/megatron` recipe提供one-step-off／two-step-off流水，但其README明确承认会牺牲严格on-policy。我们要求每次更新同步当前学生权重，生成该更新的全部样本后再更新一次；不能直接启用跨更新预生成，并将其称为完全不改逻辑的提速。每更新的mini-batch次数、ppo_epochs和loss归一化也要与当前协议对齐，任务reward与额外reference KL应关闭。

另有一项用户约束：当前verl标准vLLM server会显式传入max_num_seqs，蒸馏配置也设置该字段。若移植进本项目，应去掉这项传参，继续让vLLM使用框架默认值。

## 建议路线

短期保留当前已验证的实现，先测默认调度、阶段卸载后的真实长回答吞吐。之前的主要耗时在vLLM生成，换训练框架并不会自动加快相同引擎的相同采样工作。

如后续需要更大batch、多卡训练或更完善的资源管理，再做独立verl适配分支：

1. 固定版本与独立环境，先用Ouro固定N完成rollout→同步→HF FSDP更新的小闭环；核对实际轮数和循环核心梯度。
2. 接入我们已有的完整词表终点reverse KL，比较同一批文本的损失与梯度，暂不启用packing或异步跨更新流水。
3. 加入教师全轮状态、原生CDF与JS映射，再对齐浅终点的prefix／last策略与各层梯度。
4. 最后接双卡共享教师资源、长度均衡、packing和更快权重传输，分别报告采样／教师／更新／通信耗时。

本次CPU接口探测只证明官方小模型的普通HF前向及Huginn梯度参数语义，不证明verl分布式、FSDP、vLLM或16k训练已经兼容。原始结果在`audit/verl_opd_compatibility_20261117/cpu_interface_probes.json`，源码版本和哈希在同目录source_manifest.json。

## 上游依据

- [OPD官方说明](https://github.com/verl-project/verl/blob/07490cb5ce1fa1818e0d98540abc9bccd02534ac/docs/algo/opd.md)
- [损失注册与采样KL](https://github.com/verl-project/verl/blob/07490cb5ce1fa1818e0d98540abc9bccd02534ac/verl/trainer/distillation/losses.py#L48)
- [教师prefill及返回格式](https://github.com/verl-project/verl/blob/07490cb5ce1fa1818e0d98540abc9bccd02534ac/verl/experimental/teacher_loop/teacher_manager.py#L51)
- [HF模型加载](https://github.com/verl-project/verl/blob/07490cb5ce1fa1818e0d98540abc9bccd02534ac/verl/workers/engine/fsdp/transformer_impl.py#L287)
- [通用fused forward的骨干假设](https://github.com/verl-project/verl/blob/07490cb5ce1fa1818e0d98540abc9bccd02534ac/verl/models/transformers/dense_common.py#L38)
- [师生资源池](https://github.com/verl-project/verl/blob/07490cb5ce1fa1818e0d98540abc9bccd02534ac/verl/trainer/ppo/v1/trainer_base.py#L885)
- [vLLM参数构造](https://github.com/verl-project/verl/blob/07490cb5ce1fa1818e0d98540abc9bccd02534ac/verl/workers/rollout/vllm_rollout/vllm_async_server.py#L350)
- [CPU卸载与权重同步](https://github.com/verl-project/verl/blob/07490cb5ce1fa1818e0d98540abc9bccd02534ac/verl/workers/engine_workers.py#L790)
- [推荐版本组合](https://github.com/verl-project/verl/blob/07490cb5ce1fa1818e0d98540abc9bccd02534ac/pyproject.toml#L88)
- [vLLM0.32以前支持的模型表](https://github.com/vllm-project/vllm/blob/v0.32.0/vllm/model_executor/models/registry.py#L799)
- [旧GKD异步recipe及严格on-policy限制](https://github.com/verl-project/verl-recipe/blob/acc239fc978bf001e1957b7148b36e69df63cc19/gkd/megatron/README.md)
