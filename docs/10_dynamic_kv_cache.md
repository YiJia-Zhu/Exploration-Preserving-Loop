# 动态学生的精确KV缓存

默认`EXECUTION_MODE=hf_dynamic`使用PyTorch与HF Transformers，保留完整词表监督和真实提前停止。vLLM固定K读出与top-20教师近似已作为[显式可选方案](14_vllm_top20_fastpath.md)保留，未接入SGLang。Ouro和Huginn的动态学生默认在rollout与部署评测中开启`use_cache=true`；主方案的JS标签、阶段KL、停止BCE和本地停止阈值均不变。

## 怎样避免重算

每条回答独立保存各深度的KV、中间状态，以及该深度已经计算到的前缀长度。prompt先正常预填充；下一个token只追加新位置。

例如，3个prompt位置只执行了第一轮，下一token需要4轮：第一轮只计算新token，第二至四轮分别补齐4个位置。总计`1+4+4+4=13`个token-loops。之后同深度的token只需追加自身，不再重算这4个历史位置。

补算使用上一深度已保存的真实状态；注意力使用正确的绝对位置和右下对齐因果掩码。因果模型中，历史位置不依赖未来token，因此固定权重与初态后，可按深度及层归纳验证它等价于完整前缀前向。不存在浅层KV替代深层KV，也不会跳过应有的历史计算。实现见[cache.py](../ep_loop/cache.py)、[ouro.py](../ep_loop/ouro.py)和[huginn.py](../ep_loop/huginn.py)。

每个位置在每个循环深度最多计算一次，KV和状态存储采用按需增长缓冲区。代价是保留更多中间状态和深层KV。若后续token反复需要K轮，历史位置最终也需补到K轮，总工作量可能接近固定K；平均出口降低不代表总计算同比下降。

新HF研究线`hf_adaptive`也让教师采用这一精确缓存，按原生CDF逐位置实际停算，真实执行量与教师历史补算分别记录。交叉线仍完整计算教师K轮后选逻辑终点；两者不能混称为相同计算方式。详见[三线协议](15_three_route_protocol.md)。

2026-11-15针对16,384回答长度的训练OOM，限制长缓冲区每次额外扩容至最多1,024位置，并将Ouro历史补算分成最多256位置的块，保持真实历史状态与正确因果掩码。DDP共享梯度bucket，allocator默认使用expandable segments。真实模型在已初始化Adam状态、额外梯度存储下通过17,408总位置的缓存与反向压力检查，详见[显存修复与重启记录](../audit/long_context_recovery_20261115/README.md)。该工程检查不代表训练任务收益。

## Huginn与数值约定

Huginn保留原生随机初态，每个新位置只抽样一次并随缓存保留；教师、学生、不同回答之间独立。旧完整前缀重算会重新抽取历史位置，所以同一seed下的新旧回答不要求相同。验证使用同一组原生抽样初态，比较相同条件分布，不将其改成零初态。`test_time_noise>0`的额外噪声注入暂不支持精确学生缓存。

Huginn的coda KV按循环深度分别保存，避免浅层读出污染深层读出。教师继续使用自己的原生KL停止和latest-m4缓存，学生缓存不会改变教师语义。训练阶段完整前缀回放与梯度检查点仍保留。

精确表示相同的数学计算，不保证浮点逐位相同。BF16在单token解码、批量补算与完整前缀之间会有不同舍入，可能影响接近阈值的退出和采样；测试同时记录完整词表JS、隐藏状态和logits误差。FP32用于更严格的数值核验。

## 生命周期与成本记录

缓存只属于一次`generate()`，不跨回答、教师、模型或参数更新复用。训练回放使用正常可微前向，不读取推理KV。BF16权重转换在单条回答内复用，退出时清空autocast缓存，避免脱离梯度的权重转换影响随后训练；不强制确定性GPU算子。

- `executed_token_loops`：prompt、新位置和历史深度补算的前向总量。
- `backfill_token_loops`：其中用于历史位置缺失深度的补算量。
- `execution_depths`：当前新输入的真实循环数，不把补算摊成大于K的轮数。
- `coda_token_positions`：Huginn额外的coda读出位置数。
- `cache_policy`：动态学生缓存为`exact_lazy_depth`，原生缓存为`native`，重算为`none`。

这些量不含反向、梯度检查点重算与LM head，完整效率仍看实测耗时和峰值显存。默认配置、ablation脚本、训练日志和评测输出均已接入。

## 已完成的短测

RTX PRO 5000 72GB、真实本地权重、BF16、prompt 128 tokens、续写32 tokens、K=4、种子42，预热后各3次取中位数。两条路径均复用单回答内的BF16权重转换，表内差异来自KV与状态缓存。混合深度按`1,4,2,3`循环。

| 模型／强制深度 | 完整前缀重算 | 精确缓存 | 加速比 |
|---|---|---|---|
| Ouro 1.4B，全1轮 | 0.412秒 | 0.250秒 | 1.65× |
| Ouro 1.4B，混合深度 | 0.924秒 | 0.615秒 | 1.50× |
| Huginn，全1轮 | 0.373秒 | 0.267秒 | 1.39× |
| Huginn，混合深度 | 0.745秒 | 0.530秒 | 1.40× |

表内缓存包含全部历史补算：浅层工作量从4592降为159 token-loops；混合深度从11496降为634，其中427为历史补算。内核启动等开销仍存在，耗时不会随token-loops同比下降。缓存与权重转换复用需要额外显存，混合短测峰值约为Ouro 8.16 GiB、Huginn 20.12 GiB。

结果见[Ouro BF16](../audit/dynamic_cache_20261114/pretrained_ouro_bf16_cast_cache.json)与[Huginn BF16](../audit/dynamic_cache_20261114/pretrained_huginn_bf16.json)。完整词表对照的最大JS分别约为`4.86e-4`和`4.39e-4` nats；这是BF16舍入下的检查结果，不承诺采样或退出逐位一致。另有[Ouro FP32](../audit/dynamic_cache_20261114/pretrained_ouro_fp32.json)和[Huginn FP32](../audit/dynamic_cache_20261114/pretrained_huginn_fp32.json)对照，最大JS均小于`1.5e-7` nats。

58项单元测试通过，包含Huginn从1轮补到32轮。两模型×冻结/current教师均完成CPU短训练、重复运行和optimizer/RNG断点恢复，指标与最终权重精确一致；GPU上也核对了生成前后共享模型的BF16梯度一致，见[训练检查](../audit/dynamic_cache_20261114/training_validation.json)与[梯度检查](../audit/dynamic_cache_20261114/cuda_gradient_validation.json)。保存后评测默认缓存及题目顺序变化检查通过，见[评测检查](../audit/dynamic_cache_20261114/evaluation_validation.json)。

## 启动与验证

原有训练和评测命令自动启用缓存，不需要更换启动方式。ablation开头可调整：

```bash
GPUS=7 OURS_USE_CACHE=1 bash scripts/run_ablation.sh --dry-run
# 对照旧完整前缀重算，同时控制训练rollout和评测：
GPUS=7 OURS_USE_CACHE=0 bash scripts/run_ablation.sh --dry-run
```

直接使用JSON时，分别调整`rollout.use_cache`和`evaluation.use_cache`。旧`loopopd_recompute.json`配置仍可作诊断；两组正式比较默认都保留各自缓存。

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m unittest discover -s tests -v

# 默认是随机小模型；检查深度变化和补算，不下载模型。
.venv/bin/python scripts/benchmark_dynamic_cache.py \
  --family both --device cpu --output /tmp/tiny_dynamic_cache.json

# 真实权重、相同强制出口和续写token，预热后交替测速，各3次。
CUDA_VISIBLE_DEVICES=7 PYTHONHASHSEED=42 .venv/bin/python scripts/benchmark_dynamic_cache.py \
  --family ouro --pretrained --device cuda --mixed-precision bf16 \
  --prompt-tokens 128 --new-tokens 32 --loops 4 --repeats 3 \
  --output /tmp/ouro_dynamic_cache.json
```

验证材料存于[audit/dynamic_cache_20261114](../audit/dynamic_cache_20261114/)。受控测速强制相同出口和续写，包含prefill、decode、补算与LM head，不计分词、采样和加载。它比较精确缓存与旧重算执行成本，不证明我们的方法在任务上优于LoopOPD。真实Huginn短测使用K=4；完整K=32的深度补算语义另有随机小模型测试。
