# 训练与推理运行时间

## 统一复测协议

主推理表使用原始 Ouro-1.4B-Thinking、BF16 权重和 KV、temperature 1.0、top-p 0.7、同一题目的相同 token IDs 和请求种子。使用同一块空闲卡依次运行各引擎，记录权重 SHA-256、卡的 UUID、版本、attention 后端和调度设置。TP1，GPU 引擎的显存比例均为 0.75，关闭前缀缓存、投机解码和量化。每组均先预热。主指标的墙钟从客户端把已构造的请求交给对应后端开始，到完整 token IDs 返回为止，包含请求传输、排队、prefill、decode 和生成接口返回。加载模型、预热、落盘、文本解码和判题在计时外。吞吐为全组生成 token 数除以全组墙钟。C1 将八条请求逐条跑完后累计各次墙钟，C8 将八条请求一次提交。两种并发各有自己的表。TTFT 和每个后续 token 的平均耗时同时保留，使用各后端原生观测点，跨框架首选统一的客户端墙钟。[基准入口](../scripts/benchmark_runtime.py)

| 协议 | 固定条件和工作量 | 同表对照与主指标 |
|---|---|---|
| **固定工作量** | 原来的 8 道 AIME 2024，原始短输入强制各生成 1,024 token，8,192 和 14,336 token 历史前缀各强制续写 128 token。忽略 EOS。每组预热后计三轮，报中位数和逐轮时间 | 固定两轮和固定四轮分开。C1 对比 vLLM、RLT 和 HF，四轮可加入 `τ=1` 的两种缓存布局。C8 对比 vLLM 与 RLT。报总耗时、输出 token/s、TTFT 和 TPOT |
| **自然生成** | 同一组 8 道题，每题一条，EOS 或 16,384 个新 token 时停止，上下文上限 17,408 | 固定两轮与四轮分别看。动态组统一四轮 prefill、相同阈值，配对比较 exact backfill 与 zero-backfill Ragged。报耗时、实际输出量、逐题长度与截断、正确数、平均解码深度、token-loop 和物理 KV 槽位 |
| **重复提示与并行** | 前缀缓存采用同四个归档提示，每题 16 条独立请求，各生成 256 token，单卡一次提交 64 条。缓存组从启动后未见过测量提示的状态计首批。TP 对比用固定工作量，绑定同一对卡、同一总请求量、输入、长度和批次边界 | 分别报告框架内缓存开关和 TP2 × 1 与 TP1 × 2。vLLM 原生 `n=16` 属于单独的同引擎提交方式对照。缓存组同时报告命中计数，TP 组报两卡合计吞吐 |
| **训练与离线回放** | 引擎对照冻结相同起始权重和训练目标，固定教师、题目及顺序、种子、四卡分配、每步 32 × 8、16,384 上限、梯度累积与检查点策略。完成预热更新后重新从同一起点收集稳定步，保存和评测单独计时。离线回放共用逐 token JSON 轨迹 | 各计 rollout、教师前向、学生重放和反向、优化器、完整更新墙钟及生成／训练 token/s。静态 LoopOPD 和 RAD-OPD 按各自策略报告实际轨迹长度和循环工作量。离线组件表固定轨迹，整步表保留真实在线采样 |

固定长度可对齐前向 token 数，采样实现与内核可能仍产生不同文本。自然生成另外保留输出和判题记录，便于区分每 token 更快与生成量更少。平均 decode 深度排除由 prefill 产生的首个 token。物理 KV 槽位按已用的「位置 × 轮次」统计。RLT `last_exited` 可作为等深引擎对照，动态 Ragged 与 exact backfill 对照走仓库 PyTorch 实现。

引擎基线固定为 vLLM 0.26 原生 Ouro + FlashInfer，以及发布版 RLT v0.2.0 `c08b9d7` + Triton、同步 CUDA Graph、KV block 64。vLLM Triton、RLT block 16 和 128 作为显式配置复测，同一配置贯穿各输入长度。先对照各配置本身，再单列按上下文选择最快参数的筛选。历史 94 题 × 2 的 DAPO/AIME 采样属于独立的准确率与系统开关筛选，同套题内共用采样参数，两套题分开评分。历史 n-gram 与 FP8 数字保留在该筛选表。RAD Phase 0 的 30 题 × 4、阈值扫描和 P@1/P@4 也使用独立的评测规模。[94 题筛选入口](../scripts/benchmark_ouro_accuracy.py) · [RAD Phase 0 入口](../scripts/evaluate_stage_distillation.py)

### GPU 复测命令

以 `GPU=0` 为设备参数示例。每次运行使用新的输出目录。加 `--dry-run` 可生成和校验输入清单，实际执行会检查设备占用。

```bash
set -e
GPU=0
OUT=audit/runtime_matched_gpu

for depth in 2 4; do
  for profile in short 8192 14336; do
    .venv/bin/python scripts/benchmark_runtime.py \
      --gpu "$GPU" --profile "$profile" --loops "$depth" --concurrency 1 \
      --case vllm rlt hf --output "$OUT/${profile}_n${depth}_c1"
    .venv/bin/python scripts/benchmark_runtime.py \
      --gpu "$GPU" --profile "$profile" --loops "$depth" --concurrency 8 \
      --case vllm vllm-triton rlt rlt-block16 rlt-block128 --output "$OUT/${profile}_n${depth}_c8"
  done
  .venv/bin/python scripts/benchmark_runtime.py \
    --gpu "$GPU" --profile natural --loops "$depth" --concurrency 8 \
    --case vllm rlt --output "$OUT/natural_n${depth}_c8"
done

.venv/bin/python scripts/benchmark_runtime.py \
  --gpu "$GPU" --profile natural --loops 2 --concurrency 1 \
  --case vllm rlt hf --output "$OUT/natural_n2_c1"

.venv/bin/python scripts/benchmark_runtime.py \
  --gpu "$GPU" --profile natural --loops 4 --concurrency 1 --tau 0.20 \
  --case vllm rlt hf backfill ragged --output "$OUT/natural_n4_c1"

.venv/bin/python scripts/benchmark_runtime.py \
  --gpu "$GPU" --profile repeated --loops 2 --concurrency 64 \
  --case vllm vllm-prefix-cache rlt rlt-prefix-cache --output "$OUT/repeated_n2_c64"
```

四轮固定长度可在 C1 的 `--case` 中加 `backfill-dense ragged-dense`。两卡 TP 入口可传入 `--manifest "$OUT/8192_n2_c8/manifest.json"` 复用同一份八题、前缀和采样参数，结果按两卡吞吐另成表。[两卡基准入口](../scripts/benchmark_vllm_parallel.py)

### 本轮实测范围

本轮已尝试在 GPU 上启动固定两轮 C8。设备查询返回 `NVML: Unknown Error`，当前进程无法访问 `/dev/nvidiactl`，该轮在引擎启动前结束。新一轮 vLLM/RLT 数据及四卡整步训练数据待 GPU 可用后补齐。CPU 复测使用 AMD EPYC 9A75、16 个固定线程、单请求，原始 1.4B 权重与同一批请求种子，采用单列的小工作量。[GPU 访问记录](../audit/runtime_matched_20261122/gpu_access.log) · [GPU 启动日志](../audit/runtime_matched_20261122/gpu_short2.log) · [CPU 运行日志](../audit/runtime_matched_20261122/cpu.log)

#### CPU 固定工作量，原始短输入

同 8 道题各生成 32 token。每个配置预热后计两轮。每轮生成 256 token，表中为两轮的中位数。

| PyTorch 路径 | 墙钟 | 输出 token/s | 首 token 延迟 | 后续 token 耗时 |
|:---|---:|---:|---:|---:|
| HF 固定两轮 | 20.62 s | 12.41 | 234 ms | 75 ms |
| HF 固定四轮 | 39.86 s | 6.42 | 460 ms | 146 ms |
| Exact backfill，四轮，τ=1 | 39.33 s | 6.51 | 468 ms | 142 ms |
| Ragged KV，四轮，τ=1 | 42.66 s | 6.00 | 471 ms | 157 ms |

三个四轮配置均执行 5,840 token-loop。每次测量的八条输出在三个配置间逐 token 一致。等深 Ragged KV 较 exact backfill 多用 8.5% 墙钟，增加主要在首 token 之后。[两轮原始结果](../audit/runtime_matched_20261122/cpu_short2/) · [四轮原始结果](../audit/runtime_matched_20261122/cpu_short4/)

#### CPU 固定工作量，8K 前缀

使用归档中第一题的 8,192 token 前缀，先用同长度输入预热，再测两次固定四轮、续写 16 token。两组输出一致，都执行 32,828 token-loop。

| PyTorch 路径 | 墙钟 | 输出 token/s | 首 token 延迟 | 后续 token 耗时 |
|:---|---:|---:|---:|---:|
| Exact backfill，τ=1 | 44.43 s | 0.360 | 39.92 s | 287 ms |
| Ragged KV，τ=1 | 38.72 s | 0.413 | 32.50 s | 405 ms |

这条输入上，Ragged KV 的首 token 提前了 7.42 秒，后续每 token 耗时增加约 41%。短续写的总墙钟主要由首 token 延迟决定。[8K 原始结果](../audit/runtime_matched_20261122/cpu_8k/)

#### CPU 自然 EOS 小样

同 8 道题、EOS 启用，每条上限缩至 32 token。三个配置均在 8 条上触及上限，各输出 256 token。四轮 prefill，动态组统一使用原生退出阈值 τ=0.20。

| PyTorch 路径 | 墙钟 | 输出 token/s | 后续 token 耗时 | 平均 decode 轮次 | 计算 token-loop | 回填 token-loop | 物理 KV 槽位 |
|:---|---:|---:|---:|---:|---:|---:|---:|
| HF 固定四轮 | 40.60 s | 6.30 | 149 ms | 4.00 | 5,840 | 0 | 5,840 |
| 动态退出 + exact backfill | 19.70 s | 12.99 | 62 ms | 1.70 | 5,551 | 281 | 5,551 |
| 动态退出 + Ragged KV | 24.22 s | 10.57 | 83 ms | 2.02 | 5,349 | 0 | 5,349 |

Ragged KV 相比固定四轮省约 40% 的墙钟。同阈值下，这批题在 Ragged KV 中退出得更深，耗时比 exact backfill 多 4.52 秒。共用的四轮 prefill 占 4,848 token-loop，解码阶段另外用 501 和 703 token-loop，分别对应 Ragged 和 exact backfill，后者含回填。这里的 32 token 上限用于计时验证，完整答题质量需要 16,384-token 对照。[自然 EOS 小样原始结果](../audit/runtime_matched_20261122/cpu_natural/)

## 历史记录

### EP-D-LoopOPD 十步训练

原始 Ouro-1.4B-Thinking，四张 RTX PRO 5000 72 GB。每步 32 题 × 8 条回答，学生固定两轮、教师实际计算四轮。生成上限 16,384 token。学生 rollout 使用 BF16 KV 与 n-gram5，教师读出和评测使用普通推理。10 个完整更新的墙钟区间合计 **4 小时 22 分 39 秒**，平均 **26 分 16 秒／步**，共生成 22,398,863 个 token。生成阶段平均 17 分 58 秒，约占整步的 **68.4%**。[实际训练配置](../audit/ep_d_ngram_launch_20261119_182732/production.config.json) · [十步统计与原始日志位置](../audit/ep_d_ngram_launch_20261119_182732/phase_timing_10_steps.json)

| 步 | 学生生成 | 教师骨干尾段 | HF 训练尾段 | 整步 |
|---:|---:|---:|---:|---:|
| 1 | 19:33 | 1:32 | 7:30 | 28:38 |
| 2 | 16:02 | 0:53 | 6:21 | 23:25 |
| 3 | 18:55 | 0:58 | 7:31 | 27:32 |
| 4 | 16:12 | 0:54 | 6:27 | 23:40 |
| 5 | 20:45 | 1:08 | 8:15 | 30:23 |
| 6 | 17:04 | 0:53 | 6:40 | 24:46 |
| 7 | 17:49 | 0:58 | 7:05 | 26:00 |
| 8 | 18:11 | 0:58 | 7:29 | 26:45 |
| 9 | 17:48 | 0:58 | 7:15 | 26:06 |
| 10 | 17:16 | 0:58 | 6:59 | 25:24 |
| **平均** | **17:58** | **1:01** | **7:09** | **26:16** |

按四个 rank 中各阶段最晚的结束时刻切分墙钟。先完成生成的 rank 可以提前进入教师与 HF 阶段，所以表中的尾段是前一阶段全卡完成之后剩余的时间。完整词表读出、JS、reverse KL、反向与同步计入 HF 尾段。整步还包括状态迁移、日志和快照保存，第 1 步包含引擎首次初始化。相同批次耗时会重复出现在逐回答日志中，汇总使用阶段事件。

两次独立 AIME 2024 评测各生成 480 条回答，双卡普通 vLLM，学生固定两轮。第 5 步评测耗时 154.3 分钟，第 10 步 158.2 分钟。评测在训练更新之外单独计时。[评测时间与生成量](../audit/checkpoint_eval_20261120_004854/comparison.json)

### 历史训练路径对照

三次运行使用相同型号的四卡、256 条回答／步和 16,384 新 token 上限。生成文本、每步实际 token 量与部分调度设置有差异。

| 路线 | 完整步数 | 平均输出 token／步 | 生成墙钟／步 | 整步墙钟 |
|---|---:|---:|---:|---:|
| 纯 HF，教师复用学生两轮状态后继续深算 | 1 | 1,978,018 | 47:48 | 55:34 |
| vLLM 普通生成，独立四轮教师读出 | 2 | 2,132,351 | 24:24 | 32:36 |
| 当前 vLLM + n-gram5 训练生成 | 10 | 2,239,886 | 17:58 | 26:16 |

纯 HF 的共享续算计入训练尾段。旧 vLLM 与当前主运行之间的整步耗时差约 6 分 20 秒，主要落在生成阶段。两次训练的轨迹和样本数有别，n-gram 单项性能参考下面的独立筛选。[HF 原始计时](../audit/hf_shared_timing_20261118_235314/stopped_timing_summary.json) · [旧 vLLM 分阶段计时](../audit/ep_d_loopopd_launch_20261118_213142/step_phase_breakdown.json)

### 解码与并行筛选

原始 Ouro 模型固定两轮，对 DAPO 开发集 64 题和 AIME 2024 的 30 题各采样两次。三组使用相同题目、请求种子和生成上限，输出自然停止或达到 16,384 token。

| 解码 | 整批耗时 | 输出 token/s | DAPO Avg@2 | AIME 2024 Avg@2 |
|---|---:|---:|---:|---:|
| BF16 普通生成 | 2,662.0 秒 | 736.4 | 26.56% | 21.67% |
| BF16 + n-gram5 | 1,679.3 秒 | 1,114.7 | 28.91% | 26.67% |
| FP8 KV + n-gram5 | 936.1 秒 | 2,118.7 | 27.34% | 20.00% |

BF16 + n-gram5 的输出吞吐约为原来的 1.51 倍。两套题的准确率差按题重采样，95% 区间均跨零。FP8 KV 吞吐更高，但在相同 n-gram 配置下 AIME 点估计比 BF16 低 6.67 个百分点。正式十步运行选用 BF16。[对照数据和置信区间](../audit/vllm_accuracy_recheck_20261119_151324/accuracy_summary.json)

- **张量并行。** 同两张 PCIe GPU、固定生成 128 token，8K／16K 前缀短测中，单个 TP2 引擎比两个 TP1 副本分别多用 23.8%／22.7% 时间。当前主线采用 TP1。[原始对照](../audit/vllm026_tp2_20261118/benchmark.json)
- **重复提示。** 4 题 × 16 条、每条 256 token 的官方 vLLM 短测中，原生多样本接口与独立请求的耗时几乎相同。开前缀缓存时中位约 2.71 秒，关缓存时约 3.93 秒。RLT 在另一轮同规模短测中开缓存平均约 34.36 秒，关缓存约 35.69 秒。两项测试的计时边界各自一致，按各引擎内部的开关对照解读。[官方 vLLM](../audit/vllm_native_n16_20261120/comparison.json) · [vLLM-RLT](../audit/rlt_prefix_cache_20261120/comparison.json)

### 官方 vLLM 与 RLT 固定两轮推理

原始 Ouro-1.4B-Thinking，同一组 8 道 AIME 2024 题一次提交，每题采样一次。每个引擎单独使用一张 RTX PRO 5000 72 GB（SM120），权重与 KV 均为 BF16，prefill 和 decode 固定两轮。temperature 为 1.0，top-p 为 0.7，使用相同题目和逐请求种子。两边均关闭投机解码和前缀缓存。预热后计整批时间。短测每题生成 1,024 token，长测遇 EOS 或达到 16,384 新 token 时结束，最大上下文为 17,408。[完整配置与题目](../audit/rlt_v020_bf16_vs_vllm_20261120_152301/workload.json)

| 引擎与配置 | 短测 8,192 token 耗时 | 短测 token/s | 长测耗时 | 长测输出 token | 长测 token/s |
|:---|---:|---:|---:|---:|---:|
| 官方 vLLM 0.26 + FlashInfer，首次发布版对照 | 8.89 秒 | 921.4 | 347.2 秒 | 96,586 | 278.1 |
| RLT v0.2.0 + Triton，KV block 16 | 18.60 秒 | 440.5 | 628.6 秒 | 104,876 | 166.9 |
| 官方 vLLM 0.26 + FlashInfer，block 64 同轮复测 | 8.90 秒 | 920.8 | 354.7 秒 | 96,581 | 272.3 |
| RLT v0.2.0 + Triton，KV block 64 | 17.61 秒 | 465.1 | 534.4 秒 | 104,876 | 196.3 |

RLT 使用官方 `v0.2.0` 标签（`c08b9d7`）、同步调度、`last_exited` KV 和 CUDA Graph。KV block 从 16 改为 64 后，长测耗时减少 **94.2 秒（15.0%）**，短测和长测的 8 条输出各自都逐 token 一致。复测中，官方 vLLM 的短测吞吐为 RLT block 64 的 **1.98 倍**。长测两边的实际输出量及结束顺序有差异，吞吐分别为 **272.3** 和 **196.3 token/s**。[发布版首测](../audit/vllm_rlt_v020_release_20261120_144351/comparison.json) · [复测与逐题校验](../audit/rlt_v020_bf16_vs_vllm_20261120_152301/verified_comparison.json)

#### 等工作量的上下文筛选

短输入使用原始题目，每题续写 1,024 token。长输入在相同题目后拼接保存的推理 token，构造固定 8,192 或 14,336 token 前缀，每题续写 128 token。固定输出量测试忽略 EOS，生成量分别为 8,192 和 1,024 token。表中耗时包括 prefill 和 decode。RLT 的最快值来自已测的不同 KV block、packed prefill 和 BF16 融合算子设置。[等工作量对照](../audit/rlt_v020_bf16_vs_vllm_20261120_152301/fixed_work_comparison.json) · [官方 vLLM 后端复测](../audit/rlt_bf16_decode_profile_20261120_171707/verified_summary.json)

| 工作量 | 官方 vLLM + FlashInfer | 官方 vLLM + Triton | RLT + Triton 已测最快 |
|:---|---:|---:|---:|
| 原始短输入，8 × 1,024 新 token | 8.85 秒 | 8.59 秒 | 16.94 秒 |
| 8,192 token 前缀，8 × 128 新 token | 5.86 秒 | 6.90 秒 | 约 9.5 秒 |
| 14,336 token 前缀，8 × 128 新 token | 10.27 秒 | 13.52 秒 | 17.55 秒 |

RLT 的 paged attention 在稳态 decode 窗口占已记录 GPU 活动时间的 **62%（8K）和 75%（14K）**，三个被测上下文的 CUDA Graph 均正常重放。Block 128 相比 64 在固定长前缀的续写吞吐提高约 5–6%，完整长测已验证到 block 64。上述结果覆盖固定两轮推理，Ouro 按 token 动态退出的混合深度调度需单独测量。[Profiling 与开关筛选](../audit/rlt_bf16_decode_profile_20261120_171707/verified_summary.json) · [KV block 对照](../audit/rlt_v020_runtime_probe_20261120_151153/validated_metrics.json)

早期直接导入 `bed3400` 源码的 RLT 测试中，官方 vLLM 和 RLT 短测分别用 8.87 和 17.87 秒。长测分别用 396.60 和 497.52 秒，输出 112,877 和 83,830 token，吞吐分别为 284.6 和 168.5 token/s。[早期试跑设置](../audit/ouro_base_8q_backend_speed_20261120_143308/workload.json) · [早期试跑结果](../audit/ouro_base_8q_backend_speed_20261120_143308/comparison.json)
