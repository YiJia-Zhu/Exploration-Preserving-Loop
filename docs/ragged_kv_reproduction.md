# Phase 0 · Ragged KV 零训练复现

先复现 Figure 1(c) 中非 Thinking Ouro 在 BBH 上的低深度 P@1 掉点，再用 Ouro-1.4B-Thinking 在 AIME 2024 上比较 Ragged KV 与原生 Ouro KV 的准确率。图中 Ouro-1.4B 的 `τ=0.20` 约在 1.75 轮，P@1 约 40%，P@4 约 70%。`τ=0.28` 约在 2.25 轮，P@1 约 58%，P@4 约 80%。这些是读图近似值。[Ragged KV 论文 v1，Figure 1(c) 与 Appendix E.1](https://arxiv.org/abs/2610.09013v1)

## BBH · Figure 1(c)

| 项目 | 首批 | 论文口径 |
|:--|:--|:--|
| 模型 | 原始 `ByteDance/Ouro-1.4B`，固定 revision `574fa66` | 图中另有非 Thinking 2.6B。代码支持其原始权重 `1ed0425` |
| 数据 | `lukaemon/bbh` test，27 个子任务各 5 道，抽样种子 1234，共 135 道 | 与 E.1 一致。每任务独立用 Python `Random(1234).sample`，论文给出种子和题数，未列出完整样本索引或 RNG 实现 |
| 提示 | BIG-Bench-Hard 官方任务前缀加 3 个 CoT 示例，追加当前题和 `A: Let's think step by step.` | 3-shot CoT 纯文本，与论文示例同格式。准备数据时冻结各提示、题目索引和源文件哈希 |
| 采样 | 每题 4 个回答，温度 1.0，top-p 0.7，关闭 top-k（论文 0，RLT 接口 -1），EOS 或 1,024 个新 token | 与 E.1 一致。论文未列出生成种子，采用本仓库种子 42，由题目提示哈希与样本号派生，同一题和样本跨组配对 |
| 对照 | 首批固定 2／4 轮。Ragged KV 用满 4 轮 prefill，decode 允许 1–4 轮，按原生退出 CDF 扫 `0.10, 0.15, 0.20, 0.25, 0.30` | 论文固定 1／2／3／4 轮，Ragged KV 阈值 `0.20, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.75`。本地加测 `0.10` 和 `0.15` |
| 判分 | 对生成文本的最后一个 `the answer is …` 严格提取，到句号或换行并与 target 精确匹配。未提取到答案记错 | E.1 使用答案提取及精确匹配。最后一处匹配和标点边界是本地实现细节 |
| 运行 | 固定组用官方 vLLM `0.26.0+cu129` 和 FlashInfer。动态组用 vLLM-RLT v0.2.0 `c08b9d7`、Triton、同步 CUDA Graph、KV block 64。两组均用单张 H100、BF16、`gpu_memory_utilization=0.90`，由引擎采用默认 `max_num_seqs`，评测入口每批最多提交 8 条，关闭前缀缓存和投机解码 | 论文在 H100 执行。权重选用论文链接的公开模型 |

**Ragged KV** 用 vLLM-RLT 的 `LAST_EXITED` 执行原生 CDF 早退。退出轮的 KV 值复制到未执行轮的缓存页，后续注意力读取论文式 (7) 指定的值。BBH 按实际执行的轮数统计计算量。物理 KV 页按 vLLM-RLT 实际分配计。[vLLM-RLT 发布版](https://github.com/InferMatrixLab/vllm-rlt/tree/c08b9d79834bb8b4ff283b15bae2c14c559f6b52) · [KV 复制实现](https://github.com/InferMatrixLab/vllm-rlt/blob/c08b9d79834bb8b4ff283b15bae2c14c559f6b52/vllm_rlt/core/kv_cache_manager.py#L751-L777)

首批 7 组共 `135 × 4 × 7 = 3,780` 条生成，最多 3,870,720 个新 token。补齐固定 1／3 轮后共 9 组、4,860 条生成，已完成的首批结果可复用。2.6B 使用独立计划和相同题目。[BBH 数据集](https://huggingface.co/datasets/lukaemon/bbh) · [官方 CoT 提示](https://github.com/suzgunmirac/BIG-Bench-Hard/tree/9ee07bd481feebf959a6b59d61ea57bdcf30964d/cot-prompts) · [Ouro-1.4B 权重](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1)

动态组的主对照用 `τ=0.20` 的结果减去 `τ=0.30` 的结果。每组报告 P@1（每题四条的正确比例再按题平均）、P@4（四条中至少一条正确的题占比）、按 token 加权的平均 decode 深度、首轮退出比例、输出长度、截断率、未抽取到答案的比例和实际 token-loop。深度统计排除由 prefill 预测的首个生成 token。差值按子任务分层、在同一批题上做 5,000 次配对 bootstrap，报 95% 区间。判断左段现象时同时看 `0.20` 的深度是否更低、P@1 差值是否为负，以及 P@4 的降幅是否更小。`0.30` 与固定 2／4 轮对照时同步报告实际深度差和推理引擎。

### 首批结果与论文对照

本次复判使用同一批 3,780 条已保存输出。每条在首个 `\nQ:` 处截断，在当前题范围内取最后一处 `the answer is …` 与标准答案精确匹配。截断前无可提取答案的样本记错。平均解码轮数仍按原运行的完整输出统计。论文 E.1 说明了精确匹配规则，续写新题时的评分边界未公开。[原始输出](../ep_loop_runs/ragged_bbh_rlt_1p4b/shard_00.sqlite) · [原始汇总](../ep_loop_runs/ragged_bbh_rlt_1p4b/summary_core.json) · [论文 E.1](https://arxiv.org/html/2610.09013v1#A5.SS1)

论文列从 Figure 15 的非 Thinking Ouro-1.4B 蓝色图点估读，包含浅色散点。差值用舍入前的本次复判值减论文读图值。`τ=0.10` 和 `τ=0.15` 是本次加测。[论文 Figure 15](https://arxiv.org/html/2610.09013v1#A5.F15)

| 推理方式 | 轮数或阈值 | 论文平均解码轮数 | 本次平均解码轮数 | 论文 P@1 / P@4（%） | 本次复判 P@1 / P@4（%） | P@1 / P@4 差值（百分点） |
|---|:---:|---:|---:|---:|---:|---:|
| 固定循环，官方 vLLM | 2 轮 | 2.00 | 2.00 | 26.67 / 43.70 | 31.67 / 53.33 | +5.00 / +9.63 |
| 固定循环，官方 vLLM | 4 轮 | 4.00 | 4.00 | 65.56 / 81.48 | 67.22 / 84.44 | +1.67 / +2.96 |
| Ragged KV，vLLM-RLT | `τ=0.10` | — | 1.60 | — | 40.19 / 68.15 | — |
| Ragged KV，vLLM-RLT | `τ=0.15` | — | 1.73 | — | 43.52 / 67.41 | — |
| Ragged KV，vLLM-RLT | `τ=0.20` | 1.75 | 1.89 | 40.00 / 70.37 | 40.93 / 68.15 | +0.93 / −2.22 |
| Ragged KV，vLLM-RLT | `τ=0.25` | 2.15 | 2.27 | 47.41 / 67.41 | 51.11 / 74.81 | +3.70 / +7.41 |
| Ragged KV，vLLM-RLT | `τ=0.30` | 2.28 | 2.21 | 57.22 / 78.52 | 59.26 / 81.48 | +2.04 / +2.96 |

固定 4 轮、`τ=0.20` 和 `τ=0.30` 的两项 Pass 与论文均相差约 3 个百分点以内。固定 2 轮与 `τ=0.25` 的 P@4 差距较大，分别为 9.63 和 7.41 个百分点。

同样从 `τ=0.30` 降到 `τ=0.20`，论文的 P@1 下降约 17.22 个百分点，本次复判下降 18.33 个百分点（配对 95% CI 为 −23.70 至 −13.15）。P@4 在两边也下降，论文约下降 8.15 个百分点，本次下降 13.33 个百分点。复判复现了这一段的 P@1 掉点趋势，固定组和动态组各自的引擎及 BBH 抽样差异见上文。

## AIME 2024 · 目标模型与原生 Ouro KV 对照

使用 AIME 2024 的全部 30 道题与原始 `Ouro-1.4B-Thinking`（revision `3aaa222`）。各组使用 0-shot Thinking chat template，温度 1.0，top-p 0.7，关闭 top-k，种子 42，每题 4 条，EOS 或 16,384 个新 token。从同一条回答截取 2K／4K／8K／16K 评分，用严格 `\boxed{整数}` 判分。论文 Table 6 的 AIME 使用 8 条／题、32,768 上限与 Qwen 判分。这里检验的是 RAD-OPD 计划中实际要训练的模型和预算。[现有 RAD-OPD 配置](../configs/rad_opd.json) · [论文 Table 6](https://arxiv.org/html/2610.09013v1#A4.T6)

固定 1／2／4 轮和**原生 Ouro KV** 用官方 vLLM `0.26.0+cu129` 和 FlashInfer。**Ragged KV** 用 vLLM-RLT v0.2.0 `c08b9d7` 的 `LAST_EXITED`、Triton、同步 CUDA Graph 和 KV block 64。两套引擎均用单张 H100、BF16、`gpu_memory_utilization=0.90` 和引擎默认 `max_num_seqs`，vLLM-RLT 生效值为 8。每批最多提交 8 条，关闭前缀缓存和投机解码。两个动态组满 4 轮 prefill，从第 4 轮采样首个 token，decode 按原生退出门扫 `0.15, 0.20, 0.25, 0.30, 0.45, 0.75`。原生 Ouro KV 每步执行 4 轮，保存每轮的 KV 并用退出门选出的 logits 采样。Ragged KV 以实际退出轮次统计执行深度。[AIME 评测入口](../scripts/evaluate_stage_distillation.py) · [vLLM-RLT 日志](../ep_loop_runs/aime2024_phase0_vllm_Ah7meM4r/responses.vllm-rlt.4.engine.log)

### AIME 2024 最终结果

本轮于北京时间 2026-11-26 21:28 完成，固定 3 组、Ragged KV 6 组，每组 120 条，共 1,080 条。生成批次累计耗时 **20 小时 47 分 11 秒**，固定 4 轮合并续跑前后的已完成批次。[数据清单](../ep_loop_runs/aime2024_phase0_20261125_FI3s1w/aime/manifest.json) · [逐条结果](../ep_loop_runs/aime2024_phase0_vllm_Ah7meM4r/responses.jsonl) · [汇总](../ep_loop_runs/aime2024_phase0_vllm_Ah7meM4r/responses.summary.json) · [批次日志](../ep_loop_runs/aime2024_phase0_vllm_Ah7meM4r/responses.events.jsonl)

平均解码轮数按 token 加权，排除由 prefill 预测的首个 token。平均 token-loop 计入 prefill 和 decode 的有效计算轮次。耗时是已完成批次的墙钟时间之和，以时、分、秒表示。

| 实验组 | 生成耗时 | 平均解码轮数 | 首轮退出比例 | 平均输出 token | 达到 16K 上限 | 平均 token-loop |
|---|---:|---:|---:|---:|---:|---:|
| Dense 4-Loop | 1:41:56 | 4.000 | 0.00% | 10,797 | 40/120 | 43,788 |
| Static 2-Loop | 1:05:26 | 2.000 | 0.00% | 13,472 | 67/120 | 27,244 |
| Static 1-Loop | 0:25:14 | 1.000 | 100.00% | 8,336 | 48/120 | 8,486 |
| Ragged KV `τ=0.15` | 2:37:08 | 1.760 | 67.38% | 15,671 | 108/120 | 28,190 |
| Ragged KV `τ=0.20` | 2:39:09 | 1.763 | 67.21% | 15,878 | 114/120 | 28,595 |
| Ragged KV `τ=0.25` | 3:13:23 | 2.178 | 29.11% | 15,944 | 112/120 | 35,324 |
| Ragged KV `τ=0.30` | 2:35:44 | 2.356 | 0.00% | 11,896 | 60/120 | 28,631 |
| Ragged KV `τ=0.45` | 2:45:49 | 2.609 | 0.00% | 11,579 | 53/120 | 30,814 |
| Ragged KV `τ=0.75` | 3:43:21 | 3.832 | 0.00% | 11,084 | 44/120 | 43,072 |

下表各格为 Pass@1 / Pass@4（%）。Pass@1 按答对的回答数除以 120，Pass@4 按四次中至少答对一次的题数除以 30。

| 实验组 | 2K | 4K | 8K | 16K |
|---|---:|---:|---:|---:|
| Dense 4-Loop | 6.67 / 10.00 | 15.83 / 26.67 | 35.83 / 56.67 | 49.17 / 73.33 |
| Static 2-Loop | 1.67 / 6.67 | 9.17 / 13.33 | 13.33 / 26.67 | 25.00 / 50.00 |
| Static 1-Loop | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 | 0.00 / 0.00 |
| Ragged KV `τ=0.15` | 0.83 / 3.33 | 1.67 / 3.33 | 1.67 / 3.33 | 4.17 / 10.00 |
| Ragged KV `τ=0.20` | 0.83 / 3.33 | 0.83 / 3.33 | 0.83 / 3.33 | 1.67 / 6.67 |
| Ragged KV `τ=0.25` | 0.83 / 3.33 | 1.67 / 3.33 | 2.50 / 3.33 | 2.50 / 3.33 |
| Ragged KV `τ=0.30` | 3.33 / 6.67 | 13.33 / 23.33 | 29.17 / 53.33 | 36.67 / 63.33 |
| Ragged KV `τ=0.45` | 5.83 / 6.67 | 17.50 / 33.33 | 36.67 / 56.67 | 45.83 / 63.33 |
| Ragged KV `τ=0.75` | 4.17 / 10.00 | 15.83 / 23.33 | 35.00 / 53.33 | 47.50 / 70.00 |

### 下一轮对照依据

- `τ=0.15` 和 `0.20` 都落在目标深度 1.5–2.0，平均约 1.76 轮。两组首轮退出约 67%，分别有 90% 和 95% 的回答达到 16K 上限。该区间的准确率和总生成量需结合截断情况理解。
- `τ=0.25 → 0.30` 是本轮最明显的变化区间。平均解码增加约 0.18 轮，首轮退出从 29.11% 降到 0%，16K Pass@1 提高 34.17 个百分点，达到上限的回答从 112 条降到 60 条。后续若细扫阈值，这两个点可作为区间端点。
- `τ=0.45` 在 8K 的 Pass@4 与 Dense 4-Loop 相同，Pass@1 对应的正确回答多 1 条。16K 的 Pass@1／Pass@4 分别低约 3.3／10 个百分点。每条平均 token-loop 降约 30%，单卡生成耗时约为 Dense 4-Loop 的 1.63 倍。两个组分别使用 vLLM-RLT 和官方 vLLM，下一轮规划用时可参考对应引擎的实测耗时。
- **原生 Ouro KV** 六个阈值待跑。按同阈值、同题和同样本号与 Ragged KV 配对，重点看 Ragged KV 实际平均解码深度落在 `[1.5, 2.0]` 的点。原生 Ouro KV 的读出轮次单独统计，实际解码深度为 4。[RAD-OPD Phase 0](rad_opd_mvp.md#3-phase-0零训练证伪1-至-2-天)

## 命令

准备脚本把原始数据和官方提示缓存到 `.cache/huggingface_datasets/BBH`，评测计划默认读取 `data/bbh_frontier/eval.jsonl`。准备输入与校验模型文件均在 CPU 上完成。重新冻结计划时指定一个新目录。

```bash
.venv/bin/python scripts/prepare_ragged_bbh.py
.venv/bin/python scripts/reproduce_ragged_bbh.py plan --model Ouro-1.4B \
  --model-dir ../huggingface_models/Ouro-1.4B --shards 1 \
  --output ep_loop_runs/ragged_bbh_rlt_1p4b
```

单卡计划使用分片 0。每条结果提交到该分片的 SQLite，重启同一分片时跳过已提交的回答。首批汇总完成后可用 `--stage full` 补跑并汇总完整曲线。

```bash
plan=ep_loop_runs/ragged_bbh_rlt_1p4b/plan.json
.venv/bin/python -u scripts/reproduce_ragged_bbh.py run --plan "$plan" --stage core --gpu 0 --shard 0
.venv/bin/python scripts/reproduce_ragged_bbh.py summarize --plan "$plan" --stage core
```

AIME 下列命令复现本轮的 9 组，使用实际评测的 30 题 JSONL 和新的输出路径。移除 `--skip-native` 即包含原生 Ouro KV 的六个阈值。对同一输出路径使用 `--resume` 可从已保存的 token IDs 恢复统计并生成其余回答。

```bash
.venv/bin/python scripts/evaluate_stage_distillation.py --config configs/rad_opd.json --phase-zero --skip-native --write-token-ids \
  --data ep_loop_runs/aime2024_phase0_20261125_FI3s1w/aime/aime2024.jsonl \
  --output ep_loop_runs/ragged_aime2024/responses.jsonl
```
