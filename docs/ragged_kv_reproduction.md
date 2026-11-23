# Phase 0 · Ragged KV 零训练复现

先复现 Figure 1(c) 中非 Thinking Ouro 在 BBH 上的低深度 P@1 掉点，再检查目标模型 Ouro-1.4B-Thinking 在 AIME 2024 上是否也出现掉点，以及精确回填能恢复多少。图中 Ouro-1.4B 的 `τ=0.20` 约在 1.75 轮，P@1 约 40%，P@4 约 70%。`τ=0.28` 约在 2.25 轮，P@1 约 58%，P@4 约 80%。这些是读图近似值。[Ragged KV 论文 v1，Figure 1(c) 与 Appendix E.1](https://arxiv.org/abs/2610.09013v1)

## BBH · Figure 1(c)

| 项目 | 首批 | 论文口径 |
|:--|:--|:--|
| 模型 | 原始 `ByteDance/Ouro-1.4B`，固定 revision `574fa66` | 图中另有非 Thinking 2.6B。代码支持其原始权重 `1ed0425` |
| 数据 | `lukaemon/bbh` test，27 个子任务各 5 道，抽样种子 1234，共 135 道 | 与 E.1 一致。每任务独立用 Python `Random(1234).sample`，论文给出种子和题数，未列出完整样本索引或 RNG 实现 |
| 提示 | BIG-Bench-Hard 官方任务前缀加 3 个 CoT 示例，追加当前题和 `A: Let's think step by step.` | 3-shot CoT 纯文本，与论文示例同格式。各提示、题目索引和源文件哈希已冻结 |
| 采样 | 每题 4 个回答，温度 1.0，top-p 0.7，关闭 top-k（论文 0，RLT 接口 -1），EOS 或 1,024 个新 token | 与 E.1 一致。论文未列出生成种子，采用本仓库种子 42，由题目提示哈希与样本号派生，同一题和样本跨组配对 |
| 对照 | 固定 2／4 轮。动态组统一四轮 prefill，decode 允许 1–4 轮，按原生退出 CDF 扫 `0.20, 0.25, 0.28, 0.30` | 完整固定 1／2／3／4 轮，Dense-Ragged 阈值 `0.20, 0.25, 0.28, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60, 0.75` |
| 判分 | 对生成文本的最后一个 `the answer is …` 严格提取，到句号或换行并与 target 精确匹配。未提取到答案记错 | E.1 使用答案提取及精确匹配。最后一处匹配和标点边界是本地实现细节 |
| 运行 | vLLM-RLT v0.2.0 `c08b9d7`，BF16 权重与 KV，Triton，同步 CUDA Graph，KV block 64，A800 四卡按题分片，每卡最多 16 条并发，关闭前缀缓存和投机解码 | 论文在 H100 执行。权重选用论文链接的公开模型，运行时使用独立的 vLLM-RLT 实现 |

vLLM-RLT 的 `LAST_EXITED` 执行原生 CDF 早退。它把退出轮的 KV 值复制到未执行轮的缓存页，后续注意力看到的值与论文式 (7) 的别名规则一致。RLT 有每轮的物理缓存页，BBH 结果用于复现低深度精度曲线和计算轮次。论文的物理页压缩需用页别名实现单独验证。[vLLM-RLT 发布版](https://github.com/InferMatrixLab/vllm-rlt/tree/c08b9d79834bb8b4ff283b15bae2c14c559f6b52) · [KV 复制实现](https://github.com/InferMatrixLab/vllm-rlt/blob/c08b9d79834bb8b4ff283b15bae2c14c559f6b52/vllm_rlt/core/kv_cache_manager.py#L751-L777)

首批 6 组共 `135 × 4 × 6 = 3,240` 条生成，最多 3,317,760 个新 token。补齐当前模型的完整 14 组时总计 7,560 条，已保存的 3,240 条可复用。2.6B 使用独立计划和相同题目。 [BBH 数据集](https://huggingface.co/datasets/lukaemon/bbh) · [官方 CoT 提示](https://github.com/suzgunmirac/BIG-Bench-Hard/tree/9ee07bd481feebf959a6b59d61ea57bdcf30964d/cot-prompts) · [Ouro-1.4B 权重](https://huggingface.co/ByteDance/Ouro-1.4B/tree/574fa66cb8bf5abdc979642d01cf2b79b16bfab1)

主对照是 `Ragged(0.20) − Ragged(0.28)`。每组报告 P@1（每题四条的正确比例再按题平均）、P@4（四条中至少一条正确的题占比）、按 token 加权的平均 decode 深度、首轮退出比例、输出长度、截断率、未抽取到答案的比例和实际 token-loop。深度统计排除由 prefill 预测的首个生成 token。差值按子任务分层、在同一批题上做 5,000 次配对 bootstrap，报 95% 区间。判断左段现象时同时看 `0.20` 的深度是否更低、P@1 差值是否为负，以及 P@4 的降幅是否更小。`0.28` 与固定 2／4 轮给出近预算和满深参照，随结果报告实际深度差。

## AIME 2024 · 目标模型和回填对照

使用现有的 `data/aime/aime2024.jsonl` 全部 30 道题，原始 `Ouro-1.4B-Thinking`（revision `3aaa222`），0-shot Thinking chat template，温度 1.0，top-p 0.7，top-k 0，种子 42，每题 4 条，EOS 或 16,384 个新 token。按现有配置加载 FP32 权重并以 BF16 autocast 执行 PyTorch SDPA。从同一条回答截取 2K／4K／8K／16K 评分，用严格 `\boxed{整数}` 判分。论文 Table 6 的 AIME 使用 8 条／题、32,768 上限与 Qwen 判分。这里检验的是 RAD-OPD 计划中实际要训练的模型和预算。[AIME 数据清单](../data/aime/manifest.json) · [现有 RAD-OPD 配置](../configs/rad_opd.json) · [论文 Table 6](https://arxiv.org/abs/2610.09013v1)

固定 1／2／4 轮作为锚点。精确回填需要重新计算缺失的深轮状态，统一用本仓库 PyTorch 运行这个机制对照。动态精确回填与动态零回填均用四轮 prefill、相同原生门，扫 `0.15, 0.20, 0.25, 0.30, 0.45, 0.75`，共 15 组和 1,800 条生成。按同阈值、同题和同样本号报告配对对照，也按实测平均 decode 深度比较两条曲线在 `[1.5, 2.0]` 的重叠区间。比较精度时同时报告回填额外的 token-loop。观察到回填在相近深度下优于静态浅轮、零回填的 P@1 低于回填且 P@4 降幅更小，才具备计划所需的证据。[RAD-OPD Phase 0](rad_opd_mvp.md#3-phase-0零训练证伪1-至-2-天) · [AIME 评测入口](../scripts/evaluate_stage_distillation.py)

## 命令

准备脚本把原始数据、官方提示和冻结题集存到 `~/data/huggingface_datasets/BBH`。评测计划默认读取其中的 `frontier_135_seed1234/eval.jsonl`。准备输入与校验模型文件均在 CPU 上完成。重新冻结计划时指定一个新目录。

```bash
.venv/bin/python scripts/prepare_ragged_bbh.py
.venv/bin/python scripts/reproduce_ragged_bbh.py plan --model Ouro-1.4B --shards 4 \
  --output ep_loop_runs/ragged_bbh_rlt_1p4b_20261123
```

正式评测需确认后执行。每个进程绑定一张空闲卡和一个分片。以下为分片 0 的调用，对其余分片指定各自的 GPU。每条结果提交到该分片的 SQLite，重启同一分片时跳过已提交的回答。首批汇总完成后可用 `--stage full` 补跑并汇总完整曲线。

```bash
plan=ep_loop_runs/ragged_bbh_rlt_1p4b_20261123/plan.json
.venv/bin/python -u scripts/reproduce_ragged_bbh.py run --plan "$plan" --stage core --gpu 0 --shard 0
.venv/bin/python scripts/reproduce_ragged_bbh.py summarize --plan "$plan" --stage core
```

AIME 使用现有原始权重入口，按以下参数启动。

```bash
.venv/bin/python scripts/evaluate_stage_distillation.py --config configs/rad_opd.json --phase-zero --write-token-ids \
  --data data/aime/aime2024.jsonl --output ep_loop_runs/ragged_aime2024/responses.jsonl
```
