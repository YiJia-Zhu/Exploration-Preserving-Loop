# Exploration-Preserving Loop Distillation

当前默认路线为 **RAD-OPD**。Ouro-1.4B-Thinking 用原生 gate 和零回填 Ragged KV 在策略生成。训练时每题抽取一个退出阈值，冻结的满四轮教师在相同文本上打分，学生沿实际退出轨迹可微重放并做全词表 Reverse KL。退出头参数冻结。[方案](docs/rad_opd_mvp.md) · [默认配置](configs/rad_opd.json)

## RAD-OPD 入口

检查本地模型和 DAPO 去重训练池。

```bash
.venv/bin/python scripts/train_stage_distillation.py --check-assets
```

Phase 0 用原始权重执行固定 1／2／4 轮、动态精确回填和动态零回填对照。

```bash
.venv/bin/python scripts/evaluate_stage_distillation.py --phase-zero \
  --data data/aime/aime2024.jsonl --output ep_loop_runs/rad_phase0/responses.jsonl
```

训练单个 checkpoint，然后扫退出阈值。重复传 `--threshold` 可选取部分工作点。评测汇总包含 P@1、P@4、多预算切片、平均 decode 深度、Depth-1 比例和每层实际占用的 KV 槽。切片从同一条长回答截取，decode 深度统计从处理首个生成 token 开始。

```bash
.venv/bin/python scripts/train_stage_distillation.py --device cuda --output ep_loop_runs/rad_opd
.venv/bin/python scripts/evaluate_stage_distillation.py \
  --checkpoint ep_loop_runs/rad_opd/checkpoints/update_000100 \
  --data data/aime/aime2024.jsonl --output ep_loop_runs/rad_opd/curve.responses.jsonl
```

消融配置可以继承 `configs/rad_opd.json`。设置 `algorithm.rad_detach_reused_kv=true` 截断浅轮 KV 复用梯度，或设置 `algorithm.rad_tau.fixed` 训练单个阈值。离线对照通过 `data.rad_offline_responses` 指向冻结四轮教师生成的 JSONL，每行包含 `problem_sha256`、`sample` 和 `tokens`，覆盖训练池及每题所需样本。用评测入口的 `--loops 4 --write-token-ids` 导出时，显式传原始权重并将数据校验设置对齐教师语料。离线训练沿给定文本重新计算学生退出轨迹。

当前 rollout 用 PyTorch 维护紧凑物理 KV，并在 attention 读取时临时 gather。训练重放用 `torch.where` 保留梯度，CUDA SDPA 选择可用的 attention kernel。物理槽统计不计分配器余量、索引和临时 gather。

## 备选路线

固定深度配置保留在 `configs/a_loopopd.json` 和 `configs/vllm_fixed_*.json`。此前的 JS 动态学生可显式选择 `configs/hf_dynamic_js.json`，其余 ablation 入口继续使用各自的配置。

以下命令预览实验计划，只检查配置。

```bash
bash scripts/run_opd_comparison.sh --dry-run
OPD_MODE=d_loopopd ABLATION_GROUPS=top_n bash scripts/run_opd_comparison.sh --dry-run
ROUTE=hf bash scripts/run_three_route_ablation.sh --dry-run
ROUTE=hybrid bash scripts/run_three_route_ablation.sh --dry-run
```

固定轮数评测入口为 [evaluate_fastpath.py](scripts/evaluate_fastpath.py)。评估已保存的结果时，可从[评测计划](audit/checkpoint_eval_20261120_004854/plan.json)取得 checkpoint、数据和配置。使用 [vLLM-RLT 配置](configs/evaluation_vllm_rlt.json)可切换到本机的 vLLM-RLT 引擎，每张 GPU 运行一个副本。

历史 EP-D-LoopOPD 十步 checkpoint 的固定两轮 AIME 2024 评测得到 **Avg@16 24.17% · Pass@16 56.67%**。[评测核验](audit/checkpoint_eval_20261120_004854/validation.json) · [历史计时](docs/runtime_timing.md) · [论文与证据索引](ref/README.md)
