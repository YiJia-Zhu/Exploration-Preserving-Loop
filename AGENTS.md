# Exploration-Preserving-Loop

本项目研究 Ouro 循环语言模型的低深度动态早退。RAD-OPD 的目标是在较少的实际解码轮数下提高准确率。主方案训练一个学生模型，评测时扫描退出阈值 `τ`。方法和实验设计见 [RAD-OPD 方案](docs/rad_opd_mvp.md)。

## 实验组和术语

Prefill 处理完整输入，包括示例、题目和提示词里已有的回答开头，并预测回答的首个 token。原生 Ouro KV 和 Ragged KV 都用满 4 轮 prefill，并从第 4 轮采样首个 token。

| 方案组名 | 执行含义 |
|---|---|
| **Dense 4-Loop** | Prefill 和 decode 都固定 4 轮，计算每轮自己的 KV。 |
| **Static 2-Loop / 1-Loop** | Prefill 和 decode 都固定在所选的 2 轮或 1 轮。 |
| **原生 Ouro KV** | 官方 vLLM 每步执行 4 轮并缓存各轮 KV。退出门选择当前步的采样轮次。 |
| **Ragged KV** | vLLM-RLT `LAST_EXITED` 在 decode 按退出门早退。深轮读取历史 token 最后实际执行轮次的 KV。 |

Ragged KV 的实际执行轮数取早退轮次。vLLM-RLT 把退出轮的 KV 值复制到深轮页，PyTorch 训练实现通过索引读取同一份 KV。显存统计依据各后端实际分配的物理 KV。[BBH 运行说明](docs/ragged_kv_reproduction.md)

## 实验顺序

1. **Phase 0 BBH**。用非 Thinking `Ouro-1.4B`，27 个子任务各 5 题，每题 4 次。首批固定 2／4 轮走官方 vLLM，动态 `τ=0.10 / 0.15 / 0.20 / 0.25 / 0.30` 走 vLLM-RLT。7 组共 3,780 条已完成。[BBH 结果与论文对照](docs/ragged_kv_reproduction.md#首批结果与论文对照)
2. **Phase 0 AIME 2024**。用原始 `Ouro-1.4B-Thinking`，30 题、每题 4 次。固定 1／2／4 轮和六个 Ragged KV 阈值，共 9 组、1,080 条已完成，按 2K／4K／8K／16K 切片评分。原生 Ouro KV 六个阈值的对照待跑。[AIME 结果](docs/ragged_kv_reproduction.md#aime-2024-最终结果)
3. **RAD-OPD MVP**。本机试跑直接读取原始 DAPO-Math-17k Parquet，教师使用初始权重，在 AIME 2024 上评测第 16 步权重。[试跑配置](configs/rad_opd_pilot.json) · [逐步刷新教师的配置](configs/rad_opd_pilot_current_teacher.json)

## RAD-OPD 训练

1. 学生和教师由同一原始权重初始化，冻结满 4 轮教师和学生原生退出门。
2. 学生按训练采样的 `τ` 做 Ragged KV rollout，保存生成 token 和实际退出轮次。
3. 教师在同一段学生输出上用满 4 轮打分。学生沿实际退出轨迹重放，深轮读取早退 token 的退出轮 KV。复用路径保留梯度，即方案中的 No-Detach。
4. 在学生实际退出轮对齐冻结教师的满深输出，以 Reverse KL `KL(学生 || 教师)` 更新学生主干。

## 入口与结果口径

- BBH 数据准备和评测用 `scripts/prepare_ragged_bbh.py`、`scripts/reproduce_ragged_bbh.py`。本机 BBH 两个引擎都用显存比例 `0.90`，各取默认 `max_num_seqs`，脚本每批最多 8 条，关闭投机解码。vLLM-RLT 和训练依赖在 `.venv`，官方 vLLM 在 `ep_loop_envs/vllm026`。[环境说明](docs/core_environment.md) · [吞吐记录](docs/runtime_timing.md)
- AIME Phase 0 用 `scripts/evaluate_stage_distillation.py --phase-zero`。固定组和原生 Ouro KV 用官方 vLLM 0.26，Ragged KV 用 vLLM-RLT。试跑训练用 `scripts/train_stage_distillation.py`，每步依次执行 vLLM-RLT 学生生成、官方 vLLM 四轮教师前向、PyTorch 可微重放和反传。
- 首批 BBH 复判用已有输出，在首个新 `\nQ:` 前取最后一次 `the answer is …` 做精确匹配。原运行的耗时和循环量仍按完整生成统计。
- 报告使用表中的组名和实测值。字段 `experiment_group` 记录组名，`τ`、实际平均解码轮数、平均读出轮次、总 token-loop 和物理 KV 占用各有单独的统计字段。P@1 是每题四条的正确比例再按题平均，P@4 是四条里至少一条答对的题目比例。论文读图值和本地结果注明各自口径。
