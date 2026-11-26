# Exploration-Preserving-Loop

本项目研究 Ouro 循环语言模型的低深度动态早退。RAD-OPD 的目标是在较少的实际解码轮数下提高准确率。主方案训练一个学生模型，评测时扫描退出阈值 `τ`。方法和实验设计见 [RAD-OPD 方案](docs/rad_opd_mvp.md)。

## 实验组和术语

Prefill 处理完整输入，包括示例、题目和提示词里已有的回答开头，并预测回答的首个 token。原版四轮 Ouro 和本方案的两个动态组都用满 4 轮 prefill。Decode 随后逐 token 生成回答，两个动态组在这一阶段都可以早退，分别按下表处理缺失的深轮 KV。

| 方案组名 | 执行含义 |
|---|---|
| **Dense 4-Loop** | Prefill 和 decode 都固定 4 轮，计算每轮自己的 KV。 |
| **Static 2-Loop / 1-Loop** | Prefill 和 decode 都固定在所选的 2 轮或 1 轮。 |
| **Dynamic + Backfill** | Decode 可以早退。后续 token 需要历史 token 的深轮 KV 时，补算缺失轮次。 |
| **Dynamic + Zero-Backfill Ragged KV** | Decode 可以早退。深轮读取历史 token 最后实际执行轮次的 KV，跳过缺失轮次的计算。方案要求每个早退 token 只保留实际执行轮次的物理 KV 槽。 |

按实际执行轮数和 KV 来源判断实验组。原版 Hugging Face Ouro 在满深度前向后按 `exit_threshold` 选读出轮次。BBH 动态组用 vLLM-RLT `LAST_EXITED`，实际跳过深轮计算，把退出轮的 KV 复制到深轮页。后续读取的值符合第 4 组的复用规则，物理页仍按轮分配。用这组结果评估准确率和计算轮数，显存节省以实际物理 KV 占用为准。[BBH 运行说明](docs/ragged_kv_reproduction.md)

## 实验顺序

1. **Phase 0 BBH**。用非 Thinking `Ouro-1.4B`，27 个子任务各 5 题，每题 4 次。首批固定 2／4 轮走官方 vLLM，动态 `τ=0.10 / 0.15 / 0.20 / 0.25 / 0.30` 走 vLLM-RLT。7 组共 3,780 条已完成。[BBH 结果与论文对照](docs/ragged_kv_reproduction.md#首批结果与论文对照)
2. **Phase 0 AIME 2024**。用原始 `Ouro-1.4B-Thinking`，30 题、每题 4 次。固定 1／2／4 轮和六个 Ragged KV 阈值，共 9 组、1,080 条已完成，按 2K／4K／8K／16K 切片评分。Backfill 对照待定。[AIME 结果](docs/ragged_kv_reproduction.md#aime-2024-最终结果)
3. **RAD-OPD MVP**。配置使用 DAPO 去重题池训练，在 AIME 2024 上评测。默认设置和阈值取自 [RAD-OPD 配置](configs/rad_opd.json)。

## RAD-OPD 训练

1. 学生和教师由同一原始权重初始化，冻结满 4 轮教师和学生原生退出门。
2. 学生按训练采样的 `τ` 做零回填 Ragged KV rollout，保存生成 token 和实际退出轮次。
3. 教师在同一段学生输出上用满 4 轮打分。学生沿实际退出轨迹重放，深轮读取早退 token 的退出轮 KV。复用路径保留梯度，即方案中的 No-Detach。
4. 在学生实际退出轮对齐冻结教师的满深输出，以 Reverse KL `KL(学生 || 教师)` 更新学生主干。

## 入口与结果口径

- BBH 数据准备和评测用 `scripts/prepare_ragged_bbh.py`、`scripts/reproduce_ragged_bbh.py`。本机 BBH 两个引擎都用显存比例 `0.90`，各取默认 `max_num_seqs`，脚本每批最多 8 条，关闭投机解码。vLLM-RLT 和训练依赖在 `.venv`，官方 vLLM 在 `ep_loop_envs/vllm026`。[环境说明](docs/core_environment.md) · [吞吐记录](docs/runtime_timing.md)
- AIME Phase 0 用 `scripts/evaluate_stage_distillation.py --phase-zero`。固定组走官方 vLLM，零回填走 vLLM-RLT，精确回填调用 `ep_loop/ouro.py` 的 PyTorch 实现。训练用 `scripts/train_stage_distillation.py`，当前 RAD-OPD rollout 和教师路径用 PyTorch，训练所用的零回填实现在 `ep_loop/ragged.py`。
- 首批 BBH 复判用已有输出，在首个新 `\nQ:` 前取最后一次 `the answer is …` 做精确匹配。原运行的耗时和循环量仍按完整生成统计。
- 报告时用方案里的组名和实测值。`τ`、实际平均解码轮数、总 token-loop 和物理 KV 占用各有含义。P@1 是每题四条的正确比例再按题平均，P@4 是四条里至少一条答对的题目比例。论文读图值和本地结果注明各自口径。
