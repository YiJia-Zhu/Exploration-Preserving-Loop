> 历史记录：本文描述 2026-11-08 的查询干预和 LoRA 候选辅助。当前方案已改为[原生动态退出与 JS 阶段蒸馏](loop_distillation_research_proposal_zh.md)，本文数字与原始方法保留追溯，不作为新方法的实验结果。

> 当前核心代码与 Ouro/Huginn 接口核验见[实现说明](08_implementation_and_reproduction.md)；本文件的历史结果不变。

# 2026-11-08：Ouro / LoopOPD 首轮可行性实验

**本轮已经实际拉取代码、加载模型、执行干预和短训。结果支持工程可行性，以及部分跨轮更新的任务价值；12 步短训尚未显示中间辅助优于 LoopOPD。**没有完成完整 EP，也没有证明 LoopOPD 丢失了必要的多候选探索。

## 1. 拉取了什么，改了什么

- 官方模型：[ByteDance/Ouro-1.4B-Thinking](https://huggingface.co/ByteDance/Ouro-1.4B-Thinking)，revision `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d`。权重与模型源码现统一存放于 `/home/zhenning/data/huggingface_models/Ouro-1.4B-Thinking`，项目内 `models/Ouro-1.4B-Thinking/` 为软链接。权重 SHA256 与官方 LFS 一致，见[下载记录](../audit/model_download.json)。
- 查询干预官方代码：[Muuzzziiy/cql-handoff](https://github.com/Muuzzziiy/cql-handoff)，commit `75b52dbfdb160eb5da59b7e7897bbeb0bb16ec91`。复用其原生读出、层／head 定位及 SDPA 查询替换接口；没有把仓库自带结果冒充新实验。
- [LoopOPD 项目页](https://github.com/wanglne/LoopOPD-page) commit `4994c802abfae6dd891bbd700b4f2bd8c530360d` 仍标注代码和模型 coming soon。本轮的训练实现根据论文目标编写，属于受控适配。模型卡所列 Ouro GitHub 地址返回 404，使用可获得的官方 HF 源码。
- 新增 [runtime.py](../ep_loop/runtime.py)：可微的各轮读出、固定深度控制、FP32 LoRA、关闭 adapter 的冻结教师、完整词表 reverse KL、候选 + OTHER 保留损失，以及每步重新采样的两轮学生 rollout。
- 新增[诊断脚本](../scripts/run_diagnostic.py)、[短训脚本](../scripts/run_training_pilot.py)、[结果核验脚本](../scripts/verify_pilot.py)及[目标函数测试](../tests/test_losses.py)。原始模型源码与上游仓库未改写；适配逻辑单独保留。

详细设计见[执行协议](../experiments/pilot_protocol_20261108.md)，环境与代码来源见[环境记录](../audit/pilot_environment.json)。Python 3.10.20、PyTorch 2.8.0/CUDA 12.8、Transformers 4.57.1；使用物理 GPU 4，NVIDIA RTX PRO 5000 72GB。

## 2. 工程检查

最初按模型卡建议使用 Transformers 4.54.1，缓存构造出现 `key_cache` 只读属性错误，未进入任务评测。失败日志保留在[首次运行](../experiments/runs/20261108_diagnostic_v1/run.log)。切换到 4.57.1 后重跑，没有修改数据或放宽成功阈值。

[D0 记录](../experiments/runs/20261108_diagnostic_v2/engineering.json)的 14 项检查通过：

- 两个输入、四个循环出口，原生读出与独立停止前向的 logits 逐项完全相同。
- 两个 query 恒等替换检查完全相同。
- 两／四轮缓存解码与完整前缀比较，full-vocab top-1 一致，KL 为 0.00031–0.00070，低于预设 0.01。BF16 下两种计算形状的 logits 有小差异，未声称逐位相同。

实际诊断和训练均采用完整前缀重算。下载的 chat template 会输出字面量 `\n`；本轮使用论文的原始 completion 模板，没有据此评价原生 chat 长推理表现，也没有静默修改模板。

## 3. 新场景中的中间更新是否有用

独立生成 64 个场景，seed 为 20261108；每个场景包含两跳问题和两个单跳控制，共 192 个问题。64 个场景的语义内容也不同。题目形式为“Carol 有 B 盒；B 盒内是黄色；Carol 的盒子里是什么颜色？”

**下表全部是四候选排序准确率，不是真实生成的答案正确率。**采用固定候选可直接检查各轮读出，不能由此推断数学或自然问答能力。

| 两跳题条件 | 第一轮 | 第二轮 | 第三轮 | 第四轮 |
|---|---:|---:|---:|---:|
| 原始模型 | 25/64，39.06% | 48/64，75.00% | 50/64，78.13% | 54/64，84.38% |
| 后续轮的回答位置 query 固定为第一轮值 | 25/64，39.06% | 25/64，39.06% | 24/64，37.50% | 27/64，42.19% |
| 无关事实位置做同规模 query 冻结 | 25/64，39.06% | 47/64，73.44% | 50/64，78.13% | 54/64，84.38% |

冻结实验保留实际计算轮数。第四轮的回答位置冻结损伤为 **42.19 个百分点**，场景级配对 bootstrap 95% 区间为 **[29.69, 54.69]**；无关位置冻结的平均损伤为零。区间基于 5000 次探索性重采样，未作多重比较调整。

单跳控制也有损伤：第四轮“人→盒子”为 100%→95.31%，“盒子→颜色”为 95.31%→81.25%。因此冻结有一部分一般检索影响；不能把全部损伤都归于两跳组合。扣除较大的单跳损伤，组合特异损伤仍为 28.13 个百分点，核验脚本会重新计算。

在 25 个“第一轮错、第三轮对”的场景中，把同题第三轮各层／head 的查询移入第一轮，修复 **23/25（92%）**；等长度其他题 donor 修复 **5/25（20%）**。这组 donor 从所有等长度输入中轮换选择，与原论文只在目标集合中选择的协议有差异。全体 64 题移植后的第一轮准确率为 47/64（73.44%），没有只报告筛选后子集。后轮 donor 需要额外计算，结果不能当作免费的一轮加速。

逐题输入、概率和干预记录见[诊断目录](../experiments/runs/20261108_diagnostic_v2/)，汇总见[summary.json](../experiments/runs/20261108_diagnostic_v2/summary.json)。

### 一个实际分布变化例子

场景 3 的正确链是 `Carol → B 盒 → yellow`。下面的概率为完整词表中小写、带前导空格的 ` yellow` token 的原始概率：

| 条件 | 四候选中排名第一 | P(` yellow`) |
|---|---|---:|
| 第一轮 | green | 0.234% |
| 第二轮 | yellow | 2.137% |
| 第三轮 | yellow | 3.702% |
| 第四轮 | yellow | 5.125% |
| 冻结回答位置 query 后的第四轮 | green | 0.0114% |

这个样例说明某次候选修正与查询更新的任务作用相连。它是汇总实验中的解释性例子，不能单独证明多个候选同时保留是必要探索。

## 4. LoopOPD 与中间辅助是否能训练

在冻结底座上训练 q/v 的 rank-8、alpha-16 LoRA，共 **1,572,864 个可训练参数**。关闭 adapter 后的原模型作为四轮教师；两组都从相同的零输出 adapter 开始，使用 AdamW、学习率 1e-4、梯度范数裁剪 1。

训练池为 32 个新场景，与 64 个留出场景按原始文本及忽略事实排列的语义内容去重。每臂实际执行 **12 次更新、使用 12 个不同训练场景、生成 172 个 token**。每次更新前由当前两轮学生重新采样，温度 1，最多 16 token，无 top-k/top-p。每个回答的 endpoint KL 按长度归一化。

- LoopOPD 适配：完整词表 `KL(S2 || 冻结 T4)`，仅监督生成 token 对应的前缀。
- 固定中间辅助：上述目标 + `0.2 × KL(聚合 T2 || 聚合 S1)`，只在首个回答前缀使用四个颜色 token 加 OTHER。这里 `g=1`，没有因果价值门控，没有关键修正项，是完整 EP 的前置对照。

两组 12 步梯度均有限且非零，零 adapter 初始前向与原模型完全一致，训练后关闭 adapter 的教师参考 logits 完全不变。保存后重新加载两个 adapter，首个留出题四个出口的候选 logits 均与原记录逐项一致。四轮学生也随共享 adapter 改变；表中的训练后第四轮是当前学生，冻结教师始终是初始模型。

| 64 题留出集，四候选排序 | 第一轮 | 第二轮（目标部署预算） | 第三轮 | 第四轮 |
|---|---:|---:|---:|---:|
| 训练前 | 39.06% | 75.00% | 78.13% | 84.38% |
| LoopOPD 适配，12 步 | 46.88% | **75.00%** | 81.25% | 89.06% |
| LoopOPD + 固定中间辅助，12 步 | 46.88% | **75.00%** | 82.81% | 89.06% |

两组第二轮均为一题改善、一题退步，净值为零。第三轮的一题差异不支持独立有效性结论。第一轮全词表平均熵从 7.377 降至 4.315／4.275，同时候选排序准确率提高，这也说明熵下降本身不能被当作探索损伤。

保存的训练记录与 adapter：

- [LoopOPD 适配](../experiments/runs/20261108_loopopd_pilot/)，[权重](../experiments/runs/20261108_loopopd_pilot/adapter.pt)。
- [固定中间辅助](../experiments/runs/20261108_middle_pilot/)，[权重](../experiments/runs/20261108_middle_pilot/adapter.pt)。

诊断程序计时 22.20 秒，两臂各 10.57／10.43 秒，均包含各自模型加载及内部评测，不包含环境安装和下载。训练峰值 PyTorch 显存约 3.15 GiB。这是很小的工程实验预算。

## 5. 这次可以得出什么

**已经支持：**当前 Ouro-Thinking checkpoint 的循环读出可可靠测量；一些中间 query 更新对合成两跳检索有因果作用；LoopOPD 目标和候选保留辅助可以在同一代码中实际反向传播和更新。

**尚未支持：**LoopOPD 损伤了原有探索；多个有价值候选需要同时保留；固定中间辅助优于 LoopOPD；完整 EP 在数学或代码任务上有效。候选排序、全词表熵和真实采样的解法覆盖不能互换。当前短训也未控制不同方法最终实际总 FLOPs，因此不作效率优越性结论。

下一步应先在原生、经过格式检查的生成任务上做 D2：识别候选对学生后续成功率的影响，再决定 EP 的门控。之后使用多个训练 seed、较充分训练、固定／随机辅助及同成本对照。当前结果不支持直接投入复杂动态选择器或宣布成功。

## 6. 重跑

本工作区已配置 `.venv`，底层 Torch 复用现有 `tokenrouter-blackwell` 环境；新机器需先安装匹配 GPU 的 Torch/CUDA，再安装 [requirements-pilot.txt](../requirements-pilot.txt)。cql-handoff 仓库需位于 `external/cql-handoff`，并检出上述固定 commit；模型可由[下载脚本](../scripts/download_pilot_model.py)获取。

```bash
.venv/bin/python -m unittest discover -s tests -v

CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/run_diagnostic.py \
  --out experiments/runs/my_diagnostic --n-scenes 64

CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/run_training_pilot.py \
  --out experiments/runs/my_loopopd \
  --evaluation-inputs experiments/runs/my_diagnostic/inputs.json --mode loopopd

CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/run_training_pilot.py \
  --out experiments/runs/my_middle \
  --evaluation-inputs experiments/runs/my_diagnostic/inputs.json --mode middle

# 核验本次保存的结果；加此参数会重新加载并验证两个 adapter 的输出。
CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/verify_pilot.py --replay-checkpoints
python scripts/validate_package.py
```

运行前选择空闲 GPU；脚本对已有逐题输出拒绝覆盖。核验脚本当前固定读取本次三个正式目录；其结果见[运行核验](../audit/runtime_validation.json)，目标函数测试日志见[测试记录](../audit/objective_tests.log)。
