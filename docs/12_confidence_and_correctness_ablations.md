# 置信度与正确性消融

主方法仍直接基于LoopOPD：原生教师终点、JS阈值阶段、段末reverse KL与学生停止BCE。下面均是显式开启的消融，不改变默认JS方案，不把动态学生改成固定轮数。

## 一次只替换边界信号

| `BOUNDARY_SIGNAL` | 原始前缀还是探针分支 | 信号与局限 |
|---|---|---|
| `js`（默认） | 原始前缀 | 完整词表相邻轮JS；不保证变化有益 |
| `max_probability` | 原始前缀 | `c=max(q)`，用相邻轮`abs(Δc)`分段；答案身份变化但信心不变时可能漏检 |
| `normalized_entropy` | 原始前缀 | `c=1−H(q)/log(V)`，用`abs(Δc)`分段；低熵不保证正确 |
| `probe_know` | 插入“现在能否正确回答”的prompt | 读取Yes/No标签序列的联合概率并归一化；是自报信心，需校准 |

最大概率和归一化熵复用教师逐轮原始分布，无额外backbone前向。`probe_know`探针只看到当前学生文本前缀，不看到学生后续回答或标准答案；阶段蒸馏目标仍是原始前缀上的教师分布。探针prompt保存在[confidence.py](../ep_loop/confidence.py)，修改后应视作新的实验配置并保留源码哈希。

探针改变输入；为了测量不同深度，会在探针分支逐个读取深度d。该读出不是对原始隐态中“已经知道答案”的直接测量。主rollout和部署仍由学生自己的退出头动态停止，上述逐深度控制只属于测量仪器。独立的强制回答诊断仍可显式开启，但不构造候选答案真假信心，也不参与阶段标签生成。

所有信号使用绝对变化，既保留信心上升也保留下降；超过阈值才分段，保留全部边界，不做Top-S。原生终点`n_t`之外的轮次不进入标签；无边界是一段。各组保持相同KL、BCE与教师规则。原始JS始终单独统计，`mean_boundary_change`才是所选信号的平均变化，避免把置信度变化称为JS。

探针各深度读出来自同一条循环轨迹，不逐深度从头重跑；Yes/No都是单token时共用一次前向，多token标签分别teacher-force并共用Huginn原生随机初态。这样既减少重复计算，也避免把每轮重新抽样的初态差异当成信心变化。

`JS_THRESHOLD=0.05`与`CONFIDENCE_THRESHOLD=0.05`只是各自未校准的开发起点，两种单位不可直接比较。先做离线诊断，再按开发集实际成本选择可比点。探针额外token-loops、coda位置数、前向次数与日志阶段单列；不应将其成本归到原始教师轨迹后隐去。Huginn探针使用隔离的原生随机初态流，退出后恢复正常Torch RNG。

配置入口：`g_dynamic_max_probability.json`、`g_dynamic_entropy.json`、`g_dynamic_probe_know.json`；Huginn均有`huginn_`前缀版本。

## 正确性只筛教师答对的题目

`question filter`表示题目筛选：初始冻结教师先独立按原生动态策略答题，再决定该训练题目是否保留。学生生成的回答全部参与蒸馏，无论回答是否正确；错误前缀同样需要教师提供纠正信号。学生回答过滤已经从代码、配置与启动脚本移除。

| `QUESTION_FILTER` | 含义 |
|---|---|
| `all`（默认） | 不根据教师是否答对过滤题目 |
| `teacher_correct` | 只保留初始冻结教师独立答对的训练题目，B/E共享相同池 |
| `random_matched` | 用同一教师缓存确定保留数量，再用seed42随机选相同数量的训练题目，控制样本量 |

不把“教师曾经答对”当成每个当前学生前缀或中间分布都正确。教师缓存只保存题目接受标签，不复用其退出深度和阶段标签。训练/dev/eval先划分，只过滤训练；开发与测试保留全部题目。

开启教师筛题时，当前验证器对boxed提示严格匹配最后一个`\boxed{整数}`；公开DAPO原始提示同时接受其要求的`Answer:`整数行（优先采用最后一个Answer行），默认还要求教师EOS完整终止。符号或代码任务需先接入合适验证器。`FILTER_REQUIRE_EOS=0`可允许教师带正确且可解析答案的截断回答；该参数在`QUESTION_FILTER=all`时不产生正确性过滤。

缓存校验种子、模型revision、原生退出、采样、题目与标准答案哈希。重复ID、缺标签、答案变化或旧schema缓存均拒绝。一次教师作答成本写入缓存旁`.summary.json`；比较时报告保留题目数量、一次筛题成本和完整训练／推理成本。题目数量减少不应悄悄变成更多更新；同数量随机题目对照用于区分正确性与样本量因素。

## 启动参数在sh开头

```bash
# all=默认不过滤；teacher_correct=教师答对才保留训练题目。
QUESTION_FILTER="${QUESTION_FILTER:-all}"
FILTER_REQUIRE_EOS="${FILTER_REQUIRE_EOS:-1}" # 仅开启教师筛题时生效。
```

默认只读检查：

```bash
bash scripts/run_ablation.sh --dry-run

# 教师正确性筛题消融；执行时先在训练划分生成一次共享缓存。
QUESTION_FILTER=teacher_correct bash scripts/run_ablation.sh --dry-run
QUESTION_FILTER=random_matched bash scripts/run_ablation.sh --dry-run

# 置信度边界消融与教师筛题独立；学生始终动态退出。
BOUNDARY_SIGNAL=max_probability CONFIDENCE_THRESHOLD=0.05 bash scripts/run_ablation.sh --dry-run
BOUNDARY_SIGNAL=probe_know bash scripts/run_ablation.sh --dry-run
```

移除`--dry-run`才执行。`--prepare-only`只准备划分和配置，不生成教师答案或加载权重；手动启动teacher_correct配置前，需先生成共享缓存。

独立JSON配置使用`distillation.question_filter`与`distillation.require_eos`，缓存路径为`data.correctness_cache`。已有`e_dynamic_js_teacher_correct.json`及对应LoopOPD、Huginn版本；启动前按实际缓存位置修改路径。旧配置若要求学生回答过滤会明确报错。

## 先做少量前缀诊断

```bash
CUDA_VISIBLE_DEVICES=7 PYTHONHASHSEED=42 .venv/bin/python scripts/diagnose_confidence.py \
  --config configs/e_dynamic_js.json --data /path/to/train_dev_diagnostic.jsonl \
  --limit 64 --positions-per-response 4 --max-new-tokens 128 \
  --signals js max_probability normalized_entropy probe_know \
  --forced-answer-check --output /path/to/confidence_diagnostic.jsonl
```

该脚本从原生动态教师续写得到前缀，每条回答均匀抽样若干位置，保存各轮原始信号及探针信心。`--forced-answer-check`另外在各深度分支短生成最终答案，再用标准答案验证，帮助观察错→对／对→错；标准答案仅进入外部验证器。它会增加计算，且测到的是带强制提示及额外生成的可答性，不能直接作为因果“必要探索”标签。该诊断默认最多生成16 tokens，可用`--forced-answer-max-tokens`修改；短截断回答不能视为完整答案正确率。

默认诊断只计算JS、最大概率、归一化熵，不运行昂贵的prompt探针或强制回答。`--smoke --device cpu`明确使用随机小模型。记录保存完整阶段事件、进度、环境、信号与成本，不能把此诊断称为真实任务效果验证。

诊断记录保存probe_know的Yes/No原始标签log概率，方便检查模型是否只是在两个低概率选项之间归一化。正式探针训练不逐深度写这些明细，也不为日志增加每轮GPU标量同步。
