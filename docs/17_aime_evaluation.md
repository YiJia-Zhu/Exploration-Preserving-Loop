# AIME评测协议

三个启动脚本默认`EVAL_SUITE=aime`，当前先评AIME2024：30题、每题16次，每个模型预算生成480条回答。AIME2025／2026数据保留，可扩展为三年各30题×16次、合计1,440条，与LoopOPD附录D.2的评测规模对应。分别报告各年的Avg@16＝正确回答数／480、Pass@16＝至少答对一次的题数／30，不把三年合并分数替代分年份结果。

## 数据与提示

| 年份 | 公开来源 | 固定revision |
|---|---|---|
| 2024 | [HuggingFaceH4/aime_2024](https://huggingface.co/datasets/HuggingFaceH4/aime_2024) | `2fe88a2f1091d5048c0f36abc874fb997b3dd99a` |
| 2025 | [opencompass/AIME2025](https://huggingface.co/datasets/opencompass/AIME2025)，I与II各15题 | `a6ad95f611d72cf628a80b58bd0432ef6638f958` |
| 2026 | [MathArena/aime_2026](https://huggingface.co/datasets/MathArena/aime_2026) | `d61f486805306e6e6b5913504c07763a99e1b112` |

原始文件集中缓存于`/local_nvme/data/zhenning/huggingface_datasets/AIME年份/revision/`；项目`data/aime/`包含合并90题的eval.jsonl、各年独立JSONL与manifest。脚本校验源文件SHA256、每年30题、唯一题目ID与0..999整数答案。原题后只加论文的逐步推理／boxed答案提示，源数据中的solution不进入模型输入。2025一条带角度符号的答案及2024前导零答案转成规范整数，原值与转换记录在manifest。

```bash
# 本地已有缓存时准备；缺文件则明确报错。
.venv/bin/python scripts/prepare_aime_data.py

# 显式下载缺失的固定版本源文件；不下载模型。
.venv/bin/python scripts/prepare_aime_data.py --download
```

这三份公开数据覆盖对应年份的30题；尚未与作者内部评测文件逐项比较，不宣称文件完全相同。

## 参数与评分

启动脚本开头可调：`EVAL_DATA_PATH=data/aime/aime2024.jsonl`、`EVAL_EXAMPLES=30`、`EVAL_SAMPLES_PER_PROBLEM=16`、`EVAL_TEMPERATURE=1.0`、`EVAL_TOP_P=0.7`、`EVAL_MAX_NEW_TOKENS=16384`。seed42按题目内容与采样编号派生，不受GPU分片和完成顺序影响。数据、源码与参数哈希均记录。

vLLM评测每题只分词一次，然后构造16个采样请求，当前AIME24两卡各240条一次提交，扩展三年时各720条，使用框架默认序列并发。每条结果保留benchmark、题目ID、sample、sampling_seed、回答、退出深度和正确性。summary在每个深度下保留整体统计及benchmarks.AIME2024／AIME2025／AIME2026的独立统计，显式包含avg_at_16和pass_at_16。

当前评分读取最后一个boxed答案并比较整数值，支持前导零、整值小数和角度记号。没有接入通用符号等价验证，例如等价算式未必被接受。论文写明数值与符号等价判分，且长度上限为31,744；这些差异必须随结果披露。达到长度上限的回答保留在评测中并记录截断状态，不筛选正确回答。

## 训练重合与旧结果

默认`TRAIN_DECONTAMINATION_PATH=data/aime/eval.jsonl`，即使只评AIME24也保持排除三年AIME重合题，避免扩展评测时训练池变化。准备训练数据时，用规范化后的完整AIME问题文本检查是否包含在DAPO消息中；只规范空白、Unicode兼容字符及left／right命令，不声称识别改写或语义近重复。本次发现1道AIME2026问题与训练源原文重合，已从新训练池排除，ID记录在data_manifest.json。17,391源题减17道超长题、1道重合题和64道备用开发题，得到17,309道训练题。当前新目录带`_aime30_clean`；旧源文件与数据划分保留，旧运行结果和checkpoint已按后续清理要求删除，当前GPU6、7运行保留。

```bash
# 默认AIME计划；不启动模型。
ROUTE=vllm bash scripts/run_three_route_ablation.sh --dry-run

# 只准备新训练划分、AIME配置和来源记录。
ROUTE=vllm bash scripts/run_three_route_ablation.sh --prepare-only

# 扩展到三年AIME，训练池保持不变。
ROUTE=vllm EVAL_DATA_PATH=data/aime/eval.jsonl EVAL_EXAMPLES=90 \
  bash scripts/run_three_route_ablation.sh --dry-run

# 历史DAPO开发评测，默认恢复64题×1次。
ROUTE=vllm EVAL_SUITE=dapo_dev bash scripts/run_three_route_ablation.sh --dry-run
```

统一入口仍限制整个panel最多10小时，包含初始教师／学生参考、最终评测和交叉线各固定预算。当前一次AIME24评测回答数是旧64题×1协议的7.5倍，三年版本为22.5倍，10小时不保证训练100步及全部评测都完成。旧DAPO分数与新AIME分数分别报告。
