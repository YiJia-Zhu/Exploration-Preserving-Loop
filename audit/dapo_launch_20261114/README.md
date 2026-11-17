# DAPO 64题、16,384 tokens 消融启动记录

后续状态：本次初始评测已完成（40/64，62.5%），训练第2次更新因OOM失败，未保存checkpoint。2026-11-15已修复显存并创建独立运行目录，从初始权重重训且复用初始评测；当前进度见[重启记录](../long_context_recovery_20261115/README.md)与`experiments/ablations/latest_run.json`。下文保留首次启动时记录。

2026-11-14 20:08:42（Asia/Shanghai）启动。后台launcher PID为`2888544`，完整记录见[launch.json](launch.json)。本文件记录启动时检查结果，后续实时状态以进程和事件日志为准。

运行目录：`experiments/ablations/20261114_200842_ouro_e_js_dapo64_len16384_seed42_gpu23/`。

| 项目 | 本轮实际配置 |
|---|---|
| 模型 | Ouro-1.4B-Thinking，revision `3aaa2224253a92ca45cf2e3d427c360e1ef9c93d` |
| GPU | 物理GPU 2、3；初始与快照评测使用GPU 2，训练使用两卡DDP |
| 方法 | E：JS threshold阶段监督，冻结教师，学生学习逐token动态停止；最大4轮 |
| JS／退出 | JS=0.05 nats；教师CDF=0.9；学生停止概率=0.5；停止BCE权重=1 |
| 训练数据 | seed42抽取256题；与开发评测题目按prompt内容互斥 |
| 评测数据 | DAPO留出64题，即本轮全部开发集；每题1次采样 |
| 长度 | prompt最多1,024；训练与评测max_new_tokens均为16,384，遇EOS提前结束 |
| 更新与batch | 20次更新；4题×每题4条回答=全局16；每rank处理2题×4条回答 |
| 优化 | AdamW LR=1e-6，WD=0.01，betas=(0.9,0.999)，eps=1e-8，无warmup，无梯度裁剪 |
| 精度与缓存 | FP32主参数，BF16计算；gradient checkpointing；readout chunk=128；精确动态KV缓存 |
| 采样 | 训练T=1/top_p=1；评测T=1/top_p=0.7；seed42，不启用严格确定性算子 |
| 筛选 | 不按教师正确性筛题，不按学生回答正确性过滤 |
| 顺序 | 原生动态初始评测 → E组训练 → 第10与20次更新快照评测 |
| 多卡等待 | collective timeout为21,600秒，容纳长回答造成的rank耗时差异 |

## 数据来源与处理

固定公开仓库`BytedTsinghua-SIA/DAPO-Math-17k`，revision `65877096c24ffa7abc4e4fa5edb95cf3413a5674`，文件`data/dapo-math-17k.parquet`。共享原始文件保存至`/home/zhenning/data/huggingface_datasets/DAPO-Math-17k/`，下载SHA256校验通过，见[download_manifest.json](download_manifest.json)。

原文件1,791,700行、17,917个源ID。按完整prompt消息去重得到17,398道题，其中7道带有冲突整数答案。显式使用`--conflicting-answers drop`排除这7题的全部副本，不选择其中某个标签。保留17,391题后写入`data/dapo_math17k/train.jsonl`；排除的prompt哈希与冲突答案见[source_manifest.json](source_manifest.json)，未规范化或改写原prompt。

使用Ouro tokenizer筛掉17道超过1,024 tokens的prompt，剩余17,374题按seed42洗牌，选择训练256题、开发64题，评测与开发集相同。训练与评测无重复prompt。源文件、模型版本、分区内容的SHA256均写入运行目录的`data_manifest.json`。

该处理可复现公开数据上的本轮消融；尚未独立确认其与论文内部训练子集完全一致。开发评测用于选方法，不作为最终测试集。

```bash
.venv/bin/python scripts/prepare_stage_data.py \
  --input /home/zhenning/data/huggingface_datasets/DAPO-Math-17k/dapo-math-17k.parquet \
  --output data/dapo_math17k/train.jsonl \
  --deduplicate --conflicting-answers drop --expected-examples 17391 \
  --source-repo BytedTsinghua-SIA/DAPO-Math-17k \
  --source-revision 65877096c24ffa7abc4e4fa5edb95cf3413a5674

# 后续新运行；不重复启动本记录中的同一输出目录。
GPUS=2,3 bash scripts/run_ablation.sh
```

## 评测修复与验证

公开DAPO原prompt明确要求最后输出`Answer:`。验证器对该提示接受最后一个`Answer:`整数行或boxed整数答案；存在Answer行时以最后一行为准，拒绝模糊或无效的最终答案。其他boxed提示维持严格boxed解析。逐回答记录实际verifier名称，不把正文中任意整数当作答案。教师筛题与置信度诊断复用同一解析，当前运行未启用这两项。

72项CPU测试通过，见[tests.log](tests.log)，涵盖DAPO最终答案格式、严格boxed兼容、冲突题整体排除与默认拒绝冲突。shell语法和只读资源检查通过；训练与评测配置均验证16,384上限。启动后模型加载成功，已观察到GPU 2上的真实token生成；这不代表训练或评测已经完成。

启动时的代码、脚本与配置已保存到`launch_sources.tar.gz`，文件哈希见[launch_source_sha256.json](launch_source_sha256.json)。运行目录内还保存解析后的配置、launcher副本、数据manifest和阶段事件。

```bash
# 当前初始评测的进度、token进度和报错。
tail -f experiments/ablations/20261114_200842_ouro_e_js_dapo64_len16384_seed42_gpu23/logs/eval_initial_native.log

# 全流程日志，包含后续训练与快照评测。
tail -f audit/dapo_launch_20261114/launch.console.log

# 到训练阶段后，查看每rank阶段起止时间与逐更新统计。
tail -f experiments/ablations/20261114_200842_ouro_e_js_dapo64_len16384_seed42_gpu23/ours/events_rank0.jsonl
```
