# vLLM固定计算、门控读出与top-20近似验证

实现及命令见[独立路径说明](../../docs/14_vllm_top20_fastpath.md)，结构化汇总见[validation.json](validation.json)。未修改已安装vLLM／Torch／共享模型文件；官方Ouro源码已复制至项目内vendor，并在独立副本适配当前vLLM 0.11。

生产配置与原方案相比，算法只增加教师top-20原始概率＋均匀尾部的近似。原门控、阶段映射、停止BCE、reverse KL方向、数据划分、batch16、LR1e-6、seed42、20更新及16384上限均保留。新执行路径实际计算K轮，按原门控选读出；实际成本按K报告。

| 检查 | 证据 | 结果 |
|---|---|---|
| 84项CPU测试 | [all_tests.log](all_tests.log) | 通过 |
| 4题×128 tokens短吞吐 | [compiled_benchmark_v2.json](compiled_benchmark_v2.json) | HF37.39、vLLM280.37 tokens/s，约7.50倍 |
| 同前缀门控与分布 | 同上 | 4/4深度相同，最大top-20重建分布JS为3.12e-4 |
| 真实模型两次更新 | [two_real_updates/validation.json](two_real_updates/validation.json) | 全局batch16，64-token诊断通过；教师冻结、有限梯度 |
| 两卡真实DDP两次更新 | [two_gpu_real_updates/summary.json](two_gpu_real_updates/summary.json) | GPU2/3，每次全局16回答，32-token诊断通过 |
| 权重同步生效 | [weight_sentinel.json](weight_sentinel.json) | actor内门控bias更新后读出分别变为1／4；共享模型未改 |
| 双卡分片评测 | [two_gpu_eval.summary.json](two_gpu_eval.summary.json) | 4题×32-token调度／聚合通过 |
| 17,408总位置 | [long_context_and_approximation.json](long_context_and_approximation.json) | 两请求各17406历史位置＋2次decode，无OOM |
| 4题、完整16384上限 | [four_questions_16384.summary.json](four_questions_16384.summary.json) | 共38072 tokens，含初始化530.9秒，1条达到上限，无OOM |
| top-20误差审计 | [approximation_diverse.json](approximation_diverse.json) | 4道不同历史题512位置，样本内边界／目标深度一致；最大JS误差0.0964，不表示处处等价 |
| 原配置不变项 | [invariants.json](invariants.json) | model/rollout/optimization/evaluation/distillation/data/seed一致 |
| 源码与环境 | [source_sha256.json](source_sha256.json)、[runtime_provenance.json](runtime_provenance.json) | 独立源码hash、共享库版本／hash保留 |

原HF同checkpoint、同4道题生成37593 tokens，生成耗时1126.1秒；新双卡路径生成38072 tokens、含初始化530.9秒，本次约2.1倍。由于并发度、初始化计时和随机续写不同，这不是严格单卡延迟对照；短测7.50倍不能外推到16k，也不由4题宣称任务收益。

`compiled_benchmark.json`是首次比较时误用HF trace默认0.9阈值的失败诊断，已显式改为学生阈值0.5；只使用v2作为通过证据。早期同题四轨迹top-20审计有1个位置的目标深度不同；近似导致的边界变化属于允许改变的分布表示，JS阈值仍保持0.05，没有重新调参。

真实DDP诊断为节省磁盘，仅在测试脚本内用明确不可恢复的小marker代替完整checkpoint。生产快照／optimizer保存代码未改。新入口按原保留规则与临时同步文件保守预检约75 GiB，当前共享盘约16 GiB空闲，因此没有再次启动完整20步。

全部验证进程已结束，GPU2、3释放。启动入口：

```bash
bash scripts/run_fastpath_ablation.sh --dry-run
# 有足够checkpoint空间后运行同配置的完整消融。
GPUS=2,3 bash scripts/run_fastpath_ablation.sh
```
