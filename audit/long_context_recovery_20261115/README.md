# 16,384 tokens训练显存修复与重启

完成状态（2026-11-16 16:21:40，Asia/Shanghai）：20次训练更新和第10／20次快照评测均已完成，checkpoint与恢复状态文件齐全。初始模型40/64（62.50%），第10次35/64（54.69%），第20次39/64（60.94%）；实际出口均值依次为4.00、3.43、3.24。最后一次评测计入补算和回答长度后的总前向token-loops较初始增加8.77%，不能仅由出口下降推断计算节约。核验见[completion_summary.json](completion_summary.json)。任务产物完成后，外层Bash因运行期间脚本文件长度改变而收尾报错；现已将启动方式改成exec Python，避免重新读取旧文件偏移。下文保留修复与重启时记录。

2026-11-15 12:43:32（Asia/Shanghai）在GPU 2、3重启，launcher PID `832291`，见[launch.json](launch.json)。当前运行目录为：

`experiments/ablations/20261115_124332_ouro_e_js_dapo64_len16384_seed42_gpu23_memfix/`

最新运行指针另存于`experiments/ablations/latest_run.json`。本报告记录启动时的验证；实时进度以当前运行的事件与metrics日志为准。

## 前一轮实际结果

- 2026-11-15 02:32完成原生模型DAPO留出64题评测：40/64正确，62.5%；50题EOS，14题达到长度上限。生成505,144 tokens，总生成时间23,033.6秒。该结果属于未训练原生模型，不是训练后学生或LoopOPD蒸馏结果。
- 双卡训练完成1次更新，全局16条回答、153,925 tokens；目标平均深度1.3883，学生实际出口平均3.6662。单次更新不足以说明方法有效。
- 第2次更新期间，rank 0在03:58生成长回答时发生OOM，堆栈定位到`TensorBuffer.append()`的KV扩容。异常时已分配59.32 GiB、另有11.13 GiB保留未分配；launcher最终于10:16退出。
- 原运行按第10次更新保存，因此没有模型／优化器checkpoint。此次从初始权重和seed42重新训练20次更新，未声称从第1次更新恢复。

## 修复

1. KV／状态缓冲区在短序列仍几何增长，长序列每次最多额外扩容1,024个位置，避免临近16k时将全部深层KV容量翻倍。
2. Ouro深层历史补算按最多256个位置分块，维持同一绝对位置、因果掩码和真实KV；不替换成浅层KV，不改变所需token-loops。
3. DDP开启`gradient_as_bucket_view=True`，共享梯度与通信bucket存储；保持原有梯度累积和全局batch。
4. 默认allocator采用`expandable_segments:True`减少不同回答长度之间的碎片。复现provenance记录该变量。
5. 异常退出不再走正常的分布式销毁路径，使torchrun尽快获知失败并结束其他rank；正常完成仍清理进程组。
6. 脚本新增`KEEP_LATEST_CHECKPOINTS=2`：每次更新保存完整模型／optimizer／各rank RNG，保留最近两次，并永久保留既定第10／20次快照。只删除当前运行目录内已被新完整快照替代的非定期快照。

训练目标、模型、数据划分、训练／评测采样、20更新、全局batch16和16,384生成上限均未改变；已比较前后解析配置中的model、algorithm、rollout、evaluation、optimization五节完全相同。分块计算的浮点舍入可能与原路径不同，沿用best-effort seed42约定，不承诺逐位一致。

## 已完成验证

- 74项单元测试通过：[tests.log](tests.log)。新增长缓冲区容量边界和多块晚期深层补算与原生完整前缀的逐深度比较。
- 两进程CPU DDP完成两次更新，核验共享梯度存储路径：[ddp_bucket_view_smoke.log](ddp_bucket_view_smoke.log)。
- 三次CPU更新保存并轮换快照，确认仅剩update 2、3；既定快照规则不变。
- GPU 2上使用真实Ouro-1.4B权重、FP32主参数、BF16计算、完整冻结教师及已初始化Adam状态，测试1,024 prompt＋16,384回答位置。额外预留两份完整FP32梯度大小的GPU存储后，完成从1轮延迟补齐4轮，以及重复增长到17,408位置；随后在保留一份DDP大小存储时完成完整回答的阶段损失与反向传播，梯度有限。最大PyTorch分配54.77 GiB，最大保留56.34 GiB。见[gpu_validation.json](gpu_validation.json)与[验证脚本](validate_long_context.py)。

GPU压力验证使用强制深度与固定token测试内存，不能证明采样质量或训练收益。生产运行仍由学生自身停止头选择实际深度。

## 重启与评测复用

此次显式`EVAL_INITIAL=0`，复用前一轮已完成的初始原生模型评测，避免重复约6.4小时计算。新旧模型版本、数据分区hash及评测设置一致；新运行`evaluation/initial_native.jsonl`和`.summary.json`软链接到原结果，复用声明写入`reused_initial_evaluation.json`。旧训练日志与失败状态完整保留，不拼接到新训练metrics中。

第10、20次更新后的评测仍自动执行。每次运行解析配置、数据manifest、代码快照、进程和环境记录均保留。启动后已确认两个rank完成模型加载并在物理GPU 2、3生成真实训练回答。

```bash
# 当前训练、保存与后续评测的全程日志。
tail -f audit/long_context_recovery_20261115/launch.console.log

# 更新完成后才追加一行；逐回答进度和阶段耗时见events_rank*.jsonl。
tail -f experiments/ablations/20261115_124332_ouro_e_js_dapo64_len16384_seed42_gpu23_memfix/ours/metrics.jsonl
```
