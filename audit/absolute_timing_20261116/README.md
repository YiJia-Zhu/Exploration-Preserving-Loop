# 节点绝对时间与周期记录

2026-11-16针对日志时间不直观的问题增加记录。旧事件的`timestamp/started_at/finished_at`为UTC，`local_time`为北京时间；旧tqdm只显示相对耗时。时区相差8小时与多卡输出交错容易造成误读，尚无证据表明系统时钟本身不准确。

## 新进程默认行为

- 每个节点开始和结束写UTC及`started_at_local/finished_at_local`，本地字段与控制台均含明确的`+08:00`时区。
- 事件保存同一次主机时间采样的`timestamp/local_time/unix_time_ns`，另存`monotonic_ns`；阶段耗时仍用单调计时器，不为日志额外同步CUDA。
- 每次tqdm实际刷新带北京时间和rank，保留原来的约1秒刷新节流。
- 生成、教师前向、JS划分、学生回放、损失、反向、参数更新、指标归约、checkpoint保存等节点均写纯文本时间线。各rank独立保存`events_rankN.timeline.log`，无需在混合的终端输出里猜测来源。
- 默认每30秒写一次heartbeat，包含当前活跃节点、开始时间、耗时和生成进度。生成过程仅更新Python整数；不新增逐token文件I/O或GPU张量读取。
- 子节点继承更新编号、题目ID与回答编号。脚本顶部新增`LOG_STAGE_TIMESTAMPS=1`及`LOG_HEARTBEAT_SECONDS=30`。

## 当前运行接入方式

当前运行`20261115_124332_ouro_e_js_dapo64_len16384_seed42_gpu23_memfix`已完成20次训练更新，仍在评测第20次更新快照。进程启动时已经导入旧logger，因此保留该评测进程，使用只读旁路观察器附加时间记录。

观察器在2026-11-16 16:15:43（Asia/Shanghai）启动，PID `562370`，见[watcher_launch.json](watcher_launch.json)。每2秒检查新增事件，每30秒写当前节点及最新可见进度；launcher结束后自动退出。原始事件时间与观察时间分开记录；从旧tqdm读取的token数同时附带控制台文件mtime，不冒充GPU精确采样。

运行目录中新增：

- `timeline.live.log`：当前节点转换后的绝对时间，以及后续节点变化与周期记录。
- `timeline.live.jsonl`：相同信息的结构化版本。
- `timeline.history.log`：将已有节点记录统一转成北京时间并按事件时间排序的历史快照，保留rank与来源文件。只转换已经记录的时间，不补造缺失时间。

旧日志、训练参数、checkpoint、模型和随机状态均未改写；没有重启训练或评测。历史转换的输入前缀hash见[history_manifest.json](history_manifest.json)，原日志仍可继续追加。

```bash
tail -f experiments/ablations/20261115_124332_ouro_e_js_dapo64_len16384_seed42_gpu23_memfix/timeline.live.log
```

## 验证

78项测试通过，见[tests.log](tests.log)，包含UTC／北京时间一致性、子阶段上下文、rank 1时间线、心跳线程及收尾、进度时间戳、旧日志半行读取、观察器节点更新与随launcher结束退出、生成结果和RNG不变。

两进程CPU小模型完成两次更新，最终权重与日志改动前的同配置检查逐位相同，除时间相关字段外的训练指标相同。CPU模拟不构成真实任务收益或GPU kernel精确计时证据。当前旁路观察器存活并持续写心跳；核验见[validation.json](validation.json)。
