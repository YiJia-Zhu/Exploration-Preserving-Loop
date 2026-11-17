# 日志、进度与阶段时间

训练、评测、一次教师筛选和数据转换默认开启tqdm。多卡训练只在rank 0显示进度，各rank独立保存日志；原始指标和生成记录继续保留，不把动态时间戳混入用于复现比较的rollout内容。

## 进度显示

- 训练总进度：已完成的optimizer更新数／配置总更新数，恢复时从checkpoint中的更新数开始。
- 每次更新：当前rank已处理的回答数／本rank应处理回答数，包含生成、监督前向与反传。
- 每次生成：已生成token数／最大生成长度，显示速度、预计时间与最新显示时的退出深度。遇到EOS会提前结束，不要求填满最大长度。
- 评测：所有策略／深度、题目和样本的总回答进度，以及当前回答的token进度。
- 教师筛选与数据准备：题目／行数进度。

默认最多约每秒刷新一次，不逐token写JSONL。关闭终端进度不会关闭时间线、结果或异常记录。

2026-11-16起，每次tqdm实际刷新均显示带时区的绝对时间与rank，例如`[2026-11-16T16:00:00.123+08:00] [rank=0]`。每个节点开始、结束均写时间；长节点默认每30秒写一次heartbeat，包含当前活跃阶段、各节点开始时间、已用时间和当前生成token数。token计数只在内存中更新，heartbeat才写文件，不逐token增加文件I/O。

## 文件位置

使用`run_ablation.sh`时，设运行根目录为`<run>`：

| 文件 | 内容 |
|---|---|
| `<run>/launcher.events.jsonl` | 启动参数、预检结果、数据划分、配置保存及各训练／评测子进程的时间与返回码 |
| `<run>/launcher.console.log` | launcher运行期间的控制台输出和转发的子进程输出 |
| `<run>/logs/train_baseline.log`、`train_ours.log` | 对应子进程合并后的原始stdout/stderr，包括tqdm、库提示和traceback |
| `<run>/logs/eval_<label>.log` | 每次独立评测的完整子进程输出 |
| `<run>/baseline/events_rankN.jsonl`、`<run>/ours/events_rankN.jsonl` | 各rank的详细阶段时间线 |
| 对应的`console_rankN.log` | Python控制台及可重定向的logging handler输出，保留进度刷新字符 |
| 对应的`events_rankN.timeline.log` | 每个rank独立、无tqdm刷新控制字符的绝对时间线，含所有节点起止与周期heartbeat |
| 对应的`metrics.jsonl`、`rollouts_rankN.jsonl` | 原有每次更新的指标、每条生成结果及真实计算量 |
| `<run>/evaluation/<label>.events.jsonl`、`.console.log` | 该次评测的阶段与控制台日志；结果仍是同名`.jsonl`和`.summary.json` |

独立调用训练入口时，日志直接位于`--output`目录。独立评测、筛选或数据转换时，日志位于输出文件旁，后缀为`.events.jsonl`和`.console.log`。

`--dry-run`与`--check`保持只读，不新建日志文件。launcher预检通过后创建持久日志，并记录之前的预检起止时间；子进程计时包含Python启动与导入，训练内部事件从自身`run_start`开始。直接调用Python函数的测试／受控测速默认不显示token进度，避免干扰测试与速度测量；实际训练和评测入口会传入observer。

## 时间线包含什么

训练覆盖配置与资源检查、分布式初始化、模型／tokenizer加载、师生准备、数据编码、optimizer/DDP准备、恢复权重与RNG、每次更新、每条回答生成、prompt prefill、后续token decode、教师前向、JS阶段划分、学生回放、损失读出、反传、optimizer更新、指标归约与checkpoint保存。

每个阶段有`stage_start`和`stage_end`，包含阶段ID、父阶段ID、开始／结束时间、耗时与状态。每次生成另存题目ID、样本编号、更新编号、实际长度、EOS状态、执行量、补算量和缓存策略。结束阶段记录对应检查点路径或子进程返回码。

子节点继承`update/id/sample`等上下文，便于将教师前向、回放、反传对应到具体回答。事件还保存`unix_time_ns`与`monotonic_ns`。`started_at/finished_at`继续采用带时区的UTC以兼容既有数据，同时新增`started_at_local/finished_at_local`，控制台与纯文本timeline均明确使用北京时间。UTC与北京时间相差8小时，不应直接比较去掉时区后的字符串。

所有事件包含UTC时间`timestamp`及带`+08:00`时区的`local_time`，本地时区默认`Asia/Shanghai`。`elapsed_seconds`与阶段耗时使用单调计时器，避免系统时间调整导致负耗时。耗时是**主机侧墙钟时间**：不为日志额外同步CUDA，阶段耗时不能当作精确CUDA kernel profiling。原有生成总耗时仍沿用其结束时同步方式。

新增消融也写入同一日志：`confidence_probes`记录探针前向与计算量；`update_metrics`记录边界信号与探针成本。学生回答不按正确性过滤，全部保留生成与蒸馏记录；teacher_correct缓存旁的summary记录教师筛题接受率与一次筛题成本。

异常／KeyboardInterrupt会结束已开始的阶段、记录状态、错误与traceback。恢复同一输出目录时追加新`session_id`，保留已有日志。SIGKILL无法执行Python收尾，最后已刷新的`stage_start`可用于定位中断位置。

## 调整方式

ablation脚本开头定义：

```bash
SHOW_PROGRESS=1
TOKEN_PROGRESS=1
PROGRESS_MIN_INTERVAL=1.0
LOG_TIMEZONE=Asia/Shanghai
LOG_STAGE_TIMESTAMPS=1
LOG_HEARTBEAT_SECONDS=30
```

例如，只保留更新／回答进度：

```bash
GPUS=2,3 TOKEN_PROGRESS=0 bash scripts/run_ablation.sh
```

直接调用Python入口时使用对应环境变量：

```bash
EP_LOOP_PROGRESS=1 EP_LOOP_TOKEN_PROGRESS=1 EP_LOOP_PROGRESS_MIN_INTERVAL=1.0 \
  .venv/bin/python scripts/train_stage_distillation.py \
  --smoke --device cpu --output experiments/runs/logging_smoke
```

日志不读取模型张量，不改变种子或生成策略；训练中用于记录loss的CPU标量复用已有取值。JSONL按阶段／回答与周期heartbeat写入，不新增逐token文件I/O、GPU同步或严格确定性算子。原验证材料见[audit/logging_20261114](../audit/logging_20261114/)。

新增heartbeat使用后台线程，只读取Python侧的阶段栈与整数进度；默认每30秒写一条JSONL与纯文本记录，不同步CUDA。设`LOG_HEARTBEAT_SECONDS=0`关闭周期记录，`LOG_STAGE_TIMESTAMPS=0`减少控制台的细粒度节点输出；结构化起止事件与独立timeline仍保留。直接运行Python时使用`EP_LOOP_STAGE_TIMESTAMPS`、`EP_LOOP_HEARTBEAT_SECONDS`。

## 已运行任务的只读时间线

已经启动的进程不会重新加载修改后的logger。`scripts/watch_run_timeline.py`可以在不重启任务的情况下，增量读取原始事件，每2秒检查新节点，默认每30秒生成当前状态，输出`<run>/timeline.live.log`与`.jsonl`。观察时间和原始事件时间分开记录；旧tqdm提取的最后token数还标记原控制台文件mtime，不伪装成实时GPU采样。

```bash
.venv/bin/python scripts/watch_run_timeline.py \
  --run experiments/ablations/<run_tag> --launcher-pid <launcher_pid> \
  --heartbeat-seconds 30 --poll-seconds 2
```

launcher结束后观察进程自动退出；`--once`只写当前快照。当前第20次更新快照评测已附加该观察进程，具体PID和路径见[绝对时间记录](../audit/absolute_timing_20261116/README.md)。原始训练／评测日志不改写。

已通过63项测试；两模型×冻结/current教师的短训结果与开启日志前的生成及最终权重精确一致。同目录恢复追加、多卡独立日志、评测／筛选进度、子进程失败码及启动异常traceback均有检查。训练与分布式验证使用随机小模型和CPU，不构成正式任务训练结果。
