# 三线实现与10小时vLLM消融启动记录

本轮以三条研究路线的语义为边界，首轮实际运行vLLM固定预算线。研究定义、入口及参数见[三线协议](../../docs/15_three_route_protocol.md)。

- HF：教师原生CDF真实停算；学生HF逐token动态停止，保留精确历史KV补算。
- vLLM：教师固定K，学生生成、梯度回放与部署都实际固定N；完整词表JS/KL，无停止BCE；终点K始终保留。
- 交叉线：教师固定K计算并选择原生逻辑终点，学生HF动态on-policy训练；同checkpoint通过vLLM评多个固定N。

`budget_tests.log`中104项测试通过。三线CPU两更新检查分别在`hf_smoke/`、`vllm_fixed_smoke/`、`hybrid_smoke/`；实际Ouro N=1、2、4编译引擎检查在`fixed_engine_validation.json`；`ddp_fixed_v2/`完成真实双卡、全局batch16的两次短更新（诊断生成上限32，无正式可恢复checkpoint），确认教师K4、学生N2、完整词表及停止损失为0。这些验证支持实现可行性，不构成任务效果证据。

## 当前已启动的限时实验

启动记录：`budget_launch.json`；源码归档：`budget_sources.tar.gz`及`budget_source_sha256.json`。

- 2026-11-17 01:36:09（北京时间）启动，协调进程PID 301549；GPU 2、3启动前空闲。
- 统一panel：`experiments/three_route_ablations/20261117_013609_vllm_K4_N2_seed42_10h_panel/`。
- 顺序：endpoint → uniform_n → top_n；各10次更新，每组独立从原始Ouro-1.4B-Thinking初始化。
- 冻结初始教师K4；学生N2；全局4题×4回答=16；seed42；完整词表JS与reverse KL；无教师题目或学生回答正确性过滤。
- 训练256题；每组最终step10在同一DAPO开发集64题上评测；训练／开发互斥。
- 训练／评测生成上限均16384，prompt上限1024；训练T1/top-p1，评测T1/top-p0.7；LR1e-6、WD0.01、bf16计算、FP32主参数。
- vLLM编译图、Flash Attention和paged KV用于on-policy生成及部署评测；HF双卡DDP用于完整分布梯度回放；actor睡眠时释放KV占用，更新前同步学生权重并检查版本。
- 为节省预算，EVAL_INITIAL=0；仅评最终10步，不重复生成未训练模型的K／N参考。
- 每更新保存模型、optimizer与各rank RNG，写完后原子生成`checkpoint.complete.json`；滚动保留最近1份，最终点永久保留。
- MAX_RUNTIME_HOURS=10涵盖整轮所有阶段。独立watchdog记录于panel的`time_budget.json`，截止北京时间2026-11-17 11:36:09；提前15秒开始停止本次进程树。可能在部分组未完成时达到上限；完成情况以各组summary、metrics及评测summary为准。

查看当前日志与状态：

```bash
cat experiments/latest_run.json
tail -f audit/three_routes_20261117/budget_013609.console.log
cat experiments/three_route_ablations/20261117_013609_vllm_K4_N2_seed42_10h_panel/time_budget.json
```

此前01:21启动的20步计划在训练更新前，因新增10小时要求于01:29停止；保留`launch.json`、`launch.console.log`和`prebudget_stop.json`，原panel标记为`stopped_for_reconfiguration`。没有把其参考评测与新10步实验混报。

初次双卡短训因vLLM CuMem池继承HF的expandable_segments设置失败，随后仅隔离actor环境变量；修复后的`ddp_fixed_v2`通过，原失败日志保留。未修改环境site-packages或共享模型目录。
