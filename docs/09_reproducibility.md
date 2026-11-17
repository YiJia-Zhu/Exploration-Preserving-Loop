# 种子42与尽量可复现的运行

当前训练、评测和ablation默认使用基础种子42。优先保留现有计算效率，不强制GPU算子逐位确定，也不为复现切换注意力实现、混合精度或梯度检查点。

| 随机来源 | 当前控制方式 |
|---|---|
| 数据划分与题目采样 | 独立Python RNG，基础种子42；记录题目内容哈希和划分文件哈希 |
| Python、NumPy、Torch CPU/CUDA | 启动时统一设种子；训练期间连续使用随机流，不逐token重设 |
| 多卡初始化 | 各rank先以42初始化，保证初始模型一致 |
| 多卡训练随机状态与采样 | 以42加rank固定偏移产生独立随机流；数据顺序仍共用42 |
| 评测与一次教师筛选 | 用基础种子、题目内容哈希、样本编号生成稳定种子；不依赖前面回答长度或题目执行顺序 |
| Python字符串哈希 | `run_ablation.sh`在启动Python前导出 `PYTHONHASHSEED=42`；直接调用Python时记录实际环境值 |
| 恢复训练 | 保存每rank数据RNG、采样generator、Python、NumPy、Torch和当前CUDA设备的RNG状态 |

每次运行记录环境版本、实际数学后端标志、GPU、world size、代码哈希及Git版本。评测逐条保存 `sampling_seed`，便于重新定位同一题的同一条采样。正确性缓存策略也包含种子及其生成规则；旧的无种子缓存需要重建，默认未启用该筛选。

三个启动脚本开头均可设置`CPU_THREADS`，默认16，表示每个进程的CPU算子线程数。HF训练、HF评测和vLLM子进程采用相同设置；同步设置OMP、MKL、OpenBLAS和NumExpr线程上限，并记录实际Torch线程数与环境值。两张卡仍启动两个训练rank，另有各自的vLLM引擎进程；CPU_THREADS不决定模型进程数量，也不创建16个数据worker。CPU线程可以参与张量转换等CPU算子，逐条Python数据准备及校验流程仍为串行；JS、完整词表读出和训练仍在GPU执行。线程上限不代表持续使用同等数量的物理核心，速度变化需要单独测量。

为了效率，不新增 `torch.use_deterministic_algorithms(True)`、强制确定性cuDNN或cuBLAS workspace设置；不会每步扫描环境或计算模型权重的大文件哈希。现有模型精度与后端配置继续决定具体算子路径，元数据收集只在运行开始执行一次。

同一环境、配置和卡数下尽量复现；不同GPU、CUDA/PyTorch版本、并行规模或训练中的非确定性归约可能仍有差异。耗时和显存测量也不保证一致。Huginn仍按原生规则采样初态，只固定随机来源，不将其改成零状态。

动态学生现在默认使用[精确深度缓存](10_dynamic_kv_cache.md)。Huginn的每个新位置只初始化一次并缓存，历史位置不重抽；这改变了旧完整前缀重算的随机数消耗，同一seed不代表新旧路径生成相同。逐题种子与恢复机制仍有效，必须连同`use_cache`和源码版本一起记录。缓存不跨回答或参数更新保留，恢复时从保存的全局RNG重建新回答；BF16不同前向形状仍可能产生正常舍入差异。

[置信度探针消融](12_confidence_and_correctness_ablations.md)使用按前缀与基础种子派生的独立Torch随机流，退出探针后恢复原训练CPU／当前CUDA流。直接置信度不额外采样。正确性筛选不改变数据种子；teacher_correct与random_matched使用同一个冻结初始教师缓存和seed42，学生回答不按正确性过滤，恢复沿用原有optimizer与更新计数。

建议通过脚本启动，使哈希种子也在Python启动前生效：

```bash
CPU_THREADS=16 GPUS=2,3 bash scripts/run_ablation.sh
```

直接运行Python入口时可使用 `PYTHONHASHSEED=42` 前缀；其余随机源按配置中的 `seed` 设置。历史诊断与旧实验记录保留原来的种子，不重写其历史结果。
