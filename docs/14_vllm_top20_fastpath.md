# 执行模式：默认HF完整动态与可选vLLM／top-20近似

研究路线已另行实现于[三线协议](15_three_route_protocol.md)：`vllm_fixed`是真正固定N计算与完整词表监督；`hf_adaptive`使教师也真实提前停止；`hybrid`用HF动态训练后按多个固定N进行vLLM部署。本文下述`vllm_top20`仍是旧的固定K读出近似路径，不等同于新vLLM研究线。

主方法默认`EXECUTION_MODE=hf_dynamic`，保留原HF真实逐token提前停止、完整词表JS与KL和精确深度KV缓存。vLLM方案保留为显式可选消融：它将实际停算变为固定K轮计算后的动态读出，并使用top-20教师近似，不能作为主方法的完全等价加速。

| EXECUTION_MODE | 生成／评测 | 教师JS与KL目标 | 学生实际计算 |
|---|---|---|---|
| `hf_dynamic`（默认） | 原HF实现 | 完整词表 | 逐token真实提前停止，另计历史补算 |
| `vllm_top20`（可选） | vLLM批量／双卡分题 | top-20＋均匀尾部近似 | 每个token实际计算K轮，再选择读出 |
| `hf_top20`（可选） | HF实现 | top-20＋均匀尾部近似 | 逐token真实提前停止，另计历史补算 |

两个sh入口不指定模式都使用`hf_dynamic`。模式是显式选择单位，不能只把`FASTPATH_ROLLOUT_BACKEND`改成hf就将近似方案当作主方法；该变量需与选定模式一致。近似教师分布可能改变JS边界、阶段数、停止标签和训练梯度。下文的vLLM与top-20工程检查仅属于可选方案。

## 文件与源码边界

- 新入口：`scripts/run_fastpath_ablation.sh`。
- 可微训练：`ep_loop/fastpath/training.py`、`objective.py`，从原实现复制后局部适配，原文件哈希保存在`vendor/*source_manifest.json`。
- vLLM模型适配：`ep_loop/fastpath/vllm_ouro.py`。官方v0.12.0原文件、URL、SHA256和许可证保存在`ep_loop/fastpath/vendor/`，只修改项目内副本。
- 使用环境现有vLLM 0.11.0、Torch 2.8.0，不升级／修改site-packages，不复制CUDA库或整套环境。模型继续引用共享软链接。

## 保留的算法与参数

| 项目 | 设置 |
|---|---|
| 模型 | Ouro-1.4B-Thinking，原固定revision |
| 教师 | 原冻结教师；原生CDF阈值0.9；K=4 |
| 学生读出 | 原本地停止概率达到0.5时选最早轮次，否则选K |
| 阶段 | 所有JS>0.05边界，阶段按顺序映射到学生；不做Top-S |
| 损失 | 学生到教师的reverse KL＋1.0×停止BCE；原阶段／token／回答归一化 |
| 更新 | 每次4题×4回答=16，两卡各8回答，全部梯度汇总后执行一次AdamW |
| 优化 | 20更新，LR1e-6、WD0.01、原betas/eps、warmup0、无梯度裁剪、FP32主参数＋BF16计算 |
| 数据 | 同一seed42划分的256训练题与64道DAPO开发评测题，不改prompt或分区 |
| 长度与采样 | prompt1024，训练／评测回答上限16384；训练T1/top-p1，评测T1/top-p0.7，每题1次 |
| 筛选 | 默认不筛教师题目，也不筛学生回答 |
| checkpoint | 最近两次更新轮换保存，并保留第10／20次完整恢复快照 |

入口配置已与当前标准运行的解析配置比较：model、rollout、optimization、evaluation、distillation、data、seed完全相同，algorithm只增加`teacher_top_k=20`和分布近似元数据。执行后端和并发度单独记录。

## 固定计算K轮，按原门控选读出

vLLM每个位置实际执行完整K轮，各轮／各层使用独立KV，复用同一组Transformer权重。取得每轮的规范化状态和门控，在GPU内选择原学生规则对应的状态，再经过LM head和原temperature/top-p采样。不是先按第K轮采样，再事后更换分布。

教师模式仍使用原生累计CDF，学生模式仍使用本地停止阈值；两者没有合并成同一种退出规则。生产路径使用固定计算图、编译与CUDA graph、批量调度，省去逐轮GPU到CPU门控判断。

这是固定执行深度、动态读出选择。成本日志必须按实际K轮计算，不能把所选轮次当作已经跳过的计算；输出同时记录`exit_depths`与`executed_token_loops`。

## top-20近似的精确定义

对教师每个位置、每轮，保留20个token ID与其原始概率，另存剩余总概率。没有对top-20重新归一化，也没有将缺失token概率设为零。词表大小V时，未保留的token均分剩余概率：

`q_approx(v) = q(v)`（v在top-20），否则为`remaining_mass / (V-20)`。

相邻轮的JS在各自top-20的并集和共同剩余桶上计算，等于上述重建分布的JS。阶段规则和阈值仍为原定义；近似可能改变边界，因此也可能改变由边界导出的目标深度，这属于分布近似的直接影响。

蒸馏仍计算`KL(p_student || q_approx_teacher)`，学生p保留完整词表和尾部熵，用解析尾项避免构造完整教师分布。没有改成forward KL、交叉熵或只在20个token内归一化学生。

教师LM head仍需产生临时logits并归一化才能取得真实top-20概率；该近似主要减少需保留／传输的分布与JS运算，不会消除全部LM-head计算。4轮×16384位置时，int32 ID＋FP32 logprob＋尾部标量约10.25 MiB；完整FP32词表分布约12 GiB。这是分布表示的大小比较，原HF实现本来也按chunk计算，并不同时持有12 GiB。

## On-policy与权重生命周期

每个更新内学生权重原本就保持不变，因此先由vLLM生成该更新的全部回答，再按原顺序回放、累积梯度，仍使用同一版本学生策略。每次更新前同步全部学生权重，RPC确认版本后才允许生成。禁用前缀共享缓存；生成后sleep丢弃KV并把vLLM权重移至CPU，随后执行原教师标注和可微学生回放。

临时同步文件每rank最多一份，加载完成立即删除，不保留每更新的密集rollout分布。两个rank的vLLM actor各自只看到分配的物理GPU，清除父torchrun的rank/store环境变量，使用独立通信组。

seed仍为42，训练数据抽样仍使用原独立`data_rng`。vLLM每回答的种子由持久rank generator产生，状态仍进入checkpoint；该映射写入provenance。不同后端的采样器与BF16内核可产生不同随机实现／舍入，不能声称HF与vLLM输出逐token完全相同。这不改变temperature、top-p或门控算法；同前缀分布和退出的一致性另行验证。

## 评测并行与计时

`evaluate_fastpath.py`把原问题按原索引分给GPU 2、3。每题seed继续使用原`sample_seed(base_seed, problem_sha256, sample)`，与分片／完成顺序无关。输出最后恢复原索引顺序，准确率聚合规则和整数验证器复用原实现。

当前所有vLLM路径使用框架默认的请求调度上限；评测一次提交每卡完整分片，不再在外部按2条等待。各GPU的`.engine.log`每约10秒记录绝对时间及已生成token数量，主事件日志保留阶段起止与周期心跳。批量吞吐用`vllm_batch_metrics.batch_seconds`与评测`wall_seconds`；训练`generation_seconds_sum`含共享批次的请求等待，不能当作单卡纯计算时间。并行吞吐也不能直接与原独立单请求延迟混比。

## 已执行的验证与边界

报告目录：`audit/vllm_top20_20261116/`。

- 4道真实DAPO题，每条最多128 tokens的短测：HF逐条约37.39 tokens/s，vLLM四请求批量约280.37 tokens/s，吞吐比7.50。加载／编译／预热不计入该比值；生成内容因跨后端采样不同，不宣称逐条延迟或16k速度同样提升。
- 4个相同前缀的所选轮次一致，top-20重建分布之间最大JS约3.12e-4，检查阈值5e-4。见`compiled_benchmark_v2.json`。首次比较误用了HF trace默认退出阈值0.9，已显式改为学生0.5；原错误比较保留为`compiled_benchmark.json`，不作为通过证据。
- 两次真实权重更新，全局batch16、诊断回答上限64，冻结教师、有限梯度、版本0/1同步、sleep/wake和临时文件删除通过，见`two_real_updates/validation.json`。
- 另在独立actor内将门控bias设为正／负大值，RPC同步后分别观察到所有读出变成第1／4轮，验证权重更新确实影响计算，而非仅返回版本号；见`weight_sentinel.json`。只改变测试actor内存，不修改共享模型或生产权重。
- 物理GPU 2、3上运行复制后的真实DDP训练器，两次更新、每次全局16回答、诊断上限32通过，见`two_gpu_real_updates/summary.json`。仅诊断替换checkpoint写入为明确标注不可恢复的小文件，避免重复写16 GiB；生产保存逻辑未更改。
- 双卡分题评测完成4题×32-token调度检查，种子与原问题哈希对应，见`two_gpu_eval.summary.json`；截断短测不用于任务准确率结论。
- 两请求各17,406历史位置＋2次新token的KV／长上下文检查通过，见`long_context_and_approximation.json`。这验证17,408总位置容量与长前缀decode，不是完整16k自回归测速。
- 使用原第20步checkpoint、原采样参数与16,384回答上限，在GPU 2、3完成4道真实DAPO题，生成38,072 tokens，总墙钟530.9秒（含两卡初始化／编译），其中1条达到长度上限，无OOM，见`four_questions_16384.summary.json`。原HF同checkpoint、同4题生成37,593 tokens，生成耗时1126.1秒；本次端到端约2.1倍，包含双卡调度收益且续写并不逐token相同，不能作为严格单卡延迟比较或完整64题结论。
- 对4道不同历史训练题的512个位置，top-20与完整分布的JS边界、目标深度在此样本中一致；最大JS数值误差约0.0964，低置信位置top-20质量可能很低，详见`approximation_diverse.json`。早期同题四轨迹检查出现过1个目标深度不一致，故不能将小样本一致率解释为处处等价；阈值仍保持0.05。
- 单元测试包括top-20概率质量、稀疏JS与重建分布的一致性、reverse KL及梯度的稠密参考、原门控选择、保留全词表时还原原损失／停止标签／梯度，以及HF缓存优化后的token与随机流精确一致。

该可选方案版本的84项单元测试与工程检查汇总为`audit/vllm_top20_20261116/validation.json`。验证结束后actor均已退出，物理GPU 2、3释放。后续默认模式及路由核验见`audit/default_hf_dynamic_20261116/`。

近似JS误差与阶段一致率仅在报告列出的历史位置上测量，不证明所有题目都等价；阈值未重新调参。完整训练效果仍需独立消融。

## 启动

```bash
# 默认：完整HF动态方法，不启用top-20。
bash scripts/run_ablation.sh --dry-run

# 显式可选：固定K计算、动态读出与top-20。
EXECUTION_MODE=vllm_top20 bash scripts/run_ablation.sh --dry-run

# 真实权重的短吞吐检查，不复制环境或模型。
CUDA_VISIBLE_DEVICES=2 CUDA_DEVICE_ORDER=PCI_BUS_ID \
  .venv/bin/python scripts/benchmark_vllm_top20.py \
  --gpu 2 --examples 4 --tokens 128 --output /tmp/fastpath_benchmark.json

# 正式消融入口；GPU与原超参数、vLLM显存参数均在脚本开头。
GPUS=2,3 EXECUTION_MODE=vllm_top20 bash scripts/run_ablation.sh

# 单独选择top-20教师近似；生成与评测保留HF真实提前停止。
EXECUTION_MODE=hf_top20 bash scripts/run_ablation.sh --dry-run
```

`hf_top20`可隔离教师近似与vLLM固定计算后端的影响；其评测也使用HF真实停止，不再静默调用vLLM。两个近似模式目前支持Ouro及JS阶段方案；默认`hf_dynamic`继续支持Huginn、置信度探针与原重算诊断。

`configs/default.json`明确记录HF完整主方法。直接调用通用或fastpath训练入口时，均按配置选择：默认配置进入原HF训练器；可选配置为`configs/e_dynamic_js_vllm_top20.json`与`configs/e_dynamic_js_hf_top20.json`。直接调用fastpath评测入口时，默认配置也委托原HF评测器；vLLM配置必须显式提供。JSON方案配置继承原研究参数，当前20步／batch16／16384消融设置由sh开头的超参数覆盖。

当前共享盘约剩16 GiB，按原最近两个＋定期快照以及临时同步文件保守估算，生产入口预检需要约75 GiB空闲。已运行短训练验证和4题完整长度评测，未再次启动完整20步；不通过缩短生产生成上限、减少batch或关闭optimizer保存绕过磁盘限制。
