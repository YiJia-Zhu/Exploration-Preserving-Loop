# vLLM教师、原生监督终点与阶段显存切换

用户本轮要求：固定轮次教师可用vLLM、监督依靠JS与原生退出机制、inactive模型按阶段释放显存、暂去均匀阶段对照。

当前实现：

- 默认两组endpoint与top_n；均匀对齐仅保留为显式可选旧消融。
- endpoint控制保持LoopOPD式固定K终点；top_n教师固定执行K轮，按Ouro原生stick-breaking CDF选择每token逻辑终点n_t，仅在此前选JS最大的边界，追加n_t，共min(N,n_t)个目标。
- 浅终点默认prefix，遵循此前“监督到1层”的要求；固定N部署仍读N，存在监督与读出差异；可设SHALLOW_TARGET_PLACEMENT=last。未把这个选项当作已被证明更优。
- vLLM教师对学生的完整已知前缀prefill，保留各轮规范化状态与gate；额外一个接口采样token丢弃。传输状态与gate，不传全部位置的密集logits；HF侧只保留初始教师的冻结LM head以分块重建完整分布。
- HF学生与AdamW状态在推理阶段移至CPU。vLLM学生生成后sleep，vLLM教师完成后sleep，随后恢复HF学生做可微回放／反向／DDP更新。模型参数对象与optimizer动量保留，不重复初始化；按更新重建梯度桶。
- vLLM融合残差算子可原地修改上一轮状态，保留中间轮读出时已加入快照。固定最终轮读出的学生无需额外复制。
- 所有vLLM入口仍无max_num_seqs参数。

验证见vllm_teacher_execution_validation.json：109项CPU检查通过；vllm_teacher_staged_ddp_v2完成真实双卡2更新（全局16、诊断32-token cap），有限梯度、教师冻结、停止损失0、权重同步与CPU卸载／恢复通过。第二次更新HF父进程仅余约16.25 MiB已分配张量，optimizer CUDA张量数0；不包含vLLM actor及CUDA运行时显存。vLLM sleep仍保留编译图和运行时，GPU占用不是零。

vllm_teacher_long_context.json：两条17407位置的已知前缀，固定4轮，17408总上下文限制，真实分块prefill与全轮读出通过，教师batch约3.38秒；不当作完整16k逐token生成的吞吐。

vllm_teacher_fidelity_v2.json：两个真实DAPO前缀，369位置gate出口一致，末尾32位置JS监督选段一致；完整词表分布最大JS约0.001128。最初严格JS<0.001检查未过，原CHECK_FAILED保留。BF16后端舍入不同，不能声称HF/vLLM数值完全相同或小样本选段一致处处成立。中间probe曾在无快照时观察到选段不一致，原记录保留。一次GPU0验证与新启动的外部任务竞争显存后失败，未干预外部任务，随后在空闲GPU6验证。

诊断checkpoint明确标记为不可恢复的小文件，生产完整保存函数没有替换。新协议尚未重跑正式panel；此前三组30／64、27／64、30／64属于旧教师终点协议。
