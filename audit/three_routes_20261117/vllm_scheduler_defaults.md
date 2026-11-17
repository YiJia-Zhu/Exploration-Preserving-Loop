# 使用vLLM默认调度

2026-11-17按用户要求，当前项目所有vLLM训练、评测、测速和验证入口均不再显式设置`max_num_seqs`，也不提供对应的环境变量、配置生成或actor参数入口。现有历史运行记录保留其当时设置。

训练每次更新把各rank的全部回答请求一次提交，仍在生成前同步当前学生权重；不跨更新预生成。评测每卡完整分片一次提交，取消每两条请求等待一次的外部循环。每题seed及最终结果排序保持原协议。

默认调度触发分块预填充时，vLLM对尚未完成前缀的请求会计算logits并丢弃临时采样。深度诊断现在读取runner已有的CPU丢弃标记，只记录实际生成的token；不会改写logits、采样、调度或KV。

验证：

- `vllm_scheduler_default_tests.log`：105项测试通过，包含交叉线四种固定深度、整分片单次提交、actor无显式并发参数及分块预填充深度记录。
- `vllm_scheduler_default_plan.log`：vLLM三组dry-run通过，不再生成并发数量参数。
- `vllm_scheduler_default_validation.json`：真实Ouro权重、编译的固定N=2引擎、GPU2；一次提交8条训练请求和32条1024-token前缀请求均通过；后者触发分块预填充；16-token诊断上限；包含sleep/wake。
- 初次32条检查发现深度记录多计数，修复前日志保留于`vllm_scheduler_default_validation_first_attempt.*`。

该检查支持默认调度下的工程可行性，不提供长回答吞吐提升比例或新的任务准确率。
