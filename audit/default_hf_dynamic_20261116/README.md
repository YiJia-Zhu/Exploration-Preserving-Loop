# 默认恢复完整HF动态方法，近似路径显式可选

用户确认：原HF真实逐token停止必须保留并设为默认，固定K读出＋top-20近似仅作选择方案。本次不改变已完成实验、checkpoint或既有评测结果。

## 执行模式

| 模式 | 默认 | 生成／评测实现 | 教师分布 | 学生实际停算 |
|---|---|---|---|---|
| `hf_dynamic` | 是 | 原HF实现 | 完整词表JS和reverse KL | 是，另计精确历史补算 |
| `vllm_top20` | 否，需显式选择 | vLLM批量与双卡分题 | top-20原始概率＋均匀尾部 | 否，固定K计算后选择读出 |
| `hf_top20` | 否，需显式选择 | HF实现 | top-20原始概率＋均匀尾部 | 是，另计精确历史补算 |

`EXECUTION_MODE`在sh开头定义，默认`hf_dynamic`。两个sh入口不指定模式都进入原HF完整实现；默认不启动vLLM actor或加载vLLM运行时。只有显式选择近似模式才使用`experiments/ablations_fastpath/`，运行标签和解析配置记录实际模式。

```bash
# 默认主方法。
GPUS=2,3 bash scripts/run_ablation.sh

# 可选：固定K计算＋动态读出＋top-20教师近似。
GPUS=2,3 EXECUTION_MODE=vllm_top20 bash scripts/run_ablation.sh

# 可选：只近似教师分布，仍用HF真实停止生成和评测。
GPUS=2,3 EXECUTION_MODE=hf_top20 bash scripts/run_ablation.sh
```

三者沿用脚本其余设置：Ouro、冻结教师、K上限4、本地停止阈值0.5、JS阈值0.05、BCE权重1、训练256题／DAPO开发64题、20更新、全局batch16、训练及评测16384、seed42、GPU 2、3、不筛题和学生回答。默认学生达到停止条件就停止当前token的后续循环，未改为固定四轮执行；Ouro教师的原生CDF读出仍沿用之前的实现。

## 路由与一致性保护

- `configs/default.json`显式记录HF完整模式，`teacher_top_k=null`、`teacher_distribution=full_vocab`。
- `ep_loop/execution.py`集中识别、校验模式。不能把top-20配置标成默认完整模式，后端与模式不一致时报错。
- 默认通用训练和fastpath兼容训练入口都调用`ep_loop.training`，使用原`ep_loop.objective`和`ep_loop.inference`。可选配置才调用独立fastpath训练器。
- fastpath评测入口使用默认配置时委托原HF评测器；HF＋top-20的生成与评测也都使用真实停算。修复了过去该选项在评测时仍进入vLLM的歧义。
- 过去保存的显式top-20配置即使没有新的模式字段，也按原分布及后端元数据保留其身份，不改写历史。
- 两个可选JSON配置为`configs/e_dynamic_js_vllm_top20.json`和`configs/e_dynamic_js_hf_top20.json`。JSON继承原研究参数；本轮20步消融的实际数值由sh头部覆盖。

## 验证

91项测试通过：[tests.log](tests.log)。包含两个sh入口的默认和显式模式路由、完整词表与top-20互斥校验、旧配置兼容、训练器委托及评测默认路由。

随机小模型CPU两步训练：原通用入口与兼容fastpath入口在默认配置下最终权重逐位相同，除时间字段外的逐更新指标相同。显式HF＋top-20也完成两步训练。默认兼容评测入口完成CPU保存后评测。没有为本次默认切换重跑完整20步或GPU实验。

四份只读计划为`default_plan.txt`、`compat_default_plan.txt`、`vllm_top20_plan.txt`、`hf_top20_plan.txt`；核验摘要见[validation.json](validation.json)。README、工作流、缓存说明、实现说明、执行选项与主方案已同步，主方案原CRLF换行保留。
