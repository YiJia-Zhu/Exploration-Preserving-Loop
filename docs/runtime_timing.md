所有组均在单张 H100 上运行，关闭投机解码，每条请求生成 256 token。vLLM 和 vLLM-RLT 使用 `gpu_memory_utilization=0.90`。耗时和吞吐取预热后 3 次测量的中位数。

| 引擎 | 运行方式 | 实际平均解码轮数 | 整批耗时 | 吞吐 |
|---|---|---:|---:|---:|
| 官方 vLLM | 固定 2 轮，并发 4 | 2.00 | 1.44 s（4 条合计） | 709.4 token/s |
| vLLM-RLT | 固定 2 轮，并发 4 | 2.00 | 3.57 s（4 条合计） | 287.0 token/s |
| 官方 vLLM | 固定 4 轮，并发 4 | 4.00 | 2.79 s（4 条合计） | 367.3 token/s |
| vLLM-RLT | 固定 4 轮，并发 4 | 4.00 | 6.54 s（4 条合计） | 156.5 token/s |
| vLLM-RLT | Ragged KV，`τ=0.20`，并发 4 | **2.28** | **4.44 s**（4 条合计） | **230.6 token/s** |
| 官方 vLLM | 原生 Ouro KV，`τ=0.20`，并发 4 | 4.00 | 2.76 s（4 条合计） | 370.4 token/s |
| vLLM-RLT | Ragged KV，`τ=0.20`，并发 4 | 2.36 | 18.99 s（16 条合计） | 215.7 token/s |
| vLLM-RLT | Ragged KV，`τ=0.20`，并发 8（默认） | 2.35 | 11.01 s（16 条合计） | 372.2 token/s |
| vLLM-RLT | Ragged KV，`τ=0.20`，并发 16 | 2.37 | 6.89 s（16 条合计） | 594.8 token/s |

原生 Ouro KV 的数据来自临时测速入口，使用 vLLM 0.26 V2 Model Runner、FlashInfer 和 CUDA Graph。每步执行四轮，首个 token 从第 4 轮采样，后续由退出门选择读出轮次。Ragged KV 的实际平均解码轮数按 vLLM-RLT 的早退轮次统计。
