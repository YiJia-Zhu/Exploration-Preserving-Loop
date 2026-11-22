# Step 5 / Step 10 评测

AIME2024，30题×16次；固定N2，BF16权重与KV，普通推理，temperature=1.0、top_p=0.7、上限16384 token。
accuracy 按480条回答统计，等于本轮 Avg@16；Pass@16 表示至少一次答对的题目比例。

| step | 正确回答 | accuracy / Avg@16 | Pass@16 | 平均输出token | 总输出token | 平均循环数 | 截断数 | 时间(分钟) |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 110/480 | 22.92% | 60.00% | 12974.3 | 6227682 | 2.0 | 254 | 154.3 |
| 10 | 116/480 | 24.17% | 56.67% | 13210.3 | 6340964 | 2.0 | 276 | 158.2 |

每条回答 JSONL 均包含问题、输入 messages、参考答案、模型回答全文、token数、循环次数、随机seed和正确性。

- step 5：[问题与回答](/local_nvme/data/zhenning/working_tmp/Exploration-Preserving-Loop/ep_loop_runs/Exploration-Preserving-Loop/ep_d_loopopd/20261119_182732_EP_D_LoopOPD_ngram5_K4_N2_B256_u100_seed42_gpu0145_top_n/ours/evaluation/update_000005/aime2024.responses.jsonl) · [每题一条](/local_nvme/data/zhenning/working_tmp/Exploration-Preserving-Loop/ep_loop_runs/Exploration-Preserving-Loop/ep_d_loopopd/20261119_182732_EP_D_LoopOPD_ngram5_K4_N2_B256_u100_seed42_gpu0145_top_n/ours/evaluation/update_000005/aime2024.responses.questions.jsonl)

- step 10：[问题与回答](/local_nvme/data/zhenning/working_tmp/Exploration-Preserving-Loop/ep_loop_runs/Exploration-Preserving-Loop/ep_d_loopopd/20261119_182732_EP_D_LoopOPD_ngram5_K4_N2_B256_u100_seed42_gpu0145_top_n/ours/evaluation/update_000010/aime2024.responses.jsonl) · [每题一条](/local_nvme/data/zhenning/working_tmp/Exploration-Preserving-Loop/ep_loop_runs/Exploration-Preserving-Loop/ep_d_loopopd/20261119_182732_EP_D_LoopOPD_ngram5_K4_N2_B256_u100_seed42_gpu0145_top_n/ours/evaluation/update_000010/aime2024.responses.questions.jsonl)
