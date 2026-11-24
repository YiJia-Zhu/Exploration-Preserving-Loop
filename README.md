# Exploration-Preserving Loop Distillation

当前默认路线为 **RAD-OPD**。Ouro-1.4B-Thinking 用原生 gate 和零回填 Ragged KV 在策略生成。训练时每题抽取一个退出阈值，冻结的满四轮教师在相同文本上打分，学生沿实际退出轨迹可微重放并做全词表 Reverse KL。退出头参数冻结。[方案](docs/rad_opd_mvp.md) · [默认配置](configs/rad_opd.json)

## Python 环境

以下命令在仓库根目录运行，适用于 Linux x86_64。运行默认 RAD-OPD 训练和 Hugging Face 评测时，安装「默认训练和评测」这一节即可。

### 默认训练和评测

使用本机的 Python 3.10 和 `venv` 模块创建 `.venv`。主环境使用 PyTorch `2.8.0+cu128` 和 Transformers `4.57.1`。安装 PyTorch 的 cu128 wheel 时会一并安装 CUDA 12.8 运行库。GPU 运行需要宿主机的 NVIDIA 驱动支持 CUDA 12.8。本机验证的组合为 Python 3.10.12、RTX 3090 和驱动 580.178.04。[训练依赖](requirements-stage.txt)

如果项目是迁移过来的，先备份旧 `.venv`，再用目标机器的解释器重建。

```bash
python3.10 -m venv .venv
.venv/bin/python -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.8.0+cu128'
.venv/bin/python -m pip install -r requirements-stage.txt pytest ninja
.venv/bin/python -m pip check
.venv/bin/python -c 'import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())'
```

最后一行打印 PyTorch 版本、CUDA 运行库版本和 GPU 可用状态。启动 GPU 训练前，确认最后一项为 `True`。

### 固定轮数 vLLM 推理

`configs/vllm_fixed_*.json` 使用 `ep_loop_envs/vllm026` 中的独立推理环境。该环境使用 Python 3.12、vLLM `0.26.0`、PyTorch `2.11.0+cu130` 和 FlashInfer `0.6.14`。宿主机需要支持 CUDA 13.0 的 NVIDIA 驱动，JIT 编译需要 GCC/G++。先安装 `uv` 并让 shell 能找到 `uv` 命令。[uv 安装说明](https://docs.astral.sh/uv/getting-started/installation/) · [推理依赖](requirements-vllm026.txt)

```bash
uv venv --python 3.12 ep_loop_envs/vllm026
UV_BIN=uv bash scripts/install_vllm026.sh
ep_loop_envs/vllm026/bin/python -c 'from importlib.metadata import version; print(version("vllm"), version("flashinfer-python"))'
```

训练和评测入口仍用 `.venv/bin/python` 启动，固定轮数配置会通过 `vllm_python_bin` 启动独立推理进程。

### vLLM-RLT 推理与 BBH 复现

BBH 复现使用 vLLM-RLT `v0.2.0`，固定到提交 `c08b9d79834bb8b4ff283b15bae2c14c559f6b52`。安装到复现脚本指定的 `benchmark-venv`，使用 Python 3.10、PyTorch `2.8.0+cu128` 和随 PyTorch 安装的 Triton `3.4.0`。[BBH 复现说明](docs/ragged_kv_reproduction.md)

```bash
RLT_RELEASE=audit/vllm_rlt_v020_release_20261120_144351
git clone --depth 1 --branch v0.2.0 https://github.com/InferMatrixLab/vllm-rlt.git "$RLT_RELEASE/vllm-rlt"
python3.10 -m venv "$RLT_RELEASE/benchmark-venv"
RLT_PY="$RLT_RELEASE/benchmark-venv/bin/python"
"$RLT_PY" -m pip install --index-url https://download.pytorch.org/whl/cu128 'torch==2.8.0+cu128'
"$RLT_PY" -m pip install -r requirements-stage.txt -e "$RLT_RELEASE/vllm-rlt"
"$RLT_PY" -m pip check
.venv/bin/python -c 'from ep_loop.fastpath.rlt_actor import verified_release; print(verified_release())'
```

使用 `configs/evaluation_vllm_rlt.json` 评测时，`scripts/evaluate_fastpath.py` 从 `external/vllm-rlt` 加载源码。可将该路径链接到上面的源码目录，然后用 `"$RLT_PY"` 启动评测脚本。

## 数据和模型

项目统一从 `data/` 读取预处理数据。迁移后把 `data` 直接链接到存放 `aime/`、`bbh_frontier/`、`dapo_math17k/` 和 `dapo_ablation/` 的目录。`models/` 通过 `.gitkeep` 随仓库保留，各模型按目录建立相对软链接。以下命令在项目根目录运行。

```bash
DATA_DIR=../huggingface_datasets/Exploration-Preserving-Loop
MODELS_DIR=../huggingface_models
ln -srT -- "$DATA_DIR" data
ln -srT -- "$MODELS_DIR/Ouro-1.4B-Thinking" models/Ouro-1.4B-Thinking
ln -srT -- "$MODELS_DIR/Ouro-1.4B" models/Ouro-1.4B
```

需要重新下载默认模型时，下载脚本会在相邻的 `huggingface_models/` 下保存固定版本并建立软链接。共享模型盘可通过 `--model-dir` 指定。

```bash
.venv/bin/python scripts/download_pilot_model.py
```

## RAD-OPD 入口

接好 `data` 和模型链接后，检查本地模型和 17,309 条 DAPO 去重训练池。资产齐备时输出 `"ready": true`。

```bash
.venv/bin/python scripts/train_stage_distillation.py --check-assets
```

全量回归测试还会读取 `models/huginn-0125/` 下的官方 `raven_config_minimal.py` 和 `raven_modeling_minimal.py`。准备好模型源码后运行 `.venv/bin/python -m pytest -q`。[Huginn 固定版本](configs/huginn_math_base.json)

Phase 0 先用非 Thinking Ouro 和 BBH 复现 Ragged KV 图 1(c)，随后在 AIME 2024 上用原始 Thinking 权重执行固定 1／2／4 轮、动态精确回填和动态零回填对照。[复现方案](docs/ragged_kv_reproduction.md)

```bash
.venv/bin/python scripts/evaluate_stage_distillation.py --phase-zero \
  --data data/aime/aime2024.jsonl --output ep_loop_runs/rad_phase0/responses.jsonl
```

训练单个 checkpoint，然后扫退出阈值。重复传 `--threshold` 可选取部分工作点。评测汇总包含 P@1、P@4、多预算切片、平均 decode 深度、Depth-1 比例和每层实际占用的 KV 槽。切片从同一条长回答截取，decode 深度统计从处理首个生成 token 开始。

```bash
.venv/bin/python scripts/train_stage_distillation.py --device cuda --output ep_loop_runs/rad_opd
.venv/bin/python scripts/evaluate_stage_distillation.py \
  --checkpoint ep_loop_runs/rad_opd/checkpoints/update_000100 \
  --data data/aime/aime2024.jsonl --output ep_loop_runs/rad_opd/curve.responses.jsonl
```

消融配置可以继承 `configs/rad_opd.json`。设置 `algorithm.rad_detach_reused_kv=true` 截断浅轮 KV 复用梯度，或设置 `algorithm.rad_tau.fixed` 训练单个阈值。离线对照通过 `data.rad_offline_responses` 指向冻结四轮教师生成的 JSONL，每行包含 `problem_sha256`、`sample` 和 `tokens`，覆盖训练池及每题所需样本。用评测入口的 `--loops 4 --write-token-ids` 导出时，显式传原始权重并将数据校验设置对齐教师语料。离线训练沿给定文本重新计算学生退出轨迹。

当前 rollout 用 PyTorch 维护紧凑物理 KV，并在 attention 读取时临时 gather。训练重放用 `torch.where` 保留梯度，CUDA SDPA 选择可用的 attention kernel。物理槽统计不计分配器余量、索引和临时 gather。

## 备选路线

固定深度配置保留在 `configs/a_loopopd.json` 和 `configs/vllm_fixed_*.json`。此前的 JS 动态学生可显式选择 `configs/hf_dynamic_js.json`，其余 ablation 入口继续使用各自的配置。

以下命令预览实验计划，只检查配置。

```bash
bash scripts/run_opd_comparison.sh --dry-run
OPD_MODE=d_loopopd ABLATION_GROUPS=top_n bash scripts/run_opd_comparison.sh --dry-run
ROUTE=hf bash scripts/run_three_route_ablation.sh --dry-run
ROUTE=hybrid bash scripts/run_three_route_ablation.sh --dry-run
```

固定轮数评测入口为 [evaluate_fastpath.py](scripts/evaluate_fastpath.py)。评估已保存的结果时，可从[评测计划](audit/checkpoint_eval_20261120_004854/plan.json)取得 checkpoint、数据和配置。使用 [vLLM-RLT 配置](configs/evaluation_vllm_rlt.json)可切换到本机的 vLLM-RLT 引擎，每张 GPU 运行一个副本。

历史 EP-D-LoopOPD 十步 checkpoint 的固定两轮 AIME 2024 评测得到 **Avg@16 24.17% · Pass@16 56.67%**。[评测核验](audit/checkpoint_eval_20261120_004854/validation.json) · [历史计时](docs/runtime_timing.md) · [论文与证据索引](ref/README.md)
