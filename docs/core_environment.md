# 本机 RAD-OPD 核心环境

项目训练依赖、vLLM-RLT、verl 和 FlashAttention 统一安装在仓库根目录的 **`.venv`** 中。本机为 H100 80GB，驱动 `570.172.08`，CUDA 12.8 工具链位于 `/usr/local/cuda-12.8`。

| 组件 | 版本 | 说明 |
| --- | --- | --- |
| Python | `3.10.12` | 系统解释器创建的独立 venv |
| PyTorch / Triton | `2.8.0+cu128` / `3.4.0` | CUDA 12.8 wheel 栈 |
| vLLM-RLT | `0.2.0` | 固定到方案指定的源码提交 |
| verl | 标签 `v0.8.0`，包版本 `0.8.0.dev0` | editable 安装 |
| Transformers | `4.57.1` | 与现有 Ouro 代码一致 |
| NumPy | `1.26.4` | 同时满足项目、RLT 和 verl |
| FlashAttention | `2.8.3` | Torch 2.8、CUDA 12、CXX11 ABI TRUE 官方 wheel |
| TensorDict / Ray | `0.10.0` / `2.54.0` | verl 核心依赖 |

vLLM-RLT 要求 NumPy `>=1.24`，verl `v0.8.0` 要求 `<2`，因此可以使用同一环境。项目用到的 NumPy 数值、随机数和状态保存接口均已在 1.26 提供；`requirements-pilot.txt` 已同步固定为 `1.26.4`。

源码使用 editable 安装：

- vLLM-RLT：`external/vllm-rlt`，标签 `v0.2.0`，提交 `c08b9d79834bb8b4ff283b15bae2c14c559f6b52`。
- verl：`external/verl`，标签 `v0.8.0`，提交 `7aed6b230776f963fa09509c10d9c3a767d1102c`。该官方标签中的版本文件是 `0.8.0.dev`，所以 Python 包元数据显示 `0.8.0.dev0`。

为兼容现有 `verified_release()` 和 BBH 入口，RLT 源码实际保存在 `audit/vllm_rlt_v020_release_20261120_144351/vllm-rlt`，`external/vllm-rlt` 链接到该目录；同目录的 `benchmark-venv` 链接到同一个 `.venv`。

## 使用

在仓库根目录激活环境：

```bash
source .venv/bin/activate
```

也可以直接指定解释器：

```bash
.venv/bin/vllm-rlt-serve --help
.venv/bin/python -c 'from ep_loop.fastpath.rlt_actor import verified_release; print(verified_release())'
.venv/bin/python -c 'import verl, vllm_rlt; print(verl.__file__, vllm_rlt.__file__)'
```

本机系统默认 `nvcc` 指向 CUDA 12.2。若后续编译 CUDA 扩展，在当前终端指定已有的 12.8 工具链：

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
```

## 重建依赖

统一依赖入口为 [requirements-core.txt](../requirements-core.txt)，其中包含现有训练依赖。

需要代理时，仅在当前终端配置：

```bash
export HTTPS_PROXY=http://127.0.0.1:7892
export HTTP_PROXY=http://127.0.0.1:7892
export NO_PROXY=127.0.0.1,localhost
export UV_HTTP_TIMEOUT=3600
```

在全新检出的仓库中，先准备固定版本源码：

```bash
RLT_RELEASE=audit/vllm_rlt_v020_release_20261120_144351
git clone --depth 1 --branch v0.2.0 https://github.com/InferMatrixLab/vllm-rlt.git "$RLT_RELEASE/vllm-rlt"
mkdir -p external
ln -srT "$RLT_RELEASE/vllm-rlt" external/vllm-rlt
git clone --depth 1 --branch v0.8.0 https://github.com/volcengine/verl.git external/verl
```

安装到一个环境中：

```bash
uv venv --python /usr/bin/python3.10 --seed .venv
uv pip install --python .venv/bin/python --torch-backend cu128 \
  -r requirements-core.txt -e external/verl -e external/vllm-rlt
ln -srT .venv audit/vllm_rlt_v020_release_20261120_144351/benchmark-venv
```

FlashAttention 使用 Dao-AILab 官方 `v2.8.3` 发布的 Python 3.10、CUDA 12、Torch 2.8、CXX11 ABI TRUE 预编译 wheel。安装前确认 `torch._C._GLIBCXX_USE_CXX11_ABI` 为 `True`；wheel 不需要本机编译。

```bash
uv pip install --python .venv/bin/python \
  'https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3%2Bcu12torch2.8cxx11abiTRUE-cp310-cp310-linux_x86_64.whl'
```

## 验证与范围

本机验证已通过：147 个包的依赖一致性检查，RLT 的 Ouro-1.4B-Thinking 满四轮与动态两轮退出短生成，FlashAttention 反向传播、verl FSDP 引擎导入、单卡 NCCL/FSDP 优化器更新，以及 11 项 NumPy/随机状态/检查点回归测试。RLT 服务和 verl 训练入口的 `--help` 均可运行。

```bash
uv pip check --python .venv/bin/python
.venv/bin/python -m pytest -q tests/test_reproducibility.py tests/test_checkpoint_tensorboard.py
```

本次安装的下载清单、官方 GPU wheel SHA256、冻结依赖 `freeze.txt`、命令日志和 GPU 验证脚本保存在 `.cache/core-env-install/`。完整安装记录为 `installation.json`，详细 GPU 结果为 `validation-rlt.json` 和 `validation-verl.json`。

RAD-OPD 的现有训练入口仍是 `scripts/train_stage_distillation.py`。verl 的内置 rollout 后端没有注册本项目的 vLLM-RLT；后续使用 verl 编排 RAD-OPD 时，还需接入 RLT rollout 和 No-Detach Ragged 重放。这里安装并验证的是核心运行环境。

## 本机官方 vLLM 推理环境

固定轮数的 vLLM 配置使用 `ep_loop_envs/vllm026/bin/python`。在 H100 和驱动 `570.172.08` 上安装的是官方 `vllm 0.26.0+cu129`，配套 Python `3.12.14`、PyTorch `2.11.0+cu129`、Transformers `5.5.3` 和 FlashInfer `0.6.14`。

在仓库根目录可直接检查版本。需要在终端中使用官方 vLLM CLI 时，激活该环境。

```bash
source ep_loop_envs/vllm026/bin/activate
vllm --version
```

本机运行 FlashInfer 时可用 CUDA 12.8 工具链编译 GPU 内核。

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH="$CUDA_HOME"
export PATH="$CUDA_HOME/bin:$PATH"
```

重建本机的 CUDA 12.9 环境时，选用官方 `cu129` wheel。

```bash
uv venv --python 3.12 --seed ep_loop_envs/vllm026
uv pip install --python ep_loop_envs/vllm026/bin/python --torch-backend cu129 \
  --index https://wheels.vllm.ai/0.26.0/cu129 \
  'vllm==0.26.0+cu129' 'torch==2.11.0+cu129' \
  'torchvision==0.26.0+cu129' 'torchaudio==2.11.0+cu129' \
  'transformers==5.5.3' 'flashinfer-python==0.6.14'
uv pip check --python ep_loop_envs/vllm026/bin/python
```

本机已通过 194 个包的依赖检查，以及 Ouro-1.4B-Thinking 的固定两轮 FlashAttention 3 和固定四轮 FlashInfer 短生成。固定四轮验证包含 CUDA Graph 捕获。结果、依赖快照和安装记录保存在 `.cache/vllm026-install/`。
