# Environment

本教程假设你在单机 4 卡 B200/GB200 类机器上训练。GPU 名字如果显示为 `L20A` 不重要，关键看 `compute capability`、`bf16/fp8`、`NCCL` 和实际吞吐。

## 目录变量

```bash
export TRAIN_ROOT=/sgl-workspace/zkx/train
export MEGATRON_ROOT=$TRAIN_ROOT/Megatron-LM
export COURSE_ROOT=$TRAIN_ROOT/megatron-from-scratch-server-course
export RUN_ROOT=$COURSE_ROOT/runs
```

## 必查项

```bash
nvidia-smi
python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda", torch.version.cuda)
print("cuda available", torch.cuda.is_available())
print("device count", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(i, p.name, "cc", f"{p.major}.{p.minor}", "mem_gb", round(p.total_memory/1024**3, 1))
PY
```

## Megatron import check

```bash
cd $MEGATRON_ROOT
python - <<'PY'
import megatron
from megatron.core import parallel_state
print("Megatron import ok")
PY
```

## Transformer Engine and Apex

B200/GB200 主路径必须安装 `Transformer Engine` 和 `Apex`。不要用 `--transformer-impl local`、`--no-gradient-accumulation-fusion` 或 `--disable-jit-fuser` 作为默认方案；这些只适合定位问题。

本环境使用 `torch 2.9.1+cu130`，对应 CUDA 13：

```bash
python -m pip install --no-build-isolation "transformer-engine[pytorch,core_cu13]"
python -m pip install --upgrade nvidia-cublas
git clone https://github.com/NVIDIA/apex.git /tmp/nvidia-apex
cd /tmp/nvidia-apex
APEX_CPP_EXT=1 APEX_CUDA_EXT=1 python -m pip install -v --no-build-isolation .
```

`transformer-engine==2.15.0` 需要较新的 `libcublasLt.so.13` 符号，例如 `cublasLtGroupedMatrixLayoutInit_internal`。如果 import 报这个符号缺失，优先升级 `nvidia-cublas`，并让 `scripts/common_env.sh` 把 pip CUDA13 library path 放到 `LD_LIBRARY_PATH` 前面。

安装后必须看到：

```bash
cd $COURSE_ROOT
bash scripts/00_healthcheck.sh
```

关键输出包括：

```text
transformer_engine 2.15.0
transformer_engine_torch_ok True
megatron_have_te True
megatron_te_spec_provider_ok True
```

再单独确认 Apex fused extension：

```bash
python - <<'PY'
import apex
import fused_weight_gradient_mlp_cuda
import fused_layer_norm_cuda
import amp_C
print("apex fused extensions ok")
PY
```

## Distributed smoke test

```bash
cd $COURSE_ROOT
bash scripts/00_healthcheck.sh
```

这个脚本只检查环境，不启动长时间训练。通过后再运行 `the-verdict.txt` real-data smoke test。

## 推荐环境变量

```bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=WARN
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTHONFAULTHANDLER=1
```

`CUDA_DEVICE_MAX_CONNECTIONS=1` 是 Megatron 常用设置，能改善部分 `tensor parallel` / `sequence parallel` 通信调度。

## Compile Cache

Megatron、PyTorch Inductor、Triton、FlashInfer 和 dataset helper 会在启动时编译或加载本地扩展。不要把这些编译产物放在网络文件系统或不稳定挂载上，否则可能出现 `Stale file handle`、`.so` 加载失败或重复编译。

本教程的 `scripts/common_env.sh` 默认把编译缓存放到本机临时盘：

```bash
export COMPILE_CACHE_DIR=/tmp/megatron-server-course-cache-${USER:-user}
export TORCH_EXTENSIONS_DIR=$COMPILE_CACHE_DIR/torch_extensions
export TRITON_CACHE_DIR=$COMPILE_CACHE_DIR/triton
export FLASHINFER_CACHE_DIR=$COMPILE_CACHE_DIR/flashinfer
```

这不是关闭 Megatron 功能，也不是 fallback；它是服务器训练的正常环境设置。`runs/cache/runtime` 仍用于课程数据相关 cache，例如 `HF_HOME`。

## Performance Packages

如果 `scripts/00_healthcheck.sh` 提示 `Transformer Engine and Apex are not installed`，不要继续主线训练；先修环境。正式训练建议使用包含 `transformer_engine`、`apex`、匹配 CUDA/NCCL 与 Megatron 版本的 NVIDIA PyTorch container 或等价环境。

## Server workflow

服务器上不依赖 Jupyter。推荐使用：

```bash
tmux new -s megatron-course
cd /sgl-workspace/zkx/train/megatron-from-scratch-server-course

python scripts/10_prepare_from_scratch_datasets.py \
  --train-root /sgl-workspace/zkx/train \
  --output-dir data/from_scratch

bash scripts/11_preprocess_gpt_data.sh \
  data/from_scratch/pretrain_the_verdict.jsonl \
  runs/data/the_verdict

DATA_SPLIT=100,0,0 EVAL_ITERS=0 \
bash scripts/21_run_pretrain_real_4gpu.sh runs/data/the_verdict_text_document
```

长任务必须写日志到文件。后续所有脚本都会默认创建 `runs/logs`、`runs/checkpoints`、`runs/tensorboard`。
