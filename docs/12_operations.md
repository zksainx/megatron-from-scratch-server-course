# Operations

这章把教程流程整理成服务器作业规范。

## Run layout

```text
runs/
├── cache/
├── checkpoints/
├── data/
├── logs/
├── tensorboard/
└── reports/
```

## Naming

建议 run name 包含：

```text
{stage}-{model_preset}-{data}-{precision}-{date}
```

例如：

```text
pretrain-edu1b-tiny-bf16-20260529
sft-qwen3-edu1b-math-bf16-20260529
grpo-qwen3-math-kl001-20260529
```

## Before a long run

1. `bash scripts/00_healthcheck.sh`
2. `the-verdict.txt` real-data smoke run。
3. 小规模真实 corpus 抽样 run。
4. 确认 TensorBoard 正常。
5. 确认 checkpoint 能恢复。
6. 确认 evaluation script 能读模型输出。

## During a run

1. 保存完整 command。
2. 保存 git commit 或 `git status`。
3. 定期 sample output。
4. 记录 OOM、NaN、hang 的上下文。

## After a run

1. 解析 logs。
2. 生成 loss curve。
3. 跑 benchmark。
4. 跑 evaluation。
5. 保存 run report。
6. 清理无用中间 checkpoint，但不要删除最优 checkpoint 和复现实验所需配置。

## Troubleshooting

| Symptom | First check |
|---|---|
| `torchrun` hang | NCCL env, port conflict, visible GPUs |
| OOM | reduce `micro_batch_size`, `seq-length`, enable recompute |
| NaN loss | lower lr, disable fp8, check data, enable loss checks |
| slow startup | dataset cache build |
| low GPU util | dataloader, tiny model, too frequent eval/save |
| cannot resume | checkpoint dir, model args changed, tokenizer changed |
| Triton `.so` stale file handle | move `COMPILE_CACHE_DIR`, `TRITON_CACHE_DIR`, `TORCH_EXTENSIONS_DIR` to local disk such as `/tmp` |
| `cublasLtGroupedMatrixLayoutInit_internal` missing | upgrade `nvidia-cublas` and ensure pip CUDA13 lib path is before `/usr/local/cuda/lib64` |
| `fused_weight_gradient_mlp_cuda module is not found` | install NVIDIA Apex with `APEX_CPP_EXT=1 APEX_CUDA_EXT=1` |

## Runtime Cache Policy

训练主路径会使用 Megatron fused/JIT path。为避免 PyTorch Inductor/Triton 扩展在网络文件系统上生成损坏句柄，`scripts/common_env.sh` 默认：

```bash
COMPILE_CACHE_DIR=/tmp/megatron-server-course-cache-${USER:-user}
```

把 `TORCH_EXTENSIONS_DIR`、`TRITON_CACHE_DIR`、`FLASHINFER_CACHE_DIR` 和 `HOME` 放在本机临时盘。不要用 `--disable-jit-fuser` 来掩盖这个问题；先修 cache placement，再重跑 smoke test。

## Required Kernel Stack

主线训练必须满足：

```text
Transformer Engine import ok
Megatron HAVE_TE=True
Apex fused_weight_gradient_mlp_cuda import ok
gradient_accumulation_fusion=True
disable_jit_fuser=False
```

如果其中任意一项失败，先修安装或 library path。不要把 `local transformer implementation`、关闭 `gradient_accumulation_fusion` 或禁用 JIT 当成默认教程路径。

## Fallback Policy

主线训练保持 Megatron native path：`GPT2BPETokenizer`、fused kernels、JIT/fusion、Transformer Engine 可用时优先使用。只有在排障时才使用 fallback：

1. `TOKENIZER_MODE=offline_token_ids`: Hugging Face metadata 访问异常时，用教程 BPE 预编码 token ids。
2. `--disable-jit-fuser`: 仅用于定位 PyTorch Inductor/Triton cache 问题。
3. local HF tokenizer: 仅用于 `SFTTokenizer` 在无外网或模型 tokenizer 未下载时的格式验证。
