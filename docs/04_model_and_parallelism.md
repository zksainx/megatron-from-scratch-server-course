# Model And Parallelism

这章对应 `LLMs-from-scratch` Ch 3/4/5 bonus 与 `reasoning-from-scratch` Qwen3 appendix。我们不再手写 `Transformer block`，而是理解它如何映射到 Megatron 参数。

## GPT/Llama-style architecture

| Concept | Megatron argument |
|---|---|
| `number of layers` | `--num-layers` |
| `embedding dimension / hidden size` | `--hidden-size` |
| `feed-forward dimension` | `--ffn-hidden-size` |
| `attention heads` | `--num-attention-heads` |
| `Grouped-Query Attention` | `--group-query-attention --num-query-groups` |
| `RoPE` | `--position-embedding-type rope --rotary-base ...` |
| `SwiGLU` | `--swiglu` |
| `RMSNorm` | `--normalization RMSNorm` |
| `causal attention backend` | `--attention-backend flash/fused/auto` |
| tied embeddings | omit or set `--untie-embeddings-and-output-weights` |

## Why not implement attention yourself

教学上，手写 `self-attention` 能解释机制；服务器训练上，手写实现通常会浪费硬件。Megatron/Transformer Engine 提供：

1. optimized `FlashAttention` / fused kernels。
2. `tensor parallel` 版 linear layers。
3. `sequence parallel` 与 communication overlap。
4. `activation checkpointing`。
5. `fp8` kernels for Hopper/Blackwell。

## 4-GPU parallelism presets

| Preset | 用途 | TP | PP | CP | DP | 说明 |
|---|---:|---:|---:|---:|---:|---|
| `edu-350m` | 教学与真实数据 smoke run | 1 | 1 | 1 | 4 | 最稳定，吞吐看 DP |
| `edu-1b` | 充分利用 4 卡训练小模型 | 1 | 1 | 1 | 4 | batch 更大，适合 pretraining |
| `b200-8b-fp8` | B200/GB200 资源利用 | 2 | 1 | 2 | 1 | 长上下文和 fp8，数据并行少 |
| `debug-tiny` | 快速验证脚本 | 1 | 1 | 1 | 4 | 极小模型，不能代表性能 |

`TP * PP * CP * DP = world_size`。单机 4 卡时 `world_size=4`。

## Precision

| Precision | 适用 |
|---|---|
| `fp32` | debugging only |
| `bf16` | 默认训练精度，稳定 |
| `fp8` | B200/GB200/Hopper 上用于高吞吐，需要 Transformer Engine |

建议顺序：

1. `the-verdict real-data + bf16` 验证训练栈和数据管线。
2. `sampled real-data + bf16` 验证较大数据吞吐。
3. `sampled real-data + fp8` 建立硬件吞吐参考。
4. `full real-data + fp8` 做正式 benchmark。

## Batch sizes

Megatron 的有效 batch：

```text
global_batch_size = micro_batch_size * data_parallel_size * gradient_accumulation_steps
```

如果 `TP=2, CP=2, DP=1`，那么 global batch 主要依赖 gradient accumulation。吞吐可能高，但 optimizer update 更少；教学时要明确区分 `tokens/sec` 与 `samples/update`。

## Memory controls

常用参数：

```bash
--recompute-granularity selective
--use-distributed-optimizer
--overlap-grad-reduce
--overlap-param-gather
--sequence-parallel
```

`sequence-parallel` 通常和 `TP>1` 搭配。`recompute` 用更多计算换显存。
