# Pretraining

`pretraining` 是 next-token prediction。`LLMs-from-scratch` Ch 5 手写训练循环；本教程用 `Megatron-LM/pretrain_gpt.py`。

## Stage 1: real-data smoke test

先从 `LLMs-from-scratch` 的 `the-verdict.txt` 生成 JSONL：

```bash
python scripts/10_prepare_from_scratch_datasets.py \
  --train-root /sgl-workspace/zkx/train \
  --output-dir data/from_scratch
```

再预处理：

```bash
bash scripts/11_preprocess_gpt_data.sh \
  data/from_scratch/pretrain_the_verdict.jsonl \
  runs/data/the_verdict
```

这个脚本默认使用 Megatron native `GPT2BPETokenizer`，tokenizer 文件来自 `LLMs-from-scratch` 的 GPT-2 BPE bonus 材料。只有在 Hugging Face metadata 访问异常时，才显式设置 `TOKENIZER_MODE=offline_token_ids` 进入 fallback。

`the-verdict.txt` 是极小教学语料。做 4GPU real-data smoke test 时，先使用 train-only split：

```bash
DATA_SPLIT=100,0,0 EVAL_ITERS=0 \
bash scripts/21_run_pretrain_real_4gpu.sh runs/data/the_verdict_native_text_document
```

这个命令验证：

1. `torchrun` 能启动 4 个 rank。
2. Megatron 使用真实 `.bin/.idx` dataset。
3. `GPT2BPETokenizer`、Transformer Engine、Apex fused kernels、JIT/fusion 可用。
4. model 能 forward/backward。
5. checkpoint 和 TensorBoard 目录可写。

更大的 pretraining corpus 再使用默认 `DATA_SPLIT=90,9,1` 并打开 validation/test。

最后训练：

```bash
bash scripts/21_run_pretrain_real_4gpu.sh runs/data/the_verdict_text_document
```

## What to watch

训练日志中重点看：

| Metric | 意义 |
|---|---|
| `lm loss` | next-token cross entropy |
| `learning rate` | scheduler 是否按预期 warmup/decay |
| `grad norm` | 梯度是否爆炸 |
| `iteration time` | 每步耗时 |
| `tokens/sec` | 训练吞吐 |
| `mem` | 显存占用 |

## Loss expectation

`the-verdict.txt` 是教程级小语料，loss 不代表通用模型质量，只用于验证 pipeline。正式训练时：

1. `train loss` 应整体下降，但会有波动。
2. `valid loss` 才能反映泛化。
3. 数据太小会迅速 overfit。
4. 如果 tiny real-data loss 不稳定，优先排查环境、训练参数或 tokenizer/data mismatch。

## Continue training

Megatron 使用：

```bash
--save runs/checkpoints/pretrain
--load runs/checkpoints/pretrain
```

如果目录内有 checkpoint，会自动尝试恢复。想从头开始训练，换一个新的 checkpoint 目录，不要直接删除正在用的 checkpoint。

## Scaling to real corpus

真实数据建议：

1. 把 raw corpus 转成 JSONL，每行一个 `{"text": ...}`。
2. 先抽样 10k-100k 行做 pipeline test。
3. 固定 tokenizer 后再全量 preprocess。
4. 先用 10k-100k 行真实抽样得到 smoke benchmark。
5. 再比较全量 real-data throughput，定位 dataloader 和 filesystem 瓶颈。
