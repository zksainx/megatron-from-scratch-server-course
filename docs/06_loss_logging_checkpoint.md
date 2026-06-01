# Loss Logging Checkpoint

这章对应 `LLMs-from-scratch` 的 training loop、scheduler、checkpoint，以及 `reasoning-from-scratch` 的 RL/SFT tracking 思路。

## Loss

`pretraining loss` 是 `cross entropy`。Megatron 日志里的 `lm loss` 通常是按 non-masked token 归一化后的 loss。

对于 `SFT`：

1. 如果使用带 chat template 的 tokenizer（如 Llama/Qwen）并选择对应的 `--sft-tokenizer-prompt-format`（如 `nemotron-h-aligned`），prompt token 被 mask，只有 assistant response 参与 loss（`response-only loss`）。
2. 本教程使用 GPT-2 tokenizer 的 `identity` prompt format，所有 token 都参与 loss（不区分 prompt/response）。这是 GPT-2 tokenizer 缺少 chat template 的限制。
3. padding token 被 mask。

## Logging

本教程脚本默认打开：

```bash
--log-interval 1
--tensorboard-dir runs/tensorboard/...
--log-throughput
--log-params-norm
--log-num-zeros-in-grad
```

启动 TensorBoard：

```bash
tensorboard --logdir runs/tensorboard --host 0.0.0.0 --port 6006
```

如果服务器没有浏览器，用 SSH tunnel：

```bash
ssh -L 6006:localhost:6006 user@server
```

## Checkpoint

Megatron checkpoint 常见目录：

```text
runs/checkpoints/pretrain/
├── iter_0000001/
├── latest_checkpointed_iteration.txt
└── ...
```

检查 checkpoint：

```bash
bash scripts/40_inspect_checkpoint.sh runs/checkpoints/pretrain
```

本教程真实 pretraining 脚本默认 checkpoint 目录是：

```bash
bash scripts/40_inspect_checkpoint.sh runs/checkpoints/pretrain_real
```

解析训练日志：

```bash
python scripts/30_parse_training_log.py runs/logs/pretrain_real.log
```

成功的 smoke run 至少会解析出 2 条 `lm_loss`。

## Save interval

教学建议：

```bash
--save-interval 20
--eval-interval 20
--eval-iters 5
```

正式训练建议按 wall-clock 设计，例如每 15-30 分钟保存一次，避免频繁 checkpoint 拖慢训练。

## Failure recovery

1. 如果训练中断，保留 `--load` 指向同一 checkpoint 目录。
2. 如果 checkpoint 写坏，切换到前一个 `iter_*`。
3. 如果 tokenizer 或 model shape 改了，不要加载旧 checkpoint。
4. 如果 optimizer state 导致问题，可尝试只加载 model weights：使用 `--no-load-optim --no-load-rng` 跳过 optimizer state 和 random state 恢复。
