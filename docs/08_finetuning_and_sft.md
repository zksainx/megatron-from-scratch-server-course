# Finetuning And SFT

这章覆盖 `LLMs-from-scratch` Ch 6/7：classification finetuning 与 instruction finetuning。Megatron 主线使用 `SFTDataset`，数据格式是 `messages` JSONL。

## Instruction SFT

从 `LLMs-from-scratch` 的 instruction dataset 生成 `messages` JSONL：

```bash
python scripts/10_prepare_from_scratch_datasets.py \
  --train-root /sgl-workspace/zkx/train \
  --output-dir data/from_scratch
```

运行 SFT：

```bash
bash scripts/22_run_sft_4gpu.sh \
  data/from_scratch/sft_instruction_data.jsonl \
  runs/tokenizers/gpt2_from_scratch
```

如果没有传入 tokenizer，脚本会从 `LLMs-from-scratch/ch02/02_bonus_bytepair-encoder/gpt2_model` 生成本地 Hugging Face tokenizer 目录，避免服务器离线时访问 Hugging Face Hub。

推荐在教程中显式传入 `runs/tokenizers/gpt2_from_scratch`，这样从新的 shell 复制命令也不依赖预先导出的 `COURSE_ROOT`。

## SFT data rule

每行：

```json
{"messages":[{"role":"system","content":"..."},{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

要求：

1. 第一条通常是 `system`。
2. `user` 和 `assistant` 交替。
3. assistant 内容不能为空。
4. 多轮对话可以放在同一个 `messages` list。

## Loss masking 说明

本教程使用 GPT-2 tokenizer 的 `--sft-tokenizer-prompt-format identity`，训练 loss 包含所有 token（不区分 prompt/response）。这是因为 GPT-2 tokenizer 没有 chat template，无法自动识别 prompt/response 边界。

要实现 response-only loss masking，需要使用带 chat template 的 tokenizer（如 Llama/Qwen 的 HuggingFace tokenizer）并选择对应的 prompt format（如 `nemotron-h-aligned`）。

## Classification as generation

把 classification 样本改成：

```json
{"messages":[{"role":"system","content":"Return exactly one label: spam or not_spam."},{"role":"user","content":"Free money now!!!"},{"role":"assistant","content":"spam"}]}
```

训练后评估：

1. 用固定 prompt 生成 label。
2. 清洗输出，只保留允许 label。
3. 计算 accuracy、precision、recall、F1。

这种方法牺牲了专门 classification head 的简单性，但统一了 instruction SFT 和 reasoning SFT。

## Preference data and DPO

`LLMs-from-scratch` Ch 7 bonus 介绍 `DPO`，数据通常是：

```json
{"prompt":"...","chosen":"...","rejected":"..."}
```

本地 Megatron 当前主线更偏 `SFT` 与 `GRPO/RL`。如果要做 DPO，可以把它作为扩展：先用 from-scratch DPO notebook 理解 objective，再接入支持 DPO 的 training stack，或在 Megatron 上新增 data loader 与 loss。这个教程不伪造一个 Megatron DPO 脚本。

## LoRA / PEFT

`LoRA` 是 parameter-efficient finetuning，适合显存受限或多任务 adapter。你当前 4 卡 B200/GB200 更适合先掌握 full-rank SFT；LoRA 可作为后续扩展。

## Fine-tuning checklist

1. 从 pretraining checkpoint 或 HF converted checkpoint 开始。
2. tokenizer 必须和 checkpoint 匹配。
3. SFT learning rate 通常比 pretraining 小（本教程脚本使用 `--lr 1.0e-5 --min-lr 1.0e-6`）。
4. 保存频率更高，因为 SFT 数据小、过拟合快。
5. 每个 checkpoint 都要跑 instruction eval 和 reasoning eval。
