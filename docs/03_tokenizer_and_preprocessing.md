# Tokenizer And Preprocessing

`tokenizer` 是从 raw text 到 token id 的边界。`LLMs-from-scratch` 里会手写或使用 BPE 来理解这个过程；本教程主线使用 Megatron 支持的 `GPT2BPETokenizer`、`HuggingFaceTokenizer` 和 `SFTTokenizer`。

## Tokenizer choices

| 场景 | tokenizer | 用途 |
|---|---|---|
| GPT-2 style pretraining | `GPT2BPETokenizer` | 需要 `encoder.json`（vocab）和 `vocab.bpe`（merges） |
| Qwen/Llama style pretraining | `HuggingFaceTokenizer` | 使用 Hugging Face tokenizer 目录或模型名 |
| instruction SFT | `SFTTokenizer` | 基于 HF tokenizer 加 chat template 与 response loss mask |

## Preprocess command

```bash
python scripts/10_prepare_from_scratch_datasets.py \
  --train-root /sgl-workspace/zkx/train \
  --output-dir data/from_scratch

bash scripts/11_preprocess_gpt_data.sh \
  data/from_scratch/pretrain_the_verdict.jsonl \
  runs/data/the_verdict
```

默认脚本调用 Megatron native preprocessing：

```bash
python $MEGATRON_ROOT/tools/preprocess_data.py \
  --input data/from_scratch/pretrain_the_verdict.jsonl \
  --output-prefix runs/data/the_verdict \
  --json-keys text \
  --tokenizer-type GPT2BPETokenizer \
  --vocab-file $GPT2_VOCAB_FILE \
  --merge-file $GPT2_MERGE_FILE \
  --append-eod \
  --workers 8
```

这个路径是主线：`tokenization` 由 Megatron 执行，tokenizer 文件来自 `LLMs-from-scratch/ch02/02_bonus_bytepair-encoder/gpt2_model`。`$GPT2_VOCAB_FILE` 和 `$GPT2_MERGE_FILE` 在 `configs/4gpu_edu_pretrain.env` 中定义，分别指向 `encoder.json` 和 `vocab.bpe`。

## Offline Token-id Fallback

```bash
TOKENIZER_MODE=offline_token_ids \
  bash scripts/11_preprocess_gpt_data.sh \
  data/from_scratch/pretrain_the_verdict.jsonl \
  runs/data/the_verdict
```

`offline_token_ids` 会先用 `LLMs-from-scratch` 自带 GPT-2 BPE 实现编码，再让 Megatron `NullTokenizer` 读取 token ids。这个路径只用于 Hugging Face metadata 访问异常或内网隔离排障，不是默认训练路线。

主线学习时不要先用这个模式。只有当 native `GPT2BPETokenizer` preprocessing 失败，并且你已经确认 `LLMs-from-scratch/ch02/02_bonus_bytepair-encoder/gpt2_model/encoder.json` 和 `vocab.bpe` 存在时，才用它缩小问题范围。`vocab.json/merges.txt` 是 SFT helper 生成 Hugging Face tokenizer 目录时使用的文件名，不是 pretraining preprocessing 的输入文件名。

输出会生成：

```text
runs/data/the_verdict_text_document.bin
runs/data/the_verdict_text_document.idx
```

训练时传入的是不带 `.bin/.idx` 的 prefix：

```bash
bash scripts/21_run_pretrain_real_4gpu.sh runs/data/the_verdict_text_document
```

## Sequence length

`seq-length` 是每个 training sample 的 token 数。它决定：

1. `attention` 计算量近似按 `O(seq_length^2)` 增长。
2. `activation memory` 随 `micro_batch_size * seq_length * hidden_size * layers` 增长。
3. reasoning model 通常需要更长 `context length`，但教学 smoke test 应先用 512 或 1024。

## Data cache

Megatron 会为 dataset 生成 index cache。大规模训练建议固定：

```bash
--data-cache-path runs/cache/pretrain
```

真实大数据训练可先用 `tools/prepare_cache.py` 预构建 cache，避免 rank 0 在训练启动时独自建索引。

## Common errors

| 现象 | 常见原因 | 处理 |
|---|---|---|
| 找不到 `.idx` | `data-path` 传错，必须传 prefix | 确认结尾是 `_text_document` |
| tokenizer vocab mismatch | checkpoint 与 tokenizer 不一致 | 使用 checkpoint 对应 tokenizer |
| loss 全是 NaN | fp8/bf16 不稳定、lr 太大、数据异常 | 先用 `bf16`、小 lr、打开 loss check |
| startup 很慢 | building dataset index | 预构建 cache 或先小数据验证 |
