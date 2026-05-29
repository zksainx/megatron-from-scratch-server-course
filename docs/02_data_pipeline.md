# Data Pipeline

`LLMs-from-scratch` 从小文本开始解释 `tokenization`、`sliding window` 和 `DataLoader`。在 Megatron 中，等价流程被拆成两个阶段：

1. `raw data`: 你维护 JSONL / CSV / parquet / plain text。
2. `indexed dataset`: Megatron 读取 `.bin/.idx`，训练时按 `sequence length` 取样。

## Pretraining data format

Megatron `tools/preprocess_data.py` 的标准输入是 JSONL，每行至少有 `text` 字段：

```json
{"text": "Large language models predict the next token."}
{"text": "Training data is packed into fixed-length sequences."}
```

每行是一篇 `document`。加 `--append-eod` 后，文档末尾会追加 `EOD token`，帮助模型学习文档边界。

## Instruction / SFT data format

Megatron 当前 `SFTDataset` 读取 JSONL，每行有 `messages` 字段：

```json
{"messages":[{"role":"system","content":"You are a helpful assistant."},{"role":"user","content":"Explain gradient clipping."},{"role":"assistant","content":"Gradient clipping limits gradient norm to stabilize training."}]}
```

`SFTDataset` 会把 prompt token 的 label mask 掉，只对 assistant response 计算 loss，具体取决于 `SFTTokenizer` 的 `prompt format`。

## Classification data as SFT

`LLMs-from-scratch` Ch 6 使用 `classification head` 做 spam classification。Megatron GPT 主线不专门加 classification head。服务器实践中更通用的做法是把分类任务转成 generative `SFT`：

```json
{"messages":[{"role":"system","content":"Classify the message as spam or not_spam."},{"role":"user","content":"You won a free prize. Click now!"},{"role":"assistant","content":"spam"}]}
```

评估时用 exact match / accuracy / F1 统计 assistant 输出。

## Reasoning data

`reasoning-from-scratch` 的 reasoning trace 可以直接放到 assistant response：

```json
{"messages":[{"role":"system","content":"Solve math problems. Put final answer in \\boxed{}."},{"role":"user","content":"What is 17+25?"},{"role":"assistant","content":"17+25=42. Therefore, \\boxed{42}."}]}
```

如果要训练短回答模型，可以把 `chain-of-thought` 留给 teacher generation，然后 distill 成更短 response。

## Dataset quality checklist

1. `deduplication`: instruction tuning 前先去重，避免 benchmark contamination。
2. `length histogram`: 检查 token length，决定 `seq-length`。
3. `split`: train/valid/test 固定随机种子，避免调参污染 test。
4. `format validation`: JSONL 必须一行一个 JSON object。
5. `license`: 真实训练前确认数据授权。

## From-scratch tutorial datasets

主线数据来自两个教程仓库：

| Dataset | Source | 用途 |
|---|---|---|
| `the-verdict.txt` | `LLMs-from-scratch/ch02/01_main-chapter-code` | `pretraining` text corpus |
| `instruction-data.json` | `LLMs-from-scratch/ch07/01_main-chapter-code` | instruction `SFT` |
| `instruction-data-with-response.json` | `LLMs-from-scratch/ch07/01_main-chapter-code` | instruction evaluation |
| `instruction-data-with-preference.json` | `LLMs-from-scratch/ch07/04_preference-tuning-with-dpo` | `DPO` concept / preference data |
| `math500_test.json` | `reasoning-from-scratch/ch03/01_main-chapter-code` | reasoning evaluation |
| `math_train_sample.json` | `reasoning-from-scratch/ch08/02_generate_distillation_data` | reasoning SFT/distillation sample |
| `sample_ollama_outputs.json` | `reasoning-from-scratch/ch08/02_generate_distillation_data` | teacher-output distillation sample |

生成 Megatron-friendly JSONL：

```bash
python scripts/10_prepare_from_scratch_datasets.py \
  --train-root /sgl-workspace/zkx/train \
  --output-dir data/from_scratch
```

输出：

```text
data/from_scratch/pretrain_the_verdict.jsonl
data/from_scratch/sft_instruction_data.jsonl
data/from_scratch/eval_instruction_responses.jsonl
data/from_scratch/preference_dpo_format.jsonl
data/from_scratch/eval_math500.jsonl
data/from_scratch/sft_reasoning_math_train_sample.jsonl
data/from_scratch/sft_reasoning_distillation_sample.jsonl
```
