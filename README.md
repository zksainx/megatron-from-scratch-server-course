# Megatron Server Course From Scratch

这是一套把 `LLMs-from-scratch` 与 `reasoning-from-scratch` 的知识主线迁移到服务器环境的训练教程。核心思想是：不再手写 `Transformer` 组件，而是用 `Megatron-LM` 承担 `model implementation`、`distributed training`、`checkpointing` 与 `mixed precision`；学习重点转到完整训练流程。

本教程默认工作目录如下：

```bash
/sgl-workspace/zkx/train
├── LLMs-from-scratch
├── reasoning-from-scratch
├── Megatron-LM
└── megatron-from-scratch-server-course
```

## 学习目标

完成本教程后，你应该能独立跑通以下流程：

1. `environment check`: 确认 4 卡 B200/GB200、CUDA、NCCL、PyTorch、Transformer Engine 与 Megatron 可用。
2. `data preparation`: 从原始 text / instruction / math reasoning 数据生成训练数据。
3. `tokenization and preprocessing`: 使用 Megatron 的 `tools/preprocess_data.py` 生成 `.bin/.idx`，理解 `document`、`sample`、`sequence length`、`data split`。
4. `pretraining`: 用 `pretrain_gpt.py` 在 4 卡上跑教程真实数据 smoke test 与后续真实数据训练。
5. `loss and logging`: 看 `lm loss`、`learning rate`、`grad norm`、`tokens/sec`、`TFLOPs`、`TensorBoard`。
6. `checkpointing`: 保存、恢复、检查 checkpoint，理解 `torch_dist` 与 `iteration` 目录。
7. `benchmark`: 用不同规模的真实 `.bin/.idx` 数据比较吞吐、I/O 与 GPU utilization。
8. `supervised finetuning`: 将 instruction / classification / reasoning trace 数据转为 `messages` JSONL，用 `SFTDataset` 做 `SFT`。
9. `evaluation`: 覆盖 `perplexity-style validation`、`instruction response evaluation`、`MATH-500 verifier`、`MMLU`、`LLM-as-a-judge`、`leaderboard`。
10. `reasoning post-training`: 覆盖 `inference-time scaling`、`self-consistency`、`self-refinement`、`GRPO/RLVR`、`distillation`。

## 课程顺序

建议按顺序阅读并执行：

```text
docs/00_course_map.md
docs/01_environment.md
docs/02_data_pipeline.md
docs/03_tokenizer_and_preprocessing.md
docs/04_model_and_parallelism.md
docs/05_pretraining.md
docs/06_loss_logging_checkpoint.md
docs/07_benchmarking.md
docs/08_finetuning_and_sft.md
docs/09_evaluation.md
docs/10_reasoning_workflow.md
docs/11_distillation_and_export.md
docs/12_operations.md
```

## 最小执行路径

先跑环境检查：

```bash
cd /sgl-workspace/zkx/train/megatron-from-scratch-server-course
bash scripts/00_healthcheck.sh
```

`healthcheck` 必须显示 `transformer_engine_torch_ok True`、`megatron_have_te True`。

再从两个教程自带数据集生成 Megatron-friendly 数据：

```bash
python scripts/10_prepare_from_scratch_datasets.py \
  --train-root /sgl-workspace/zkx/train \
  --output-dir data/from_scratch
```

预处理 `LLMs-from-scratch` 的 `the-verdict.txt` 并启动真实数据训练：

```bash
bash scripts/11_preprocess_gpt_data.sh \
  data/from_scratch/pretrain_the_verdict.jsonl \
  runs/data/the_verdict

bash scripts/21_run_pretrain_real_4gpu.sh \
  runs/data/the_verdict_text_document
```

`the-verdict.txt` 是教学 tiny corpus。做最小 smoke test 时可以先只跑 train split，避免极小语料在 validation/test split 上重复采样：

```bash
DATA_SPLIT=100,0,0 EVAL_ITERS=0 \
bash scripts/21_run_pretrain_real_4gpu.sh \
  runs/data/the_verdict_text_document
```

这个真实数据 smoke test 必须至少出现 `iteration 1`、`iteration 2`、`lm loss` 和 `successfully saved checkpoint`，否则不要进入长训练。

默认 preprocessing 使用 Megatron native `GPT2BPETokenizer`，并读取 `LLMs-from-scratch` 自带的 `encoder.json/vocab.bpe`。如果服务器临时无法访问 Hugging Face tokenizer metadata，可显式设置 `TOKENIZER_MODE=offline_token_ids` 使用 fallback；fallback 不作为主线训练方式。

默认训练不关闭 Megatron/PyTorch JIT/fusion。`scripts/common_env.sh` 会把 Triton、Torch extensions、FlashInfer 等编译缓存放到 `/tmp/megatron-server-course-cache-*`，避免网络文件系统上的 stale handle 问题。

然后用 `LLMs-from-scratch` 的 instruction dataset 做 `SFT` 数据格式检查或训练：

```bash
bash scripts/22_run_sft_4gpu.sh \
  data/from_scratch/sft_instruction_data.jsonl \
  runs/tokenizers/gpt2_from_scratch
```

## 目录说明

```text
configs/      4-GPU B200-oriented presets and environment templates
data/         rebuild instructions; generated JSONL files are ignored by Git
docs/         Chinese tutorial, English technical terms
scripts/      server-friendly shell/Python scripts
runs/         ignored runtime artifacts: logs, checkpoints, cache, preprocessed data
```

## 与两个 from-scratch 教程的关系

本教程不会重复实现 `Multi-Head Attention`、`LayerNorm`、`GELU/SwiGLU`、`KV cache`、`GRPO loss` 等内部组件。你仍然会学习这些概念，但实现入口改为 Megatron 参数、数据格式、训练脚本、日志、评估与 checkpoint 操作。

完整映射见 [docs/00_course_map.md](docs/00_course_map.md)。

## 使用的教程数据集

本教程主线使用下列本地文件：

```text
LLMs-from-scratch/ch02/01_main-chapter-code/the-verdict.txt
LLMs-from-scratch/ch07/01_main-chapter-code/instruction-data.json
LLMs-from-scratch/ch07/01_main-chapter-code/instruction-data-with-response.json
LLMs-from-scratch/ch07/04_preference-tuning-with-dpo/instruction-data-with-preference.json
reasoning-from-scratch/ch03/01_main-chapter-code/math500_test.json
reasoning-from-scratch/ch08/02_generate_distillation_data/math_train_sample.json
reasoning-from-scratch/ch08/02_generate_distillation_data/sample_ollama_outputs.json
```
