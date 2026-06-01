# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repository Is

A server-side training course that maps the knowledge from `LLMs-from-scratch` and `reasoning-from-scratch` onto Megatron-LM. Instead of hand-writing Transformer components, learners use Megatron for model implementation, distributed training, checkpointing, and mixed precision. The focus is the end-to-end pipeline: data → preprocess → pretrain → SFT → evaluate → post-train (GRPO/distillation).

The workspace layout matters — sibling repos are required:

```
/sgl-workspace/zkx/train/
├── LLMs-from-scratch          # source datasets (the-verdict.txt, instruction-data.json, etc.)
├── reasoning-from-scratch     # source datasets (math500_test.json, etc.)
├── Megatron-LM                # training framework — scripts cd here to run pretrain_gpt.py
└── megatron-from-scratch-server-course   # this repo
```

## Key Commands

```bash
# Environment healthcheck (must show transformer_engine_torch_ok True, megatron_have_te True)
bash scripts/00_healthcheck.sh

# Generate tutorial JSONL from source repos
python scripts/10_prepare_from_scratch_datasets.py \
  --train-root /sgl-workspace/zkx/train --output-dir data/from_scratch

# Preprocess JSONL → Megatron .bin/.idx
bash scripts/11_preprocess_gpt_data.sh <input.jsonl> <output_prefix>

# 4-GPU pretraining (requires .bin/.idx DATA_PREFIX)
bash scripts/21_run_pretrain_real_4gpu.sh <DATA_PREFIX>

# Minimal smoke test (skip validation/test splits on tiny corpus)
DATA_SPLIT=100,0,0 EVAL_ITERS=0 bash scripts/21_run_pretrain_real_4gpu.sh <DATA_PREFIX>

# 4-GPU SFT
bash scripts/22_run_sft_4gpu.sh <sft_messages.jsonl> [tokenizer_model_dir]

# Static course validation
python scripts/90_validate_course.py
```

## Architecture & Script Numbering

Scripts follow a numbered convention indicating pipeline stage:
- `00_*` — environment checks
- `1x_*` — data preparation and preprocessing
- `2x_*` — training (pretrain, SFT, GRPO)
- `3x_*` — log parsing
- `4x_*` — checkpoint inspection
- `5x_*` — evaluation bridges
- `9x_*` — course validation

All training scripts source two files in order:
1. `configs/4gpu_edu_pretrain.env` — model hyperparams, parallelism defaults, tokenizer paths
2. `scripts/common_env.sh` — sets `COURSE_ROOT`, `MEGATRON_ROOT`, `PYTHONPATH`, cache directories

`common_env.sh` redirects compile caches (Torch extensions, Triton, FlashInfer) to `/tmp/megatron-server-course-cache-*` to avoid NFS stale-handle issues, and sets `PYTHONPATH` to include Megatron-LM.

## Configuration

Two env presets in `configs/`:
- `4gpu_edu_pretrain.env` — default teaching config: 12-layer 768-hidden GPT, seq_length=512, 20 iters, GPT2BPETokenizer using encoder.json/vocab.bpe from LLMs-from-scratch
- `4gpu_b200_llama_fp8.env` — performance preset: 32-layer 4096-hidden Llama-style with GQA, TP=2/CP=2, fp8, seq_length=4096

All env vars use `${VAR:-default}` pattern — override any variable from shell before sourcing.

## Data Flow

1. **Source data** lives in sibling repos (never committed here)
2. `scripts/10_prepare_from_scratch_datasets.py` → `data/from_scratch/*.jsonl`
3. For pretraining: `scripts/11_preprocess_gpt_data.sh` → `runs/data/*_text_document.{bin,idx}`
4. For SFT: JSONL with `messages` key fed directly to `--data-path`, using `--tokenizer-type SFTTokenizer`
5. Tokenizer fallback: set `TOKENIZER_MODE=offline_token_ids` for offline BPE encoding when HF is unavailable

## Runtime Artifacts

Everything under `runs/` is gitignored: logs, checkpoints (`torch_dist` format), TensorBoard files, preprocessed data, caches. Training scripts auto-create subdirectories there.

## Docs

13 sequential Chinese-language tutorials (English technical terms) in `docs/00_course_map.md` through `docs/12_operations.md`. The course map (`docs/00_course_map.md`) contains the complete chapter-by-chapter mapping from both source courses to this one.

## Important Constraints

- Training scripts `cd` into `$MEGATRON_ROOT` before running `torchrun` — paths must be absolute or resolved before the `cd`.
- Default pretraining must use `GPT2BPETokenizer` (not mock data, not NullTokenizer). JIT/fusion must not be disabled.
- Smoke tests on tiny corpora (e.g., the-verdict.txt) should use `DATA_SPLIT=100,0,0 EVAL_ITERS=0` to avoid split issues.
- `scripts/90_validate_course.py` enforces structural invariants: required files, dataset keys, tokenizer choices, and no disabled JIT in training scripts.
