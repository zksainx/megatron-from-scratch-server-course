#!/usr/bin/env python3
"""Static validation for the server course artifacts."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path


COURSE_ROOT = Path(__file__).resolve().parents[1]
TRAIN_ROOT = Path(os.environ.get("TRAIN_ROOT", "/sgl-workspace/zkx/train"))

REQUIRED_FILES = [
    "README.md",
    "docs/00_course_map.md",
    "docs/01_environment.md",
    "docs/02_data_pipeline.md",
    "docs/03_tokenizer_and_preprocessing.md",
    "docs/04_model_and_parallelism.md",
    "docs/05_pretraining.md",
    "docs/06_loss_logging_checkpoint.md",
    "docs/07_benchmarking.md",
    "docs/08_finetuning_and_sft.md",
    "docs/09_evaluation.md",
    "docs/10_reasoning_workflow.md",
    "docs/11_distillation_and_export.md",
    "docs/12_operations.md",
    "scripts/10_prepare_from_scratch_datasets.py",
    "scripts/13_encode_pretrain_with_from_scratch_bpe.py",
    "scripts/14_prepare_local_hf_gpt2_tokenizer.py",
    "scripts/11_preprocess_gpt_data.sh",
    "scripts/21_run_pretrain_real_4gpu.sh",
    "scripts/22_run_sft_4gpu.sh",
]

SOURCE_DATASETS = [
    "LLMs-from-scratch/ch02/01_main-chapter-code/the-verdict.txt",
    "LLMs-from-scratch/ch07/01_main-chapter-code/instruction-data.json",
    "LLMs-from-scratch/ch07/01_main-chapter-code/instruction-data-with-response.json",
    "LLMs-from-scratch/ch07/04_preference-tuning-with-dpo/instruction-data-with-preference.json",
    "reasoning-from-scratch/ch03/01_main-chapter-code/math500_test.json",
    "reasoning-from-scratch/ch08/02_generate_distillation_data/math_train_sample.json",
    "reasoning-from-scratch/ch08/02_generate_distillation_data/sample_ollama_outputs.json",
]

TERMS = [
    "tokenization",
    "pretraining",
    "SFT",
    "checkpoint",
    "benchmark",
    "MATH-500",
    "MMLU",
    "GRPO",
    "distillation",
    "inference-time scaling",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_jsonl(path: Path, key: str) -> None:
    require(path.exists(), f"missing generated jsonl: {path}")
    with path.open("r", encoding="utf-8") as f:
        first = f.readline()
    require(bool(first), f"empty jsonl: {path}")
    obj = json.loads(first)
    require(key in obj, f"{path} first row missing key {key!r}")


def main() -> None:
    for rel in REQUIRED_FILES:
        require((COURSE_ROOT / rel).exists(), f"missing course file: {rel}")

    for rel in SOURCE_DATASETS:
        require((TRAIN_ROOT / rel).exists(), f"missing source dataset: {rel}")

    all_docs = "\n".join(p.read_text(encoding="utf-8") for p in (COURSE_ROOT / "docs").glob("*.md"))
    for term in TERMS:
        require(term in all_docs, f"coverage term not found in docs: {term}")

    preprocess_script = (COURSE_ROOT / "scripts/11_preprocess_gpt_data.sh").read_text(encoding="utf-8")
    real_train_script = (COURSE_ROOT / "scripts/21_run_pretrain_real_4gpu.sh").read_text(encoding="utf-8")
    require("TOKENIZER_MODE=${TOKENIZER_MODE:-megatron_gpt2}" in preprocess_script, "preprocess default must be Megatron native GPT-2")
    require("--tokenizer-type GPT2BPETokenizer" in real_train_script, "real training must use Megatron GPT2BPETokenizer by default")
    require("--disable-jit-fuser" not in real_train_script, "real training must not disable JIT/fusion by default")
    require("--mock-data" not in real_train_script, "real training must not use mock data")
    require("DATA_SPLIT=${DATA_SPLIT:-90,9,1}" in real_train_script, "real training must allow split override")

    generated_dir = COURSE_ROOT / "data/from_scratch"
    if not generated_dir.exists():
        subprocess.run(
            [
                "python",
                str(COURSE_ROOT / "scripts/10_prepare_from_scratch_datasets.py"),
                "--train-root",
                str(TRAIN_ROOT),
                "--output-dir",
                str(generated_dir),
            ],
            check=True,
        )

    validate_jsonl(generated_dir / "pretrain_the_verdict.jsonl", "text")
    validate_jsonl(generated_dir / "sft_instruction_data.jsonl", "messages")
    validate_jsonl(generated_dir / "preference_dpo_format.jsonl", "prompt")
    validate_jsonl(generated_dir / "eval_math500.jsonl", "problem")
    validate_jsonl(generated_dir / "sft_reasoning_distillation_sample.jsonl", "messages")
    validate_jsonl(generated_dir / "sft_reasoning_math_train_sample.jsonl", "messages")

    print("course_validation_ok True")


if __name__ == "__main__":
    main()
