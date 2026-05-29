#!/usr/bin/env python3
"""Build a local Hugging Face GPT-2 tokenizer directory from tutorial BPE files."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, default=Path("/sgl-workspace/zkx/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/tokenizers/gpt2_from_scratch"))
    args = parser.parse_args()

    source_dir = args.train_root / "LLMs-from-scratch/ch02/02_bonus_bytepair-encoder/gpt2_model"
    vocab_src = source_dir / "encoder.json"
    merges_src = source_dir / "vocab.bpe"
    if not vocab_src.exists() or not merges_src.exists():
        raise FileNotFoundError(f"missing GPT-2 BPE files under {source_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(vocab_src, args.output_dir / "vocab.json")
    shutil.copyfile(merges_src, args.output_dir / "merges.txt")

    tokenizer_config = {
        "model_max_length": 1024,
        "tokenizer_class": "GPT2Tokenizer",
        "unk_token": "<|endoftext|>",
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
    }
    special_tokens_map = {
        "bos_token": "<|endoftext|>",
        "eos_token": "<|endoftext|>",
        "unk_token": "<|endoftext|>",
        "pad_token": "<|endoftext|>",
    }
    config = {
        "model_type": "gpt2",
        "vocab_size": 50257,
        "n_positions": 1024,
        "n_ctx": 1024,
        "n_embd": 768,
        "n_layer": 12,
        "n_head": 12,
        "bos_token_id": 50256,
        "eos_token_id": 50256,
    }

    (args.output_dir / "tokenizer_config.json").write_text(json.dumps(tokenizer_config, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "special_tokens_map.json").write_text(json.dumps(special_tokens_map, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(args.output_dir.resolve())


if __name__ == "__main__":
    main()

