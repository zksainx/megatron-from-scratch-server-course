#!/usr/bin/env python3
"""Encode text JSONL with the GPT-2 BPE implementation shipped in LLMs-from-scratch.

The output keeps Megatron's JSONL shape but stores token ids as a space-separated
string so Megatron's NullTokenizer can preprocess it fully offline.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any


def load_bpe_encoder(train_root: Path) -> Any:
    bpe_path = train_root / "LLMs-from-scratch/ch02/02_bonus_bytepair-encoder/bpe_openai_gpt2.py"
    spec = importlib.util.spec_from_file_location("from_scratch_bpe_openai_gpt2", bpe_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load BPE module: {bpe_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model_dir = train_root / "LLMs-from-scratch/ch02/02_bonus_bytepair-encoder"
    return module.get_encoder("gpt2_model", str(model_dir))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, default=Path("/sgl-workspace/zkx/train"))
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--eod-id", type=int, default=50256)
    args = parser.parse_args()

    encoder = load_bpe_encoder(args.train_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    tokens = 0
    with args.input.open("r", encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            obj = json.loads(line)
            ids = encoder.encode(obj["text"]) + [args.eod_id]
            tokens += len(ids)
            dst.write(json.dumps({"text": " ".join(str(x) for x in ids)}, ensure_ascii=False) + "\n")
            rows += 1

    print(f"wrote {rows} rows, {tokens} token ids -> {args.output}")


if __name__ == "__main__":
    main()

