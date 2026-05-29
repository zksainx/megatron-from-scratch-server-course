#!/usr/bin/env python3
"""Create Megatron-friendly datasets from the two from-scratch repositories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows):5d} rows -> {path}")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def chunk_text(text: str, chunk_chars: int) -> list[str]:
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}".strip() if current else para
        if len(candidate) <= chunk_chars:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = para
    if current:
        chunks.append(current)
    return chunks


def instruction_to_messages(item: dict[str, Any], response_key: str = "output") -> dict[str, Any]:
    user = item["instruction"].strip()
    if item.get("input"):
        user = f"{user}\n\nInput:\n{item['input'].strip()}"
    return {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user},
            {"role": "assistant", "content": str(item[response_key]).strip()},
        ]
    }


def math_problem_to_messages(item: dict[str, Any], response_key: str) -> dict[str, Any]:
    return {
        "messages": [
            {
                "role": "system",
                "content": "Solve the math problem. Show concise reasoning and put the final answer in \\boxed{}.",
            },
            {"role": "user", "content": item["problem"].strip()},
            {"role": "assistant", "content": str(item[response_key]).strip()},
        ]
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-root", type=Path, default=Path("/sgl-workspace/zkx/train"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/from_scratch"))
    parser.add_argument("--chunk-chars", type=int, default=1800)
    args = parser.parse_args()

    train_root = args.train_root.resolve()
    output_dir = args.output_dir.resolve()

    verdict_path = train_root / "LLMs-from-scratch/ch02/01_main-chapter-code/the-verdict.txt"
    instruction_path = train_root / "LLMs-from-scratch/ch07/01_main-chapter-code/instruction-data.json"
    instruction_response_path = train_root / "LLMs-from-scratch/ch07/01_main-chapter-code/instruction-data-with-response.json"
    preference_path = train_root / "LLMs-from-scratch/ch07/04_preference-tuning-with-dpo/instruction-data-with-preference.json"
    math500_path = train_root / "reasoning-from-scratch/ch03/01_main-chapter-code/math500_test.json"
    math_train_path = train_root / "reasoning-from-scratch/ch08/02_generate_distillation_data/math_train_sample.json"
    ollama_outputs_path = train_root / "reasoning-from-scratch/ch08/02_generate_distillation_data/sample_ollama_outputs.json"

    required = [
        verdict_path,
        instruction_path,
        instruction_response_path,
        preference_path,
        math500_path,
        math_train_path,
        ollama_outputs_path,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("missing tutorial dataset files:\n" + "\n".join(missing))

    verdict_text = verdict_path.read_text(encoding="utf-8")
    verdict_rows = [{"text": chunk} for chunk in chunk_text(verdict_text, args.chunk_chars)]
    write_jsonl(output_dir / "pretrain_the_verdict.jsonl", verdict_rows)

    instruction_data = load_json(instruction_path)
    sft_rows = [instruction_to_messages(item) for item in instruction_data]
    write_jsonl(output_dir / "sft_instruction_data.jsonl", sft_rows)

    instruction_response_data = load_json(instruction_response_path)
    eval_rows = [
        {
            "instruction": item["instruction"],
            "input": item.get("input", ""),
            "reference": item["output"],
            "model_response": item.get("model_response", ""),
        }
        for item in instruction_response_data
    ]
    write_jsonl(output_dir / "eval_instruction_responses.jsonl", eval_rows)

    preference_data = load_json(preference_path)
    preference_rows = [
        {
            "prompt": (item["instruction"] + ("\n\nInput:\n" + item["input"] if item.get("input") else "")).strip(),
            "chosen": item["chosen"],
            "rejected": item["rejected"],
        }
        for item in preference_data
    ]
    write_jsonl(output_dir / "preference_dpo_format.jsonl", preference_rows)

    math500 = load_json(math500_path)
    math500_rows = [
        {
            "problem": item["problem"],
            "answer": item["answer"],
            "solution": item.get("solution", ""),
            "subject": item.get("subject", ""),
            "level": item.get("level", ""),
            "unique_id": item.get("unique_id", ""),
        }
        for item in math500
    ]
    write_jsonl(output_dir / "eval_math500.jsonl", math500_rows)

    math_train = load_json(math_train_path)
    math_train_rows = [
        {
            "messages": [
                {
                    "role": "system",
                    "content": "Solve the math problem. Put the final answer in \\boxed{}.",
                },
                {"role": "user", "content": item["problem"].strip()},
                {"role": "assistant", "content": f"\\boxed{{{item['answer']}}}"},
            ]
        }
        for item in math_train
    ]
    write_jsonl(output_dir / "sft_reasoning_math_train_sample.jsonl", math_train_rows)

    ollama_outputs = load_json(ollama_outputs_path)
    distill_rows = []
    for item in ollama_outputs:
        response = item.get("message_content") or item.get("message_thinking") or ""
        if item.get("message_content") and item.get("message_thinking"):
            response = f"{item['message_thinking'].strip()}\n\n{item['message_content'].strip()}"
        distill_rows.append(math_problem_to_messages({"problem": item["problem"], "response": response}, "response"))
    write_jsonl(output_dir / "sft_reasoning_distillation_sample.jsonl", distill_rows)


if __name__ == "__main__":
    main()

