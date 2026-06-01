#!/usr/bin/env python3
"""Extract common metrics from a Megatron training log."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


PATTERNS = {
    "lm_loss": re.compile(r"lm loss[:=]\s*([0-9.eE+-]+)"),
    "lr": re.compile(r"learning rate[:=]\s*([0-9.eE+-]+)"),
    "throughput_tflops": re.compile(r"throughput per GPU.*?([0-9.eE+-]+)"),
    "tokens_per_sec": re.compile(r"(?:tokens/sec|tokens per second).*?([0-9.eE+-]+)", re.IGNORECASE),
    "grad_norm": re.compile(r"grad norm[:=]\s*([0-9.eE+-]+)"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=Path)
    args = parser.parse_args()

    text = args.log.read_text(encoding="utf-8", errors="replace")
    print(f"log: {args.log}")
    for name, pattern in PATTERNS.items():
        values = [float(m.group(1)) for m in pattern.finditer(text)]
        if values:
            print(f"{name}: count={len(values)} first={values[0]:.6g} last={values[-1]:.6g} min={min(values):.6g} max={max(values):.6g}")
        else:
            print(f"{name}: not_found")


if __name__ == "__main__":
    main()

