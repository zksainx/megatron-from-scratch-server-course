#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 CHECKPOINT_DIR" >&2
  exit 2
fi

CKPT_DIR=$1
test -d "$CKPT_DIR"

echo "[checkpoint] $CKPT_DIR"
find "$CKPT_DIR" -maxdepth 2 -type f | sort | sed -n '1,120p'

if [[ -f "$CKPT_DIR/latest_checkpointed_iteration.txt" ]]; then
  echo "[latest] $(cat "$CKPT_DIR/latest_checkpointed_iteration.txt")"
fi

