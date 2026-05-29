#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 PREDICTIONS_JSON" >&2
  echo "This bridge calls reasoning-from-scratch verifier scripts. The predictions file format must match that script." >&2
  exit 2
fi

PREDICTIONS_JSON=$1
TRAIN_ROOT=${TRAIN_ROOT:-/sgl-workspace/zkx/train}
EVAL_DIR="$TRAIN_ROOT/reasoning-from-scratch/ch03/02_math500-verifier-scripts"

cd "$EVAL_DIR"
python evaluate_json.py "$PREDICTIONS_JSON"

