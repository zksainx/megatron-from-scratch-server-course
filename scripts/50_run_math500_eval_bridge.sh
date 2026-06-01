#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 PREDICTIONS_JSON" >&2
  echo "This bridge calls reasoning-from-scratch verifier scripts. The predictions file format must match that script." >&2
  exit 2
fi

PREDICTIONS_JSON=$1

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
COURSE_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$COURSE_ROOT/scripts/common_env.sh"

test -f "$PREDICTIONS_JSON" || { echo "error: predictions file not found: $PREDICTIONS_JSON" >&2; exit 1; }

EVAL_DIR="$TRAIN_ROOT/reasoning-from-scratch/ch03/02_math500-verifier-scripts"

cd "$EVAL_DIR"
python evaluate_json.py "$PREDICTIONS_JSON"

