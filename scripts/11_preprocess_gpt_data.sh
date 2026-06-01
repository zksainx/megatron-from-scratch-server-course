#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 INPUT_JSONL OUTPUT_PREFIX [WORKERS]" >&2
  exit 2
fi

INPUT_JSONL=$1
OUTPUT_PREFIX=$2
WORKERS=${3:-8}
TOKENIZER_MODE=${TOKENIZER_MODE:-megatron_gpt2}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
COURSE_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$COURSE_ROOT/configs/4gpu_edu_pretrain.env"
source "$COURSE_ROOT/scripts/common_env.sh"

mkdir -p "$(dirname "$OUTPUT_PREFIX")"
INPUT_JSONL=$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$INPUT_JSONL")
OUTPUT_PREFIX=$(python -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUTPUT_PREFIX")

if [[ "$TOKENIZER_MODE" == "offline_token_ids" ]]; then
  ENCODED_JSONL="${OUTPUT_PREFIX}.token_ids.jsonl"
  python "$COURSE_ROOT/scripts/13_encode_pretrain_with_from_scratch_bpe.py" \
    --train-root "$TRAIN_ROOT" \
    --input "$INPUT_JSONL" \
    --output "$ENCODED_JSONL"
  INPUT_FOR_MEGATRON="$ENCODED_JSONL"
  TOKENIZER_ARGS=(--tokenizer-type NullTokenizer --vocab-size 50257)
else
  INPUT_FOR_MEGATRON="$INPUT_JSONL"
  TOKENIZER_ARGS=(--tokenizer-type GPT2BPETokenizer --vocab-file "$GPT2_VOCAB_FILE" --merge-file "$GPT2_MERGE_FILE" --append-eod)
fi

cd "$MEGATRON_ROOT"
python tools/preprocess_data.py \
  --input "$INPUT_FOR_MEGATRON" \
  --output-prefix "$OUTPUT_PREFIX" \
  --json-keys text \
  "${TOKENIZER_ARGS[@]}" \
  --workers "$WORKERS" \
  --log-interval 10

echo "[preprocess] data prefix for training: ${OUTPUT_PREFIX}_text_document"
