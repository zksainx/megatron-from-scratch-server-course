#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 DATA_PREFIX" >&2
  echo "example: $0 runs/data/the_verdict_text_document" >&2
  exit 2
fi

DATA_PREFIX=$1

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
COURSE_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$COURSE_ROOT/configs/4gpu_edu_pretrain.env"
source "$COURSE_ROOT/scripts/common_env.sh"

if [[ "$DATA_PREFIX" != /* ]]; then
  DATA_PREFIX="$COURSE_ROOT/$DATA_PREFIX"
fi

LOG_DIR="$RUN_ROOT/logs"
CKPT_DIR="$RUN_ROOT/checkpoints/pretrain_real"
TB_DIR="$RUN_ROOT/tensorboard/pretrain_real"
CACHE_DIR="$RUN_ROOT/cache/pretrain_real"
mkdir -p "$LOG_DIR" "$CKPT_DIR" "$TB_DIR" "$CACHE_DIR"
DATA_SPLIT=${DATA_SPLIT:-90,9,1}

test -f "${DATA_PREFIX}.bin"
test -f "${DATA_PREFIX}.idx"

cd "$MEGATRON_ROOT"

torchrun \
  --nproc_per_node "$GPUS_PER_NODE" \
  --nnodes "$NUM_NODES" \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  pretrain_gpt.py \
  --num-layers "$NUM_LAYERS" \
  --hidden-size "$HIDDEN_SIZE" \
  --ffn-hidden-size "$FFN_HIDDEN_SIZE" \
  --num-attention-heads "$NUM_ATTENTION_HEADS" \
  --seq-length "$SEQ_LENGTH" \
  --max-position-embeddings "$SEQ_LENGTH" \
  --attention-backend auto \
  --micro-batch-size "$MICRO_BATCH_SIZE" \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --train-iters "$TRAIN_ITERS" \
  --lr "$LR" \
  --min-lr "$MIN_LR" \
  --lr-decay-style cosine \
  --lr-warmup-fraction 0.01 \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --clip-grad 1.0 \
  --bf16 \
  --data-path "$DATA_PREFIX" \
  --tokenizer-type GPT2BPETokenizer \
  --vocab-file "$GPT2_VOCAB_FILE" \
  --merge-file "$GPT2_MERGE_FILE" \
  --data-cache-path "$CACHE_DIR" \
  --split "$DATA_SPLIT" \
  --eval-iters "$EVAL_ITERS" \
  --eval-interval "$EVAL_INTERVAL" \
  --save-interval "$SAVE_INTERVAL" \
  --log-interval 1 \
  --log-throughput \
  --log-params-norm \
  --log-num-zeros-in-grad \
  --ckpt-format torch_dist \
  --save "$CKPT_DIR" \
  --load "$CKPT_DIR" \
  --tensorboard-dir "$TB_DIR" \
  2>&1 | tee "$LOG_DIR/pretrain_real.log"
