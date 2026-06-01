#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 SFT_MESSAGES_JSONL [TOKENIZER_MODEL]" >&2
  echo "example: $0 data/from_scratch/sft_instruction_data.jsonl /path/to/hf-tokenizer" >&2
  exit 2
fi

SFT_JSONL=$1

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
COURSE_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$COURSE_ROOT/configs/4gpu_edu_pretrain.env"
source "$COURSE_ROOT/scripts/common_env.sh"

TOKENIZER_MODEL=${2:-${TOKENIZER_MODEL:-$COURSE_ROOT/runs/tokenizers/gpt2_from_scratch}}

if [[ "$SFT_JSONL" != /* ]]; then
  SFT_JSONL="$COURSE_ROOT/$SFT_JSONL"
fi

LOG_DIR="$RUN_ROOT/logs"
CKPT_DIR="$RUN_ROOT/checkpoints/sft"
TB_DIR="$RUN_ROOT/tensorboard/sft"
CACHE_DIR="$RUN_ROOT/cache/sft"
mkdir -p "$LOG_DIR" "$CKPT_DIR" "$TB_DIR" "$CACHE_DIR"
test -f "$SFT_JSONL"
if [[ ! -f "$TOKENIZER_MODEL/tokenizer_config.json" ]]; then
  python "$COURSE_ROOT/scripts/14_prepare_local_hf_gpt2_tokenizer.py" \
    --train-root "$TRAIN_ROOT" \
    --output-dir "$TOKENIZER_MODEL" >/dev/null
fi

cd "$MEGATRON_ROOT"

torchrun \
  --nproc_per_node "$GPUS_PER_NODE" \
  --nnodes "$NUM_NODES" \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "$MASTER_PORT" \
  pretrain_gpt.py \
  --sft \
  --num-layers "$NUM_LAYERS" \
  --hidden-size "$HIDDEN_SIZE" \
  --ffn-hidden-size "$FFN_HIDDEN_SIZE" \
  --num-attention-heads "$NUM_ATTENTION_HEADS" \
  --seq-length "$SEQ_LENGTH" \
  --max-position-embeddings "$SEQ_LENGTH" \
  --attention-backend auto \
  --micro-batch-size 1 \
  --global-batch-size "$GLOBAL_BATCH_SIZE" \
  --train-iters "$TRAIN_ITERS" \
  --lr 1.0e-5 \
  --min-lr 1.0e-6 \
  --lr-decay-style cosine \
  --weight-decay 0.1 \
  --adam-beta1 0.9 \
  --adam-beta2 0.95 \
  --clip-grad 1.0 \
  --bf16 \
  --data-path "$SFT_JSONL" \
  --tokenizer-type SFTTokenizer \
  --tokenizer-model "$TOKENIZER_MODEL" \
  --sft-tokenizer-prompt-format identity \
  --no-create-attention-mask-in-dataloader \
  --data-cache-path "$CACHE_DIR" \
  --split "${DATA_SPLIT:-90,9,1}" \
  --eval-iters "$EVAL_ITERS" \
  --eval-interval "$EVAL_INTERVAL" \
  --save-interval "$SAVE_INTERVAL" \
  --log-interval 1 \
  --log-throughput \
  --ckpt-format torch_dist \
  --save "$CKPT_DIR" \
  --load "$CKPT_DIR" \
  --tensorboard-dir "$TB_DIR" \
  2>&1 | tee "$LOG_DIR/sft.log"
