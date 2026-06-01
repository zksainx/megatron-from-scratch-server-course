#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
COURSE_ROOT=$(cd -- "$SCRIPT_DIR/.." && pwd)
source "$COURSE_ROOT/configs/4gpu_edu_pretrain.env"
source "$COURSE_ROOT/scripts/common_env.sh"

cat <<'MSG'
This is a GRPO/RLVR command skeleton, not a one-click smoke test.
You must provide a converted Megatron checkpoint, matching tokenizer, and an RL environment config.

Required environment variables:
  PRETRAINED_CHECKPOINT=/path/to/megatron/checkpoint
  TOKENIZER_MODEL=/path/to/hf/tokenizer
  PROMPT_DATA=/path/to/prompts.jsonl
  ENV_CONFIG=/sgl-workspace/zkx/train/Megatron-LM/examples/rl/environment_configs/math.yaml

The local Megatron entry point is train_rl.py.
MSG

: "${PRETRAINED_CHECKPOINT:?set PRETRAINED_CHECKPOINT}"
: "${TOKENIZER_MODEL:?set TOKENIZER_MODEL}"
: "${PROMPT_DATA:?set PROMPT_DATA}"
: "${ENV_CONFIG:?set ENV_CONFIG}"

RUN_NAME=${RUN_NAME:-grpo_math_4gpu}
mkdir -p "$RUN_ROOT/logs" "$RUN_ROOT/checkpoints/$RUN_NAME" "$RUN_ROOT/tensorboard/$RUN_NAME" "$RUN_ROOT/cache/$RUN_NAME"

cd "$MEGATRON_ROOT"

torchrun \
  --nproc_per_node "$GPUS_PER_NODE" \
  --nnodes "$NUM_NODES" \
  --node_rank "$NODE_RANK" \
  --master_addr "$MASTER_ADDR" \
  --master_port "${MASTER_PORT:-6020}" \
  train_rl.py \
  --perform-rl-step \
  --data-path "$PROMPT_DATA" \
  --pretrained-checkpoint "$PRETRAINED_CHECKPOINT" \
  --tokenizer-type HuggingFaceTokenizer \
  --tokenizer-model "$TOKENIZER_MODEL" \
  --langrl-env-config "$ENV_CONFIG" \
  --num-layers "$NUM_LAYERS" \
  --hidden-size "$HIDDEN_SIZE" \
  --ffn-hidden-size "$FFN_HIDDEN_SIZE" \
  --num-attention-heads "$NUM_ATTENTION_HEADS" \
  --seq-length "$SEQ_LENGTH" \
  --inference-max-seq-length "$SEQ_LENGTH" \
  --max-position-embeddings "$SEQ_LENGTH" \
  --attention-backend flash \
  --micro-batch-size 1 \
  --global-batch-size 16 \
  --grpo-group-size 4 \
  --grpo-prompts-per-step 4 \
  --grpo-iterations 1 \
  --grpo-kl-beta "${GRPO_KL_BETA:-0.01}" \
  --train-samples 64 \
  --lr 1.0e-6 \
  --bf16 \
  --save "$RUN_ROOT/checkpoints/$RUN_NAME" \
  --load "$RUN_ROOT/checkpoints/$RUN_NAME" \
  --tensorboard-dir "$RUN_ROOT/tensorboard/$RUN_NAME" \
  --data-cache-path "$RUN_ROOT/cache/$RUN_NAME" \
  --log-interval 1 \
  --save-interval 20 \
  --eval-interval 20 \
  2>&1 | tee "$RUN_ROOT/logs/$RUN_NAME.log"
