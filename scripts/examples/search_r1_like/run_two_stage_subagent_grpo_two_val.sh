#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="$SCRIPT_DIR/run_two_stage_subagent_grpo.sh"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

if [ ! -f "$BASE_SCRIPT" ]; then
  echo "Base script not found: $BASE_SCRIPT"
  exit 1
fi

ROOT_DIR="${ROOT_DIR:-/ai/cqn}"
PROJECT_NAME="${PROJECT_NAME:-search_subagent_grpo}"
BASE_EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_0p6b_grpo_chain_verify_val2samples}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${BASE_EXPERIMENT_NAME}_${RUN_TS}"

# Quick validation mode defaults
VAL_ONLY="${VAL_ONLY:-true}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-80}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"

OUT_DIR="$ROOT_DIR/ckpt/$PROJECT_NAME/$EXPERIMENT_NAME"

echo "============================================"
echo "Quick validation check (first 80 samples)"
echo "EXPERIMENT_NAME: $EXPERIMENT_NAME"
echo "VAL_MAX_SAMPLES: $VAL_MAX_SAMPLES"
echo "VAL_BATCH_SIZE: $VAL_BATCH_SIZE"
echo "OUTPUT_DIR: $OUT_DIR"
echo "============================================"

VAL_ONLY="$VAL_ONLY" \
VAL_BEFORE_TRAIN="$VAL_BEFORE_TRAIN" \
VAL_BATCH_SIZE="$VAL_BATCH_SIZE" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
ROLLOUT_DATA_DIR="$OUT_DIR/rollout_data" \
VALIDATION_DATA_DIR="$OUT_DIR/validation_data" \
IO_TRACE_LOG_PATH="$PROJECT_DIR/tmp_logs/verl_io_trace_${EXPERIMENT_NAME}.jsonl" \
TRAIN_LOG="/root/shared_planing/tmp/verl_train_${EXPERIMENT_NAME}.log" \
bash "$BASE_SCRIPT" \
  data.val_max_samples="$VAL_MAX_SAMPLES"

echo "Done. Check:"
echo "  $OUT_DIR/validation_data"
echo "  $PROJECT_DIR/tmp_logs/verl_io_trace_${EXPERIMENT_NAME}.jsonl"
