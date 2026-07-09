#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
BASE_SCRIPT="$PROJECT_DIR/scripts/examples/search_r1_like/run_two_stage_subagent_grpo.sh"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
PROJECT_NAME="${PROJECT_NAME:-search_subagent_grpo}"

BASE_MODEL_PATH="${BASE_MODEL_PATH:-/ai/cqn/model/Qwen3-1.7B}"
VAL_DATA="${VAL_DATA:-$PROJECT_DIR/data/hotpotqa_2wiki_musique_train/val_mixed_900.parquet}"
STEP50_CKPT="${STEP50_CKPT:-/ai/cqn/s3/checkpoints/search_subagent_grpo/qwen3_1p7B_grpo_20260418_014208/global_step_50}"

COMMON_ENV=(
  "PROJECT_NAME=$PROJECT_NAME"
  "ACTOR_MODEL_PATH=$BASE_MODEL_PATH"
  "TOKENIZER_PATH=$BASE_MODEL_PATH"
  "VAL_DATA=$VAL_DATA"
  "VAL_ONLY=true"
  "VAL_BEFORE_TRAIN=true"
  "VAL_MAX_SAMPLES=-1"
  "VALIDATION_SHUFFLE=false"
  "TOTAL_TRAINING_STEPS=1"
  "ORCHESTRATOR_MAX_ROUNDS=4"
  "VALIDATION_ORCHESTRATOR_MAX_ROUNDS=3"
  "POLICY_ROLLOUT_N=4"
  "VALIDATION_POLICY_ROLLOUT_N=1"
)

echo "[compare] Running base-model validation..."
env \
  "${COMMON_ENV[@]}" \
  "EXPERIMENT_NAME=qwen3_1p7B_val_compare_base_${TIMESTAMP}" \
  "RESUME_MODE=disable" \
  "RESUME_FROM_PATH=" \
  bash "$BASE_SCRIPT"

echo "[compare] Running step-50 checkpoint validation..."
env \
  "${COMMON_ENV[@]}" \
  "EXPERIMENT_NAME=qwen3_1p7B_val_compare_step50_${TIMESTAMP}" \
  "RESUME_MODE=resume_path" \
  "RESUME_FROM_PATH=$STEP50_CKPT" \
  bash "$BASE_SCRIPT"

echo "[compare] Done."
echo "[compare] Base validation data: /ai/cqn/s3/ckpt/$PROJECT_NAME/qwen3_1p7B_val_compare_base_${TIMESTAMP}/validation_data"
echo "[compare] Step-50 validation data: /ai/cqn/s3/ckpt/$PROJECT_NAME/qwen3_1p7B_val_compare_step50_${TIMESTAMP}/validation_data"
