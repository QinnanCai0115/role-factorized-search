#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
BASE_SCRIPT="$SCRIPT_DIR/run_two_stage_subagent_grpo.sh"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "/ai/cqn/miniconda3/envs/verl/bin/python" ]; then
    PYTHON_BIN="/ai/cqn/miniconda3/envs/verl/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

SFT_STEP="${SFT_STEP:-250}"
SFT_RUN_DIR="${SFT_RUN_DIR:-/ai/cqn/s3/ckpt/search_subagent_policy_sft/qwen3_1p7b_policy_sft_20260429_200022}"
SFT_LOAD_MODE="${SFT_LOAD_MODE:-merged_hf}" # merged_hf | resume
SFT_MERGED_MODEL_DIR="${SFT_MERGED_MODEL_DIR:-$SFT_RUN_DIR/merged_hf_global_step_$SFT_STEP}"
SFT_RESUME_PATH="${SFT_RESUME_PATH:-$SFT_RUN_DIR/verl_resume_layout/global_step_$SFT_STEP}"
SFT_TOKENIZER_DIR="${SFT_TOKENIZER_DIR:-$SFT_RUN_DIR/global_step_$SFT_STEP/huggingface}"
BASE_ACTOR_MODEL_PATH="${BASE_ACTOR_MODEL_PATH:-/ai/cqn/model/Qwen3-1.7B}"

VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-900}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
VALIDATION_SHUFFLE="${VALIDATION_SHUFFLE:-false}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="${ROOT_DIR:-/ai/cqn/s3}"
PROJECT_NAME="${PROJECT_NAME:-search_subagent_policy_sft_val}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_1p7b_policy_sft_step${SFT_STEP}_${SFT_LOAD_MODE}_val${VAL_MAX_SAMPLES}_${RUN_TS}}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/ckpt/$PROJECT_NAME/$EXPERIMENT_NAME}"

# Ray creates AF_UNIX sockets under _temp_dir/session_*/sockets, and Linux
# limits that path to 107 bytes. Keep this directory intentionally short.
TMP_RUN_DIR="${TMP_RUN_DIR:-/tmp/sftval_${SFT_STEP}_${RUN_TS}}"

case "$SFT_LOAD_MODE" in
  merged_hf)
    if [ ! -d "$SFT_MERGED_MODEL_DIR" ]; then
      echo "Merged HF SFT model not found: $SFT_MERGED_MODEL_DIR" >&2
      exit 1
    fi
    ACTOR_MODEL_PATH_TO_USE="$SFT_MERGED_MODEL_DIR"
    TOKENIZER_PATH_TO_USE="$SFT_MERGED_MODEL_DIR"
    RESUME_MODE_TO_USE="disable"
    RESUME_FROM_PATH_TO_USE=""
    ;;
  resume)
    if [ ! -d "$SFT_RESUME_PATH" ]; then
      echo "SFT resume path not found: $SFT_RESUME_PATH" >&2
      exit 1
    fi
    if [ ! -d "$SFT_TOKENIZER_DIR" ]; then
      echo "SFT tokenizer path not found: $SFT_TOKENIZER_DIR" >&2
      exit 1
    fi
    if [ ! -d "$BASE_ACTOR_MODEL_PATH" ]; then
      echo "Base actor model not found: $BASE_ACTOR_MODEL_PATH" >&2
      exit 1
    fi
    ACTOR_MODEL_PATH_TO_USE="$BASE_ACTOR_MODEL_PATH"
    TOKENIZER_PATH_TO_USE="$SFT_TOKENIZER_DIR"
    RESUME_MODE_TO_USE="resume_path"
    RESUME_FROM_PATH_TO_USE="$SFT_RESUME_PATH"
    ;;
  *)
    echo "Invalid SFT_LOAD_MODE=$SFT_LOAD_MODE (expected merged_hf|resume)" >&2
    exit 1
    ;;
esac

echo "============================================"
echo "Policy SFT val-only evaluation"
echo "SFT run dir:       $SFT_RUN_DIR"
echo "SFT step:          $SFT_STEP"
echo "Load mode:         $SFT_LOAD_MODE"
echo "Actor model:       $ACTOR_MODEL_PATH_TO_USE"
echo "Tokenizer:         $TOKENIZER_PATH_TO_USE"
echo "Resume mode:       $RESUME_MODE_TO_USE"
echo "Resume path:       ${RESUME_FROM_PATH_TO_USE:-<none>}"
echo "Val max samples:   $VAL_MAX_SAMPLES"
echo "Experiment:        $EXPERIMENT_NAME"
echo "Output dir:        $OUT_DIR"
echo "Tmp dir:           $TMP_RUN_DIR"
echo "============================================"

env \
  ROOT_DIR="$ROOT_DIR" \
  PROJECT_NAME="$PROJECT_NAME" \
  EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  RUN_TS="$RUN_TS" \
  VAL_ONLY=true \
  VAL_BEFORE_TRAIN=true \
  VAL_MAX_SAMPLES="$VAL_MAX_SAMPLES" \
  VAL_BATCH_SIZE="$VAL_BATCH_SIZE" \
  VALIDATION_SHUFFLE="$VALIDATION_SHUFFLE" \
  TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}" \
  POLICY_USE_API=false \
  ACTOR_MODEL_PATH="$ACTOR_MODEL_PATH_TO_USE" \
  TOKENIZER_PATH="$TOKENIZER_PATH_TO_USE" \
  RESUME_MODE="$RESUME_MODE_TO_USE" \
  RESUME_FROM_PATH="$RESUME_FROM_PATH_TO_USE" \
  N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}" \
  CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}" \
  ROLLOUT_DATA_DIR="$OUT_DIR/rollout_data" \
  VALIDATION_DATA_DIR="$OUT_DIR/validation_data" \
  IO_TRACE_LOG_PATH="$OUT_DIR/io_trace.jsonl" \
  IO_TRACE_MAX_CHARS="${IO_TRACE_MAX_CHARS:-0}" \
  IO_TRACE_MAX_ITEMS="${IO_TRACE_MAX_ITEMS:-0}" \
  IO_TRACE_MAX_SAMPLES="${IO_TRACE_MAX_SAMPLES:-900}" \
  TMPDIR="${TMPDIR:-$TMP_RUN_DIR/tmp}" \
  HF_HOME="${HF_HOME:-$TMP_RUN_DIR/hf_cache}" \
  HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$TMP_RUN_DIR/hf_datasets}" \
  RAY_TMPDIR="${RAY_TMPDIR:-$TMP_RUN_DIR/ray}" \
  RAY_PLASMA_DIRECTORY="${RAY_PLASMA_DIRECTORY:-$TMP_RUN_DIR/ray_plasma}" \
  TRAIN_LOG="$OUT_DIR/train.log" \
  bash "$BASE_SCRIPT" \
    actor_rollout_ref.model.lora_rank=0 \
    actor_rollout_ref.actor.strategy="${ACTOR_STRATEGY:-fsdp2}" \
    actor_rollout_ref.actor.fsdp_config.fsdp_size="${ACTOR_FSDP_SIZE:-1}" \
    'actor_rollout_ref.actor.checkpoint.load_contents=[model]' \
    +actor_rollout_ref.rollout.custom.backbone_judge_max_chars=0 \
    "$@"

LATEST_VAL_JSONL="$(find "$OUT_DIR/validation_data" -maxdepth 1 -type f -name '*.jsonl' | sort | tail -n 1 || true)"
if [ -n "$LATEST_VAL_JSONL" ] && [ -f "$LATEST_VAL_JSONL" ]; then
  ROUND_TRACE_JSONL="$OUT_DIR/validation_data/round_traces_$(basename "$LATEST_VAL_JSONL")"
  "$PYTHON_BIN" "$SCRIPT_DIR/extract_round_traces.py" \
    --input "$LATEST_VAL_JSONL" \
    --output "$ROUND_TRACE_JSONL" || true
  echo "Validation JSONL: $LATEST_VAL_JSONL"
  echo "Round traces:     $ROUND_TRACE_JSONL"
fi

echo "Done. Check:"
echo "  $OUT_DIR/validation_data"
echo "  $OUT_DIR/io_trace.jsonl"
echo "  $OUT_DIR/train.log"
