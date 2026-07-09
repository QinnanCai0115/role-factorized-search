#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="$SCRIPT_DIR/run_two_stage_subagent_grpo.sh"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "/ai/cqn/miniconda3/envs/verl/bin/python" ]; then
    PYTHON_BIN="/ai/cqn/miniconda3/envs/verl/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

SFT_STEP="${SFT_STEP:-250}"
SFT_RUN_DIR="${SFT_RUN_DIR:-"/ai/cqn/s3/ckpt/search_subagent_policy_sft/qwen3_1p7b_policy_sft_20260429_200022/"}"
SFT_CKPT="${SFT_CKPT:-$SFT_RUN_DIR/global_step_$SFT_STEP}"
BASE_ACTOR_MODEL_PATH="${BASE_ACTOR_MODEL_PATH:-/ai/cqn/model/Qwen3-1.7B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$SFT_CKPT/huggingface}"

ROOT_DIR="${ROOT_DIR:-/ai/cqn/s3}"
PROJECT_NAME="${PROJECT_NAME:-search_subagent_grpo}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_1p7b_sft_step${SFT_STEP}_val900_${RUN_TS}}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/ckpt/$PROJECT_NAME/$EXPERIMENT_NAME}"
# Ray creates AF_UNIX sockets under _temp_dir/session_*/sockets, and Linux
# limits that path to 107 bytes. Keep this directory intentionally short.
TMP_RUN_DIR="${TMP_RUN_DIR:-/tmp/v250_${RUN_TS}}"

RESUME_ROOT="${RESUME_ROOT:-$PROJECT_DIR/tmp_sft_runs/ppo_resume_layout/sft_step${SFT_STEP}}"
RESUME_STEP_DIR="$RESUME_ROOT/global_step_$SFT_STEP"
ACTOR_LINK="$RESUME_STEP_DIR/actor"

if [ ! -d "$SFT_CKPT" ]; then
  echo "SFT checkpoint not found: $SFT_CKPT" >&2
  exit 1
fi
if [ ! -d "$BASE_ACTOR_MODEL_PATH" ]; then
  echo "Base actor model not found: $BASE_ACTOR_MODEL_PATH" >&2
  exit 1
fi

mkdir -p "$RESUME_STEP_DIR" "$OUT_DIR"
if [ -e "$ACTOR_LINK" ] && [ ! -L "$ACTOR_LINK" ]; then
  echo "Refusing to overwrite non-symlink actor path: $ACTOR_LINK" >&2
  exit 1
fi
if [ ! -L "$ACTOR_LINK" ]; then
  ln -s "$SFT_CKPT" "$ACTOR_LINK"
elif [ "$(readlink "$ACTOR_LINK")" != "$SFT_CKPT" ]; then
  echo "Actor symlink points elsewhere: $ACTOR_LINK -> $(readlink "$ACTOR_LINK")" >&2
  echo "Expected: $SFT_CKPT" >&2
  exit 1
fi

echo "============================================"
echo "SFT step-$SFT_STEP val-only evaluation"
echo "SFT checkpoint: $SFT_CKPT"
echo "Resume path:    $RESUME_STEP_DIR"
echo "Actor base:     $BASE_ACTOR_MODEL_PATH"
echo "Tokenizer:      $TOKENIZER_PATH"
echo "Experiment:     $EXPERIMENT_NAME"
echo "Output dir:     $OUT_DIR"
echo "Tmp dir:        $TMP_RUN_DIR"
echo "============================================"

VAL_ONLY=true \
VAL_BEFORE_TRAIN=true \
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-900}" \
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}" \
VALIDATION_SHUFFLE="${VALIDATION_SHUFFLE:-false}" \
POLICY_USE_API=false \
ACTOR_MODEL_PATH="$BASE_ACTOR_MODEL_PATH" \
TOKENIZER_PATH="$TOKENIZER_PATH" \
RESUME_MODE=resume_path \
RESUME_FROM_PATH="$RESUME_STEP_DIR" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
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
  'actor_rollout_ref.actor.checkpoint.load_contents=[model]' \
  actor_rollout_ref.model.lora_rank=0 \
  +actor_rollout_ref.rollout.custom.backbone_judge_max_chars=0 \
  "$@"

VAL_JSONL="$OUT_DIR/validation_data/$SFT_STEP.jsonl"
ROUND_TRACE_JSONL="$OUT_DIR/validation_data/round_traces_step${SFT_STEP}.jsonl"
if [ -f "$VAL_JSONL" ]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/extract_round_traces.py" \
    --input "$VAL_JSONL" \
    --output "$ROUND_TRACE_JSONL"
  echo "Round traces: $ROUND_TRACE_JSONL"
else
  echo "Validation JSONL not found for postprocess: $VAL_JSONL" >&2
fi

echo "Done. Check:"
echo "  $OUT_DIR/validation_data"
echo "  $OUT_DIR/io_trace.jsonl"
echo "  $OUT_DIR/train.log"
