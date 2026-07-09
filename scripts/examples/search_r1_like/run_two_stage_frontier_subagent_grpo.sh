#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
BASE_SCRIPT="$SCRIPT_DIR/run_two_stage_subagent_grpo.sh"

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="${ROOT_DIR:-/ai/cqn/s3}"
PROJECT_NAME="${PROJECT_NAME:-search_subagent_grpo_frontier}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-main_reasoner_frontier_subagent_${RUN_TS}}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/ckpt/$PROJECT_NAME/$EXPERIMENT_NAME}"

# Ray creates AF_UNIX sockets under _temp_dir/session_*/sockets, and Linux
# limits that path to 107 bytes. Keep this directory intentionally short.
TMP_RUN_DIR="${TMP_RUN_DIR:-/tmp/frontier_${RUN_TS}}"

# Main agent/backbone and frontier subagent both use the frontier reasoner by default.
MAIN_AGENT_MODEL="${MAIN_AGENT_MODEL:-${BACKBONE_API_MODEL:-deepseek-reasoner}}"
FRONTIER_SUBAGENT_MODEL="${FRONTIER_SUBAGENT_MODEL:-${POLICY_API_MODEL:-$MAIN_AGENT_MODEL}}"
MAIN_AGENT_API_URL="${MAIN_AGENT_API_URL:-${BACKBONE_API_URL:-https://api.deepseek.com/v1}}"
FRONTIER_SUBAGENT_API_URL="${FRONTIER_SUBAGENT_API_URL:-${POLICY_API_URL:-$MAIN_AGENT_API_URL}}"
MAIN_AGENT_API_TIMEOUT="${MAIN_AGENT_API_TIMEOUT:-${BACKBONE_API_TIMEOUT:-120}}"
FRONTIER_SUBAGENT_API_TIMEOUT="${FRONTIER_SUBAGENT_API_TIMEOUT:-${POLICY_API_TIMEOUT:-$MAIN_AGENT_API_TIMEOUT}}"

# Frontier subagent is API-backed, so default to validation/evaluation mode.
VAL_ONLY="${VAL_ONLY:-true}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-true}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-900}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
VALIDATION_SHUFFLE="${VALIDATION_SHUFFLE:-false}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-1}"

# With an API-backed frontier subagent, the local actor is only infrastructure.
# Keep the default light; override these env vars if you need a different layout.
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-1}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ACTOR_LORA_RANK="${ACTOR_LORA_RANK:-0}"
ACTOR_STRATEGY="${ACTOR_STRATEGY:-fsdp2}"
ACTOR_FSDP_SIZE="${ACTOR_FSDP_SIZE:-1}"
ACTOR_CHECKPOINT_LOAD_CONTENTS="${ACTOR_CHECKPOINT_LOAD_CONTENTS:-[model]}"

echo "============================================"
echo "Two-stage main agent + frontier subagent run"
echo "Main/backbone model:       $MAIN_AGENT_MODEL"
echo "Frontier subagent model:   $FRONTIER_SUBAGENT_MODEL"
echo "Experiment:                $EXPERIMENT_NAME"
echo "Output dir:                $OUT_DIR"
echo "Val only:                  $VAL_ONLY"
echo "Val max samples:           $VAL_MAX_SAMPLES"
echo "Tmp dir:                   $TMP_RUN_DIR"
echo "============================================"

env \
  ROOT_DIR="$ROOT_DIR" \
  PROJECT_NAME="$PROJECT_NAME" \
  EXPERIMENT_NAME="$EXPERIMENT_NAME" \
  RUN_TS="$RUN_TS" \
  OUT_DIR="$OUT_DIR" \
  VAL_ONLY="$VAL_ONLY" \
  VAL_BEFORE_TRAIN="$VAL_BEFORE_TRAIN" \
  VAL_MAX_SAMPLES="$VAL_MAX_SAMPLES" \
  VAL_BATCH_SIZE="$VAL_BATCH_SIZE" \
  VALIDATION_SHUFFLE="$VALIDATION_SHUFFLE" \
  TOTAL_TRAINING_STEPS="$TOTAL_TRAINING_STEPS" \
  N_GPUS_PER_NODE="$N_GPUS_PER_NODE" \
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  BACKBONE_API_MODE="${BACKBONE_API_MODE:-openai_compatible}" \
  BACKBONE_API_URL="$MAIN_AGENT_API_URL" \
  BACKBONE_API_MODEL="$MAIN_AGENT_MODEL" \
  BACKBONE_API_TIMEOUT="$MAIN_AGENT_API_TIMEOUT" \
  BACKBONE_API_MAX_CONCURRENT="${BACKBONE_API_MAX_CONCURRENT:-4}" \
  BACKBONE_API_MAX_RETRIES="${BACKBONE_API_MAX_RETRIES:-10}" \
  BACKBONE_API_NO_PROXY="${BACKBONE_API_NO_PROXY:-1}" \
  POLICY_USE_API=true \
  POLICY_API_MODE="${POLICY_API_MODE:-openai_compatible}" \
  POLICY_API_URL="$FRONTIER_SUBAGENT_API_URL" \
  POLICY_API_MODEL="$FRONTIER_SUBAGENT_MODEL" \
  POLICY_API_TIMEOUT="$FRONTIER_SUBAGENT_API_TIMEOUT" \
  POLICY_API_MAX_RETRIES="${POLICY_API_MAX_RETRIES:-3}" \
  POLICY_API_TEMPERATURE="${POLICY_API_TEMPERATURE:-0.0}" \
  POLICY_API_MAX_TOKENS="${POLICY_API_MAX_TOKENS:-}" \
  POLICY_API_NO_PROXY="${POLICY_API_NO_PROXY:-1}" \
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
    actor_rollout_ref.model.lora_rank="$ACTOR_LORA_RANK" \
    actor_rollout_ref.actor.strategy="$ACTOR_STRATEGY" \
    actor_rollout_ref.actor.fsdp_config.fsdp_size="$ACTOR_FSDP_SIZE" \
    "actor_rollout_ref.actor.checkpoint.load_contents=$ACTOR_CHECKPOINT_LOAD_CONTENTS" \
    +actor_rollout_ref.rollout.custom.backbone_judge_max_chars=0 \
    "$@"

echo "Done. Check:"
echo "  $OUT_DIR/validation_data"
echo "  $OUT_DIR/io_trace.jsonl"
echo "  $OUT_DIR/train.log"
