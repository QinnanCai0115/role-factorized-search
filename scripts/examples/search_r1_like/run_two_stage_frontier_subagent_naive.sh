#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "/ai/cqn/miniconda3/envs/verl/bin/python" ]; then
    PYTHON_BIN="/ai/cqn/miniconda3/envs/verl/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="${ROOT_DIR:-/ai/cqn/s3}"
PROJECT_NAME="${PROJECT_NAME:-search_subagent_grpo_frontier_naive}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-main_reasoner_frontier_subagent_naive_${RUN_TS}}"
OUT_DIR="${OUT_DIR:-$ROOT_DIR/ckpt/$PROJECT_NAME/$EXPERIMENT_NAME}"

VAL_DATA="${VAL_DATA:-$PROJECT_DIR/data/hotpotqa_2wiki_musique_train/val_mixed_900.parquet}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-900}"
VAL_OFFSET="${VAL_OFFSET:-0}"

# Main/backbone and frontier subagent both use the frontier reasoner by default.
MAIN_AGENT_MODEL="${MAIN_AGENT_MODEL:-${BACKBONE_API_MODEL:-deepseek-reasoner}}"
FRONTIER_SUBAGENT_MODEL="${FRONTIER_SUBAGENT_MODEL:-${POLICY_API_MODEL:-$MAIN_AGENT_MODEL}}"
API_URL="${API_URL:-${BACKBONE_API_URL:-https://api.deepseek.com/v1}}"
API_TIMEOUT="${API_TIMEOUT:-${BACKBONE_API_TIMEOUT:-120}}"
API_MAX_RETRIES="${API_MAX_RETRIES:-3}"
API_KEY="${API_KEY:-${DEEPSEEK_API_KEY:-}}"
DEEPSEEK_ENV_FILE="${DEEPSEEK_ENV_FILE:-/ai/cqn/s3/.secrets/deepseek.env}"

RETRIEVAL_URL="${RETRIEVAL_URL:-http://162.30.4.229:8765/search}"
TOPK="${TOPK:-3}"
RETRIEVAL_TIMEOUT="${RETRIEVAL_TIMEOUT:-180}"
RETRIEVAL_MAX_CONCURRENT="${RETRIEVAL_MAX_CONCURRENT:-64}"

MAX_ORCHESTRATOR_ROUNDS="${MAX_ORCHESTRATOR_ROUNDS:-4}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-3}"
MAX_PARALLEL_CALLS="${MAX_PARALLEL_CALLS:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"

BACKBONE_TEMPERATURE="${BACKBONE_TEMPERATURE:-0.0}"
POLICY_TEMPERATURE="${POLICY_TEMPERATURE:-0.0}"
BACKBONE_MAX_TOKENS="${BACKBONE_MAX_TOKENS:-1024}"
POLICY_MAX_TOKENS="${POLICY_MAX_TOKENS:-1024}"
NO_PROXY="${NO_PROXY:-true}"
RESUME="${RESUME:-true}"

POLICY_ROUNDS_JSONL="${POLICY_ROUNDS_JSONL:-$OUT_DIR/policy_rounds.jsonl}"
ORCHESTRATOR_TRACES_JSONL="${ORCHESTRATOR_TRACES_JSONL:-$OUT_DIR/orchestrator_traces.jsonl}"
RUN_LOG="${RUN_LOG:-$OUT_DIR/naive_run.log}"

mkdir -p "$OUT_DIR"

echo "============================================"
echo "Naive API-only two-stage frontier run"
echo "Backbone model:        $MAIN_AGENT_MODEL"
echo "Policy model:          $FRONTIER_SUBAGENT_MODEL"
echo "API URL:               $API_URL"
echo "Input val data:        $VAL_DATA"
echo "Val max samples:       $VAL_MAX_SAMPLES"
echo "Retrieval URL:         $RETRIEVAL_URL"
echo "Experiment:            $EXPERIMENT_NAME"
echo "Output dir:            $OUT_DIR"
echo "Policy rounds JSONL:   $POLICY_ROUNDS_JSONL"
echo "Orchestrator JSONL:    $ORCHESTRATOR_TRACES_JSONL"
echo "No verl/Ray/vLLM/GPU will be started."
echo "============================================"

cd "$PROJECT_DIR"

ARGS=(
  "$SCRIPT_DIR/generate_sft_rollout.py"
  --input "$VAL_DATA"
  --output "$POLICY_ROUNDS_JSONL"
  --orchestrator_output "$ORCHESTRATOR_TRACES_JSONL"
  --env_file "$DEEPSEEK_ENV_FILE"
  --api_url "$API_URL"
  --model "$FRONTIER_SUBAGENT_MODEL"
  --backbone_model "$MAIN_AGENT_MODEL"
  --api_timeout "$API_TIMEOUT"
  --api_max_retries "$API_MAX_RETRIES"
  --temperature "$POLICY_TEMPERATURE"
  --backbone_temperature "$BACKBONE_TEMPERATURE"
  --max_tokens "$POLICY_MAX_TOKENS"
  --backbone_max_tokens "$BACKBONE_MAX_TOKENS"
  --retrieval_url "$RETRIEVAL_URL"
  --topk "$TOPK"
  --retrieval_timeout "$RETRIEVAL_TIMEOUT"
  --retrieval_max_concurrent "$RETRIEVAL_MAX_CONCURRENT"
  --max_orchestrator_rounds "$MAX_ORCHESTRATOR_ROUNDS"
  --max_assistant_turns "$MAX_ASSISTANT_TURNS"
  --max_parallel_calls "$MAX_PARALLEL_CALLS"
  --num_workers "$NUM_WORKERS"
  --limit "$VAL_MAX_SAMPLES"
  --offset "$VAL_OFFSET"
)

if [ -n "$API_KEY" ]; then
  ARGS+=(--api_key "$API_KEY")
fi

if [ "$NO_PROXY" = "true" ]; then
  ARGS+=(--no_proxy)
else
  ARGS+=(--no-no_proxy)
fi

if [ "$RESUME" = "true" ]; then
  ARGS+=(--resume)
else
  ARGS+=(--no-resume)
fi

"$PYTHON_BIN" "${ARGS[@]}" "$@" 2>&1 | tee "$RUN_LOG"

echo "Done. Check:"
echo "  $ORCHESTRATOR_TRACES_JSONL"
echo "  $POLICY_ROUNDS_JSONL"
echo "  $RUN_LOG"
