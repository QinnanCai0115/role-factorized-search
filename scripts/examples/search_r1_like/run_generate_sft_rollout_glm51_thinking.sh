#!/usr/bin/env bash
set -euo pipefail

DEBUG_XTRACE="${DEBUG_XTRACE:-0}"
if [ "$DEBUG_XTRACE" = "1" ]; then
  set -x
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
cd "$PROJECT_DIR"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "/ai/cqn/miniconda3/envs/verl/bin/python" ]; then
    PYTHON_BIN="/ai/cqn/miniconda3/envs/verl/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

INPUT="${INPUT:-/ai/cqn/datacon/data/hotpotqa_2wiki_musique_train/train_mixed_2000_sft.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/ai/cqn/datacon/data/deepseek_policy_sft_rollouts}"
RUN_NAME="${RUN_NAME:-train_mixed_2000.glm_5_1_thinking.multi_query.backbone_prompt_v2}"
OUTPUT="${OUTPUT:-$OUTPUT_DIR/$RUN_NAME.sft.jsonl}"
ORCHESTRATOR_OUTPUT="${ORCHESTRATOR_OUTPUT:-$OUTPUT_DIR/$RUN_NAME.traces.jsonl}"

BACKBONE_ENV_FILE="${BACKBONE_ENV_FILE:-.secrets/deepseek.env}"
POLICY_ENV_FILE="${POLICY_ENV_FILE:-.secrets/glm.env}"
BACKBONE_API_URL="${BACKBONE_API_URL:-https://api.deepseek.com/v1}"
POLICY_API_URL="${POLICY_API_URL:-https://open.bigmodel.cn/api/paas/v4}"
BACKBONE_MODEL="${BACKBONE_MODEL:-deepseek-reasoner}"
POLICY_MODEL="${POLICY_MODEL:-glm-5.1}"

NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_PARALLEL_POLICY_QUERIES="${MAX_PARALLEL_POLICY_QUERIES:-3}"
MAX_BACKBONE_SEARCH_QUERIES="${MAX_BACKBONE_SEARCH_QUERIES:-3}"
MAX_ORCHESTRATOR_ROUNDS="${MAX_ORCHESTRATOR_ROUNDS:-4}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-4}"
RETRIEVAL_MAX_CONCURRENT="${RETRIEVAL_MAX_CONCURRENT:-96}"
API_MAX_RETRIES="${API_MAX_RETRIES:-4}"
API_TIMEOUT="${API_TIMEOUT:-180}"
TOPK="${TOPK:-3}"
LIMIT="${LIMIT:-}"
OFFSET="${OFFSET:-0}"
RESUME="${RESUME:-true}"

mkdir -p "$OUTPUT_DIR"

echo "Input: $INPUT"
echo "Output: $OUTPUT"
echo "Trace: $ORCHESTRATOR_OUTPUT"
echo "Backbone: $BACKBONE_MODEL @ $BACKBONE_API_URL"
echo "Policy: $POLICY_MODEL @ $POLICY_API_URL (thinking enabled)"
echo "Workers: $NUM_WORKERS, max_parallel_policy_queries: $MAX_PARALLEL_POLICY_QUERIES"

ARGS=(
  "$SCRIPT_DIR/generate_sft_rollout.py"
  --input "$INPUT"
  --output "$OUTPUT"
  --orchestrator_output "$ORCHESTRATOR_OUTPUT"
  --env_file "$BACKBONE_ENV_FILE"
  --policy_env_file "$POLICY_ENV_FILE"
  --api_url "$BACKBONE_API_URL"
  --policy_api_url "$POLICY_API_URL"
  --backbone_model "$BACKBONE_MODEL"
  --policy_model "$POLICY_MODEL"
  --policy_enable_thinking
  --policy_thinking_field thinking
  --policy_thinking_type enabled
  --policy_preserve_reasoning_content
  --num_workers "$NUM_WORKERS"
  --max_parallel_policy_queries "$MAX_PARALLEL_POLICY_QUERIES"
  --max_backbone_search_queries "$MAX_BACKBONE_SEARCH_QUERIES"
  --max_orchestrator_rounds "$MAX_ORCHESTRATOR_ROUNDS"
  --max_assistant_turns "$MAX_ASSISTANT_TURNS"
  --retrieval_max_concurrent "$RETRIEVAL_MAX_CONCURRENT"
  --api_max_retries "$API_MAX_RETRIES"
  --api_timeout "$API_TIMEOUT"
  --topk "$TOPK"
  --offset "$OFFSET"
)

if [ "$RESUME" = "true" ]; then
  ARGS+=(--resume)
else
  ARGS+=(--no-resume)
fi

if [ -n "$LIMIT" ]; then
  ARGS+=(--limit "$LIMIT")
fi

PYTHONUNBUFFERED=1 "$PYTHON_BIN" "${ARGS[@]}"
