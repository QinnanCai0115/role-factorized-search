#!/usr/bin/env bash
set -euo pipefail

DEBUG_XTRACE="${DEBUG_XTRACE:-0}"
if [ "$DEBUG_XTRACE" = "1" ]; then
  set -x
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
cd "$PROJECT_DIR"
export PYTHONPATH="$PROJECT_DIR${PYTHONPATH:+:$PYTHONPATH}"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "/ai/cqn/miniconda3/envs/verl/bin/python" ]; then
    PYTHON_BIN="/ai/cqn/miniconda3/envs/verl/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
INPUT="${INPUT:-$PROJECT_DIR/data/hotpotqa_2wiki_musique_train/test_all.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/ai/cqn/s3/ckpt/search_subagent_deepseek_reasoner_direct_tool_test_all}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-test_all_deepseek_reasoner_direct_tool_${RUN_TS}}"
EXP_DIR="${EXP_DIR:-$OUTPUT_ROOT/$EXPERIMENT_NAME}"
OUT_DIR="${OUT_DIR:-$EXP_DIR/run1}"
OUTPUT="${OUTPUT:-$OUT_DIR/predictions.json}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-4125}"
VAL_OFFSET="${VAL_OFFSET:-0}"

DIRECT_MODEL="${DIRECT_MODEL:-deepseek-reasoner}"
DIRECT_API_URL="${DIRECT_API_URL:-https://api.deepseek.com/v1}"
DIRECT_ENV_FILE="${DIRECT_ENV_FILE:-.secrets/deepseek.env}"
DIRECT_API_KEY="${DIRECT_API_KEY:-${DEEPSEEK_API_KEY:-}}"
DIRECT_TEMPERATURE="${DIRECT_TEMPERATURE:-0.0}"
DIRECT_MAX_TOKENS="${DIRECT_MAX_TOKENS:-8192}"

NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_OUTER_ROUNDS="${MAX_OUTER_ROUNDS:-4}"
SEARCHES_PER_OUTER_ROUND="${SEARCHES_PER_OUTER_ROUND:-1}"
GLOBAL_SEARCH_CAP="${GLOBAL_SEARCH_CAP:-3}"
MAX_PARALLEL_SEARCH_QUERIES="${MAX_PARALLEL_SEARCH_QUERIES:-3}"
MAX_ASSISTANT_TURNS_PER_OUTER_ROUND="${MAX_ASSISTANT_TURNS_PER_OUTER_ROUND:-1}"
RETRIEVAL_URL="${RETRIEVAL_URL:-http://162.30.4.229:8765/search}"
RETRIEVAL_MAX_CONCURRENT="${RETRIEVAL_MAX_CONCURRENT:-96}"
RETRIEVAL_TIMEOUT="${RETRIEVAL_TIMEOUT:-180}"
API_MAX_RETRIES="${API_MAX_RETRIES:-4}"
API_TIMEOUT="${API_TIMEOUT:-300}"
TOPK="${TOPK:-3}"
NO_PROXY="${NO_PROXY:-true}"
SAVE_MESSAGES="${SAVE_MESSAGES:-false}"
SAVE_RAW_API_RESPONSE="${SAVE_RAW_API_RESPONSE:-false}"

mkdir -p "$EXP_DIR" "$OUT_DIR"

cat >"$EXP_DIR/experiment_config.json" <<EOF_CONFIG
{
  "input": "$INPUT",
  "output": "$OUTPUT",
  "val_max_samples": $VAL_MAX_SAMPLES,
  "val_offset": $VAL_OFFSET,
  "direct_model": "$DIRECT_MODEL",
  "direct_api_url": "$DIRECT_API_URL",
  "direct_env_file": "$DIRECT_ENV_FILE",
  "max_outer_rounds": $MAX_OUTER_ROUNDS,
  "searches_per_outer_round": $SEARCHES_PER_OUTER_ROUND,
  "global_search_cap": $GLOBAL_SEARCH_CAP,
  "max_parallel_search_queries": $MAX_PARALLEL_SEARCH_QUERIES,
  "retrieval_url": "$RETRIEVAL_URL",
  "retrieval_max_concurrent": $RETRIEVAL_MAX_CONCURRENT,
  "output_dir": "$OUT_DIR"
}
EOF_CONFIG

echo "============================================"
echo "DeepSeek reasoner direct_tool test_all rollout"
echo "Input:              $INPUT"
echo "Samples:            $VAL_MAX_SAMPLES"
echo "Model:              $DIRECT_MODEL @ $DIRECT_API_URL"
echo "Output:             $OUTPUT"
echo "Search turns:       $GLOBAL_SEARCH_CAP total, $SEARCHES_PER_OUTER_ROUND per non-final round"
echo "Parallel queries:   $MAX_PARALLEL_SEARCH_QUERIES per search turn"
echo "Retrieval:          $RETRIEVAL_URL"
echo "============================================"

args=(
  --input "$INPUT"
  --output "$OUTPUT"
  --env_file "$DIRECT_ENV_FILE"
  --api_url "$DIRECT_API_URL"
  --model "$DIRECT_MODEL"
  --api_key "$DIRECT_API_KEY"
  --api_timeout "$API_TIMEOUT"
  --api_max_retries "$API_MAX_RETRIES"
  --temperature "$DIRECT_TEMPERATURE"
  --max_tokens "$DIRECT_MAX_TOKENS"
  --retrieval_url "$RETRIEVAL_URL"
  --topk "$TOPK"
  --retrieval_timeout "$RETRIEVAL_TIMEOUT"
  --retrieval_max_concurrent "$RETRIEVAL_MAX_CONCURRENT"
  --max_outer_rounds "$MAX_OUTER_ROUNDS"
  --searches_per_outer_round "$SEARCHES_PER_OUTER_ROUND"
  --global_search_cap "$GLOBAL_SEARCH_CAP"
  --max_parallel_search_queries "$MAX_PARALLEL_SEARCH_QUERIES"
  --max_assistant_turns_per_outer_round "$MAX_ASSISTANT_TURNS_PER_OUTER_ROUND"
  --num_workers "$NUM_WORKERS"
  --limit "$VAL_MAX_SAMPLES"
  --offset "$VAL_OFFSET"
)

if [ "$NO_PROXY" = "true" ]; then
  args+=(--no_proxy)
else
  args+=(--no-no_proxy)
fi
if [ "$SAVE_MESSAGES" = "true" ]; then
  args+=(--save_messages)
fi
if [ "$SAVE_RAW_API_RESPONSE" = "true" ]; then
  args+=(--save_raw_api_response)
fi

"$PYTHON_BIN" scripts/baselines/direct_search_budget_matched.py "${args[@]}" "$@"

echo "Done. Predictions: $OUTPUT"
echo "Config: $EXP_DIR/experiment_config.json"
