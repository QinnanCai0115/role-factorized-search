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
OUTPUT_ROOT="${OUTPUT_ROOT:-/ai/cqn/s3/ckpt/deepseek_reasoner_no_search_test_all}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-test_all_deepseek_reasoner_no_search_${RUN_TS}}"
EXP_DIR="${EXP_DIR:-$OUTPUT_ROOT/$EXPERIMENT_NAME}"
OUT_DIR="${OUT_DIR:-$EXP_DIR/run1}"
OUTPUT="${OUTPUT:-$OUT_DIR/predictions.json}"
OUTPUT_JSONL="${OUTPUT_JSONL:-$OUT_DIR/predictions.jsonl}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-4125}"
VAL_OFFSET="${VAL_OFFSET:-0}"

DIRECT_MODEL="${DIRECT_MODEL:-deepseek-reasoner}"
DIRECT_API_URL="${DIRECT_API_URL:-https://api.deepseek.com/v1}"
DIRECT_ENV_FILE="${DIRECT_ENV_FILE:-.secrets/deepseek.env}"
DIRECT_API_KEY="${DIRECT_API_KEY:-${DEEPSEEK_API_KEY:-}}"
DIRECT_TEMPERATURE="${DIRECT_TEMPERATURE:-0.0}"
DIRECT_MAX_TOKENS="${DIRECT_MAX_TOKENS:-8192}"

NUM_WORKERS="${NUM_WORKERS:-8}"
API_MAX_RETRIES="${API_MAX_RETRIES:-4}"
API_TIMEOUT="${API_TIMEOUT:-300}"
NO_PROXY="${NO_PROXY:-true}"
RESUME="${RESUME:-true}"
SAVE_MESSAGES="${SAVE_MESSAGES:-false}"
SAVE_RAW_API_RESPONSE="${SAVE_RAW_API_RESPONSE:-false}"
EXTRA_BODY_JSON="${EXTRA_BODY_JSON:-}"

mkdir -p "$EXP_DIR" "$OUT_DIR"

cat >"$EXP_DIR/experiment_config.json" <<EOF_CONFIG
{
  "baseline": "DeepSeekReasoner-NoSearch",
  "input": "$INPUT",
  "output": "$OUTPUT",
  "output_jsonl": "$OUTPUT_JSONL",
  "val_max_samples": $VAL_MAX_SAMPLES,
  "val_offset": $VAL_OFFSET,
  "direct_model": "$DIRECT_MODEL",
  "direct_api_url": "$DIRECT_API_URL",
  "direct_env_file": "$DIRECT_ENV_FILE",
  "num_workers": $NUM_WORKERS,
  "tool_calls_allowed": false,
  "retrieval_url": null,
  "search_protocol": null,
  "output_dir": "$OUT_DIR"
}
EOF_CONFIG

echo "============================================"
echo "DeepSeek reasoner no-search test_all baseline"
echo "Input:              $INPUT"
echo "Samples:            $VAL_MAX_SAMPLES"
echo "Offset:             $VAL_OFFSET"
echo "Model:              $DIRECT_MODEL @ $DIRECT_API_URL"
echo "Output:             $OUTPUT"
echo "Incremental JSONL:  $OUTPUT_JSONL"
echo "Workers:            $NUM_WORKERS"
echo "Tools/retrieval:    disabled"
echo "============================================"

args=(
  --input "$INPUT"
  --output "$OUTPUT"
  --output_jsonl "$OUTPUT_JSONL"
  --env_file "$DIRECT_ENV_FILE"
  --api_url "$DIRECT_API_URL"
  --model "$DIRECT_MODEL"
  --api_key "$DIRECT_API_KEY"
  --api_timeout "$API_TIMEOUT"
  --api_max_retries "$API_MAX_RETRIES"
  --temperature "$DIRECT_TEMPERATURE"
  --max_tokens "$DIRECT_MAX_TOKENS"
  --num_workers "$NUM_WORKERS"
  --limit "$VAL_MAX_SAMPLES"
  --offset "$VAL_OFFSET"
)

if [ "$NO_PROXY" = "true" ]; then
  args+=(--no_proxy)
else
  args+=(--no-no_proxy)
fi
if [ "$RESUME" = "true" ]; then
  args+=(--resume)
else
  args+=(--no-resume)
fi
if [ "$SAVE_MESSAGES" = "true" ]; then
  args+=(--save_messages)
fi
if [ "$SAVE_RAW_API_RESPONSE" = "true" ]; then
  args+=(--save_raw_api_response)
fi
if [ -n "$EXTRA_BODY_JSON" ]; then
  args+=(--extra_body_json "$EXTRA_BODY_JSON")
fi

"$PYTHON_BIN" scripts/baselines/deepseek_reasoner_no_search.py "${args[@]}" "$@"

echo "Done. Predictions: $OUTPUT"
echo "Incremental JSONL: $OUTPUT_JSONL"
echo "Config: $EXP_DIR/experiment_config.json"
