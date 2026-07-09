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

CONDA_BASE="${CONDA_BASE:-/ai/cqn/miniconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-verl}"
CONDA_ACTIVATE="${CONDA_ACTIVATE:-true}"
if [ "$CONDA_ACTIVATE" = "true" ] && [ -f "$CONDA_BASE/bin/activate" ]; then
  # Keep vLLM, flashinfer JIT helpers, and Python packages on one consistent PATH.
  source "$CONDA_BASE/bin/activate"
  conda activate "$CONDA_ENV_NAME"
fi

if [ -z "${PYTHON_BIN:-}" ]; then
  PYTHON_BIN="$(command -v python)"
fi
PYTHON_BIN_DIR="$(cd "$(dirname "$PYTHON_BIN")" && pwd)"
export PATH="$PYTHON_BIN_DIR:$PATH"

MODEL_SIZE="${MODEL_SIZE:-1.7B}"
case "$MODEL_SIZE" in
  1.7B|1p7B|1p7b|1.7b)
    MODEL_PATH="${MODEL_PATH:-/ai/cqn/model/Qwen3-1.7B}"
    MODEL_NAME="${MODEL_NAME:-Qwen3-1.7B}"
    CUDA_VISIBLE_DEVICES_FOR_VLLM="${CUDA_VISIBLE_DEVICES_FOR_VLLM:-0}"
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
    DISABLE_QWEN_THINKING="${DISABLE_QWEN_THINKING:-true}"
    ;;
  8B|8b)
    MODEL_PATH="${MODEL_PATH:-/ai/yzx/Models/Qwen/Qwen3-8B}"
    MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
    CUDA_VISIBLE_DEVICES_FOR_VLLM="${CUDA_VISIBLE_DEVICES_FOR_VLLM:-0}"
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
    DISABLE_QWEN_THINKING="${DISABLE_QWEN_THINKING:-false}"
    ;;
  *)
    echo "Unsupported MODEL_SIZE=$MODEL_SIZE. Use MODEL_SIZE=1.7B or MODEL_SIZE=8B." >&2
    exit 2
    ;;
esac

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
INPUT="${INPUT:-$PROJECT_DIR/data/hotpotqa_2wiki_musique_train/test_all.parquet}"
LIMIT="${LIMIT:-4125}"
OFFSET="${OFFSET:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/ai/cqn/s3/ckpt/untrained_qwen_direct_search_baseline}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL_NAME}_direct_search_test_all_${RUN_TS}}"
EXP_DIR="${EXP_DIR:-$OUTPUT_ROOT/$EXPERIMENT_NAME}"
OUTPUT="${OUTPUT:-$EXP_DIR/predictions.json}"

API_HOST="${API_HOST:-0.0.0.0}"
API_PORT="${API_PORT:-8010}"
API_URL="${API_URL:-http://127.0.0.1:${API_PORT}/v1}"
API_KEY="${API_KEY:-local-vllm}"
API_TIMEOUT="${API_TIMEOUT:-300}"
API_MAX_RETRIES="${API_MAX_RETRIES:-3}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-512}"
NO_PROXY="${NO_PROXY:-true}"

MAX_OUTER_ROUNDS="${MAX_OUTER_ROUNDS:-4}"
SEARCHES_PER_OUTER_ROUND="${SEARCHES_PER_OUTER_ROUND:-1}"
GLOBAL_SEARCH_CAP="${GLOBAL_SEARCH_CAP:-3}"
MAX_PARALLEL_SEARCH_QUERIES="${MAX_PARALLEL_SEARCH_QUERIES:-3}"
MAX_ASSISTANT_TURNS_PER_OUTER_ROUND="${MAX_ASSISTANT_TURNS_PER_OUTER_ROUND:-1}"
NUM_WORKERS="${NUM_WORKERS:-1}"

TOOL_CONFIG="${TOOL_CONFIG:-$PROJECT_DIR/scripts/examples/config/tool_config/search_subagent_tool_config.yaml}"
RETRIEVAL_URL="${RETRIEVAL_URL:-http://162.30.4.229:8765/search}"
TOPK="${TOPK:-3}"
RETRIEVAL_TIMEOUT="${RETRIEVAL_TIMEOUT:-180}"
RETRIEVAL_MAX_CONCURRENT="${RETRIEVAL_MAX_CONCURRENT:-64}"

START_VLLM="${START_VLLM:-true}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-900}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.85}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-true}"
VLLM_USE_V1="${VLLM_USE_V1:-1}"
VLLM_LOG="${VLLM_LOG:-$EXP_DIR/vllm_${API_PORT}.log}"

SAVE_MESSAGES="${SAVE_MESSAGES:-false}"
SAVE_RAW_API_RESPONSE="${SAVE_RAW_API_RESPONSE:-false}"
STRIP_THINKING_FROM_HISTORY="${STRIP_THINKING_FROM_HISTORY:-true}"

mkdir -p "$EXP_DIR"

VLLM_PID=""
cleanup_vllm() {
  if [ -n "$VLLM_PID" ] && kill -0 "$VLLM_PID" >/dev/null 2>&1; then
    echo "Stopping vLLM pid=$VLLM_PID"
    kill "$VLLM_PID" >/dev/null 2>&1 || true
    wait "$VLLM_PID" >/dev/null 2>&1 || true
  fi
}
trap cleanup_vllm EXIT

wait_for_local_server() {
  local url="$1"
  local deadline=$((SECONDS + VLLM_WAIT_SECONDS))
  until curl --noproxy "*" --silent --fail --max-time 5 "$url/models" >/dev/null 2>&1; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "Timed out waiting for $url/models" >&2
      return 1
    fi
    sleep 5
  done
}

start_vllm_server() {
  mkdir -p "$(dirname "$VLLM_LOG")"
  echo "Starting vLLM: $MODEL_NAME from $MODEL_PATH on port $API_PORT (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES_FOR_VLLM)"
  local args=(
    -m vllm.entrypoints.openai.api_server
    --host "$API_HOST"
    --port "$API_PORT"
    --model "$MODEL_PATH"
    --served-model-name "$MODEL_NAME"
    --dtype "$VLLM_DTYPE"
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION"
    --max-model-len "$VLLM_MAX_MODEL_LEN"
    --max-num-seqs "$VLLM_MAX_NUM_SEQS"
    --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS"
    --disable-log-requests
  )
  if [ "$VLLM_ENABLE_PREFIX_CACHING" = "true" ]; then
    args+=(--enable-prefix-caching)
  fi

  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES_FOR_VLLM" VLLM_USE_V1="$VLLM_USE_V1" PYTHONUNBUFFERED=1 "$PYTHON_BIN" "${args[@]}" >"$VLLM_LOG" 2>&1 &
  VLLM_PID="$!"
  wait_for_local_server "$API_URL"
}

if [ "$START_VLLM" = "true" ]; then
  start_vllm_server
else
  wait_for_local_server "$API_URL"
fi

cat >"$EXP_DIR/experiment_config.json" <<EOF
{
  "baseline": "untrained_qwen_direct_search",
  "input": "$INPUT",
  "limit": $LIMIT,
  "offset": $OFFSET,
  "model_size": "$MODEL_SIZE",
  "model_path": "$MODEL_PATH",
  "model_name": "$MODEL_NAME",
  "disable_qwen_thinking": "$DISABLE_QWEN_THINKING",
  "api_url": "$API_URL",
  "cuda_visible_devices": "$CUDA_VISIBLE_DEVICES_FOR_VLLM",
  "tensor_parallel_size": $TENSOR_PARALLEL_SIZE,
  "vllm_use_v1": "$VLLM_USE_V1",
  "output": "$OUTPUT",
  "retrieval_url": "$RETRIEVAL_URL",
  "budget": {
    "max_outer_rounds": $MAX_OUTER_ROUNDS,
    "searches_per_outer_round": $SEARCHES_PER_OUTER_ROUND,
    "global_search_cap": $GLOBAL_SEARCH_CAP,
    "max_parallel_search_queries": $MAX_PARALLEL_SEARCH_QUERIES
  }
}
EOF

echo "============================================"
echo "Untrained Qwen direct-search baseline"
echo "Model:          $MODEL_NAME"
echo "No thinking:    $DISABLE_QWEN_THINKING"
echo "Model path:     $MODEL_PATH"
echo "Input:          $INPUT"
echo "Limit/offset:   $LIMIT / $OFFSET"
echo "API URL:        $API_URL"
echo "Retrieval URL:  $RETRIEVAL_URL"
echo "Output:         $OUTPUT"
echo "vLLM log:       $VLLM_LOG"
echo "============================================"

cmd=(
  "$PYTHON_BIN" "$PROJECT_DIR/scripts/baselines/direct_search_budget_matched.py"
  --input "$INPUT"
  --output "$OUTPUT"
  --env_file ""
  --api_url "$API_URL"
  --model "$MODEL_NAME"
  --api_key "$API_KEY"
  --api_timeout "$API_TIMEOUT"
  --api_max_retries "$API_MAX_RETRIES"
  --temperature "$TEMPERATURE"
  --max_tokens "$MAX_TOKENS"
  --tool_config "$TOOL_CONFIG"
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
  --limit "$LIMIT"
  --offset "$OFFSET"
)

if [ "$NO_PROXY" = "true" ]; then
  cmd+=(--no_proxy)
else
  cmd+=(--no-no_proxy)
fi
if [ "$SAVE_MESSAGES" = "true" ]; then
  cmd+=(--save_messages)
fi
if [ "$SAVE_RAW_API_RESPONSE" = "true" ]; then
  cmd+=(--save_raw_api_response)
fi
if [ "$STRIP_THINKING_FROM_HISTORY" = "true" ]; then
  cmd+=(--strip_thinking_from_history)
fi
if [ "$DISABLE_QWEN_THINKING" = "true" ]; then
  cmd+=(--disable_qwen_thinking)
fi

"${cmd[@]}"

"$PYTHON_BIN" - "$OUTPUT" <<'PY'
import json
import sys

path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)
summary = data.get("summary", {})
print("Summary JSON:")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "Done. Predictions: $OUTPUT"
echo "Done. Config:      $EXP_DIR/experiment_config.json"
