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
  source "$CONDA_BASE/bin/activate"
  conda activate "$CONDA_ENV_NAME"
fi

if [ -z "${PYTHON_BIN:-}" ]; then
  PYTHON_BIN="$(command -v python)"
fi
PYTHON_BIN_DIR="$(cd "$(dirname "$PYTHON_BIN")" && pwd)"
export PATH="$PYTHON_BIN_DIR:$PATH"

MODEL_SIZE="${MODEL_SIZE:-32B}"
case "$MODEL_SIZE" in
  32B|32b)
    MODEL_PATH="${MODEL_PATH:-/ai/yzx/Models/Qwen3-32B}"
    MODEL_NAME="${MODEL_NAME:-Qwen3-32B}"
    CUDA_VISIBLE_DEVICES_FOR_VLLM="${CUDA_VISIBLE_DEVICES_FOR_VLLM:-0}"
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
    API_PORT="${API_PORT:-8020}"
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
    VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
    VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
    NUM_WORKERS="${NUM_WORKERS:-4}"
    ;;
  8B|8b)
    MODEL_PATH="${MODEL_PATH:-/ai/yzx/Models/Qwen/Qwen3-8B}"
    MODEL_NAME="${MODEL_NAME:-Qwen3-8B}"
    CUDA_VISIBLE_DEVICES_FOR_VLLM="${CUDA_VISIBLE_DEVICES_FOR_VLLM:-1}"
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
    API_PORT="${API_PORT:-8030}"
    VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.75}"
    VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
    VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
    NUM_WORKERS="${NUM_WORKERS:-8}"
    ;;
  *)
    echo "Unsupported MODEL_SIZE=$MODEL_SIZE. Use MODEL_SIZE=32B or MODEL_SIZE=8B." >&2
    exit 2
    ;;
esac

if [ ! -d "$MODEL_PATH" ]; then
  echo "MODEL_PATH does not exist: $MODEL_PATH" >&2
  exit 2
fi

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
INPUT="${INPUT:-$PROJECT_DIR/data/hotpotqa_2wiki_musique_train/test_all.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/ai/cqn/s3/ckpt/qwen3_no_search_test_all}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-test_all_${MODEL_NAME}_no_search_${RUN_TS}}"
EXP_DIR="${EXP_DIR:-$OUTPUT_ROOT/$EXPERIMENT_NAME}"
OUT_DIR="${OUT_DIR:-$EXP_DIR/run1}"
OUTPUT="${OUTPUT:-$OUT_DIR/predictions.json}"
OUTPUT_JSONL="${OUTPUT_JSONL:-$OUT_DIR/predictions.jsonl}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-4125}"
VAL_OFFSET="${VAL_OFFSET:-0}"

API_HOST="${API_HOST:-0.0.0.0}"
API_URL="${API_URL:-http://127.0.0.1:${API_PORT}/v1}"
API_KEY="${API_KEY:-local-vllm}"
API_TIMEOUT="${API_TIMEOUT:-300}"
API_MAX_RETRIES="${API_MAX_RETRIES:-4}"
TEMPERATURE="${TEMPERATURE:-0.0}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
NO_PROXY="${NO_PROXY:-true}"
RESUME="${RESUME:-true}"

START_VLLM="${START_VLLM:-true}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-900}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-true}"
VLLM_USE_V1="${VLLM_USE_V1:-1}"
VLLM_LOG="${VLLM_LOG:-$EXP_DIR/vllm_${API_PORT}.log}"

SAVE_MESSAGES="${SAVE_MESSAGES:-false}"
SAVE_RAW_API_RESPONSE="${SAVE_RAW_API_RESPONSE:-false}"
EXTRA_BODY_JSON="${EXTRA_BODY_JSON:-}"
BASELINE_NAME="${BASELINE_NAME:-${MODEL_NAME}-NoSearch}"

mkdir -p "$EXP_DIR" "$OUT_DIR"

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

cat >"$EXP_DIR/experiment_config.json" <<EOF_CONFIG
{
  "baseline": "$BASELINE_NAME",
  "input": "$INPUT",
  "output": "$OUTPUT",
  "output_jsonl": "$OUTPUT_JSONL",
  "val_max_samples": $VAL_MAX_SAMPLES,
  "val_offset": $VAL_OFFSET,
  "model_size": "$MODEL_SIZE",
  "model_path": "$MODEL_PATH",
  "model_name": "$MODEL_NAME",
  "api_url": "$API_URL",
  "cuda_visible_devices": "$CUDA_VISIBLE_DEVICES_FOR_VLLM",
  "tensor_parallel_size": $TENSOR_PARALLEL_SIZE,
  "vllm_gpu_memory_utilization": $VLLM_GPU_MEMORY_UTILIZATION,
  "num_workers": $NUM_WORKERS,
  "tool_calls_allowed": false,
  "retrieval_url": null,
  "search_protocol": null,
  "output_dir": "$OUT_DIR"
}
EOF_CONFIG

echo "============================================"
echo "Qwen3 no-search test_all baseline"
echo "Input:              $INPUT"
echo "Samples:            $VAL_MAX_SAMPLES"
echo "Offset:             $VAL_OFFSET"
echo "Model:              $MODEL_NAME from $MODEL_PATH"
echo "API URL:            $API_URL"
echo "CUDA devices:       $CUDA_VISIBLE_DEVICES_FOR_VLLM"
echo "Output:             $OUTPUT"
echo "Incremental JSONL:  $OUTPUT_JSONL"
echo "Workers:            $NUM_WORKERS"
echo "Tools/retrieval:    disabled"
echo "vLLM log:           $VLLM_LOG"
echo "============================================"

args=(
  --baseline_name "$BASELINE_NAME"
  --input "$INPUT"
  --output "$OUTPUT"
  --output_jsonl "$OUTPUT_JSONL"
  --env_file ""
  --api_url "$API_URL"
  --model "$MODEL_NAME"
  --api_key "$API_KEY"
  --api_timeout "$API_TIMEOUT"
  --api_max_retries "$API_MAX_RETRIES"
  --temperature "$TEMPERATURE"
  --max_tokens "$MAX_TOKENS"
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
