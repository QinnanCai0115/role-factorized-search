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
OUTPUT_ROOT="${OUTPUT_ROOT:-/ai/cqn/s3/ckpt/search_subagent_qwen3_32b_deepseek_reasoner_policy_test_all}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-test_all_qwen3_32b_backbone_deepseek_reasoner_policy_${RUN_TS}}"
EXP_DIR="${EXP_DIR:-$OUTPUT_ROOT/$EXPERIMENT_NAME}"
OUT_DIR="${OUT_DIR:-$EXP_DIR/run1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-4125}"
VAL_OFFSET="${VAL_OFFSET:-0}"

BACKBONE_MODEL_PATH="${BACKBONE_MODEL_PATH:-/ai/zjm/Models/Qwen3-32B/}"
BACKBONE_MODEL="${BACKBONE_MODEL:-Qwen3-32B}"
BACKBONE_API_URL="${BACKBONE_API_URL:-http://127.0.0.1:8000/v1}"
BACKBONE_HOST="${BACKBONE_HOST:-0.0.0.0}"
BACKBONE_PORT="${BACKBONE_PORT:-8000}"
BACKBONE_CUDA_VISIBLE_DEVICES="${BACKBONE_CUDA_VISIBLE_DEVICES:-0,1}"
BACKBONE_TENSOR_PARALLEL_SIZE="${BACKBONE_TENSOR_PARALLEL_SIZE:-2}"
BACKBONE_GPU_MEMORY_UTILIZATION="${BACKBONE_GPU_MEMORY_UTILIZATION:-0.7}"
BACKBONE_MAX_TOKENS="${BACKBONE_MAX_TOKENS:-8192}"
BACKBONE_JUDGE_MAX_TOKENS="${BACKBONE_JUDGE_MAX_TOKENS:-1024}"
BACKBONE_JUDGE_API_URL="${BACKBONE_JUDGE_API_URL:-https://api.deepseek.com/v1}"
BACKBONE_JUDGE_MODEL="${BACKBONE_JUDGE_MODEL:-deepseek-reasoner}"
BACKBONE_JUDGE_ENV_FILE="${BACKBONE_JUDGE_ENV_FILE:-.secrets/deepseek.env}"
BACKBONE_TEMPERATURE="${BACKBONE_TEMPERATURE:-0.0}"
BACKBONE_ENV_FILE="${BACKBONE_ENV_FILE:-}"

POLICY_MODEL="${POLICY_MODEL:-deepseek-reasoner}"
POLICY_API_URL="${POLICY_API_URL:-https://api.deepseek.com/v1}"
POLICY_ENV_FILE="${POLICY_ENV_FILE:-.secrets/deepseek.env}"
POLICY_API_KEY_ENV_VAR="${POLICY_API_KEY_ENV_VAR:-DEEPSEEK_API_KEY}"
POLICY_TEMPERATURE="${POLICY_TEMPERATURE:-0.6}"
POLICY_MAX_TOKENS="${POLICY_MAX_TOKENS:-4096}"
POLICY_ENABLE_THINKING="${POLICY_ENABLE_THINKING:-false}"
POLICY_PRESERVE_REASONING_CONTENT="${POLICY_PRESERVE_REASONING_CONTENT:-false}"

NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_PARALLEL_POLICY_QUERIES="${MAX_PARALLEL_POLICY_QUERIES:-3}"
MAX_BACKBONE_SEARCH_QUERIES="${MAX_BACKBONE_SEARCH_QUERIES:-3}"
MAX_ORCHESTRATOR_ROUNDS="${MAX_ORCHESTRATOR_ROUNDS:-4}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-4}"
MAX_PARALLEL_CALLS="${MAX_PARALLEL_CALLS:-1}"
RETRIEVAL_URL="${RETRIEVAL_URL:-http://162.30.4.229:8765/search}"
RETRIEVAL_MAX_CONCURRENT="${RETRIEVAL_MAX_CONCURRENT:-96}"
RETRIEVAL_TIMEOUT="${RETRIEVAL_TIMEOUT:-180}"
API_MAX_RETRIES="${API_MAX_RETRIES:-4}"
API_TIMEOUT="${API_TIMEOUT:-300}"
TOPK="${TOPK:-3}"
NO_PROXY="${NO_PROXY:-true}"
RESUME="${RESUME:-true}"
SAVE_RAW_API_RESPONSE="${SAVE_RAW_API_RESPONSE:-false}"
SAVE_RAW_RETRIEVAL_RESPONSE="${SAVE_RAW_RETRIEVAL_RESPONSE:-false}"

START_BACKBONE_VLLM="${START_BACKBONE_VLLM:-false}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-900}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-true}"

BACKBONE_VLLM_LOG="${BACKBONE_VLLM_LOG:-$EXP_DIR/backbone_vllm_${BACKBONE_PORT}.log}"

mkdir -p "$EXP_DIR" "$OUT_DIR"

BACKBONE_VLLM_PID=""
cleanup_vllm() {
  local pid="${BACKBONE_VLLM_PID:-}"
  if [ -n "$pid" ] && kill -0 "$pid" >/dev/null 2>&1; then
    echo "Stopping vLLM pid=$pid"
    kill "$pid" >/dev/null 2>&1 || true
    wait "$pid" >/dev/null 2>&1 || true
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
  local model_path="$1"
  local served_name="$2"
  local host="$3"
  local port="$4"
  local cuda_devices="$5"
  local tensor_parallel_size="$6"
  local gpu_memory_utilization="$7"
  local api_url="$8"
  local log_path="$9"

  mkdir -p "$(dirname "$log_path")"
  echo "Starting vLLM: $served_name from $model_path on port $port (CUDA_VISIBLE_DEVICES=$cuda_devices)"
  local args=(
    -m vllm.entrypoints.openai.api_server
    --host "$host"
    --port "$port"
    --model "$model_path"
    --served-model-name "$served_name"
    --dtype "$VLLM_DTYPE"
    --tensor-parallel-size "$tensor_parallel_size"
    --gpu-memory-utilization "$gpu_memory_utilization"
    --max-model-len "$VLLM_MAX_MODEL_LEN"
    --max-num-seqs "$VLLM_MAX_NUM_SEQS"
    --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS"
    --disable-log-requests
  )
  if [ "$VLLM_ENABLE_PREFIX_CACHING" = "true" ]; then
    args+=(--enable-prefix-caching)
  fi

  CUDA_VISIBLE_DEVICES="$cuda_devices" PYTHONUNBUFFERED=1 "$PYTHON_BIN" "${args[@]}" >"$log_path" 2>&1 &
  BACKBONE_VLLM_PID="$!"
  wait_for_local_server "$api_url"
}

if [ "$START_BACKBONE_VLLM" = "true" ]; then
  start_vllm_server \
    "$BACKBONE_MODEL_PATH" \
    "$BACKBONE_MODEL" \
    "$BACKBONE_HOST" \
    "$BACKBONE_PORT" \
    "$BACKBONE_CUDA_VISIBLE_DEVICES" \
    "$BACKBONE_TENSOR_PARALLEL_SIZE" \
    "$BACKBONE_GPU_MEMORY_UTILIZATION" \
    "$BACKBONE_API_URL" \
    "$BACKBONE_VLLM_LOG"
else
  wait_for_local_server "$BACKBONE_API_URL"
fi

cat >"$EXP_DIR/experiment_config.json" <<EOF
{
  "input": "$INPUT",
  "val_max_samples": $VAL_MAX_SAMPLES,
  "val_offset": $VAL_OFFSET,
  "backbone_model_path": "$BACKBONE_MODEL_PATH",
  "backbone_model": "$BACKBONE_MODEL",
  "backbone_api_url": "$BACKBONE_API_URL",
  "backbone_judge_api_url": "$BACKBONE_JUDGE_API_URL",
  "backbone_judge_model": "$BACKBONE_JUDGE_MODEL",
  "backbone_cuda_visible_devices": "$BACKBONE_CUDA_VISIBLE_DEVICES",
  "backbone_tensor_parallel_size": $BACKBONE_TENSOR_PARALLEL_SIZE,
  "backbone_gpu_memory_utilization": $BACKBONE_GPU_MEMORY_UTILIZATION,
  "policy_model": "$POLICY_MODEL",
  "policy_api_url": "$POLICY_API_URL",
  "policy_env_file": "$POLICY_ENV_FILE",
  "policy_api_key_env_var": "$POLICY_API_KEY_ENV_VAR",
  "output_dir": "$OUT_DIR"
}
EOF

echo "============================================"
echo "Qwen3-32B backbone + DeepSeek reasoner policy test_all rollout"
echo "Input:              $INPUT"
echo "Samples:            $VAL_MAX_SAMPLES"
echo "Backbone:           $BACKBONE_MODEL @ $BACKBONE_API_URL"
echo "Backbone judge:     $BACKBONE_JUDGE_MODEL @ $BACKBONE_JUDGE_API_URL"
echo "Policy:             $POLICY_MODEL @ $POLICY_API_URL"
echo "Output dir:         $OUT_DIR"
echo "Backbone vLLM log:  $BACKBONE_VLLM_LOG"
echo "============================================"

INPUT="$INPUT" \
OUTPUT_ROOT="$EXP_DIR" \
RUN_NAME="run1" \
OUT_DIR="$OUT_DIR" \
BACKBONE_ENV_FILE="$BACKBONE_ENV_FILE" \
BACKBONE_API_URL="$BACKBONE_API_URL" \
BACKBONE_MODEL="$BACKBONE_MODEL" \
BACKBONE_JUDGE_ENV_FILE="$BACKBONE_JUDGE_ENV_FILE" \
BACKBONE_JUDGE_API_URL="$BACKBONE_JUDGE_API_URL" \
BACKBONE_JUDGE_MODEL="$BACKBONE_JUDGE_MODEL" \
POLICY_ENV_FILE="$POLICY_ENV_FILE" \
POLICY_API_KEY_ENV_VAR="$POLICY_API_KEY_ENV_VAR" \
POLICY_API_URL="$POLICY_API_URL" \
POLICY_MODEL="$POLICY_MODEL" \
VAL_MAX_SAMPLES="$VAL_MAX_SAMPLES" \
VAL_OFFSET="$VAL_OFFSET" \
NUM_WORKERS="$NUM_WORKERS" \
MAX_PARALLEL_POLICY_QUERIES="$MAX_PARALLEL_POLICY_QUERIES" \
MAX_BACKBONE_SEARCH_QUERIES="$MAX_BACKBONE_SEARCH_QUERIES" \
MAX_ORCHESTRATOR_ROUNDS="$MAX_ORCHESTRATOR_ROUNDS" \
MAX_ASSISTANT_TURNS="$MAX_ASSISTANT_TURNS" \
MAX_PARALLEL_CALLS="$MAX_PARALLEL_CALLS" \
RETRIEVAL_URL="$RETRIEVAL_URL" \
RETRIEVAL_MAX_CONCURRENT="$RETRIEVAL_MAX_CONCURRENT" \
RETRIEVAL_TIMEOUT="$RETRIEVAL_TIMEOUT" \
API_MAX_RETRIES="$API_MAX_RETRIES" \
API_TIMEOUT="$API_TIMEOUT" \
TOPK="$TOPK" \
POLICY_TEMPERATURE="$POLICY_TEMPERATURE" \
BACKBONE_TEMPERATURE="$BACKBONE_TEMPERATURE" \
POLICY_MAX_TOKENS="$POLICY_MAX_TOKENS" \
BACKBONE_MAX_TOKENS="$BACKBONE_MAX_TOKENS" \
BACKBONE_JUDGE_MAX_TOKENS="$BACKBONE_JUDGE_MAX_TOKENS" \
NO_PROXY="$NO_PROXY" \
RESUME="$RESUME" \
SAVE_RAW_API_RESPONSE="$SAVE_RAW_API_RESPONSE" \
SAVE_RAW_RETRIEVAL_RESPONSE="$SAVE_RAW_RETRIEVAL_RESPONSE" \
POLICY_ENABLE_THINKING="$POLICY_ENABLE_THINKING" \
POLICY_PRESERVE_REASONING_CONTENT="$POLICY_PRESERVE_REASONING_CONTENT" \
PYTHON_BIN="$PYTHON_BIN" \
bash "$SCRIPT_DIR/run_generate_sft_rollout_val900_api_policy.sh" "$@"

echo "Done. Summary: $OUT_DIR/summary.json"
