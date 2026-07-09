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
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "/ai/cqn/miniconda3/envs/verl/bin/python" ]; then
    PYTHON_BIN="/ai/cqn/miniconda3/envs/verl/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
INPUT="${INPUT:-$PROJECT_DIR/data/hotpotqa_2wiki_musique_train/test_all.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/ai/cqn/s3/ckpt/search_subagent_api_policy_test_all_mean4}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-test_all_mean4_policy_temp06_${RUN_TS}}"
EXP_DIR="${EXP_DIR:-$OUTPUT_ROOT/$EXPERIMENT_NAME}"
REPEATS="${REPEATS:-4}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-4125}"
VAL_OFFSET="${VAL_OFFSET:-0}"

BACKBONE_ENV_FILE="${BACKBONE_ENV_FILE:-.secrets/deepseek.env}"
BACKBONE_API_URL="${BACKBONE_API_URL:-https://api.deepseek.com/v1}"
BACKBONE_MODEL="${BACKBONE_MODEL:-deepseek-reasoner}"
BACKBONE_MAX_TOKENS="${BACKBONE_MAX_TOKENS:-8192}"
BACKBONE_JUDGE_MAX_TOKENS="${BACKBONE_JUDGE_MAX_TOKENS:-4096}"
BACKBONE_TEMPERATURE="${BACKBONE_TEMPERATURE:-0.0}"

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
SAVE_RAW_API_RESPONSE="${SAVE_RAW_API_RESPONSE:-false}"
SAVE_RAW_RETRIEVAL_RESPONSE="${SAVE_RAW_RETRIEVAL_RESPONSE:-false}"
RESUME="${RESUME:-false}"

SFT_POLICY_API_URL="${SFT_POLICY_API_URL:-http://127.0.0.1:8010/v1}"
GRPO_POLICY_API_URL="${GRPO_POLICY_API_URL:-http://127.0.0.1:8011/v1}"
NAIVE_POLICY_API_URL="${NAIVE_POLICY_API_URL:-http://127.0.0.1:8009/v1}"
SFT_CUDA_VISIBLE_DEVICES="${SFT_CUDA_VISIBLE_DEVICES:-0}"
GRPO_CUDA_VISIBLE_DEVICES="${GRPO_CUDA_VISIBLE_DEVICES:-0}"
NAIVE_CUDA_VISIBLE_DEVICES="${NAIVE_CUDA_VISIBLE_DEVICES:-1}"
LOCAL_VLLM_HOST="${LOCAL_VLLM_HOST:-0.0.0.0}"
SFT_VLLM_PORT="${SFT_VLLM_PORT:-8010}"
GRPO_VLLM_PORT="${GRPO_VLLM_PORT:-8011}"
NAIVE_VLLM_PORT="${NAIVE_VLLM_PORT:-8009}"
LOCAL_POLICY_API_URL=""
LOCAL_VLLM_PORT=""
START_LOCAL_VLLM="${START_LOCAL_VLLM:-true}"
VLLM_WAIT_SECONDS="${VLLM_WAIT_SECONDS:-900}"
VLLM_DTYPE="${VLLM_DTYPE:-bfloat16}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.25}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}"
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"

SFT_POLICY_PATH="${SFT_POLICY_PATH:-/ai/cqn/s3/ckpt/search_subagent_policy_sft/qwen3_1p7b_policy_sft_filtered_em1_1893_20260514_022426/merged_hf_global_step_236}"
SFT_POLICY_MODEL="${SFT_POLICY_MODEL:-qwen3_1p7b_policy_sft}"
GRPO_POLICY_PATH="${GRPO_POLICY_PATH:-/ai/cqn/s3/ckpt/search_subagent_grpo_fully_async/activaterl_from_qwen3_1p7b_policy_sft_step236_20260515_015836/merged_hf_global_step_50}"
GRPO_POLICY_MODEL="${GRPO_POLICY_MODEL:-qwen3_1p7b_policy_grpo_step100}"
NAIVE_POLICY_PATH="${NAIVE_POLICY_PATH:-/ai/cqn/model/Qwen3-1.7B}"
NAIVE_POLICY_MODEL="${NAIVE_POLICY_MODEL:-qwen3_1p7b_naive}"
DEEPSEEK_POLICY_MODEL="${DEEPSEEK_POLICY_MODEL:-deepseek-reasoner}"
DEEPSEEK_POLICY_API_URL="${DEEPSEEK_POLICY_API_URL:-$BACKBONE_API_URL}"
DEEPSEEK_POLICY_ENV_FILE="${DEEPSEEK_POLICY_ENV_FILE:-$BACKBONE_ENV_FILE}"

POLICIES="${POLICIES:-sft_local grpo_local naive_policy_api deepseek_reasoner}"

mkdir -p "$EXP_DIR"

VLLM_PID=""
cleanup_vllm() {
  if [ -n "${VLLM_PID:-}" ]; then
    if kill -0 "$VLLM_PID" >/dev/null 2>&1; then
      echo "Stopping vLLM pid=$VLLM_PID"
      kill "$VLLM_PID" >/dev/null 2>&1 || true
      wait "$VLLM_PID" >/dev/null 2>&1 || true
    fi
    VLLM_PID=""
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
  local log_path="$3"
  local cuda_devices="$4"

  cleanup_vllm
  mkdir -p "$(dirname "$log_path")"
  echo "Starting vLLM: $served_name from $model_path on port $LOCAL_VLLM_PORT (CUDA_VISIBLE_DEVICES=$cuda_devices)"
  CUDA_VISIBLE_DEVICES="$cuda_devices" PYTHONUNBUFFERED=1 "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --host "$LOCAL_VLLM_HOST" \
    --port "$LOCAL_VLLM_PORT" \
    --model "$model_path" \
    --served-model-name "$served_name" \
    --dtype "$VLLM_DTYPE" \
    --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --max-model-len "$VLLM_MAX_MODEL_LEN" \
    --max-num-seqs "$VLLM_MAX_NUM_SEQS" \
    --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
    --enable-prefix-caching \
    --disable-log-requests \
    >"$log_path" 2>&1 &
  VLLM_PID="$!"
  wait_for_local_server "$LOCAL_POLICY_API_URL"
}

run_one_repeat() {
  local policy_label="$1"
  local policy_api_url="$2"
  local policy_model="$3"
  local policy_env_file="$4"
  local repeat_id="$5"
  local out_dir="$EXP_DIR/$policy_label/repeat${repeat_id}"

  echo "============================================"
  echo "Running $policy_label repeat $repeat_id/$REPEATS"
  echo "Output: $out_dir"
  echo "Policy: $policy_model @ $policy_api_url"
  echo "============================================"

  INPUT="$INPUT" \
  OUTPUT_ROOT="$EXP_DIR/$policy_label" \
  RUN_NAME="repeat${repeat_id}" \
  OUT_DIR="$out_dir" \
  BACKBONE_ENV_FILE="$BACKBONE_ENV_FILE" \
  BACKBONE_API_URL="$BACKBONE_API_URL" \
  BACKBONE_MODEL="$BACKBONE_MODEL" \
  POLICY_ENV_FILE="$policy_env_file" \
  POLICY_API_URL="$policy_api_url" \
  POLICY_MODEL="$policy_model" \
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
  bash "$SCRIPT_DIR/run_generate_sft_rollout_val900_api_policy.sh"
}

aggregate_policy() {
  local policy_label="$1"
  "$PYTHON_BIN" - "$EXP_DIR/$policy_label" "$REPEATS" <<'PY'
import json
import statistics
import sys
from pathlib import Path
from collections import Counter

policy_dir = Path(sys.argv[1])
repeats = int(sys.argv[2])
summary_paths = [policy_dir / f"repeat{i}" / "summary.json" for i in range(1, repeats + 1)]
summaries = []
for path in summary_paths:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as f:
        summaries.append(json.load(f))

def numeric_values(key):
    return [float(s[key]) for s in summaries if s.get(key) is not None]

def mean_or_none(values):
    return statistics.fmean(values) if values else None

def stdev_or_none(values):
    return statistics.pstdev(values) if len(values) > 1 else 0.0 if values else None

def stage_usage(summary, stage):
    usage = summary.get("token_usage_by_stage", {}).get(stage, {})
    return {
        "call_count": int(usage.get("call_count", 0) or 0),
        "input_tokens": int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
        "output_tokens": int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
        "total_tokens": int(usage.get("total_tokens", 0) or 0),
    }

stages = sorted({stage for s in summaries for stage in s.get("token_usage_by_stage", {})})
token_usage_mean4 = {}
for stage in stages:
    per_repeat = [stage_usage(s, stage) for s in summaries]
    token_usage_mean4[stage] = {
        "total_over_repeats": {
            key: sum(item[key] for item in per_repeat)
            for key in ("call_count", "input_tokens", "output_tokens", "total_tokens")
        },
        "mean_per_repeat": {
            key: mean_or_none([item[key] for item in per_repeat])
            for key in ("call_count", "input_tokens", "output_tokens", "total_tokens")
        },
        "per_repeat": per_repeat,
    }

source_counts = Counter()
error_count_total = 0
for s in summaries:
    source_counts.update(s.get("final_answer_source_counts", {}))
    error_count_total += int(s.get("error_count", 0) or 0)

metric_keys = [
    "final_em_mean",
    "final_f1_mean",
    "backbone_final_answer_llm_judge_score_mean",
    "backbone_final_answer_llm_judge_scored_count",
    "policy_round_count_mean",
    "elapsed_seconds_mean",
    "error_count",
]
metrics = {}
for key in metric_keys:
    values = numeric_values(key)
    metrics[key] = {
        "mean_at_4": mean_or_none(values),
        "std_at_4": stdev_or_none(values),
        "per_repeat": values,
    }

result = {
    "policy_dir": str(policy_dir),
    "repeat_count": repeats,
    "summary_paths": [str(p) for p in summary_paths],
    "sample_count_per_repeat": [s.get("sample_count") for s in summaries],
    "metrics": metrics,
    "final_answer_source_counts_total_over_repeats": dict(source_counts),
    "final_answer_source_counts_mean_per_repeat": {
        key: value / repeats for key, value in sorted(source_counts.items())
    },
    "error_count_total_over_repeats": error_count_total,
    "token_usage_mean4": token_usage_mean4,
}
output_path = policy_dir / "mean4_summary.json"
with output_path.open("w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print(json.dumps(result, ensure_ascii=False, indent=2))
PY
}

aggregate_all() {
  "$PYTHON_BIN" - "$EXP_DIR" $POLICIES <<'PY'
import json
import sys
from pathlib import Path

exp_dir = Path(sys.argv[1])
policy_labels = sys.argv[2:]
combined = {"experiment_dir": str(exp_dir), "policies": {}}
for label in policy_labels:
    path = exp_dir / label / "mean4_summary.json"
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            combined["policies"][label] = json.load(f)
combined_path = exp_dir / "all_policies_mean4_summary.json"
with combined_path.open("w", encoding="utf-8") as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)
print(json.dumps(combined, ensure_ascii=False, indent=2))
PY
}

run_policy() {
  local policy_label="$1"
  local policy_api_url
  local policy_model
  local policy_env_file

  case "$policy_label" in
    sft_local)
      policy_api_url="$SFT_POLICY_API_URL"
      policy_model="$SFT_POLICY_MODEL"
      LOCAL_POLICY_API_URL="$SFT_POLICY_API_URL"
      LOCAL_VLLM_PORT="$SFT_VLLM_PORT"
      policy_env_file="$BACKBONE_ENV_FILE"
      if [ "$START_LOCAL_VLLM" = "true" ]; then
        start_vllm_server "$SFT_POLICY_PATH" "$SFT_POLICY_MODEL" "$EXP_DIR/$policy_label/vllm_${SFT_VLLM_PORT}.log" "$SFT_CUDA_VISIBLE_DEVICES"
      else
        wait_for_local_server "$policy_api_url"
      fi
      ;;
    grpo_local)
      policy_api_url="$GRPO_POLICY_API_URL"
      policy_model="$GRPO_POLICY_MODEL"
      LOCAL_POLICY_API_URL="$GRPO_POLICY_API_URL"
      LOCAL_VLLM_PORT="$GRPO_VLLM_PORT"
      policy_env_file="$BACKBONE_ENV_FILE"
      if [ "$START_LOCAL_VLLM" = "true" ]; then
        start_vllm_server "$GRPO_POLICY_PATH" "$GRPO_POLICY_MODEL" "$EXP_DIR/$policy_label/vllm_${GRPO_VLLM_PORT}.log" "$GRPO_CUDA_VISIBLE_DEVICES"
      else
        wait_for_local_server "$policy_api_url"
      fi
      ;;
    naive_policy_api)
      policy_api_url="$NAIVE_POLICY_API_URL"
      policy_model="$NAIVE_POLICY_MODEL"
      LOCAL_POLICY_API_URL="$NAIVE_POLICY_API_URL"
      LOCAL_VLLM_PORT="$NAIVE_VLLM_PORT"
      policy_env_file="$BACKBONE_ENV_FILE"
      if [ "$START_LOCAL_VLLM" = "true" ]; then
        start_vllm_server "$NAIVE_POLICY_PATH" "$NAIVE_POLICY_MODEL" "$EXP_DIR/$policy_label/vllm_${NAIVE_VLLM_PORT}.log" "$NAIVE_CUDA_VISIBLE_DEVICES"
      else
        wait_for_local_server "$policy_api_url"
      fi
      ;;
    deepseek_reasoner)
      cleanup_vllm
      policy_api_url="$DEEPSEEK_POLICY_API_URL"
      policy_model="$DEEPSEEK_POLICY_MODEL"
      policy_env_file="$DEEPSEEK_POLICY_ENV_FILE"
      ;;
    *)
      echo "Unknown policy label: $policy_label" >&2
      exit 2
      ;;
  esac

  for repeat_id in $(seq 1 "$REPEATS"); do
    run_one_repeat "$policy_label" "$policy_api_url" "$policy_model" "$policy_env_file" "$repeat_id"
  done
  aggregate_policy "$policy_label"

  if [ "$policy_label" = "sft_local" ] || [ "$policy_label" = "grpo_local" ]; then
    cleanup_vllm
  fi
}

cat >"$EXP_DIR/experiment_config.json" <<EOF
{
  "input": "$INPUT",
  "repeats": $REPEATS,
  "val_max_samples": $VAL_MAX_SAMPLES,
  "backbone_model": "$BACKBONE_MODEL",
  "backbone_max_tokens": $BACKBONE_MAX_TOKENS,
  "backbone_judge_max_tokens": $BACKBONE_JUDGE_MAX_TOKENS,
  "policy_temperature": $POLICY_TEMPERATURE,
  "policy_max_tokens": $POLICY_MAX_TOKENS,
  "policies": "$POLICIES",
  "sft_policy_path": "$SFT_POLICY_PATH",
  "grpo_policy_path": "$GRPO_POLICY_PATH",
  "naive_policy_path": "$NAIVE_POLICY_PATH",
  "sft_policy_api_url": "$SFT_POLICY_API_URL",
  "grpo_policy_api_url": "$GRPO_POLICY_API_URL",
  "naive_policy_api_url": "$NAIVE_POLICY_API_URL",
  "sft_cuda_visible_devices": "$SFT_CUDA_VISIBLE_DEVICES",
  "grpo_cuda_visible_devices": "$GRPO_CUDA_VISIBLE_DEVICES",
  "naive_cuda_visible_devices": "$NAIVE_CUDA_VISIBLE_DEVICES",
  "output_dir": "$EXP_DIR"
}
EOF

for policy_label in $POLICIES; do
  run_policy "$policy_label"
done
aggregate_all

echo "Done. Combined mean@4 summary: $EXP_DIR/all_policies_mean4_summary.json"
