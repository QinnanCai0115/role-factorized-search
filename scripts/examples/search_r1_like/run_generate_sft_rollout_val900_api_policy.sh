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
INPUT="${INPUT:-$PROJECT_DIR/data/hotpotqa_2wiki_musique_train/val_mixed_900.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/ai/cqn/s3/ckpt/search_subagent_api_policy_val900}"
RUN_NAME="${RUN_NAME:-deepseek_reasoner_policy_val900_${RUN_TS}}"
OUT_DIR="${OUT_DIR:-$OUTPUT_ROOT/$RUN_NAME}"
POLICY_ROUNDS_JSONL="${POLICY_ROUNDS_JSONL:-$OUT_DIR/policy_rounds.jsonl}"
ORCHESTRATOR_TRACES_JSONL="${ORCHESTRATOR_TRACES_JSONL:-$OUT_DIR/orchestrator_traces.jsonl}"
RUN_LOG="${RUN_LOG:-$OUT_DIR/run.log}"
SUMMARY_JSON="${SUMMARY_JSON:-$OUT_DIR/summary.json}"

BACKBONE_ENV_FILE="${BACKBONE_ENV_FILE:-.secrets/deepseek.env}"
BACKBONE_API_URL="${BACKBONE_API_URL:-https://api.deepseek.com/v1}"
BACKBONE_MODEL="${BACKBONE_MODEL:-deepseek-reasoner}"
BACKBONE_JUDGE_ENV_FILE="${BACKBONE_JUDGE_ENV_FILE:-$BACKBONE_ENV_FILE}"
BACKBONE_JUDGE_API_URL="${BACKBONE_JUDGE_API_URL:-https://api.deepseek.com/v1}"
BACKBONE_JUDGE_MODEL="${BACKBONE_JUDGE_MODEL:-deepseek-reasoner}"
POLICY_ENV_FILE="${POLICY_ENV_FILE:-$BACKBONE_ENV_FILE}"
POLICY_API_KEY_ENV_VAR="${POLICY_API_KEY_ENV_VAR:-POLICY_API_KEY}"
POLICY_API_URL="${POLICY_API_URL:-$BACKBONE_API_URL}"
POLICY_MODEL="${POLICY_MODEL:-$BACKBONE_MODEL}"

VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-900}"
VAL_OFFSET="${VAL_OFFSET:-0}"
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
API_TIMEOUT="${API_TIMEOUT:-180}"
TOPK="${TOPK:-3}"
POLICY_TEMPERATURE="${POLICY_TEMPERATURE:-0.2}"
BACKBONE_TEMPERATURE="${BACKBONE_TEMPERATURE:-0.0}"
POLICY_MAX_TOKENS="${POLICY_MAX_TOKENS:-4096}"
BACKBONE_MAX_TOKENS="${BACKBONE_MAX_TOKENS:-4096}"
BACKBONE_JUDGE_MAX_TOKENS="${BACKBONE_JUDGE_MAX_TOKENS:-16}"
POLICY_EXTRA_BODY_JSON="${POLICY_EXTRA_BODY_JSON:-}"
NO_PROXY="${NO_PROXY:-true}"
RESUME="${RESUME:-true}"
SAVE_RAW_API_RESPONSE="${SAVE_RAW_API_RESPONSE:-false}"
SAVE_RAW_RETRIEVAL_RESPONSE="${SAVE_RAW_RETRIEVAL_RESPONSE:-false}"
POLICY_ENABLE_THINKING="${POLICY_ENABLE_THINKING:-false}"
POLICY_PRESERVE_REASONING_CONTENT="${POLICY_PRESERVE_REASONING_CONTENT:-false}"

mkdir -p "$OUT_DIR"

echo "============================================"
echo "API-policy two-stage val900 rollout"
echo "Input:                 $INPUT"
echo "Val max samples:       $VAL_MAX_SAMPLES"
echo "Val offset:            $VAL_OFFSET"
echo "Backbone model:        $BACKBONE_MODEL @ $BACKBONE_API_URL"
echo "Backbone judge:        $BACKBONE_JUDGE_MODEL @ $BACKBONE_JUDGE_API_URL"
echo "Policy model:          $POLICY_MODEL @ $POLICY_API_URL"
echo "Backbone max tokens:   $BACKBONE_MAX_TOKENS"
echo "Judge max tokens:      $BACKBONE_JUDGE_MAX_TOKENS"
echo "Policy thinking body:  $POLICY_ENABLE_THINKING"
echo "Preserve reasoning:    $POLICY_PRESERVE_REASONING_CONTENT"
echo "Retrieval URL:         $RETRIEVAL_URL"
echo "Workers:               $NUM_WORKERS"
echo "Output dir:            $OUT_DIR"
echo "Policy rounds JSONL:   $POLICY_ROUNDS_JSONL"
echo "Orchestrator JSONL:    $ORCHESTRATOR_TRACES_JSONL"
echo "Run log:               $RUN_LOG"
echo "No verl/Ray/vLLM/GPU will be started."
echo "============================================"

ARGS=(
  "$SCRIPT_DIR/generate_sft_rollout.py"
  --input "$INPUT"
  --output "$POLICY_ROUNDS_JSONL"
  --orchestrator_output "$ORCHESTRATOR_TRACES_JSONL"
  --env_file "$BACKBONE_ENV_FILE"
  --policy_env_file "$POLICY_ENV_FILE"
  --policy_api_key_env_var "$POLICY_API_KEY_ENV_VAR"
  --backbone_judge_env_file "$BACKBONE_JUDGE_ENV_FILE"
  --api_url "$BACKBONE_API_URL"
  --policy_api_url "$POLICY_API_URL"
  --backbone_judge_api_url "$BACKBONE_JUDGE_API_URL"
  --backbone_model "$BACKBONE_MODEL"
  --policy_model "$POLICY_MODEL"
  --backbone_judge_model "$BACKBONE_JUDGE_MODEL"
  --temperature "$POLICY_TEMPERATURE"
  --backbone_temperature "$BACKBONE_TEMPERATURE"
  --max_tokens "$POLICY_MAX_TOKENS"
  --backbone_max_tokens "$BACKBONE_MAX_TOKENS"
  --backbone_judge_max_tokens "$BACKBONE_JUDGE_MAX_TOKENS"
  --policy_extra_body_json "$POLICY_EXTRA_BODY_JSON"
  --retrieval_url "$RETRIEVAL_URL"
  --topk "$TOPK"
  --retrieval_timeout "$RETRIEVAL_TIMEOUT"
  --retrieval_max_concurrent "$RETRIEVAL_MAX_CONCURRENT"
  --api_max_retries "$API_MAX_RETRIES"
  --api_timeout "$API_TIMEOUT"
  --max_orchestrator_rounds "$MAX_ORCHESTRATOR_ROUNDS"
  --max_assistant_turns "$MAX_ASSISTANT_TURNS"
  --max_parallel_policy_queries "$MAX_PARALLEL_POLICY_QUERIES"
  --max_backbone_search_queries "$MAX_BACKBONE_SEARCH_QUERIES"
  --max_parallel_calls "$MAX_PARALLEL_CALLS"
  --num_workers "$NUM_WORKERS"
  --limit "$VAL_MAX_SAMPLES"
  --offset "$VAL_OFFSET"
)

if [ "$POLICY_ENABLE_THINKING" = "true" ]; then
  ARGS+=(
    --policy_enable_thinking
    --policy_thinking_field thinking
    --policy_thinking_type enabled
  )
else
  ARGS+=(--no-policy_enable_thinking)
fi

if [ "$POLICY_PRESERVE_REASONING_CONTENT" = "true" ]; then
  ARGS+=(--policy_preserve_reasoning_content)
else
  ARGS+=(--no-policy_preserve_reasoning_content)
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

if [ "$SAVE_RAW_API_RESPONSE" = "true" ]; then
  ARGS+=(--save_raw_api_response)
fi

if [ "$SAVE_RAW_RETRIEVAL_RESPONSE" = "true" ]; then
  ARGS+=(--save_raw_retrieval_response)
fi

PYTHONUNBUFFERED=1 "$PYTHON_BIN" "${ARGS[@]}" "$@" 2>&1 | tee "$RUN_LOG"

"$PYTHON_BIN" - "$ORCHESTRATOR_TRACES_JSONL" "$SUMMARY_JSON" <<'PY'
import json
import statistics
import sys
from collections import Counter
from collections import defaultdict

trace_path, summary_path = sys.argv[1], sys.argv[2]
rows = []
with open(trace_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

def values(name):
    return [float(row[name]) for row in rows if row.get(name) is not None]

ems = values("final_em")
f1s = values("final_f1")
llm_judge_scores = values("backbone_final_answer_llm_judge_score")
elapsed = values("elapsed_seconds")
round_counts = [int(row.get("policy_round_count") or 0) for row in rows]
errors = [row.get("error") for row in rows if row.get("error")]
sources = Counter(str(row.get("final_answer_source") or "<none>") for row in rows)

def empty_usage():
    return {"call_count": 0, "by_model": {}}

def add_usage(dst, usage, model=None):
    if not isinstance(usage, dict):
        return
    dst["call_count"] = int(dst.get("call_count", 0)) + 1
    for key, value in usage.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        dst[key] = dst.get(key, 0) + value
    if model:
        by_model = dst.setdefault("by_model", {})
        model_usage = by_model.setdefault(str(model), {"call_count": 0})
        model_usage["call_count"] = int(model_usage.get("call_count", 0)) + 1
        for key, value in usage.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            model_usage[key] = model_usage.get(key, 0) + value

def add_aliases(usage):
    if not isinstance(usage, dict):
        return usage
    if "input_tokens" not in usage and "prompt_tokens" in usage:
        usage["input_tokens"] = usage["prompt_tokens"]
    if "output_tokens" not in usage and "completion_tokens" in usage:
        usage["output_tokens"] = usage["completion_tokens"]
    for model_usage in usage.get("by_model", {}).values():
        if not isinstance(model_usage, dict):
            continue
        if "input_tokens" not in model_usage and "prompt_tokens" in model_usage:
            model_usage["input_tokens"] = model_usage["prompt_tokens"]
        if "output_tokens" not in model_usage and "completion_tokens" in model_usage:
            model_usage["output_tokens"] = model_usage["completion_tokens"]
    return usage

token_usage_by_stage = defaultdict(empty_usage)
for row in rows:
    for call in row.get("api_call_stats", []):
        if not isinstance(call, dict):
            continue
        stage = str(call.get("stage") or "<unknown>")
        model = call.get("model")
        add_usage(token_usage_by_stage[stage], call.get("usage", {}), model)

token_usage_by_stage = {
    stage: add_aliases(usage)
    for stage, usage in sorted(token_usage_by_stage.items())
}
summary = {
    "trace_path": trace_path,
    "sample_count": len(rows),
    "scored_count": len(ems),
    "final_em_mean": statistics.fmean(ems) if ems else None,
    "final_f1_mean": statistics.fmean(f1s) if f1s else None,
    "backbone_final_answer_llm_judge_score_mean": statistics.fmean(llm_judge_scores) if llm_judge_scores else None,
    "backbone_final_answer_llm_judge_scored_count": len(llm_judge_scores),
    "policy_round_count_mean": statistics.fmean(round_counts) if round_counts else None,
    "elapsed_seconds_mean": statistics.fmean(elapsed) if elapsed else None,
    "error_count": len(errors),
    "final_answer_source_counts": dict(sources),
    "token_usage_by_stage": token_usage_by_stage,
    "main_agent_token_usage": token_usage_by_stage.get("backbone", empty_usage()),
    "subagent_token_usage": token_usage_by_stage.get("policy", empty_usage()),
}
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

echo "Done. Check:"
echo "  $SUMMARY_JSON"
echo "  $ORCHESTRATOR_TRACES_JSONL"
echo "  $POLICY_ROUNDS_JSONL"
echo "  $RUN_LOG"
