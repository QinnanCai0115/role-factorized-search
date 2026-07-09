#!/usr/bin/env bash
set -euo pipefail

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

SOURCE_TRACES="${SOURCE_TRACES:-$PROJECT_DIR/data/deepseek_policy_sft_rollouts/train_mixed_2000.deepseek_v4_pro.multi_query.backbone_prompt_v2.traces.jsonl}"
ORIGINAL_INPUT="${ORIGINAL_INPUT:-$PROJECT_DIR/data/hotpotqa_2wiki_musique_train/train_mixed_2000_sft.jsonl}"
RUN_NAME="${RUN_NAME:-train_mixed_2000.deepseek_v4_pro.multi_query.backbone_prompt_v2.rerun_abnormal}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/data/deepseek_policy_sft_rollouts}"
OUT_DIR="${OUT_DIR:-$OUTPUT_ROOT/$RUN_NAME}"
RERUN_INPUT="${RERUN_INPUT:-$OUT_DIR/abnormal_input.jsonl}"
POLICY_ROUNDS_JSONL="${POLICY_ROUNDS_JSONL:-$OUT_DIR/policy_rounds.jsonl}"
ORCHESTRATOR_TRACES_JSONL="${ORCHESTRATOR_TRACES_JSONL:-$OUT_DIR/orchestrator_traces.jsonl}"
RUN_LOG="${RUN_LOG:-$OUT_DIR/run.log}"
SUMMARY_JSON="${SUMMARY_JSON:-$OUT_DIR/summary.json}"

BACKBONE_ENV_FILE="${BACKBONE_ENV_FILE:-$PROJECT_DIR/.secrets/deepseek.env}"
BACKBONE_API_URL="${BACKBONE_API_URL:-https://api.deepseek.com/v1}"
BACKBONE_MODEL="${BACKBONE_MODEL:-deepseek-reasoner}"
BACKBONE_JUDGE_ENV_FILE="${BACKBONE_JUDGE_ENV_FILE:-$BACKBONE_ENV_FILE}"
BACKBONE_JUDGE_API_URL="${BACKBONE_JUDGE_API_URL:-$BACKBONE_API_URL}"
BACKBONE_JUDGE_MODEL="${BACKBONE_JUDGE_MODEL:-deepseek-reasoner}"
POLICY_ENV_FILE="${POLICY_ENV_FILE:-$BACKBONE_ENV_FILE}"
POLICY_API_URL="${POLICY_API_URL:-$BACKBONE_API_URL}"
POLICY_MODEL="${POLICY_MODEL:-deepseek-v4-pro}"

NUM_WORKERS="${NUM_WORKERS:-8}"
MAX_PARALLEL_POLICY_QUERIES="${MAX_PARALLEL_POLICY_QUERIES:-3}"
MAX_BACKBONE_SEARCH_QUERIES="${MAX_BACKBONE_SEARCH_QUERIES:-3}"
MAX_ORCHESTRATOR_ROUNDS="${MAX_ORCHESTRATOR_ROUNDS:-4}"
MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-3}"
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
NO_PROXY="${NO_PROXY:-true}"
RESUME="${RESUME:-true}"

mkdir -p "$OUT_DIR"

"$PYTHON_BIN" - "$SOURCE_TRACES" "$ORIGINAL_INPUT" "$RERUN_INPUT" <<'PY'
import json
import sys

source_traces, original_input, rerun_input = sys.argv[1:4]

abnormal = set()
with open(source_traces, "r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        row = json.loads(line)
        final_answer = str(row.get("final_answer") or "").strip()
        final_source = row.get("final_answer_source")
        policy_round_count = int(row.get("policy_round_count") or 0)
        chain = row.get("orchestrator_chain") or []
        first_response = str(chain[0].get("response", "")) if chain and isinstance(chain[0], dict) else ""
        if (
            not final_answer
            or final_source == "policy"
            or final_answer.lower().startswith("<search")
            or (policy_round_count == 0 and "<search" in first_response.lower())
        ):
            abnormal.add(int(row["source_index"]))

written = 0
with open(original_input, "r", encoding="utf-8") as src, open(rerun_input, "w", encoding="utf-8") as out:
    for idx, line in enumerate(src):
        if idx not in abnormal:
            continue
        row = json.loads(line)
        row["source_index"] = idx
        row["__source_index"] = idx
        out.write(json.dumps(row, ensure_ascii=False) + "\n")
        written += 1

print(f"abnormal_source_count={len(abnormal)}")
print(f"rerun_input_rows={written}")
print(f"rerun_input={rerun_input}")
PY

echo "============================================"
echo "DeepSeek-v4-pro abnormal rerun"
echo "Input:              $RERUN_INPUT"
echo "Backbone model:     $BACKBONE_MODEL @ $BACKBONE_API_URL"
echo "Policy model:       $POLICY_MODEL @ $POLICY_API_URL"
echo "Output dir:         $OUT_DIR"
echo "Policy rounds:      $POLICY_ROUNDS_JSONL"
echo "Traces:             $ORCHESTRATOR_TRACES_JSONL"
echo "Run log:            $RUN_LOG"
echo "============================================"

ARGS=(
  "$SCRIPT_DIR/generate_sft_rollout.py"
  --input "$RERUN_INPUT"
  --output "$POLICY_ROUNDS_JSONL"
  --orchestrator_output "$ORCHESTRATOR_TRACES_JSONL"
  --env_file "$BACKBONE_ENV_FILE"
  --policy_env_file "$POLICY_ENV_FILE"
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
  --no-policy_enable_thinking
  --no-policy_preserve_reasoning_content
)

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

PYTHONUNBUFFERED=1 "$PYTHON_BIN" "${ARGS[@]}" "$@" 2>&1 | tee "$RUN_LOG"

"$PYTHON_BIN" - "$ORCHESTRATOR_TRACES_JSONL" "$SUMMARY_JSON" <<'PY'
import json
import statistics
import sys
from collections import Counter

trace_path, summary_path = sys.argv[1], sys.argv[2]
rows = []
with open(trace_path, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))

ems = [float(r["final_em"]) for r in rows if r.get("final_em") is not None]
f1s = [float(r["final_f1"]) for r in rows if r.get("final_f1") is not None]
judges = [
    float(r["backbone_final_answer_llm_judge_score"])
    for r in rows
    if r.get("backbone_final_answer_llm_judge_score") is not None
]
summary = {
    "trace_path": trace_path,
    "sample_count": len(rows),
    "scored_count": len(ems),
    "final_em_mean": statistics.fmean(ems) if ems else None,
    "final_f1_mean": statistics.fmean(f1s) if f1s else None,
    "backbone_final_answer_llm_judge_score_mean": statistics.fmean(judges) if judges else None,
    "backbone_final_answer_llm_judge_scored_count": len(judges),
    "policy_round_count_mean": statistics.fmean(float(r.get("policy_round_count") or 0) for r in rows) if rows else None,
    "error_count": sum(1 for r in rows if not str(r.get("final_answer") or "").strip()),
    "final_answer_source_counts": dict(Counter(str(r.get("final_answer_source") or "<none>") for r in rows)),
}
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
    f.write("\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
