#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"

# Roll out the untrained Qwen3-1.7B policy that is already served on port 8001.
# This wrapper reuses the generic API-policy rollout script and does not start vLLM.

export INPUT="${INPUT:-$PROJECT_DIR/data/hotpotqa_2wiki_musique_train/train_mixed_2000_sft.jsonl}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PROJECT_DIR/data/qwen3_policy_sft_rollouts}"
export RUN_NAME="${RUN_NAME:-train_mixed_2000.qwen3_1p7b_naive.multi_query.backbone_prompt_v2}"

export BACKBONE_ENV_FILE="${BACKBONE_ENV_FILE:-$PROJECT_DIR/.secrets/deepseek.env}"
export BACKBONE_API_URL="${BACKBONE_API_URL:-https://api.deepseek.com/v1}"
export BACKBONE_MODEL="${BACKBONE_MODEL:-deepseek-reasoner}"
export BACKBONE_JUDGE_ENV_FILE="${BACKBONE_JUDGE_ENV_FILE:-$BACKBONE_ENV_FILE}"
export BACKBONE_JUDGE_API_URL="${BACKBONE_JUDGE_API_URL:-$BACKBONE_API_URL}"
export BACKBONE_JUDGE_MODEL="${BACKBONE_JUDGE_MODEL:-deepseek-reasoner}"

export POLICY_ENV_FILE="${POLICY_ENV_FILE:-}"
export POLICY_API_URL="${POLICY_API_URL:-http://127.0.0.1:8001/v1}"
export POLICY_MODEL="${POLICY_MODEL:-Qwen3-1.7B}"
export POLICY_ENABLE_THINKING="${POLICY_ENABLE_THINKING:-false}"
export POLICY_PRESERVE_REASONING_CONTENT="${POLICY_PRESERVE_REASONING_CONTENT:-false}"
export POLICY_TEMPERATURE="${POLICY_TEMPERATURE:-0.6}"
export POLICY_MAX_TOKENS="${POLICY_MAX_TOKENS:-8192}"

export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-2000}"
export VAL_OFFSET="${VAL_OFFSET:-0}"
export NUM_WORKERS="${NUM_WORKERS:-8}"
export MAX_PARALLEL_POLICY_QUERIES="${MAX_PARALLEL_POLICY_QUERIES:-3}"
export MAX_BACKBONE_SEARCH_QUERIES="${MAX_BACKBONE_SEARCH_QUERIES:-3}"
export MAX_ORCHESTRATOR_ROUNDS="${MAX_ORCHESTRATOR_ROUNDS:-4}"
export MAX_ASSISTANT_TURNS="${MAX_ASSISTANT_TURNS:-3}"
export MAX_PARALLEL_CALLS="${MAX_PARALLEL_CALLS:-1}"
export BACKBONE_MAX_TOKENS="${BACKBONE_MAX_TOKENS:-20000}"
export BACKBONE_JUDGE_MAX_TOKENS="${BACKBONE_JUDGE_MAX_TOKENS:-512}"
export API_TIMEOUT="${API_TIMEOUT:-180}"
export API_MAX_RETRIES="${API_MAX_RETRIES:-4}"
export RETRIEVAL_URL="${RETRIEVAL_URL:-http://162.30.4.229:8765/search}"
export RETRIEVAL_MAX_CONCURRENT="${RETRIEVAL_MAX_CONCURRENT:-96}"
export RETRIEVAL_TIMEOUT="${RETRIEVAL_TIMEOUT:-180}"
export TOPK="${TOPK:-3}"
export RESUME="${RESUME:-true}"

exec "$SCRIPT_DIR/run_generate_sft_rollout_val900_api_policy.sh" "$@"
