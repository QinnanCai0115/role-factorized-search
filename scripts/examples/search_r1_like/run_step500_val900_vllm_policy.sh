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

RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
STEP="${STEP:-500}"
RUN_DIR="${RUN_DIR:-/ai/cqn/s3/ckpt/search_subagent_grpo_fully_async/rl_from_qwen3_1p7b_policy_sft_step236_discrete_20260517_003523}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-$RUN_DIR/global_step_$STEP}"
ACTOR_DIR="${ACTOR_DIR:-$CHECKPOINT_DIR/actor}"
MERGED_POLICY_PATH="${MERGED_POLICY_PATH:-$RUN_DIR/merged_hf_global_step_$STEP}"

INPUT="${INPUT:-$PROJECT_DIR/data/hotpotqa_2wiki_musique_train/val_mixed_900.parquet}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/ai/cqn/s3/ckpt/search_subagent_api_policy_val900_step500}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-grpo_step${STEP}_val900_vllm_${RUN_TS}}"

if [ ! -d "$ACTOR_DIR" ]; then
  echo "Actor checkpoint not found: $ACTOR_DIR" >&2
  exit 1
fi

if ! find "$MERGED_POLICY_PATH" -maxdepth 1 -type f \( -name '*.safetensors' -o -name 'pytorch_model*.bin' \) 2>/dev/null | grep -q .; then
  echo "Merging FSDP checkpoint to HuggingFace format:"
  echo "  actor:  $ACTOR_DIR"
  echo "  target: $MERGED_POLICY_PATH"
  "$PYTHON_BIN" -m verl.model_merger merge \
    --backend fsdp \
    --local_dir "$ACTOR_DIR" \
    --target_dir "$MERGED_POLICY_PATH"
fi

echo "Running step-$STEP vLLM API-policy val900 evaluation"
echo "Merged policy: $MERGED_POLICY_PATH"
echo "Input:         $INPUT"
echo "Output root:   $OUTPUT_ROOT/$EXPERIMENT_NAME"

POLICIES="${POLICIES:-grpo_local}" \
REPEATS="${REPEATS:-1}" \
INPUT="$INPUT" \
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-900}" \
VAL_OFFSET="${VAL_OFFSET:-0}" \
OUTPUT_ROOT="$OUTPUT_ROOT" \
EXPERIMENT_NAME="$EXPERIMENT_NAME" \
GRPO_POLICY_PATH="$MERGED_POLICY_PATH" \
GRPO_POLICY_MODEL="${GRPO_POLICY_MODEL:-qwen3_1p7b_policy_grpo_step${STEP}}" \
GRPO_POLICY_API_URL="${GRPO_POLICY_API_URL:-http://127.0.0.1:8012/v1}" \
GRPO_VLLM_PORT="${GRPO_VLLM_PORT:-8012}" \
GRPO_CUDA_VISIBLE_DEVICES="${GRPO_CUDA_VISIBLE_DEVICES:-0}" \
START_LOCAL_VLLM="${START_LOCAL_VLLM:-true}" \
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.25}" \
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}" \
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-32}" \
VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}" \
POLICY_TEMPERATURE="${POLICY_TEMPERATURE:-0.6}" \
POLICY_MAX_TOKENS="${POLICY_MAX_TOKENS:-4096}" \
BACKBONE_MAX_TOKENS="${BACKBONE_MAX_TOKENS:-8192}" \
BACKBONE_JUDGE_MAX_TOKENS="${BACKBONE_JUDGE_MAX_TOKENS:-4096}" \
NUM_WORKERS="${NUM_WORKERS:-8}" \
MAX_PARALLEL_POLICY_QUERIES="${MAX_PARALLEL_POLICY_QUERIES:-3}" \
MAX_BACKBONE_SEARCH_QUERIES="${MAX_BACKBONE_SEARCH_QUERIES:-3}" \
PYTHON_BIN="$PYTHON_BIN" \
bash "$SCRIPT_DIR/run_test_all_mean4_policy_experiments.sh" "$@"
