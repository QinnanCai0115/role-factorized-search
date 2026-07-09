#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-/ai/cqn/datacon}"
DATA_DIR="$PROJECT_DIR/data/deepseek_policy_sft_rollouts/fixed_policy_replay/sft_candidates"
DATASET_BASENAME="mixed_second_search_qwen_bad_668_plus_qwen_naive_1000_plus_deepseek_one_search_500.no_think"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_1p7b_policy_sft_no_think_2168_qwen_bad_search_stop_mix_3ep_${TIMESTAMP}}"

export PYTHON_BIN="${PYTHON_BIN:-/ai/cqn/miniconda3/envs/verl/bin/python}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"

export PROJECT_NAME="${PROJECT_NAME:-search_subagent_policy_sft}"
export MODEL_PATH="${MODEL_PATH:-/ai/cqn/model/Qwen3-1.7B}"
export TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"

export TRAIN_JSONL="${TRAIN_JSONL:-$DATA_DIR/${DATASET_BASENAME}.sft.jsonl}"
export TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/${DATASET_BASENAME}.full.verl_sft.parquet}"

# Use the whole dataset for training by default.
export ENABLE_VAL="${ENABLE_VAL:-false}"
export VAL_JSONL="${VAL_JSONL:-}"
export VAL_FILE="${VAL_FILE:-}"

export AUTO_CONVERT="${AUTO_CONVERT:-true}"
export FORCE_CONVERT="${FORCE_CONVERT:-false}"
export MAX_ASSISTANT_TURNS_FILTER="${MAX_ASSISTANT_TURNS_FILTER:-3}"

# Keep these close to the 2004-run setup for comparable behavior on 2 GPUs.
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
export MAX_LENGTH="${MAX_LENGTH:-4096}"
export MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-8192}"
export PAD_MODE="${PAD_MODE:-no_padding}"
export TRUNCATION="${TRUNCATION:-error}"
export IGNORE_INPUT_IDS_MISMATCH="${IGNORE_INPUT_IDS_MISMATCH:-true}"
export NUM_WORKERS="${NUM_WORKERS:-4}"

export TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
export TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
export VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"
export SAVE_FREQ="${SAVE_FREQ:-after_each_epoch}"
export TEST_FREQ="${TEST_FREQ:--1}"
export RESUME_MODE="${RESUME_MODE:-auto}"
export LOGGER="${LOGGER:-['console','tensorboard']}"

export LR="${LR:-1e-5}"
export LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.03}"
export LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
export WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

export LORA_RANK="${LORA_RANK:-0}"
export LORA_ALPHA="${LORA_ALPHA:-32}"
export TARGET_MODULES="${TARGET_MODULES:-all-linear}"

export OUTPUT_DIR="${OUTPUT_DIR:-/ai/cqn/s3/ckpt/$PROJECT_NAME/$EXPERIMENT_NAME}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-$PROJECT_DIR/tensorboard_log/$PROJECT_NAME/$EXPERIMENT_NAME}"

printf 'Experiment: %s\n' "$EXPERIMENT_NAME"
printf 'Train JSONL: %s\n' "$TRAIN_JSONL"
printf 'Train parquet: %s\n' "$TRAIN_FILE"
printf 'Output dir: %s\n' "$OUTPUT_DIR"
printf 'TensorBoard dir: %s\n' "$TENSORBOARD_DIR"
printf 'GPUs: %s / nproc=%s\n' "$CUDA_VISIBLE_DEVICES" "$N_GPUS_PER_NODE"

exec bash "$SCRIPT_DIR/run_policy_sft.sh"
