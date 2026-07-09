#!/usr/bin/env bash
set -euo pipefail

DEBUG_XTRACE="${DEBUG_XTRACE:-0}"
if [ "$DEBUG_XTRACE" = "1" ]; then
  set -x
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
CONFIG_PATH="$PROJECT_DIR/verl/trainer/config"

cd "$PROJECT_DIR"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x "/ai/cqn/miniconda3/envs/verl/bin/python" ]; then
    PYTHON_BIN="/ai/cqn/miniconda3/envs/verl/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="${ROOT_DIR:-/ai/cqn/s3}"
PROJECT_NAME="${PROJECT_NAME:-search_subagent_policy_sft}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_1p7b_policy_sft_${TIMESTAMP}}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export CUDA_VISIBLE_DEVICES

MODEL_PATH="${MODEL_PATH:-/ai/cqn/model/Qwen3-1.7B}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$MODEL_PATH}"
TRAIN_JSONL="${TRAIN_JSONL:-$PROJECT_DIR/data/deepseek_policy_sft_rollouts/train_mixed_2000.deepseek_v4_pro.search_xml.valid_both.train2000.sft.jsonl}"
VAL_JSONL="${VAL_JSONL:-$PROJECT_DIR/data/deepseek_policy_sft_rollouts/train_mixed_2000.deepseek_v4_pro.search_xml.valid_both.val756.sft.jsonl}"
TRAIN_FILE="${TRAIN_FILE:-$PROJECT_DIR/data/deepseek_policy_sft_rollouts/train_mixed_2000.deepseek_v4_pro.search_xml.valid_both.train2000.verl_sft.parquet}"
VAL_FILE="${VAL_FILE:-$PROJECT_DIR/data/deepseek_policy_sft_rollouts/train_mixed_2000.deepseek_v4_pro.search_xml.valid_both.val756.verl_sft.parquet}"
ENABLE_VAL="${ENABLE_VAL:-true}"
if [ "$ENABLE_VAL" = "false" ]; then
  VAL_JSONL=""
  VAL_FILE=""
fi
AUTO_CONVERT="${AUTO_CONVERT:-true}"
FORCE_CONVERT="${FORCE_CONVERT:-false}"
MAX_ASSISTANT_TURNS_FILTER="${MAX_ASSISTANT_TURNS_FILTER:-3}"

TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-8192}"
PAD_MODE="${PAD_MODE:-no_padding}"
TRUNCATION="${TRUNCATION:-error}"
IGNORE_INPUT_IDS_MISMATCH="${IGNORE_INPUT_IDS_MISMATCH:-true}"
NUM_WORKERS="${NUM_WORKERS:-4}"

TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-null}"
TRAIN_MAX_SAMPLES="${TRAIN_MAX_SAMPLES:--1}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:--1}"
SAVE_FREQ="${SAVE_FREQ:-after_each_epoch}"
TEST_FREQ="${TEST_FREQ:-100}"
RESUME_MODE="${RESUME_MODE:-auto}"
LOGGER="${LOGGER:-['console','tensorboard']}"

LR="${LR:-1e-5}"
LR_WARMUP_STEPS_RATIO="${LR_WARMUP_STEPS_RATIO:-0.03}"
LR_SCHEDULER_TYPE="${LR_SCHEDULER_TYPE:-cosine}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"

LORA_RANK="${LORA_RANK:-0}"
LORA_ALPHA="${LORA_ALPHA:-32}"
TARGET_MODULES="${TARGET_MODULES:-all-linear}"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/ckpt/$PROJECT_NAME/$EXPERIMENT_NAME}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-$PROJECT_DIR/tensorboard_log/$PROJECT_NAME/$EXPERIMENT_NAME}"
export TENSORBOARD_DIR
mkdir -p "$OUTPUT_DIR"

if { [ "$AUTO_CONVERT" = "true" ] && [ ! -f "$TRAIN_FILE" ]; } || [ "$FORCE_CONVERT" = "true" ]; then
  "$PYTHON_BIN" "$SCRIPT_DIR/convert_policy_sft_jsonl_to_parquet.py" \
    --input "$TRAIN_JSONL" \
    --output "$TRAIN_FILE" \
    --max-assistant-turns "$MAX_ASSISTANT_TURNS_FILTER" \
    --require-answer-evidence
fi

if { [ "$AUTO_CONVERT" = "true" ] && [ -n "$VAL_FILE" ] && [ ! -f "$VAL_FILE" ]; } || { [ "$FORCE_CONVERT" = "true" ] && [ -n "$VAL_FILE" ]; }; then
  "$PYTHON_BIN" "$SCRIPT_DIR/convert_policy_sft_jsonl_to_parquet.py" \
    --input "$VAL_JSONL" \
    --output "$VAL_FILE" \
    --max-assistant-turns "$MAX_ASSISTANT_TURNS_FILTER" \
    --require-answer-evidence
fi

if [ ! -f "$TRAIN_FILE" ]; then
  echo "Missing TRAIN_FILE: $TRAIN_FILE"
  exit 1
fi

echo "Policy SFT mode: full fine-tuning (LORA_RANK=$LORA_RANK)"
echo "Training parquet: $TRAIN_FILE"
echo "Validation parquet: ${VAL_FILE:-null}"
echo "Checkpoint dir: $OUTPUT_DIR"
echo "TensorBoard dir: $TENSORBOARD_DIR"

VAL_FILES_ARG="data.val_files=null"
if [ -n "$VAL_FILE" ]; then
  VAL_FILES_ARG="data.val_files=$VAL_FILE"
fi

"$PYTHON_BIN" -m torch.distributed.run \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$N_GPUS_PER_NODE" \
  -m verl.trainer.sft_trainer \
  --config-path="$CONFIG_PATH" \
  --config-name=sft_trainer_engine \
  data.train_files="$TRAIN_FILE" \
  "$VAL_FILES_ARG" \
  data.messages_key=messages \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE_PER_GPU" \
  data.max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU" \
  data.train_max_samples="$TRAIN_MAX_SAMPLES" \
  data.val_max_samples="$VAL_MAX_SAMPLES" \
  data.max_length="$MAX_LENGTH" \
  data.truncation="$TRUNCATION" \
  data.pad_mode="$PAD_MODE" \
  data.num_workers="$NUM_WORKERS" \
  data.enable_thinking_default=false \
  data.ignore_input_ids_mismatch="$IGNORE_INPUT_IDS_MISMATCH" \
  model.path="$MODEL_PATH" \
  model.tokenizer_path="$TOKENIZER_PATH" \
  model.trust_remote_code=true \
  model.use_remove_padding=false \
  model.lora_rank="$LORA_RANK" \
  model.lora_alpha="$LORA_ALPHA" \
  model.target_modules="$TARGET_MODULES" \
  +model.override_config.attn_implementation=sdpa \
  engine.model_dtype=bf16 \
  engine.dtype=bfloat16 \
  engine.use_torch_compile=false \
  optim.lr="$LR" \
  optim.weight_decay="$WEIGHT_DECAY" \
  optim.lr_warmup_steps_ratio="$LR_WARMUP_STEPS_RATIO" \
  optim.lr_scheduler_type="$LR_SCHEDULER_TYPE" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.default_local_dir="$OUTPUT_DIR" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.save_freq="$SAVE_FREQ" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.logger="$LOGGER" \
  trainer.resume_mode="$RESUME_MODE" \
  trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
  trainer.nnodes=1
