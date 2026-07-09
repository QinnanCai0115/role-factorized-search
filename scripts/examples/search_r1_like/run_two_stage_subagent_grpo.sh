
DEBUG_XTRACE="${DEBUG_XTRACE:-0}"
if [ "$DEBUG_XTRACE" = "1" ]; then
  set -x
fi

set -o pipefail

ulimit -n 65535

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
CONFIG_PATH="$PROJECT_DIR/scripts/examples/config"

# Hydra in this project relies on relative search paths like "verl/trainer/config".
# Force cwd to repo root so those paths resolve no matter where this script is launched.
cd "$PROJECT_DIR" || {
  echo "Failed to enter PROJECT_DIR: $PROJECT_DIR"
  exit 1
}

# ============================
# Quick Edit Config (common)
# Edit this block first for daily runs.
# ============================
DEFAULT_DEEPSEEK_ENV_FILE="/ai/cqn/s3/.secrets/deepseek.env"
DEFAULT_ROOT_DIR="/ai/cqn/s3"
DEFAULT_PROJECT_NAME="search_subagent_grpo"

TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
DEFAULT_EXPERIMENT_NAME="deepseek_chat_policy_sft_rollout_${TIMESTAMP}"
DEFAULT_CUDA_VISIBLE_DEVICES="0,1"
DEFAULT_N_GPUS_PER_NODE="2"
DEFAULT_ACTOR_MODEL_PATH="/ai/cqn/model/Qwen3-1.7B"
DEFAULT_TOKENIZER_PATH="$DEFAULT_ACTOR_MODEL_PATH"
DEFAULT_BACKBONE_API_MODE="openai_compatible"
DEFAULT_BACKBONE_API_URL="https://api.deepseek.com/v1"
DEFAULT_BACKBONE_API_MODEL="deepseek-reasoner"
DEFAULT_BACKBONE_API_TIMEOUT="120"
DEFAULT_BACKBONE_API_MAX_CONCURRENT="16"
DEFAULT_BACKBONE_API_MAX_RETRIES="10"
DEFAULT_BACKBONE_API_NO_PROXY="1"
DEFAULT_POLICY_USE_API="true"
DEFAULT_POLICY_API_MODE="openai_compatible"
DEFAULT_POLICY_API_URL="https://api.deepseek.com/v1"
DEFAULT_POLICY_API_MODEL="deepseek-chat"
DEFAULT_POLICY_API_TIMEOUT="120"
DEFAULT_POLICY_API_MAX_RETRIES="3"
DEFAULT_POLICY_API_TEMPERATURE="0.1"
DEFAULT_POLICY_API_MAX_TOKENS=""
DEFAULT_POLICY_API_NO_PROXY="1"
DEFAULT_ROLLOUT_BACKEND="vllm"
DEFAULT_ORCHESTRATOR_MAX_ROUNDS="4"
DEFAULT_VALIDATION_ORCHESTRATOR_MAX_ROUNDS=""
DEFAULT_TRAIN_BATCH_SIZE="16"
DEFAULT_VAL_BATCH_SIZE="8"
DEFAULT_VAL_MAX_SAMPLES="8"
DEFAULT_ROLLOUT_N="1"
DEFAULT_POLICY_ROLLOUT_N="1"
DEFAULT_VALIDATION_POLICY_ROLLOUT_N=""
# Keep the PPO update less fragmented while still using micro-batches for
# gradient accumulation. With the default 2 GPUs this normalizes to 8 samples
# per GPU and accumulates 2 micro-batches of 4 samples before each optimizer step.
DEFAULT_PPO_MINI_BATCH_SIZE="16"
DEFAULT_PPO_MICRO_BATCH_SIZE_PER_GPU="4"
DEFAULT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="2"
DEFAULT_REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="2"
DEFAULT_TOTAL_TRAINING_STEPS="200"
DEFAULT_ROLLOUT_ONLY="true"
DEFAULT_FORCE_LOCAL_FILES_ONLY="1"
DEFAULT_VAL_BEFORE_TRAIN="false"
DEFAULT_VAL_ONLY="false"
DEFAULT_VALIDATION_CONTROL_FILE=""
# Auto-load API keys and related env from deepseek.env if present.
DEEPSEEK_ENV_FILE="${DEEPSEEK_ENV_FILE:-$DEFAULT_DEEPSEEK_ENV_FILE}"
if [ -f "$DEEPSEEK_ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$DEEPSEEK_ENV_FILE"
  set +a
elif [ -f "$PROJECT_DIR/.secrets/deepseek.env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$PROJECT_DIR/.secrets/deepseek.env"
  set +a
fi
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "/ai/cqn/miniconda3/envs/verl/bin/python" ]; then
    PYTHON_BIN="/ai/cqn/miniconda3/envs/verl/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi
ROOT_DIR="${ROOT_DIR:-$DEFAULT_ROOT_DIR}"
PROJECT_NAME="${PROJECT_NAME:-$DEFAULT_PROJECT_NAME}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-$DEFAULT_EXPERIMENT_NAME}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-$DEFAULT_N_GPUS_PER_NODE}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$DEFAULT_CUDA_VISIBLE_DEVICES}"
export CUDA_VISIBLE_DEVICES
ACTOR_MODEL_PATH="${ACTOR_MODEL_PATH:-$DEFAULT_ACTOR_MODEL_PATH}"
TOKENIZER_PATH="${TOKENIZER_PATH:-$DEFAULT_TOKENIZER_PATH}"
# copy_to_local asserts src must not end with '/'. Normalize for robustness.
ACTOR_MODEL_PATH="$(printf '%s' "$ACTOR_MODEL_PATH" | tr -d '\r' | sed 's:/*$::')"
TOKENIZER_PATH="$(printf '%s' "$TOKENIZER_PATH" | tr -d '\r' | sed 's:/*$::')"
FORCE_LOCAL_FILES_ONLY="${FORCE_LOCAL_FILES_ONLY:-$DEFAULT_FORCE_LOCAL_FILES_ONLY}"
BACKBONE_API_MODE="${BACKBONE_API_MODE:-$DEFAULT_BACKBONE_API_MODE}"
BACKBONE_API_URL="${BACKBONE_API_URL:-$DEFAULT_BACKBONE_API_URL}"
BACKBONE_API_MODEL="${BACKBONE_API_MODEL:-$DEFAULT_BACKBONE_API_MODEL}"
BACKBONE_API_TIMEOUT="${BACKBONE_API_TIMEOUT:-$DEFAULT_BACKBONE_API_TIMEOUT}"
BACKBONE_API_MAX_CONCURRENT="${BACKBONE_API_MAX_CONCURRENT:-$DEFAULT_BACKBONE_API_MAX_CONCURRENT}"
BACKBONE_API_MAX_RETRIES="${BACKBONE_API_MAX_RETRIES:-$DEFAULT_BACKBONE_API_MAX_RETRIES}"
BACKBONE_API_NO_PROXY="${BACKBONE_API_NO_PROXY:-$DEFAULT_BACKBONE_API_NO_PROXY}"
BACKBONE_API_KEY="${BACKBONE_API_KEY:-${DEEPSEEK_API_KEY:-}}"
POLICY_USE_API="${POLICY_USE_API:-$DEFAULT_POLICY_USE_API}"
POLICY_API_MODE="${POLICY_API_MODE:-$DEFAULT_POLICY_API_MODE}"
POLICY_API_URL="${POLICY_API_URL:-$DEFAULT_POLICY_API_URL}"
POLICY_API_MODEL="${POLICY_API_MODEL:-$DEFAULT_POLICY_API_MODEL}"
POLICY_API_TIMEOUT="${POLICY_API_TIMEOUT:-$DEFAULT_POLICY_API_TIMEOUT}"
POLICY_API_MAX_RETRIES="${POLICY_API_MAX_RETRIES:-$DEFAULT_POLICY_API_MAX_RETRIES}"
POLICY_API_TEMPERATURE="${POLICY_API_TEMPERATURE:-$DEFAULT_POLICY_API_TEMPERATURE}"
POLICY_API_MAX_TOKENS="${POLICY_API_MAX_TOKENS:-$DEFAULT_POLICY_API_MAX_TOKENS}"
POLICY_API_NO_PROXY="${POLICY_API_NO_PROXY:-$DEFAULT_POLICY_API_NO_PROXY}"
POLICY_API_KEY="${POLICY_API_KEY:-${DEEPSEEK_API_KEY:-}}"
ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-$DEFAULT_ROLLOUT_BACKEND}"
ORCHESTRATOR_MAX_ROUNDS="${ORCHESTRATOR_MAX_ROUNDS:-$DEFAULT_ORCHESTRATOR_MAX_ROUNDS}"
VALIDATION_ORCHESTRATOR_MAX_ROUNDS="${VALIDATION_ORCHESTRATOR_MAX_ROUNDS:-$DEFAULT_VALIDATION_ORCHESTRATOR_MAX_ROUNDS}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"
IO_LOG="${IO_LOG:-$PROJECT_DIR/tmp_logs/verl_io_trace_run_${RUN_TS}.jsonl}"
IO_TRACE_LOG_PATH="${IO_TRACE_LOG_PATH:-$IO_LOG}"
TRAIN_LOG="${TRAIN_LOG:-/ai/cqn/tmp/verl_train_grpo_${RUN_TS}.log}"
IO_TRACE_MAX_CHARS="${IO_TRACE_MAX_CHARS:-4000}"
IO_TRACE_MAX_ITEMS="${IO_TRACE_MAX_ITEMS:-6}"
IO_TRACE_MAX_SAMPLES="${IO_TRACE_MAX_SAMPLES:-3}"

# Defaults validated by a 1-step end-to-end smoke run. Override via env when needed.
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-$DEFAULT_TRAIN_BATCH_SIZE}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-$DEFAULT_VAL_BATCH_SIZE}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
ROLLOUT_N="${ROLLOUT_N:-$DEFAULT_ROLLOUT_N}"
POLICY_ROLLOUT_N="${POLICY_ROLLOUT_N:-$DEFAULT_POLICY_ROLLOUT_N}"
VALIDATION_POLICY_ROLLOUT_N="${VALIDATION_POLICY_ROLLOUT_N:-$DEFAULT_VALIDATION_POLICY_ROLLOUT_N}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-$DEFAULT_PPO_MINI_BATCH_SIZE}"
PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-$DEFAULT_PPO_MICRO_BATCH_SIZE_PER_GPU}"
LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-$DEFAULT_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}"
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-$DEFAULT_REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-4}"
TOOL_PARSER_FORMAT="${TOOL_PARSER_FORMAT:-search_xml}"
ACTOR_USE_REMOVE_PADDING="${ACTOR_USE_REMOVE_PADDING:-false}"
ROLLOUT_MAX_ASSISTANT_TURNS="${ROLLOUT_MAX_ASSISTANT_TURNS:-3}"
ROLLOUT_MAX_PARALLEL_CALLS="${ROLLOUT_MAX_PARALLEL_CALLS:-1}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-12288}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-$DEFAULT_TOTAL_TRAINING_STEPS}"
ROLLOUT_ONLY="${ROLLOUT_ONLY:-$DEFAULT_ROLLOUT_ONLY}"
if [ "$ROLLOUT_ONLY" = "true" ]; then
  CRITIC_WARMUP="${CRITIC_WARMUP:-999999999}"
else
  CRITIC_WARMUP="${CRITIC_WARMUP:-0}"
fi
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-$DEFAULT_VAL_BEFORE_TRAIN}"
VAL_ONLY="${VAL_ONLY:-$DEFAULT_VAL_ONLY}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-$DEFAULT_VAL_MAX_SAMPLES}"
VALIDATION_CONTROL_FILE="${VALIDATION_CONTROL_FILE:-$DEFAULT_VALIDATION_CONTROL_FILE}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
RESUME_MODE="${RESUME_MODE:-auto}"
VALIDATION_SHUFFLE="${VALIDATION_SHUFFLE:-false}"
TOOL_REWARD_SOURCE="${TOOL_REWARD_SOURCE:-backbone_binary}"
BACKBONE_JUDGE_TIMEOUT="${BACKBONE_JUDGE_TIMEOUT:-120}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-$ROOT_DIR/ckpt/$PROJECT_NAME/$EXPERIMENT_NAME/rollout_data}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-$ROOT_DIR/ckpt/$PROJECT_NAME/$EXPERIMENT_NAME/validation_data}"

RAY_PLASMA_DIRECTORY="${RAY_PLASMA_DIRECTORY:-$PROJECT_DIR/tmp_logs/ray_tmp}"
RAY_TMPDIR="${RAY_TMPDIR:-$PROJECT_DIR/tmp_logs/ray}"
RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-134217728}"
RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-false}"

TMPDIR="${TMPDIR:-$PROJECT_DIR/tmp_logs}"
HF_HOME="${HF_HOME:-$PROJECT_DIR/tmp_logs/hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_DIR/tmp_logs/hf_datasets}"
export TMPDIR HF_HOME HF_DATASETS_CACHE

mkdir -p "$TMPDIR" "$HF_HOME" "$HF_DATASETS_CACHE" "$RAY_PLASMA_DIRECTORY" "$RAY_TMPDIR"
mkdir -p "$(dirname "$IO_TRACE_LOG_PATH")"
mkdir -p "$(dirname "$TRAIN_LOG")"
mkdir -p "$ROLLOUT_DATA_DIR" "$VALIDATION_DATA_DIR"
if [ -n "$VALIDATION_CONTROL_FILE" ]; then
  mkdir -p "$(dirname "$VALIDATION_CONTROL_FILE")"
fi

if [ "$ROLLOUT_BACKEND" != "vllm" ] && [ "$ROLLOUT_BACKEND" != "sglang" ] && [ "$ROLLOUT_BACKEND" != "trtllm" ]; then
  echo "Invalid ROLLOUT_BACKEND=$ROLLOUT_BACKEND (expected: vllm|sglang|trtllm)"
  exit 1
fi

# Strip non-printable bytes to avoid hidden characters breaking Hydra parsing.
BACKBONE_API_KEY="$(printf '%s' "$BACKBONE_API_KEY" | LC_ALL=C tr -cd '[:print:]')"
POLICY_API_KEY="$(printf '%s' "$POLICY_API_KEY" | LC_ALL=C tr -cd '[:print:]')"
export BACKBONE_API_KEY
export POLICY_API_KEY
export DEEPSEEK_API_KEY="$BACKBONE_API_KEY"
export BACKBONE_API_NO_PROXY
export POLICY_API_NO_PROXY

if [ -z "$BACKBONE_API_KEY" ] || { [ "$POLICY_USE_API" = "true" ] && [ -z "$POLICY_API_KEY" ]; }; then
  echo "Missing API key: set DEEPSEEK_API_KEY, or set BACKBONE_API_KEY/POLICY_API_KEY before running."
  exit 1
fi

if [ "$FORCE_LOCAL_FILES_ONLY" = "1" ]; then
  if [ ! -d "$ACTOR_MODEL_PATH" ]; then
    echo "FORCE_LOCAL_FILES_ONLY=1 requires ACTOR_MODEL_PATH to be a local directory, got: $ACTOR_MODEL_PATH"
    exit 1
  fi
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
fi

TRAIN_DATA="${TRAIN_DATA:-$PROJECT_DIR/data/hotpotqa_2wiki_musique_train/train_mixed_9000.parquet}"
# If VAL_DATA is not provided externally, reuse train data as a minimal runnable default.
VAL_DATA="${VAL_DATA:-$PROJECT_DIR/data/hotpotqa_2wiki_musique_train/val_mixed_900.parquet}"

TOOL_CONFIG="$PROJECT_DIR/scripts/examples/config/tool_config/search_subagent_tool_config.yaml"
CUSTOM_REWARD="$PROJECT_DIR/scripts/examples/search_r1_like/orchestrator_trajectory_reward.py"

"$PYTHON_BIN" -m verl.trainer.main_ppo \
  --config-path="$CONFIG_PATH" \
  --config-name='search_multiturn_grpo' \
  algorithm.adv_estimator=grpo \
  algorithm.tool_reward_as_grpo_point=true \
  +algorithm.grpo_group_key=pair_group_id \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$VAL_DATA" \
  data.return_raw_chat=true \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.val_max_samples="$VAL_MAX_SAMPLES" \
  data.validation_shuffle="$VALIDATION_SHUFFLE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.actor.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.ref.fsdp_config.model_dtype=bf16 \
  actor_rollout_ref.model.path="${ACTOR_MODEL_PATH%/}" \
  actor_rollout_ref.model.tokenizer_path="${TOKENIZER_PATH%/}" \
  actor_rollout_ref.model.use_remove_padding="$ACTOR_USE_REMOVE_PADDING" \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.rollout.name="$ROLLOUT_BACKEND" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$ROLLOUT_MAX_NUM_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
  actor_rollout_ref.rollout.enforce_eager=true \
  actor_rollout_ref.rollout.enable_chunked_prefill=false \
  actor_rollout_ref.rollout.multi_turn.enable=true \
  actor_rollout_ref.rollout.multi_turn.format="$TOOL_PARSER_FORMAT" \
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns="$ROLLOUT_MAX_ASSISTANT_TURNS" \
  actor_rollout_ref.rollout.multi_turn.max_parallel_calls="$ROLLOUT_MAX_PARALLEL_CALLS" \
  actor_rollout_ref.rollout.multi_turn.tool_config_path="$TOOL_CONFIG" \
  actor_rollout_ref.rollout.agent.num_workers="$AGENT_NUM_WORKERS" \
  actor_rollout_ref.rollout.agent.default_agent_loop=tool_agent \
  +actor_rollout_ref.rollout.custom.enable_backbone_rollout=true \
  +actor_rollout_ref.rollout.custom.max_orchestrator_rounds="$ORCHESTRATOR_MAX_ROUNDS" \
  +actor_rollout_ref.rollout.custom.policy_rollout_n="$POLICY_ROLLOUT_N" \
  +actor_rollout_ref.rollout.custom.validation_max_orchestrator_rounds="${VALIDATION_ORCHESTRATOR_MAX_ROUNDS:-$ORCHESTRATOR_MAX_ROUNDS}" \
  +actor_rollout_ref.rollout.custom.validation_policy_rollout_n="${VALIDATION_POLICY_ROLLOUT_N:-1}" \
  +actor_rollout_ref.rollout.custom.backbone_use_api=true \
  +actor_rollout_ref.rollout.custom.backbone_api_mode="$BACKBONE_API_MODE" \
  +actor_rollout_ref.rollout.custom.backbone_api_url="$BACKBONE_API_URL" \
  +actor_rollout_ref.rollout.custom.backbone_api_model="$BACKBONE_API_MODEL" \
  +actor_rollout_ref.rollout.custom.backbone_api_timeout="$BACKBONE_API_TIMEOUT" \
  +actor_rollout_ref.rollout.custom.backbone_api_max_concurrent="$BACKBONE_API_MAX_CONCURRENT" \
  +actor_rollout_ref.rollout.custom.backbone_api_max_retries="$BACKBONE_API_MAX_RETRIES" \
  +actor_rollout_ref.rollout.custom.policy_use_api="$POLICY_USE_API" \
  +actor_rollout_ref.rollout.custom.policy_api_mode="$POLICY_API_MODE" \
  +actor_rollout_ref.rollout.custom.policy_api_url="$POLICY_API_URL" \
  +actor_rollout_ref.rollout.custom.policy_api_model="$POLICY_API_MODEL" \
  +actor_rollout_ref.rollout.custom.policy_api_timeout="$POLICY_API_TIMEOUT" \
  +actor_rollout_ref.rollout.custom.policy_api_max_retries="$POLICY_API_MAX_RETRIES" \
  +actor_rollout_ref.rollout.custom.policy_api_temperature="$POLICY_API_TEMPERATURE" \
  +actor_rollout_ref.rollout.custom.policy_api_max_tokens="${POLICY_API_MAX_TOKENS:-null}" \
  +actor_rollout_ref.rollout.custom.policy_api_no_proxy="$POLICY_API_NO_PROXY" \
  +actor_rollout_ref.rollout.custom.tool_reward_source="$TOOL_REWARD_SOURCE" \
  +actor_rollout_ref.rollout.custom.backbone_judge_timeout="$BACKBONE_JUDGE_TIMEOUT" \
  +actor_rollout_ref.rollout.custom.io_trace_log_path="$IO_TRACE_LOG_PATH" \
  +actor_rollout_ref.rollout.custom.io_trace_max_chars="$IO_TRACE_MAX_CHARS" \
  +actor_rollout_ref.rollout.custom.io_trace_max_items="$IO_TRACE_MAX_ITEMS" \
  +actor_rollout_ref.rollout.custom.io_trace_max_samples="$IO_TRACE_MAX_SAMPLES" \
  reward.custom_reward_function.path="$CUSTOM_REWARD" \
  reward.custom_reward_function.name=compute_score \
  +reward.custom_reward_function.reward_kwargs.reward_mode=backbone_binary_only \
  +reward.custom_reward_function.reward_kwargs.binary_reduction=last \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.logger=['console','tensorboard'] \
  trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
  trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
  trainer.n_gpus_per_node="$N_GPUS_PER_NODE" \
  trainer.nnodes=1 \
  trainer.total_epochs=1 \
  trainer.save_freq=10 \
  trainer.test_freq=200 \
  trainer.total_training_steps="$TOTAL_TRAINING_STEPS" \
  trainer.critic_warmup="$CRITIC_WARMUP" \
  trainer.val_only="$VAL_ONLY" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  trainer.resume_mode="$RESUME_MODE" \
  trainer.resume_from_path="${RESUME_FROM_PATH:-null}" \
  +trainer.validation_control_file="$VALIDATION_CONTROL_FILE" \
  +ray_kwargs.ray_init._temp_dir="$RAY_TMPDIR" \
  +ray_kwargs.ray_init._plasma_directory="$RAY_PLASMA_DIRECTORY" \
  +ray_kwargs.ray_init.object_store_memory="$RAY_OBJECT_STORE_MEMORY" \
  +ray_kwargs.ray_init.include_dashboard="$RAY_INCLUDE_DASHBOARD" \
  +actor_rollout_ref.rollout.enable_sleep_mode=false \
  actor_rollout_ref.rollout.free_cache_engine=false \
  "$@" 2>&1 | tee "$TRAIN_LOG"
