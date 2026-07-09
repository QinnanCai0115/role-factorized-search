#!/usr/bin/env bash
set -euo pipefail

DEBUG_XTRACE="${DEBUG_XTRACE:-0}"
if [ "$DEBUG_XTRACE" = "1" ]; then
  set -x
fi

ulimit -n 65535

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
cd "$PROJECT_DIR"

SECRET_ENV_CANDIDATES=(
  "${DEEPSEEK_ENV_FILE:-}"
  "$PROJECT_DIR/.secret/deepseek.env"
  "$PROJECT_DIR/.secrets/deepseek.env"
  "/ai/cqn/s3/.secret/deepseek.env"
  "/ai/cqn/s3/.secrets/deepseek.env"
)
for candidate_env in "${SECRET_ENV_CANDIDATES[@]}"; do
  if [ -z "$candidate_env" ] || [ ! -f "$candidate_env" ]; then
    continue
  fi
  set -a
  # shellcheck disable=SC1090
  source "$candidate_env"
  set +a
  break
done

PYTHON_BIN="${PYTHON_BIN:-}"
if [ -z "$PYTHON_BIN" ]; then
  if [ -x "/ai/cqn/miniconda3/envs/verl/bin/python" ]; then
    PYTHON_BIN="/ai/cqn/miniconda3/envs/verl/bin/python"
  else
    PYTHON_BIN="$(command -v python)"
  fi
fi

# ============================
# Main knobs
# ============================
ROOT_DIR="${ROOT_DIR:-/ai/cqn/s3}"
PROJECT_NAME="${PROJECT_NAME:-search_subagent_grpo_fully_async}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-two_stage_async_rollout_n1_${TIMESTAMP}}"

TRAIN_DATA="${TRAIN_DATA:-$PROJECT_DIR/data/hotpotqa_2wiki_musique_train/train_mixed_7000_rl.parquet}"
VAL_DATA="${VAL_DATA:-$PROJECT_DIR/data/hotpotqa_2wiki_musique_train/val_mixed_900.parquet}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/ai/cqn/model/Qwen3-1.7B}"
SFT_CKPT_ROOT="${SFT_CKPT_ROOT:-/ai/cqn/s3/ckpt/search_subagent_policy_sft/qwen3_1p7b_policy_sft_filtered_em1_1893_20260514_022426}"
SFT_STEP="${SFT_STEP:-}"
if [ -z "$SFT_STEP" ] && [ -f "$SFT_CKPT_ROOT/latest_checkpointed_iteration.txt" ]; then
  SFT_STEP="$(tr -d '[:space:]' < "$SFT_CKPT_ROOT/latest_checkpointed_iteration.txt")"
fi
SFT_STEP="${SFT_STEP:-236}"
SFT_GLOBAL_STEP_DIR="${SFT_GLOBAL_STEP_DIR:-$SFT_CKPT_ROOT/global_step_$SFT_STEP}"
SFT_VERL_RESUME_PATH="${SFT_VERL_RESUME_PATH:-$SFT_CKPT_ROOT/verl_resume_layout/global_step_$SFT_STEP}"
if [ -z "${SFT_RESUME_FROM_PATH:-}" ]; then
  if [ -d "$SFT_VERL_RESUME_PATH" ]; then
    SFT_RESUME_FROM_PATH="$SFT_VERL_RESUME_PATH"
  else
    SFT_RESUME_FROM_PATH="$SFT_GLOBAL_STEP_DIR"
  fi
fi
SFT_MERGED_HF_MODEL_PATH="${SFT_MERGED_HF_MODEL_PATH:-$SFT_CKPT_ROOT/merged_hf_global_step_$SFT_STEP}"
SFT_TOKENIZER_PATH="${SFT_TOKENIZER_PATH:-$SFT_GLOBAL_STEP_DIR/huggingface}"
if [ -z "${ACTOR_MODEL_PATH:-}" ]; then
  if [ -d "$SFT_MERGED_HF_MODEL_PATH" ]; then
    ACTOR_MODEL_PATH="$SFT_MERGED_HF_MODEL_PATH"
  else
    ACTOR_MODEL_PATH="$BASE_MODEL_PATH"
  fi
fi
ACTOR_RESUME_FROM_PATH="${ACTOR_RESUME_FROM_PATH:-}"
if [ -z "$ACTOR_RESUME_FROM_PATH" ]; then
  if [ "$ACTOR_MODEL_PATH" = "$SFT_MERGED_HF_MODEL_PATH" ]; then
    ACTOR_RESUME_FROM_PATH=null
  else
    ACTOR_RESUME_FROM_PATH="$SFT_RESUME_FROM_PATH"
  fi
fi
TOKENIZER_PATH="${TOKENIZER_PATH:-}"
if [ -z "$TOKENIZER_PATH" ]; then
  if [ "$ACTOR_MODEL_PATH" = "$SFT_MERGED_HF_MODEL_PATH" ]; then
    TOKENIZER_PATH="$ACTOR_MODEL_PATH"
  elif [ -d "$SFT_TOKENIZER_PATH" ]; then
    TOKENIZER_PATH="$SFT_TOKENIZER_PATH"
  else
    TOKENIZER_PATH="$ACTOR_MODEL_PATH"
  fi
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
NNODES="${NNODES:-1}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-2}"
TRAINER_N_GPUS_PER_NODE="${TRAINER_N_GPUS_PER_NODE:-1}"
ROLLOUT_N_GPUS_PER_NODE="${ROLLOUT_N_GPUS_PER_NODE:-1}"
export CUDA_VISIBLE_DEVICES

# Fully async requires data.train_batch_size=0 and data.gen_batch_size=1.
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-256}"
REQUIRE_BATCHES="${REQUIRE_BATCHES:-1}"
TRIGGER_PARAMETER_SYNC_STEP="${TRIGGER_PARAMETER_SYNC_STEP:-2}"
STALENESS_THRESHOLD="${STALENESS_THRESHOLD:-0}"
MAX_POLICY_VERSION_STALENESS="${MAX_POLICY_VERSION_STALENESS:-2}"
PARTIAL_ROLLOUT="${PARTIAL_ROLLOUT:-true}"
TOTAL_ROLLOUT_STEPS="${TOTAL_ROLLOUT_STEPS:-7000}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"

# Important: keep top-level policy rollout repeat at 1. Diversity should come from
# policy_rollout_n if needed, not from actor_rollout_ref.rollout.n.
ROLLOUT_N="${ROLLOUT_N:-1}"
POLICY_ROLLOUT_N="${POLICY_ROLLOUT_N:-8}"
VALIDATION_POLICY_ROLLOUT_N="${VALIDATION_POLICY_ROLLOUT_N:-1}"
POLICY_ROLLOUT_TOP_P="${POLICY_ROLLOUT_TOP_P:-0.85}"
VALIDATION_POLICY_ROLLOUT_TOP_P="${VALIDATION_POLICY_ROLLOUT_TOP_P:-$POLICY_ROLLOUT_TOP_P}"
ORCHESTRATOR_MAX_ROUNDS="${ORCHESTRATOR_MAX_ROUNDS:-4}"
VALIDATION_ORCHESTRATOR_MAX_ROUNDS="${VALIDATION_ORCHESTRATOR_MAX_ROUNDS:-$ORCHESTRATOR_MAX_ROUNDS}"
MAX_BACKBONE_SEARCH_QUERIES="${MAX_BACKBONE_SEARCH_QUERIES:-3}"
VALIDATION_MAX_BACKBONE_SEARCH_QUERIES="${VALIDATION_MAX_BACKBONE_SEARCH_QUERIES:-$MAX_BACKBONE_SEARCH_QUERIES}"
TEST_FREQ="${TEST_FREQ:--1}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-32}"
VAL_MAX_SAMPLES="${VAL_MAX_SAMPLES:-8}"
TRAIN_SHUFFLE="${TRAIN_SHUFFLE:-true}"
VALIDATION_SHUFFLE="${VALIDATION_SHUFFLE:-false}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
SHOW_PROGRESS_BAR="${SHOW_PROGRESS_BAR:-true}"

ROLLOUT_BACKEND="${ROLLOUT_BACKEND:-vllm}"
CHECKPOINT_ENGINE_BACKEND="${CHECKPOINT_ENGINE_BACKEND:-nccl}"
CHECKPOINT_ENGINE_BUCKET_MB="${CHECKPOINT_ENGINE_BUCKET_MB:-768}"
INFER_TP="${INFER_TP:-1}"
AGENT_NUM_WORKERS="${AGENT_NUM_WORKERS:-4}"
TOOL_PARSER_FORMAT="${TOOL_PARSER_FORMAT:-search_xml}"
TOOL_CONFIG="${TOOL_CONFIG:-$PROJECT_DIR/scripts/examples/config/tool_config/search_subagent_tool_config.yaml}"
ROLLOUT_MAX_ASSISTANT_TURNS="${ROLLOUT_MAX_ASSISTANT_TURNS:-3}"
ROLLOUT_MAX_PARALLEL_CALLS="${ROLLOUT_MAX_PARALLEL_CALLS:-1}"
POLICY_GENERATION_STOP_ENABLED="${POLICY_GENERATION_STOP_ENABLED:-true}"
POLICY_GENERATION_INCLUDE_STOP_STR="${POLICY_GENERATION_INCLUDE_STOP_STR:-true}"
POLICY_GENERATION_STOP_SEQUENCES="${POLICY_GENERATION_STOP_SEQUENCES:-}"
if [ -z "$POLICY_GENERATION_STOP_SEQUENCES" ]; then
  POLICY_GENERATION_STOP_SEQUENCES='["</search>","</evidence>"]'
fi
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.60}"
ROLLOUT_MAX_NUM_BATCHED_TOKENS="${ROLLOUT_MAX_NUM_BATCHED_TOKENS:-12288}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-32}"

PPO_MICRO_BATCH_SIZE_PER_GPU="${PPO_MICRO_BATCH_SIZE_PER_GPU:-4}"
LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-2}"
REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU="${REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU:-2}"
ACTOR_MAX_TOKEN_LEN_PER_GPU="${ACTOR_MAX_TOKEN_LEN_PER_GPU:-$(( (MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 2 ))}"
LOGPROB_MAX_TOKEN_LEN_PER_GPU="${LOGPROB_MAX_TOKEN_LEN_PER_GPU:-$ACTOR_MAX_TOKEN_LEN_PER_GPU}"
ACTOR_USE_REMOVE_PADDING="${ACTOR_USE_REMOVE_PADDING:-false}"
ACTOR_LR="${ACTOR_LR:-1e-6}"
ACTOR_LORA_RANK="${ACTOR_LORA_RANK:-16}"
FSDP_SIZE="${FSDP_SIZE:-$TRAINER_N_GPUS_PER_NODE}"
FSDP_MODEL_DTYPE="${FSDP_MODEL_DTYPE:-bf16}"
TRAIN_SP="${TRAIN_SP:-1}"
OFFLOAD="${OFFLOAD:-false}"

# Backbone and backbone judge use the same OpenAI-compatible API.
BACKBONE_API_MODE="${BACKBONE_API_MODE:-openai_compatible}"
BACKBONE_API_URL="${BACKBONE_API_URL:-https://api.deepseek.com/v1}"
BACKBONE_API_MODEL="${BACKBONE_API_MODEL:-deepseek-reasoner}"
BACKBONE_API_TIMEOUT="${BACKBONE_API_TIMEOUT:-120}"
BACKBONE_API_MAX_CONCURRENT="${BACKBONE_API_MAX_CONCURRENT:-8}"
BACKBONE_API_MAX_RETRIES="${BACKBONE_API_MAX_RETRIES:-3}"
BACKBONE_API_CONTINUE_ON_FAILURE="${BACKBONE_API_CONTINUE_ON_FAILURE:-true}"
BACKBONE_JUDGE_TIMEOUT="${BACKBONE_JUDGE_TIMEOUT:-120}"
BACKBONE_API_NO_PROXY="${BACKBONE_API_NO_PROXY:-1}"
BACKBONE_API_KEY="${BACKBONE_API_KEY:-${DEEPSEEK_API_KEY_MY:-${DEEPSEEK_API_KEY:-}}}"
export BACKBONE_API_KEY DEEPSEEK_API_KEY="$BACKBONE_API_KEY" BACKBONE_API_NO_PROXY

# For policy training this should stay false: policy is generated by local rollout
# servers so logprobs correspond to the trainable policy.
POLICY_USE_API="${POLICY_USE_API:-false}"
TOOL_REWARD_SOURCE="${TOOL_REWARD_SOURCE:-backbone_binary}"
POLICY_REWARD_MODE="${POLICY_REWARD_MODE:-backbone_discrete_judge}"
RETRIEVAL_EFFECTIVE_REWARD_WEIGHT="${RETRIEVAL_EFFECTIVE_REWARD_WEIGHT:-0.4}"
SUMMARY_REASONABLE_REWARD_WEIGHT="${SUMMARY_REASONABLE_REWARD_WEIGHT:-0.6}"
FORMAT_INVALID_REWARD="${FORMAT_INVALID_REWARD:--0.5}"
BOTH_GOOD_REWARD="${BOTH_GOOD_REWARD:-1.0}"
SUMMARY_ONLY_REWARD="${SUMMARY_ONLY_REWARD:-0.2}"
RETRIEVAL_ONLY_REWARD="${RETRIEVAL_ONLY_REWARD:-0.0}"
BOTH_BAD_REWARD="${BOTH_BAD_REWARD:--0.1}"
MAX_POLICY_OUTPUT_CHARS="${MAX_POLICY_OUTPUT_CHARS:-1500}"
MAX_POLICY_ANSWER_CHARS="${MAX_POLICY_ANSWER_CHARS:-256}"
MAX_POLICY_EVIDENCE_CHARS="${MAX_POLICY_EVIDENCE_CHARS:-768}"
# Added to the dense judge reward when final answer/evidence format is invalid.
# Keep it negative because the reward formula adds format_penalty directly.
POLICY_FORMAT_PENALTY="${POLICY_FORMAT_PENALTY:--0.2}"

CUSTOM_REWARD="${CUSTOM_REWARD:-$PROJECT_DIR/scripts/examples/search_r1_like/orchestrator_trajectory_reward.py}"
DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$ROOT_DIR/ckpt/$PROJECT_NAME/$EXPERIMENT_NAME}"
ROLLOUT_DATA_DIR="${ROLLOUT_DATA_DIR:-$DEFAULT_LOCAL_DIR/rollout_data}"
VALIDATION_DATA_DIR="${VALIDATION_DATA_DIR:-$DEFAULT_LOCAL_DIR/validation_data}"
IO_TRACE_LOG_PATH="${IO_TRACE_LOG_PATH:-$PROJECT_DIR/tmp_logs/fully_async_two_stage_${TIMESTAMP}.jsonl}"
TRAIN_LOG="${TRAIN_LOG:-$PROJECT_DIR/tmp_logs/fully_async_two_stage_${TIMESTAMP}.log}"
IO_TRACE_MAX_CHARS="${IO_TRACE_MAX_CHARS:-4000}"
IO_TRACE_MAX_ITEMS="${IO_TRACE_MAX_ITEMS:-6}"
IO_TRACE_MAX_SAMPLES="${IO_TRACE_MAX_SAMPLES:-3}"

RAY_TMPDIR="${RAY_TMPDIR:-$PROJECT_DIR/tmp_logs/ray}"
RAY_PLASMA_DIRECTORY="${RAY_PLASMA_DIRECTORY:-$PROJECT_DIR/tmp_logs/ray_tmp}"
RAY_OBJECT_STORE_MEMORY="${RAY_OBJECT_STORE_MEMORY:-134217728}"
RAY_INCLUDE_DASHBOARD="${RAY_INCLUDE_DASHBOARD:-false}"
TMPDIR="${TMPDIR:-$PROJECT_DIR/tmp_logs}"
HF_HOME="${HF_HOME:-$PROJECT_DIR/tmp_logs/hf_cache}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$PROJECT_DIR/tmp_logs/hf_datasets}"
export TMPDIR HF_HOME HF_DATASETS_CACHE

mkdir -p \
  "$TMPDIR" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$RAY_TMPDIR" \
  "$RAY_PLASMA_DIRECTORY" \
  "$DEFAULT_LOCAL_DIR" \
  "$ROLLOUT_DATA_DIR" \
  "$VALIDATION_DATA_DIR" \
  "$(dirname "$IO_TRACE_LOG_PATH")" \
  "$(dirname "$TRAIN_LOG")"

if [ -z "$BACKBONE_API_KEY" ]; then
  echo "Missing BACKBONE_API_KEY, DEEPSEEK_API_KEY_MY, or DEEPSEEK_API_KEY."
  exit 1
fi

if [ "$ROLLOUT_N" != "1" ]; then
  echo "This script is intended for actor_rollout_ref.rollout.n=1, got ROLLOUT_N=$ROLLOUT_N"
  exit 1
fi

if [ "$POLICY_USE_API" = "true" ]; then
  echo "POLICY_USE_API=true would bypass the local trainable policy rollout. Set POLICY_USE_API=false for training."
  exit 1
fi

if [ ! -d "$ACTOR_MODEL_PATH" ]; then
  echo "ACTOR_MODEL_PATH must be a local directory: $ACTOR_MODEL_PATH"
  exit 1
fi

if [ ! -f "$TRAIN_DATA" ]; then
  echo "TRAIN_DATA must be a parquet file: $TRAIN_DATA"
  exit 1
fi

if [ ! -f "$VAL_DATA" ]; then
  echo "VAL_DATA must be a parquet file: $VAL_DATA"
  exit 1
fi

if [ "$ACTOR_RESUME_FROM_PATH" != "null" ] && [ "$ACTOR_RESUME_FROM_PATH" != "none" ] && [ ! -d "$ACTOR_RESUME_FROM_PATH" ]; then
  echo "ACTOR_RESUME_FROM_PATH must be a checkpoint directory: $ACTOR_RESUME_FROM_PATH"
  exit 1
fi

if [ "$ACTOR_RESUME_FROM_PATH" != "null" ] && [ "$ACTOR_RESUME_FROM_PATH" != "none" ] && [ -f "$SFT_GLOBAL_STEP_DIR/fsdp_config.json" ]; then
  SFT_SHARD_WORLD_SIZE="$(grep -o '"world_size"[[:space:]]*:[[:space:]]*[0-9]*' "$SFT_GLOBAL_STEP_DIR/fsdp_config.json" | grep -o '[0-9]*' | tail -1)"
  if [ -n "$SFT_SHARD_WORLD_SIZE" ] && [ "$TRAINER_N_GPUS_PER_NODE" != "$SFT_SHARD_WORLD_SIZE" ]; then
    echo "The SFT FSDP checkpoint has world_size=$SFT_SHARD_WORLD_SIZE, but TRAINER_N_GPUS_PER_NODE=$TRAINER_N_GPUS_PER_NODE."
    echo "For two GPUs, use the merged HF SFT model instead: ACTOR_MODEL_PATH=$SFT_MERGED_HF_MODEL_PATH ACTOR_RESUME_FROM_PATH=null"
    exit 1
  fi
fi

export VLLM_USE_V1="${VLLM_USE_V1:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-1}"
export RAY_memory_usage_threshold="${RAY_memory_usage_threshold:-0.98}"

echo "Train data: $TRAIN_DATA"
echo "Validation data: $VAL_DATA"
echo "Actor model: $ACTOR_MODEL_PATH"
echo "Tokenizer: $TOKENIZER_PATH"
echo "Actor-only SFT init: $ACTOR_RESUME_FROM_PATH"
echo "GPUs: CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES trainer=$TRAINER_N_GPUS_PER_NODE rollout=$ROLLOUT_N_GPUS_PER_NODE"
echo "Checkpoint engine: $CHECKPOINT_ENGINE_BACKEND (${CHECKPOINT_ENGINE_BUCKET_MB}MB bucket)"
echo "FSDP model dtype: $FSDP_MODEL_DTYPE"
echo "Policy generation stop: enabled=$POLICY_GENERATION_STOP_ENABLED include_stop=$POLICY_GENERATION_INCLUDE_STOP_STR sequences=$POLICY_GENERATION_STOP_SEQUENCES"
echo "Experiment output: $DEFAULT_LOCAL_DIR"
echo "Progress bar: $SHOW_PROGRESS_BAR"

"$PYTHON_BIN" -m verl.experimental.fully_async_policy.fully_async_main \
  algorithm.adv_estimator=grpo \
  algorithm.tool_reward_as_grpo_point=true \
  +algorithm.grpo_group_key=pair_group_id \
  algorithm.use_kl_in_reward=false \
  algorithm.kl_ctrl.kl_coef=0.0 \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$VAL_DATA" \
  data.return_raw_chat=true \
  data.train_batch_size=0 \
  data.gen_batch_size=1 \
  data.shuffle="$TRAIN_SHUFFLE" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.val_max_samples="$VAL_MAX_SAMPLES" \
  data.validation_shuffle="$VALIDATION_SHUFFLE" \
  data.max_prompt_length="$MAX_PROMPT_LENGTH" \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=true \
  data.truncation=error \
  reward.custom_reward_function.path="$CUSTOM_REWARD" \
  reward.custom_reward_function.name=compute_score \
  +reward.custom_reward_function.reward_kwargs.reward_mode=backbone_discrete_judge \
  +reward.custom_reward_function.reward_kwargs.binary_reduction=last \
  +reward.custom_reward_function.reward_kwargs.retrieval_effective_reward_weight="$RETRIEVAL_EFFECTIVE_REWARD_WEIGHT" \
  +reward.custom_reward_function.reward_kwargs.summary_reasonable_reward_weight="$SUMMARY_REASONABLE_REWARD_WEIGHT" \
  +reward.custom_reward_function.reward_kwargs.format_invalid_reward="$FORMAT_INVALID_REWARD" \
  +reward.custom_reward_function.reward_kwargs.both_good_reward="$BOTH_GOOD_REWARD" \
  +reward.custom_reward_function.reward_kwargs.summary_only_reward="$SUMMARY_ONLY_REWARD" \
  +reward.custom_reward_function.reward_kwargs.retrieval_only_reward="$RETRIEVAL_ONLY_REWARD" \
  +reward.custom_reward_function.reward_kwargs.both_bad_reward="$BOTH_BAD_REWARD" \
  +reward.custom_reward_function.reward_kwargs.max_output_chars="$MAX_POLICY_OUTPUT_CHARS" \
  +reward.custom_reward_function.reward_kwargs.max_answer_chars="$MAX_POLICY_ANSWER_CHARS" \
  +reward.custom_reward_function.reward_kwargs.max_evidence_chars="$MAX_POLICY_EVIDENCE_CHARS" \
  actor_rollout_ref.hybrid_engine=false \
  actor_rollout_ref.model.path="${ACTOR_MODEL_PATH%/}" \
  actor_rollout_ref.model.tokenizer_path="${TOKENIZER_PATH%/}" \
  actor_rollout_ref.model.use_remove_padding="$ACTOR_USE_REMOVE_PADDING" \
  actor_rollout_ref.model.lora_rank="$ACTOR_LORA_RANK" \
  actor_rollout_ref.model.enable_gradient_checkpointing=true \
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
  actor_rollout_ref.actor.use_kl_loss=true \
  actor_rollout_ref.actor.kl_loss_coef=0.005 \
  actor_rollout_ref.actor.clip_ratio_low=0.2 \
  actor_rollout_ref.actor.clip_ratio_high=0.28 \
  actor_rollout_ref.actor.clip_ratio_c=10.0 \
  actor_rollout_ref.actor.optim.lr="$ACTOR_LR" \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$ACTOR_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="$FSDP_SIZE" \
  actor_rollout_ref.actor.fsdp_config.model_dtype="$FSDP_MODEL_DTYPE" \
  actor_rollout_ref.actor.ulysses_sequence_parallel_size="$TRAIN_SP" \
  actor_rollout_ref.actor.fsdp_config.param_offload="$OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$OFFLOAD" \
  'actor_rollout_ref.actor.checkpoint.load_contents=[model]' \
  actor_rollout_ref.ref.fsdp_config.model_dtype="$FSDP_MODEL_DTYPE" \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="$REF_LOGPROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="$LOGPROB_MAX_TOKEN_LEN_PER_GPU" \
  actor_rollout_ref.rollout.name="$ROLLOUT_BACKEND" \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n=1 \
  rollout.n=1 \
  actor_rollout_ref.rollout.top_p="$POLICY_ROLLOUT_TOP_P" \
  actor_rollout_ref.rollout.val_kwargs.top_p="$VALIDATION_POLICY_ROLLOUT_TOP_P" \
  actor_rollout_ref.rollout.calculate_log_probs=true \
  actor_rollout_ref.rollout.checkpoint_engine.backend="$CHECKPOINT_ENGINE_BACKEND" \
  actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="$CHECKPOINT_ENGINE_BUCKET_MB" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="$LOGPROB_MICRO_BATCH_SIZE_PER_GPU" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="$INFER_TP" \
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
  +actor_rollout_ref.rollout.custom.validation_max_orchestrator_rounds="$VALIDATION_ORCHESTRATOR_MAX_ROUNDS" \
  +actor_rollout_ref.rollout.custom.validation_policy_rollout_n="$VALIDATION_POLICY_ROLLOUT_N" \
  +actor_rollout_ref.rollout.custom.max_backbone_search_queries="$MAX_BACKBONE_SEARCH_QUERIES" \
  +actor_rollout_ref.rollout.custom.validation_max_backbone_search_queries="$VALIDATION_MAX_BACKBONE_SEARCH_QUERIES" \
  +actor_rollout_ref.rollout.custom.backbone_use_api=true \
  +actor_rollout_ref.rollout.custom.backbone_api_mode="$BACKBONE_API_MODE" \
  +actor_rollout_ref.rollout.custom.backbone_api_url="$BACKBONE_API_URL" \
  +actor_rollout_ref.rollout.custom.backbone_api_model="$BACKBONE_API_MODEL" \
  +actor_rollout_ref.rollout.custom.backbone_api_timeout="$BACKBONE_API_TIMEOUT" \
  +actor_rollout_ref.rollout.custom.backbone_api_max_concurrent="$BACKBONE_API_MAX_CONCURRENT" \
  +actor_rollout_ref.rollout.custom.backbone_api_max_retries="$BACKBONE_API_MAX_RETRIES" \
  +actor_rollout_ref.rollout.custom.backbone_api_continue_on_failure="$BACKBONE_API_CONTINUE_ON_FAILURE" \
  +actor_rollout_ref.rollout.custom.backbone_judge_timeout="$BACKBONE_JUDGE_TIMEOUT" \
  +actor_rollout_ref.rollout.custom.policy_use_api=false \
  +actor_rollout_ref.rollout.custom.tool_reward_source="$TOOL_REWARD_SOURCE" \
  +actor_rollout_ref.rollout.custom.policy_generation_stop_enabled="$POLICY_GENERATION_STOP_ENABLED" \
  +actor_rollout_ref.rollout.custom.policy_generation_include_stop_str="$POLICY_GENERATION_INCLUDE_STOP_STR" \
  +actor_rollout_ref.rollout.custom.policy_generation_stop_sequences="'$POLICY_GENERATION_STOP_SEQUENCES'" \
  +actor_rollout_ref.rollout.custom.policy_reward_mode="$POLICY_REWARD_MODE" \
  +actor_rollout_ref.rollout.custom.retrieval_effective_reward_weight="$RETRIEVAL_EFFECTIVE_REWARD_WEIGHT" \
  +actor_rollout_ref.rollout.custom.summary_reasonable_reward_weight="$SUMMARY_REASONABLE_REWARD_WEIGHT" \
  +actor_rollout_ref.rollout.custom.format_invalid_reward="$FORMAT_INVALID_REWARD" \
  +actor_rollout_ref.rollout.custom.both_good_reward="$BOTH_GOOD_REWARD" \
  +actor_rollout_ref.rollout.custom.summary_only_reward="$SUMMARY_ONLY_REWARD" \
  +actor_rollout_ref.rollout.custom.retrieval_only_reward="$RETRIEVAL_ONLY_REWARD" \
  +actor_rollout_ref.rollout.custom.both_bad_reward="$BOTH_BAD_REWARD" \
  +actor_rollout_ref.rollout.custom.policy_format_penalty="$POLICY_FORMAT_PENALTY" \
  +actor_rollout_ref.rollout.custom.policy_reward_max_output_chars="$MAX_POLICY_OUTPUT_CHARS" \
  +actor_rollout_ref.rollout.custom.policy_reward_max_answer_chars="$MAX_POLICY_ANSWER_CHARS" \
  +actor_rollout_ref.rollout.custom.policy_reward_max_evidence_chars="$MAX_POLICY_EVIDENCE_CHARS" \
  +actor_rollout_ref.rollout.custom.policy_continue_require_discrete_good=true \
  +actor_rollout_ref.rollout.custom.policy_continue_max_output_chars="$MAX_POLICY_OUTPUT_CHARS" \
  +actor_rollout_ref.rollout.custom.policy_continue_max_answer_chars="$MAX_POLICY_ANSWER_CHARS" \
  +actor_rollout_ref.rollout.custom.policy_continue_max_evidence_chars="$MAX_POLICY_EVIDENCE_CHARS" \
  +actor_rollout_ref.rollout.custom.io_trace_log_path="$IO_TRACE_LOG_PATH" \
  +actor_rollout_ref.rollout.custom.io_trace_max_chars="$IO_TRACE_MAX_CHARS" \
  +actor_rollout_ref.rollout.custom.io_trace_max_items="$IO_TRACE_MAX_ITEMS" \
  +actor_rollout_ref.rollout.custom.io_trace_max_samples="$IO_TRACE_MAX_SAMPLES" \
  trainer.logger="['console','tensorboard']" \
  trainer.project_name="$PROJECT_NAME" \
  trainer.experiment_name="$EXPERIMENT_NAME" \
  trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  +trainer.show_progress_bar="$SHOW_PROGRESS_BAR" \
  trainer.log_val_generations=0 \
  trainer.save_freq="${SAVE_FREQ:-50}" \
  trainer.default_local_dir="$DEFAULT_LOCAL_DIR" \
  trainer.nnodes="$NNODES" \
  trainer.n_gpus_per_node="$TRAINER_N_GPUS_PER_NODE" \
  trainer.total_epochs="$TOTAL_EPOCHS" \
  trainer.test_freq="$TEST_FREQ" \
  trainer.resume_mode="${RESUME_MODE:-disable}" \
  trainer.resume_from_path="${RESUME_FROM_PATH:-null}" \
  +trainer.actor_resume_from_path="$ACTOR_RESUME_FROM_PATH" \
  trainer.rollout_data_dir="$ROLLOUT_DATA_DIR" \
  trainer.validation_data_dir="$VALIDATION_DATA_DIR" \
  rollout.nnodes="$NNODES" \
  rollout.n_gpus_per_node="$ROLLOUT_N_GPUS_PER_NODE" \
  rollout.total_rollout_steps="$TOTAL_ROLLOUT_STEPS" \
  async_training.staleness_threshold="$STALENESS_THRESHOLD" \
  +async_training.max_policy_version_staleness="$MAX_POLICY_VERSION_STALENESS" \
  async_training.trigger_parameter_sync_step="$TRIGGER_PARAMETER_SYNC_STEP" \
  async_training.require_batches="$REQUIRE_BATCHES" \
  async_training.partial_rollout="$PARTIAL_ROLLOUT" \
  async_training.use_trainer_do_validate=false \
  +ray_kwargs.ray_init._temp_dir="$RAY_TMPDIR" \
  +ray_kwargs.ray_init._plasma_directory="$RAY_PLASMA_DIRECTORY" \
  +ray_kwargs.ray_init.object_store_memory="$RAY_OBJECT_STORE_MEMORY" \
  +ray_kwargs.ray_init.include_dashboard="$RAY_INCLUDE_DASHBOARD" \
  "$@" 2>&1 | tee "$TRAIN_LOG"
