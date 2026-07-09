SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
LOCAL_DIR="${LOCAL_DIR:-$PROJECT_DIR/data/hotpotqa_2wiki_musique_train}"
RETRIEVER=e5
SPLIT_SEED=${SPLIT_SEED:-42}

pwd

COMMON_ARGS="--local_dir $LOCAL_DIR --retriever $RETRIEVER"
TRAIN_COMMON_ARGS="$COMMON_ARGS"
TEST_COMMON_ARGS="$COMMON_ARGS"

## process multiple dataset search format train file
DATA=hotpotqa,2wikimultihopqa,musique
"${PYTHON_BIN:-python}" "$PROJECT_DIR/scripts/data_process/train_s3.py" $TRAIN_COMMON_ARGS --data_sources $DATA --sample_per_source 2000 --keep_binary_answers

# build a validation set from the generated training parquet with 10:1 split (train:val)
"${PYTHON_BIN:-python}" - <<PY
import os
import datasets

local_dir = "${LOCAL_DIR}"
retriever = "${RETRIEVER}"
seed = int("${SPLIT_SEED}")

full_path = os.path.join(local_dir, f"train_{retriever}_s3.parquet")
full_backup_path = os.path.join(local_dir, f"train_{retriever}_s3_full.parquet")
train_path = os.path.join(local_dir, f"train_{retriever}_s3.parquet")
val_path = os.path.join(local_dir, f"val_{retriever}_s3.parquet")

if not os.path.exists(full_path):
    raise FileNotFoundError(f"Expected parquet not found: {full_path}")

ds = datasets.load_dataset("parquet", data_files=full_path)["train"]
total = len(ds)
if total < 11:
    raise ValueError(f"Need at least 11 samples for 10:1 split, got {total}")

shuffled = ds.shuffle(seed=seed)
val_size = max(1, total // 11)
val_ds = shuffled.select(range(val_size))
train_ds = shuffled.select(range(val_size, total))

if os.path.abspath(full_backup_path) != os.path.abspath(full_path):
    ds.to_parquet(full_backup_path)
train_ds.to_parquet(train_path)
val_ds.to_parquet(val_path)

print(
    f"Split done (seed={seed}): total={total}, train={len(train_ds)}, val={len(val_ds)}\\n"
    f"full_backup={full_backup_path}\\ntrain={train_path}\\nval={val_path}"
)
PY

## process multiple dataset search format test file
#DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
#python scripts/data_process/test_s3.py $TEST_COMMON_ARGS --data_sources $DATA

## For a more efficient evaluation, we sample a subset (max 3000 samples per data_source) of the test set
#DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
#python scripts/data_process/test_s3_sampled.py $TEST_COMMON_ARGS #--data_sources $DATA --sample_source_parquet $LOCAL_DIR/test_e5_s3.parquet
