LOCAL_DIR=${LOCAL_DIR:-data/nq_hotpotqa_train}

pwd

COMMON_ARGS="--local_dir $LOCAL_DIR --retriever e5"
TRAIN_COMMON_ARGS="$COMMON_ARGS"
TEST_COMMON_ARGS="$COMMON_ARGS"

# process multiple dataset search format train file
DATA=nq,hotpotqa
python scripts/data_process/train_s3.py $TRAIN_COMMON_ARGS --data_sources $DATA

# process multiple dataset search format test file
DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
python scripts/data_process/test_s3.py $TEST_COMMON_ARGS --data_sources $DATA

# sampled test set for faster evaluation utilities
DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
python scripts/data_process/test_s3_sampled.py $TEST_COMMON_ARGS --data_sources $DATA --sample_source_parquet $LOCAL_DIR/test_e5_s3.parquet
