echo "Running E5 Retrieval for Training Set"
python scripts/baselines/e5_retrieval.py \
    --input_parquet data/nq_hotpotqa_train/train_e5_ug.parquet \
    --output_dir data/RAG_retrieval/train \
    --endpoint http://127.0.0.1:3000/retrieve

echo "Running E5 Retrieval for Test Set"
python scripts/baselines/e5_retrieval.py \
    --input_parquet data/nq_hotpotqa_train/test_e5_ug.parquet \
    --output_dir data/RAG_retrieval/test \
    --endpoint http://127.0.0.1:3000/retrieve


