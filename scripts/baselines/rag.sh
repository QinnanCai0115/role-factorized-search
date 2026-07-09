python scripts/baselines/bm25_retrieval.py \
    --input_parquet data/nq_hotpotqa_train/test_e5_ug.parquet \
    --rewriter none \
    --output_dir data/BM25/ \
    --endpoint http://127.0.0.1:3000/retrieve

