# python scripts/baselines/bm25_retrieval.py \
#     --input_parquet data/nq_hotpotqa_train/train_e5_u1.parquet \
#     --rewriter none \
#     --output_dir data/rag_cache/ \
#     --endpoint http://127.0.0.1:3000/retrieve

python scripts/baselines/bm25_retrieval.py \
    --input_parquet data/mirage/mirage_test.parquet \
    --rewriter nq \
    --output_dir data/mirage \
    --endpoint http://127.0.0.1:4000/retrieve

