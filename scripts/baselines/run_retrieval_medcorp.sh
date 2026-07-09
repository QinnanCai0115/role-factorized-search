# python scripts/baselines/bm25_retrieval.py \
#     --input_parquet data/mirage/mirage_test.parquet \
#     --rewriter none \
#     --output_dir data/mirage/rag_bm25 \
#     --endpoint http://127.0.0.1:4000/retrieve

python scripts/baselines/e5_retrieval.py \
    --input_parquet data/mirage/mirage_test.parquet \
    --output_dir data/mirage/rag_e5_medcorp \
    --endpoint http://127.0.0.1:3000/retrieve


