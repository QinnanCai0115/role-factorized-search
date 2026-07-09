# python scripts/evaluation/context_mirage.py \
#     --input_file data/mirage/mirage_test.parquet \
#     --result_file results/mirage_haiku_bm25.json \
#     --context_dir data/mirage/rag_bm25 \
#     --num_workers 10 \
#     --topk 3


# python scripts/evaluation/context_mirage.py \
#     --input_file data/mirage/mirage_test.parquet \
#     --result_file results/mirage_haiku_deepretrieval_bm25_medcorp.json \
#     --context_dir data/mirage/nq_deepretrieval \
#     --num_workers 10 \
#     --topk 10

# python scripts/evaluation/context_mirage.py \
#     --input_file data/mirage/mirage_test.parquet \
#     --result_file results/haiku_s3_8_3_3_mirage_medcorp.json \
#     --context_dir data/output_sequences_s3_8_3_3_mirage_medcorp_new \
#     --num_workers 10 \
#     --topk 12

python scripts/evaluation/context_mirage.py \
    --input_file data/mirage/mirage_test.parquet \
    --result_file results/haiku_search_r1_7b_mirage_medcorp.json \
    --context_dir data/output_sequences_r1_7b_mirage_medcorp \
    --num_workers 10 \
    --topk 12


# python scripts/evaluation/context_mirage.py \
#     --input_file data/mirage/mirage_test.parquet \
#     --result_file results/mirage_haiku_rag_e5_medcorp_3.json \
#     --context_dir data/mirage/rag_e5_medcorp \
#     --num_workers 10 \
#     --topk 3
