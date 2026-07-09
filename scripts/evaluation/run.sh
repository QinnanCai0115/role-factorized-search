# python scripts/evaluation/context.py \
#     --result_file results/r1_7b_haiku.json \
#     --context_dir data/output_sequences_r1_7b \
#     --num_workers 4 \
#     --topk 12

# python scripts/evaluation/context.py \
#     --result_file results/s3_14b.json \
#     --context_dir data/output_sequences_s3 \
#     --num_workers 20 \
#     --topk 12

python scripts/evaluation/context.py \
    --result_file results/haiku_s3_8_3_3_step_20_4364.json \
    --context_dir data/output_sequences_s3_8_3_3_step_20_4364 \
    --num_workers 10 \
    --topk 20


# python scripts/evaluation/context.py \
#     --result_file results/rag_e5_top12_haiku.json \
#     --context_dir data/RAG_Retrieval/test \
#     --num_workers 10 \
#     --topk 12

# python scripts/evaluation/context.py \
#     --result_file results/rag_e5_top6_14b.json \
#     --context_dir data/RAG_Retrieval/test \
#     --num_workers 10 \
#     --topk 6



# python scripts/evaluation/context.py \
#     --input_file data/nq_hotpotqa_train/train_e5_u1.parquet \
#     --result_file data/rag_cache/hotpotqa/rag_cache.json \
#     --context_dir data/rag_cache/none_deepretrieval \
#     --num_workers 20 \
#     --topk 3 

