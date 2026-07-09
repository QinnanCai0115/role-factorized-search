# python scripts/evaluation/context_no_select.py \
#     --result_file results/no_select_s3_8_3_3_general.json \
#     --context_dir data/output_sequences_s3_8_3_3_general \
#     --num_workers 5 \
#     --topk 30

# python scripts/evaluation/context_no_select.py \
#     --result_file results/no_select_s3_5_3_3_general.json \
#     --context_dir data/output_sequences_s3_5_3_3_general \
#     --num_workers 5 \
#     --topk 30

# python scripts/evaluation/context_no_select.py \
#     --result_file results/no_select_s3_3_3_3_general_.json \
#     --context_dir data/output_sequences_s3_3_3_3_general \
#     --num_workers 5 \
#     --topk 30

# python scripts/evaluation/context_no_init.py \
#     --result_file results/no_init_s3_8_3_3_general.json \
#     --context_dir data/output_sequences_s3_8_3_3_general \
#     --num_workers 5 \
#     --topk 30

# python scripts/evaluation/context_no_init.py \
#     --result_file results/no_init_s3_5_3_3_general.json \
#     --context_dir data/output_sequences_s3_5_3_3_general \
#     --num_workers 5 \
#     --topk 30

# python scripts/evaluation/context_no_init.py \
#     --result_file results/no_init_s3_3_3_3_general.json \
#     --context_dir data/output_sequences_s3_3_3_3_general \
#     --num_workers 5 \
#     --topk 30

# python scripts/evaluation/context_no_both.py \
#     --result_file results/no_both_s3_8_3_3_general.json \
#     --context_dir data/output_sequences_s3_8_3_3_general \
#     --num_workers 5 \
#     --topk 30

# python scripts/evaluation/context_no_both.py \
#     --result_file results/no_both_s3_5_3_3_general.json \
#     --context_dir data/output_sequences_s3_5_3_3_general \
#     --num_workers 5 \
#     --topk 30

# python scripts/evaluation/context_no_both.py \
#     --result_file results/no_both_s3_3_3_3_general.json \
#     --context_dir data/output_sequences_s3_3_3_3_general \
#     --num_workers 5 \
#     --topk 30

python scripts/evaluation/compute_context_stats.py \
    --context_dir data/output_sequences_s3_8_3_3_general \
    --output_file results/s3_8_3_3_general_tokens.json




