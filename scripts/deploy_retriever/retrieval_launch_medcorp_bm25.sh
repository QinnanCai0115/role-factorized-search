# export CUDA_VISIBLE_DEVICES=6,7
save_path=/shared/eng/pj20/s3_medcorp

index_file=$save_path/bm25/bm25
corpus_file=$save_path/medcorpus.jsonl
retriever_name=bm25

python s3/search/retrieval_server.py \
    --index_path $index_file \
    --corpus_path $corpus_file \
    --topk 12 \
    --retriever_name $retriever_name \
    --port 4000