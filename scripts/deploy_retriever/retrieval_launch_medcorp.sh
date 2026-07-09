export CUDA_VISIBLE_DEVICES=4,5,6,7
file_path=/shared/eng/pj20/s3_medcorp
index_file=$file_path/e5_Flat.index
corpus_file=$file_path/medcorpus.jsonl
retriever_name=e5
retriever_path=intfloat/e5-base-v2

python s3/search/retrieval_server.py --index_path $index_file \
                                            --corpus_path $corpus_file \
                                            --topk 12 \
                                            --retriever_name $retriever_name \
                                            --retriever_model $retriever_path \
                                            --faiss_gpu \
                                            --port 3000
                                            # --port 7000
