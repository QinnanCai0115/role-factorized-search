# precompute the naïve RAG Cache for training

# preconstruct dataset without RAG Retrieval
bash scripts/dataset_construct/data_process_s3_pre.sh 

# prepare Naïve RAG Retrieval for Training and Test Set
bash scripts/baselines/run_retrieval.sh 

# Optional: run Generator with RAG Retrieval and save the RAG Cache
# Set USE_RAG_CACHE=true to enable legacy cache-based filtering.
USE_RAG_CACHE=${USE_RAG_CACHE:-false}
if [ "$USE_RAG_CACHE" = "true" ]; then
	bash scripts/evaluation/run_rag_cache.sh
fi

# construct dataset with RAG Retrieval
bash scripts/dataset_construct/data_process_s3_rl.sh