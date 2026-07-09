#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
PYTHON_BIN=${PYTHON_BIN:-/root/miniconda3/envs/ret/bin/python}

# Put HF/datasets cache on a disk with enough free space.
export HF_HOME=${HF_HOME:-/root/shared_planing/hf_cache}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
export TRANSFORMERS_CACHE=${TRANSFORMERS_CACHE:-$HF_HOME/hub}
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

file_path=${FILE_PATH:-$REPO_ROOT/wiki_data}
index_file=${INDEX_FILE:-$file_path/e5_Flat.index}
corpus_file=${CORPUS_FILE:-$file_path/wiki-18.jsonl}
retriever_name=${RETRIEVER_NAME:-e5}
retriever_path=${RETRIEVER_PATH:-intfloat/e5-base-v2}
retriever_port=${RETRIEVER_PORT:-3000}

if [ ! -f "$index_file" ]; then
    echo "[ERROR] Missing FAISS index: $index_file"
    exit 1
fi

if [ ! -f "$corpus_file" ]; then
    echo "[ERROR] Missing corpus file: $corpus_file"
    exit 1
fi

if [ ! -x "$PYTHON_BIN" ]; then
    echo "[ERROR] PYTHON_BIN is not executable: $PYTHON_BIN"
    exit 1
fi

"$PYTHON_BIN" "$REPO_ROOT/s3/search/retrieval_server.py" --index_path "$index_file" \
    --corpus_path "$corpus_file" \
    --topk 12 \
    --retriever_name "$retriever_name" \
    --retriever_model "$retriever_path" \
    --faiss_gpu \
    --port "$retriever_port"
