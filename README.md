# Role-Factorized Search

This repository contains the open-release code and base data splits for role-factorized search experiments. The core setup separates the backbone/orchestrator role from the sub-agent execution role, and uses supervised fine-tuning data generated from search-oriented rollouts.

The repository is intentionally kept small and reproducible. Large rollout traces, checkpoints, logs, TensorBoard files, and private server paths are excluded from this code release and should be published separately as dataset artifacts after sanitization.

## Contents

- `verl/`: SFT training framework. The main trainer used by this project is `python -m verl.trainer.sft_trainer`.
- `scripts/examples/search_r1_like/`: experiment scripts for rollout generation, SFT data conversion, SFT training, judging, replay, and diagnostics.
- `scripts/baselines/`: baseline scripts such as no-retrieval, direct-search, Search-R1-style, and retrieval baselines.
- `scripts/evaluation/`: EM/F1, context, and evaluation utilities.
- `data/hotpotqa_2wiki_musique_train/`: base train/validation/test splits.
- `data/FlashRAG_datasets/bamboogle/test.jsonl`: Bamboogle test source.

## Key Scripts

### Generate SFT Rollouts

```bash
python scripts/examples/search_r1_like/generate_sft_rollout.py   --input data/hotpotqa_2wiki_musique_train/train_mixed_2000_sft.jsonl   --output data/qwen3_policy_sft_rollouts/train_mixed_2000.policy.sft.jsonl
```

### Convert SFT JSONL to Parquet

```bash
python scripts/examples/search_r1_like/convert_policy_sft_jsonl_to_parquet.py   --input path/to/policy.sft.jsonl   --output path/to/policy.verl_sft.parquet
```

### Run SFT

```bash
bash scripts/examples/search_r1_like/run_policy_sft.sh
```

The SFT entrypoint is:

```bash
python -m verl.trainer.sft_trainer --config-name=sft_trainer_engine
```

Relevant config:

```text
verl/trainer/config/sft_trainer_engine.yaml
```

### Score or Inspect Rollouts

```bash
python scripts/examples/search_r1_like/score_sft_rollouts_with_backbone_judge.py
python scripts/examples/search_r1_like/diagnose_policy_sft_format.py
python scripts/examples/search_r1_like/extract_round_traces.py
```

## Data Included

The base split directory contains:

```text
data/hotpotqa_2wiki_musique_train/test_all.jsonl
data/hotpotqa_2wiki_musique_train/test_all.parquet
data/hotpotqa_2wiki_musique_train/train_mixed_2000_sft.jsonl
data/hotpotqa_2wiki_musique_train/train_mixed_7000_rl.jsonl
data/hotpotqa_2wiki_musique_train/train_mixed_7000_rl.parquet
data/hotpotqa_2wiki_musique_train/train_mixed_9000.jsonl
data/hotpotqa_2wiki_musique_train/train_mixed_9000.parquet
data/hotpotqa_2wiki_musique_train/train_mixed_extra3000_balanced_sft_seed42.jsonl
data/hotpotqa_2wiki_musique_train/val_mixed_900.parquet
```

Large SFT rollout files are not included in this repository. Recommended release layout:

```text
dataset_release/
  base_splits/
  sft_jsonl/
  sft_parquet/
  summaries/
  data_card.md
```

## Notes for Reproducibility

Some scripts still expose configurable defaults for local data, model, or output paths. Before running on a new machine, set these environment variables or pass explicit command-line arguments:

```bash
export PROJECT_DIR=/path/to/role-factorized-search
export DATA_DIR=$PROJECT_DIR/data
export MODEL_PATH=/path/to/model
export OUTPUT_DIR=/path/to/output
```

No secrets, checkpoints, logs, TensorBoard files, or original git history are included in this open-release scaffold.
