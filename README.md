# Role-Factorized Search

Code and base data splits for **Think Big, Search Small: Where Capacity Matters in Hierarchical Search Agents?**

- Paper: [arXiv:2607.07548](https://arxiv.org/abs/2607.07548)
- PDF: [https://arxiv.org/pdf/2607.07548](https://arxiv.org/pdf/2607.07548)
- Authors: Qinnan Cai, Yibo Zhao, Xiang Li

## Overview

Large language model search agents often use hierarchical or multi-agent designs, but prior systems usually instantiate the main agent and sub-agents with the same model scale. This project studies where model capacity actually matters in hierarchical search.

We factorize the search pipeline into three roles:

- **Delegation**: the backbone/orchestrator decomposes the original question into sub-queries.
- **Execution**: the sub-agent performs retrieval and extracts evidence for delegated sub-queries.
- **Answer generation**: the final answer generator is held fixed to isolate delegation and execution effects.

Main findings from the paper:

- Role-factorized search consistently improves over single-agent search across model scales.
- Capacity is asymmetric across roles: scaling the delegation backbone matters much more than scaling the execution sub-agent, suggesting decomposition is the main bottleneck.
- A small 1.7B executor trained with quality-filtered trajectory distillation can match a frontier sub-agent while using substantially fewer sub-agent tokens.

## Citation

If you use this code or data, please cite:

```bibtex
@misc{cai2026thinkbigsearchsmall,
  title         = {Think Big, Search Small: Where Capacity Matters in Hierarchical Search Agents?},
  author        = {Qinnan Cai and Yibo Zhao and Xiang Li},
  year          = {2026},
  eprint        = {2607.07548},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2607.07548}
}
```

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
python scripts/examples/search_r1_like/generate_sft_rollout.py \
  --input data/hotpotqa_2wiki_musique_train/train_mixed_2000_sft.jsonl \
  --output data/qwen3_policy_sft_rollouts/train_mixed_2000.policy.sft.jsonl
```

### Convert SFT JSONL to Parquet

```bash
python scripts/examples/search_r1_like/convert_policy_sft_jsonl_to_parquet.py \
  --input path/to/policy.sft.jsonl \
  --output path/to/policy.verl_sft.parquet
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
