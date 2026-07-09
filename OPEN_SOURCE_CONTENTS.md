# Open-source contents

This clean repository was assembled from `/ai/cqn/datacon` using a whitelist.
It intentionally excludes the original `.git` history, secrets, logs, checkpoints,
TensorBoard files, temporary analysis files, generated outputs, and large rollout artifacts.

Included:
- `verl/`: the SFT training framework used by `python -m verl.trainer.sft_trainer`.
- `scripts/examples/search_r1_like/`: rollout generation, SFT conversion/training, judging, replay, and diagnostics.
- `scripts/baselines/`: baseline scripts used for comparison experiments.
- `scripts/evaluation/`: metric and evaluation utilities.
- `data/hotpotqa_2wiki_musique_train/`: base train/validation/test splits.
- `data/FlashRAG_datasets/bamboogle/test.jsonl`: Bamboogle test source.

Not included:
- The previous scaffold search/generation utility package is not part of this release; the final SFT workflow uses `verl/` and `scripts/examples/search_r1_like/`.
- Large SFT rollout traces and generated model outputs.
- Checkpoints, logs, TensorBoard files, temporary analysis files, and private secret files.

Large SFT rollout data should be released separately as a sanitized dataset artifact.
