
### Process the dataset

```bash
sh scripts/nq_hotpotqa/data_process_ug.sh
```

### Run Inference
```bash
sh generator_llms/host.sh

python scripts/inference/context.py --input_file data/nq_hotpotqa_train/test_e5_ug.parquet --result_file results/u1_step150.json --context_dir data/output_sequences_150 --num_workers 16
```

The context file is saved in the following directory:

```bash
data/output_sequences_150
├── 2wikimultihopqa_output_sequences.json
├── bamboogle_output_sequences.json
├── hotpotqa_output_sequences.json
├── musique_output_sequences.json
├── nq_output_sequences.json
├── popqa_output_sequences.json
└── triviaqa_output_sequences.json
```

Each of file has the following format:
```json
{
  "Who is the spouse of the Green performer?": {
    "context_with_info": "..."
  },
  ...
} # key is question, value is context_with_info
```