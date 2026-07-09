#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import datasets
import pyarrow as pa
import pyarrow.parquet as pq


DATASET_NAMES = ("hotpotqa", "2wikimultihopqa", "musique")


def _pick_eval_split(ds_dict: datasets.DatasetDict, data_source: str) -> tuple[str, datasets.Dataset]:
    if "test" in ds_dict:
        return "test", ds_dict["test"]
    if "dev" in ds_dict:
        return "dev", ds_dict["dev"]
    if "validation" in ds_dict:
        return "validation", ds_dict["validation"]
    raise ValueError(f"{data_source} has no test/dev/validation split")


def _normalize_question(question: str) -> str:
    text = str(question).strip()
    if text and not text.endswith("?"):
        text += "?"
    return text


def _extract_answers(example: dict) -> list[str]:
    answers = example.get("golden_answers", [])
    if isinstance(answers, list):
        return [str(x) for x in answers if str(x).strip()]
    if answers is None:
        return []
    return [str(answers)]


def _convert_example(example: dict, *, data_source: str, split: str, index: int) -> dict:
    question = _normalize_question(example.get("question", ""))
    answers = _extract_answers(example)
    answer = answers[0] if answers else ""
    supporting_facts = example.get("supporting_facts", [])

    return {
        "prompt": [{"role": "user", "content": question}],
        "data_source": data_source,
        "reward_model": {"ground_truth": answer},
        "extra_info": {
            "question": question,
            "answers": answers,
            "split": split,
            "index": index,
            "supporting_facts": supporting_facts,
        },
        "question": question,
        "answer": answer,
        "answers": answers,
        "dataset": data_source,
        "split": split,
        "supporting_facts": supporting_facts,
    }


def build_rows(sample_per_source: int, seed: int) -> list[dict]:
    rows: list[dict] = []
    for data_source in DATASET_NAMES:
        ds_dict = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", data_source)
        split_name, eval_ds = _pick_eval_split(ds_dict, data_source)
        if len(eval_ds) < sample_per_source:
            raise ValueError(
                f"{data_source} {split_name} split only has {len(eval_ds)} rows, fewer than requested {sample_per_source}"
            )
        sampled = eval_ds.shuffle(seed=seed).select(range(sample_per_source))
        for idx, example in enumerate(sampled):
            rows.append(_convert_example(example, data_source=data_source, split=split_name, index=idx))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a mixed 900-sample validation parquet from FlashRAG datasets.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/ai/cqn/s3/data/hotpotqa_2wiki_musique_train/val_mixed_900.parquet"),
        help="Output parquet path",
    )
    parser.add_argument("--sample-per-source", type=int, default=300, help="Samples to draw from each dataset")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    args = parser.parse_args()

    rows = build_rows(sample_per_source=args.sample_per_source, seed=args.seed)
    table = pa.Table.from_pylist(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")
    print(f"columns: {table.column_names}")


if __name__ == "__main__":
    main()
