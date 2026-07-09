#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def build_prompt(question: str) -> list[dict[str, str]]:
    return [{"role": "user", "content": question}]


def convert_row(row: dict) -> dict:
    question = str(row.get("question", "")).strip()
    answer = row.get("answer", "")
    dataset = str(row.get("dataset", "default")).strip() or "default"

    extra_info = {
        "question": question,
        "answer": answer,
        "id": row.get("id"),
        "dataset": dataset,
        "split": row.get("split"),
        "positive_doc_ids": row.get("positive_doc_ids", []),
    }

    converted = {
        "prompt": build_prompt(question),
        "data_source": dataset,
        "reward_model": {"ground_truth": answer},
        "extra_info": extra_info,
        "question": question,
        "answer": answer,
        "id": row.get("id"),
        "dataset": dataset,
        "split": row.get("split"),
        "positive_doc_ids": row.get("positive_doc_ids", []),
    }
    return converted


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no}: {exc}") from exc
            rows.append(convert_row(row))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Search-R1 style JSONL QA data to training parquet.")
    parser.add_argument("input", type=Path, help="Input JSONL path")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output parquet path. Defaults to replacing the input suffix with .parquet",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output or input_path.with_suffix(".parquet")

    rows = load_jsonl(input_path)
    table = pa.Table.from_pylist(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)

    print(f"wrote {len(rows)} rows to {output_path}")
    print(f"columns: {table.column_names}")


if __name__ == "__main__":
    main()
