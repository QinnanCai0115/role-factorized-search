#!/usr/bin/env python3
"""Convert policy rollout SFT JSONL records to verl MultiTurnSFT parquet."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_INPUT = Path(
    "data/deepseek_policy_sft_rollouts/train_mixed_2000.deepseek_chat.assistant_turns_le3.sft.jsonl"
)
DEFAULT_OUTPUT = Path(
    "data/deepseek_policy_sft_rollouts/train_mixed_2000.deepseek_chat.assistant_turns_le3.verl_sft.parquet"
)


def normalize_message(message: dict[str, Any]) -> dict[str, str]:
    normalized = {
        "role": str(message.get("role", "")),
        "content": "" if message.get("content") is None else str(message.get("content")),
    }
    if message.get("name") is not None:
        normalized["name"] = str(message["name"])
    return normalized


def load_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    skipped = {
        "missing_answer_evidence": 0,
        "assistant_turns": 0,
        "missing_messages": 0,
        "limit": 0,
    }

    with args.input.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)

            if args.require_answer_evidence and not bool(record.get("has_answer_evidence")):
                skipped["missing_answer_evidence"] += 1
                continue

            assistant_turns = record.get("assistant_turns")
            if args.max_assistant_turns is not None:
                if not isinstance(assistant_turns, int) or assistant_turns > args.max_assistant_turns:
                    skipped["assistant_turns"] += 1
                    continue

            messages = record.get(args.messages_key)
            if not isinstance(messages, list) or not messages:
                skipped["missing_messages"] += 1
                continue

            rows.append(
                {
                    "messages": [normalize_message(msg) for msg in messages if isinstance(msg, dict)],
                    "source_index": record.get("source_index"),
                    "policy_round_id": record.get("policy_round_id"),
                    "round": record.get("round"),
                    "uid": record.get("uid"),
                    "question": record.get("question"),
                    "ground_truth": record.get("ground_truth"),
                    "policy_input": record.get("policy_input"),
                    "final_output": record.get("final_output"),
                    "has_answer_evidence": record.get("has_answer_evidence"),
                    "assistant_turns": assistant_turns,
                    "tool_call_count": record.get("tool_call_count"),
                    "api_model": record.get("api_model"),
                    "backbone_model": record.get("backbone_model"),
                    "created_at": record.get("created_at"),
                    "source_line": line_no,
                }
            )

            if args.limit is not None and len(rows) >= args.limit:
                skipped["limit"] += 1
                break

    print(f"loaded={len(rows)}")
    print("skipped=" + json.dumps(skipped, sort_keys=True))
    return rows


def write_parquet(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output)
    print(f"wrote {len(rows)} rows to {output}")
    print(f"columns: {table.column_names}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--messages-key", default="sft_messages")
    parser.add_argument("--max-assistant-turns", type=int, default=3)
    parser.add_argument("--require-answer-evidence", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--val-output", type=Path, default=None)
    parser.add_argument("--val-count", type=int, default=0)
    parser.add_argument("--val-ratio", type=float, default=0.0)
    args = parser.parse_args()

    if args.val_ratio < 0 or args.val_ratio >= 1:
        raise ValueError("--val-ratio must be in [0, 1)")
    if args.val_count < 0:
        raise ValueError("--val-count must be non-negative")
    if args.val_count and args.val_ratio:
        raise ValueError("Use only one of --val-count or --val-ratio")

    rows = load_rows(args)
    if args.shuffle or args.val_ratio > 0 or args.val_count > 0:
        random.Random(args.seed).shuffle(rows)

    if args.val_ratio > 0 or args.val_count > 0:
        val_count = args.val_count if args.val_count > 0 else max(1, int(round(len(rows) * args.val_ratio)))
        if val_count >= len(rows):
            raise ValueError(f"validation split is too large: {val_count=} for {len(rows)} rows")
        val_rows = rows[:val_count]
        train_rows = rows[val_count:]
        write_parquet(train_rows, args.output)
        if args.val_output is None:
            args.val_output = args.output.with_name(args.output.stem + ".val.parquet")
        write_parquet(val_rows, args.val_output)
    else:
        write_parquet(rows, args.output)


if __name__ == "__main__":
    main()
