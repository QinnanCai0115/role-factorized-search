#!/usr/bin/env python3
"""Extract external orchestrator rounds and policy assistant turns from val JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _loads_maybe(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return value
    try:
        return json.loads(text)
    except Exception:
        return value


def _parse_policy_trace(value: Any) -> dict[str, Any]:
    parsed = _loads_maybe(value)
    if isinstance(parsed, dict):
        return parsed
    return {}


def _group_external_rounds(chain: Any) -> list[dict[str, Any]]:
    chain = _loads_maybe(chain)
    if not isinstance(chain, list):
        return []

    rounds: dict[int, dict[str, Any]] = {}
    order: list[int] = []
    for event in chain:
        if not isinstance(event, dict):
            continue
        try:
            round_idx = int(event.get("round"))
        except (TypeError, ValueError):
            continue

        if round_idx not in rounds:
            rounds[round_idx] = {"round": round_idx, "events": []}
            order.append(round_idx)

        stage = str(event.get("stage", "") or "")
        dst = rounds[round_idx]
        if stage == "backbone_output":
            dst["backbone_output"] = {
                "question": event.get("question", ""),
                "response": event.get("response", ""),
                "has_tool_call": bool(event.get("has_tool_call", False)),
            }
        elif stage == "policy_input":
            dst["policy_input"] = {
                "query": event.get("query", ""),
                "backbone_response": event.get("backbone_response", ""),
            }
        elif stage == "policy_output":
            dst.setdefault("policy_outputs", []).append(
                {
                    "request_id": event.get("request_id", ""),
                    "response": event.get("response", ""),
                    "full_trace_output": event.get("full_trace_output", ""),
                }
            )
        elif stage == "policy_decision":
            dst["policy_decision"] = {
                "binary_score": event.get("binary_score"),
                "has_answer_evidence": event.get("has_answer_evidence"),
                "continue_to_backbone": event.get("continue_to_backbone"),
                "terminated_after_policy": event.get("terminated_after_policy"),
                "validation_forced_continue": event.get("validation_forced_continue"),
            }
        elif stage == "backbone_final_output":
            dst["backbone_final_output"] = {
                "question": event.get("question", ""),
                "response": event.get("response", ""),
                "has_tool_call": bool(event.get("has_tool_call", False)),
            }
        else:
            dst["events"].append(event)

    return [rounds[i] for i in order]


def _extract_policy_assistant_rounds(chain: Any) -> list[dict[str, Any]]:
    chain = _loads_maybe(chain)
    if not isinstance(chain, list):
        return []

    assistant_rounds: list[dict[str, Any]] = []
    for event in chain:
        if not isinstance(event, dict) or event.get("stage") != "policy_output":
            continue
        try:
            orchestrator_round = int(event.get("round"))
        except (TypeError, ValueError):
            orchestrator_round = None

        policy_trace = _parse_policy_trace(event.get("full_trace_output", ""))
        trace_events = policy_trace.get("policy_trace", [])
        if not isinstance(trace_events, list):
            trace_events = []

        for trace_event in trace_events:
            if not isinstance(trace_event, dict):
                continue
            if trace_event.get("stage") != "assistant_output":
                continue
            assistant_rounds.append(
                {
                    "orchestrator_round": orchestrator_round,
                    "policy_request_id": event.get("request_id", ""),
                    "assistant_turn": trace_event.get("assistant_turn"),
                    "response_text": trace_event.get("response_text", ""),
                    "tool_calls": trace_event.get("tool_calls", []),
                }
            )

    return assistant_rounds


def _build_record(row: dict[str, Any], idx: int) -> dict[str, Any]:
    chain = row.get("orchestrator_chain", row.get("interaction", []))
    return {
        "record_type": "round_trace",
        "sample_index": idx,
        "step": row.get("step"),
        "uid": row.get("uid"),
        "source_uid": row.get("source_uid"),
        "request_id": row.get("request_id"),
        "data_source": row.get("data_source"),
        "score": row.get("score"),
        "ground_truth": row.get("gts", row.get("ground_truth")),
        "final_output": row.get("output"),
        "final_answer": row.get("backbone_final_answer", row.get("final_answer")),
        "final_answer_em": row.get("backbone_final_em", row.get("final_answer_em")),
        "final_answer_f1": row.get("backbone_final_f1", row.get("final_answer_f1")),
        "orchestrator_round_count": row.get("orchestrator_round_count"),
        "external_orchestrator_rounds": _group_external_rounds(chain),
        "internal_policy_assistant_rounds": _extract_policy_assistant_rounds(chain),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path, help="Validation JSONL produced by verl.")
    parser.add_argument("--output", required=True, type=Path, help="Round trace JSONL to write.")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with args.input.open("r", encoding="utf-8") as src, args.output.open("w", encoding="utf-8") as dst:
        for idx, line in enumerate(src):
            if not line.strip():
                continue
            row = json.loads(line)
            dst.write(json.dumps(_build_record(row, idx), ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} round traces to {args.output}")


if __name__ == "__main__":
    main()
