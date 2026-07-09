#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/ai/cqn/datacon}"
PYTHON_BIN="${PYTHON_BIN:-/ai/cqn/miniconda3/envs/verl/bin/python}"
BASE="${BASE:-/ai/cqn/s3/ckpt/search_subagent_api_policy_test_all_nonthinking}"
JUDGE_WORKERS="${JUDGE_WORKERS:-8}"
BACKBONE_JUDGE_MAX_TOKENS="${BACKBONE_JUDGE_MAX_TOKENS:-8192}"

cd "$PROJECT_DIR"

"$PYTHON_BIN" - "$BASE" "$JUDGE_WORKERS" "$BACKBONE_JUDGE_MAX_TOKENS" <<'PY'
import json
import os
import statistics
import sys
from argparse import Namespace
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, "/ai/cqn/datacon")
from scripts.examples.search_r1_like.generate_sft_rollout import (
    judge_backbone_final_answer,
    load_env_file,
)

base = Path(sys.argv[1])
workers = int(sys.argv[2])
judge_max_tokens = int(sys.argv[3])

for env_file in [
    "/ai/cqn/datacon/.secrets/deepseek.env",
    "/ai/cqn/s3/.secrets/deepseek.env",
]:
    load_env_file(env_file)

args = Namespace(
    backbone_judge_api_url=os.environ.get("BACKBONE_JUDGE_API_URL", "https://api.deepseek.com/v1"),
    backbone_judge_api_key=os.environ.get("BACKBONE_JUDGE_API_KEY") or os.environ.get("DEEPSEEK_API_KEY") or "",
    backbone_judge_model=os.environ.get("BACKBONE_JUDGE_MODEL", "deepseek-reasoner"),
    backbone_judge_max_tokens=judge_max_tokens,
    api_timeout=float(os.environ.get("API_TIMEOUT", "180")),
    api_max_retries=int(os.environ.get("API_MAX_RETRIES", "4")),
    no_proxy=True,
)

if not args.backbone_judge_api_key and not args.backbone_judge_api_url.startswith(("http://127.0.0.1", "http://localhost")):
    raise SystemExit("Missing judge API key: set BACKBONE_JUDGE_API_KEY or DEEPSEEK_API_KEY.")


def get_targets(row):
    targets = row.get("ground_truth_targets")
    if isinstance(targets, list) and targets:
        return [str(item) for item in targets if str(item).strip()]
    ground_truth = row.get("ground_truth")
    if isinstance(ground_truth, list):
        return [str(item) for item in ground_truth if str(item).strip()]
    if ground_truth is not None and str(ground_truth).strip():
        return [str(ground_truth).strip()]
    return []


def load_existing(path):
    existing = {}
    if not path.exists():
        return existing
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            existing[str(row.get("source_index"))] = row
    return existing


def summarize(rows, run_name, output_path):
    scores = [float(row["score"]) for row in rows if row.get("score") is not None]
    by_source = defaultdict(list)
    for row in rows:
        if row.get("score") is not None:
            by_source[str(row.get("final_answer_source") or "<none>")].append(float(row["score"]))
    return {
        "run": run_name,
        "judge_model": args.backbone_judge_model,
        "judge_max_tokens": judge_max_tokens,
        "output_path": str(output_path),
        "answered_count": len(rows),
        "judged_count": len(scores),
        "judge_score_mean": statistics.fmean(scores) if scores else None,
        "judge_score_mean_by_source": {
            source: statistics.fmean(values)
            for source, values in sorted(by_source.items())
        },
        "final_answer_source_counts": dict(Counter(str(row.get("final_answer_source") or "<none>") for row in rows)),
        "error_count": sum(1 for row in rows if row.get("error")),
        "error_counts": dict(Counter(str(row.get("error")) for row in rows if row.get("error"))),
    }


combined = []
for run in sorted(path for path in base.iterdir() if path.is_dir()):
    trace_path = run / "orchestrator_traces.jsonl"
    if not trace_path.exists():
        continue

    output_path = run / f"final_answer_gt_judge_max{judge_max_tokens}.jsonl"
    summary_path = run / f"final_answer_gt_judge_max{judge_max_tokens}.summary.json"
    existing = load_existing(output_path)

    pending = []
    with trace_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            final_answer = str(row.get("final_answer") or "").strip()
            question = str(row.get("question") or "").strip()
            targets = get_targets(row)
            source_index = row.get("source_index")
            if not final_answer or not question or not targets:
                continue
            if str(source_index) in existing:
                continue
            pending.append(
                {
                    "source_index": source_index,
                    "uid": row.get("uid"),
                    "data_source": row.get("data_source"),
                    "final_answer_source": row.get("final_answer_source") or "<none>",
                    "question": question,
                    "ground_truth_targets": targets,
                    "final_answer": final_answer,
                    "final_em": row.get("final_em"),
                    "final_f1": row.get("final_f1"),
                }
            )

    print(
        f"[max_tokens={judge_max_tokens}] {run.name}: "
        f"existing={len(existing)} pending={len(pending)} output={output_path}",
        flush=True,
    )

    def judge_one(item):
        judge = judge_backbone_final_answer(
            question=item["question"],
            ground_truth_targets=item["ground_truth_targets"],
            predicted_answer=item["final_answer"],
            args=args,
        )
        return {
            **item,
            "judge_max_tokens": judge_max_tokens,
            "judge_model": args.backbone_judge_model,
            **judge,
        }

    with output_path.open("a", encoding="utf-8") as wf:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(judge_one, item) for item in pending]
            for idx, future in enumerate(as_completed(futures), 1):
                wf.write(json.dumps(future.result(), ensure_ascii=False) + "\n")
                wf.flush()
                if idx % 50 == 0:
                    print(f"[max_tokens={judge_max_tokens}] {run.name}: {idx}/{len(pending)}", flush=True)

    rows = list(load_existing(output_path).values())
    summary = summarize(rows, run.name, output_path)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    combined.append(summary)

combined_path = base / f"final_answer_gt_judge_max{judge_max_tokens}.combined_summary.json"
combined_path.write_text(json.dumps(combined, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(f"combined_summary={combined_path}", flush=True)
PY
