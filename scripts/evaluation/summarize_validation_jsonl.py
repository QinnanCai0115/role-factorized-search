import argparse
import json
import math
from pathlib import Path
import re
from statistics import mean
import string


def _safe_mean(values):
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return None
    return mean(vals)


def _format_float(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:.6f}"


def _normalize_answer(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def _exact_match(prediction: str, gold: str) -> float:
    return 1.0 if _normalize_answer(prediction) == _normalize_answer(gold) else 0.0


def _f1_score(prediction: str, gold: str) -> float:
    pred_tokens = _normalize_answer(prediction).split()
    gold_tokens = _normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts = {}
    for tok in pred_tokens:
        pred_counts[tok] = pred_counts.get(tok, 0) + 1
    common = 0
    for tok in gold_tokens:
        cnt = pred_counts.get(tok, 0)
        if cnt > 0:
            common += 1
            pred_counts[tok] = cnt - 1

    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _extract_targets(row: dict) -> list[str]:
    gts = row.get("gts")
    if not isinstance(gts, dict):
        return []
    target = gts.get("target")
    if target is None:
        return []
    if isinstance(target, list):
        return [str(t) for t in target if t is not None]
    return [str(target)]


def _postprocess_prediction(text: str) -> str:
    if text is None:
        return ""
    value = str(text)

    # Prefer structured answer block when present.
    m = re.search(r"<answer>(.*?)</answer>", value, flags=re.IGNORECASE | re.DOTALL)
    if m:
        value = m.group(1)

    # Remove think/tool/evidence blocks and generic XML tags.
    value = re.sub(r"<think>.*?</think>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<tool_call>.*?</tool_call>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<evidence>.*?</evidence>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<[^>]+>", " ", value)
    value = value.strip()

    # Keep the first non-empty line to avoid long chain-of-thought tails.
    for line in value.splitlines():
        line = line.strip()
        if line:
            value = line
            break

    return value.strip()


def summarize_file(path: Path, *, compute_qa_metrics: bool, backbone_only: bool):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if backbone_only:
                inp = str(row.get("input") or "")
                if "You are a frozen backbone reasoning model." not in inp:
                    continue
            rows.append(row)

    metrics = {
        "step": int(path.stem),
        "count": len(rows),
        "score_mean": _safe_mean(row.get("score") for row in rows),
        "reward_mean": _safe_mean(row.get("reward") for row in rows),
        "binary_reward_mean": _safe_mean(row.get("binary_reward") for row in rows),
    }

    if compute_qa_metrics:
        raw_ems = []
        raw_f1s = []
        post_ems = []
        post_f1s = []
        qa_count = 0

        for row in rows:
            targets = _extract_targets(row)
            if not targets:
                continue

            qa_count += 1
            prediction = str(row.get("output") or "")
            post_pred = _postprocess_prediction(prediction)

            raw_ems.append(max(_exact_match(prediction, gold) for gold in targets))
            raw_f1s.append(max(_f1_score(prediction, gold) for gold in targets))
            post_ems.append(max(_exact_match(post_pred, gold) for gold in targets))
            post_f1s.append(max(_f1_score(post_pred, gold) for gold in targets))

        metrics.update(
            {
                "qa_count": qa_count,
                "raw_em": _safe_mean(raw_ems),
                "raw_f1": _safe_mean(raw_f1s),
                "post_em": _safe_mean(post_ems),
                "post_f1": _safe_mean(post_f1s),
            }
        )

    return metrics


def print_table(metrics_list):
    headers = [
        "step",
        "count",
        "score_mean",
        "reward_mean",
        "binary_reward_mean",
        "delta_score_vs_prev",
        "qa_count",
        "raw_em",
        "raw_f1",
        "post_em",
        "post_f1",
    ]
    print("\t".join(headers))

    prev_score = None
    for metrics in metrics_list:
        score = metrics["score_mean"]
        delta = None if prev_score is None or score is None else score - prev_score
        print(
            "\t".join(
                [
                    str(metrics["step"]),
                    str(metrics["count"]),
                    _format_float(score),
                    _format_float(metrics["reward_mean"]),
                    _format_float(metrics["binary_reward_mean"]),
                    _format_float(delta),
                    str(metrics.get("qa_count", "-")),
                    _format_float(metrics.get("raw_em")),
                    _format_float(metrics.get("raw_f1")),
                    _format_float(metrics.get("post_em")),
                    _format_float(metrics.get("post_f1")),
                ]
            )
        )
        prev_score = score


def main():
    parser = argparse.ArgumentParser(description="Summarize VERL validation_data JSONL files by step.")
    parser.add_argument(
        "validation_dir",
        type=Path,
        help="Directory containing step-named JSONL files, e.g. validation_data/20.jsonl",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON instead of a tab-separated table.",
    )
    parser.add_argument(
        "--compute-qa-metrics",
        action="store_true",
        help="Compute raw/postprocessed EM/F1 against gts.target from each JSONL row.",
    )
    parser.add_argument(
        "--backbone-only",
        action="store_true",
        help="Only evaluate rows whose input contains the frozen backbone prompt.",
    )
    args = parser.parse_args()

    files = sorted(
        [path for path in args.validation_dir.glob("*.jsonl") if path.stem.isdigit()],
        key=lambda p: int(p.stem),
    )
    if not files:
        raise SystemExit(f"No step JSONL files found in {args.validation_dir}")

    metrics_list = [
        summarize_file(path, compute_qa_metrics=args.compute_qa_metrics, backbone_only=args.backbone_only)
        for path in files
    ]

    if args.json:
        print(json.dumps(metrics_list, ensure_ascii=False, indent=2))
    else:
        print_table(metrics_list)


if __name__ == "__main__":
    main()
