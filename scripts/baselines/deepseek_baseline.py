import argparse
import json
import os
import re
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import datasets
import requests
from tqdm import tqdm


DEFAULT_DATASETS = ["bamboogle"]
DEFAULT_API_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-reasoner"
DEFAULT_OUTPUT_DIR = "data/deepseek_baseline"

BASE_SYSTEM_PROMPT = (
    "You are a precise QA assistant. Solve the question carefully, then output only the final answer. "
    "Do not include explanations, reasoning, or extra text. "
    "If the answer is yes/no, output only yes or no. "
    "If the answer is a person, output only the person's name. "
    "If the answer is a place, output only the place name. "
    "If the answer is a date, output only the date."
)


def normalize_answer(s: str) -> str:
    if s is None:
        return ""
    s = str(s)

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction: str, golden_answers: List[str]) -> int:
    normalized_prediction = normalize_answer(prediction)
    for golden_answer in golden_answers:
        if normalize_answer(golden_answer) == normalized_prediction:
            return 1
    return 0


def token_f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if len(pred_tokens) == 0 and len(gold_tokens) == 0:
        return 1.0
    if len(pred_tokens) == 0 or len(gold_tokens) == 0:
        return 0.0

    common: Dict[str, int] = {}
    for token in pred_tokens:
        common[token] = common.get(token, 0) + 1

    num_same = 0
    for token in gold_tokens:
        if common.get(token, 0) > 0:
            num_same += 1
            common[token] -= 1

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)

# ...existing code...
def run_single_dataset(
    dataset_name: str,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    chosen_split, examples = load_flashrag_split(dataset_name, split=args.split)

    if args.limit is not None:
        examples = examples[: args.limit]

    results: List[Optional[Dict[str, Any]]] = [None] * len(examples)
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_one_example, idx, example, dataset_name, args): idx
            for idx, example in enumerate(examples)
        }

        with tqdm(total=len(futures), desc=dataset_name) as pbar:
            for future in as_completed(futures):
                result = future.result()
                idx = result["index"]
                with lock:
                    results[idx] = result
                pbar.update(1)

    ordered_results = [result for result in results if result is not None]
    summary = summarize_results(ordered_results)
    summary["dataset"] = dataset_name
    summary["split"] = chosen_split

    output_path = os.path.join(
        args.output_dir,
        f"{sanitize_filename(dataset_name)}_predictions.json",
    )
    save_json(
        {
            "dataset": dataset_name,
            "split": chosen_split,
            "model": args.model,
            "api_url": args.api_url,
            "summary": summary,
            "results": ordered_results,
        },
        output_path,
    )

    # 新增：导出 question / deepseek_answer / golden_answers
    qa_export_path = os.path.join(
        args.output_dir,
        f"{sanitize_filename(dataset_name)}_qa_pairs.json",
    )
    qa_records = [
        {
            "question": r.get("question", ""),
            "deepseek_answer": r.get("predicted_answer", ""),
            "golden_answers": r.get("golden_answers", []),
        }
        for r in ordered_results
    ]
    save_json(
        {
            "dataset": dataset_name,
            "split": chosen_split,
            "model": args.model,
            "count": len(qa_records),
            "records": qa_records,
        },
        qa_export_path,
    )

    return summary, ordered_results

def best_f1_across_golds(prediction: str, golden_answers: List[str]) -> float:
    if not golden_answers:
        return 0.0
    return max(token_f1_score(prediction, answer) for answer in golden_answers)


def sanitize_filename(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", name)


def strip_model_answer(text: str) -> str:
    if text is None:
        return ""

    answer = str(text).strip()
    answer = answer.strip("`\"' \n\t")

    patterns = [
        r"(?is)^final answer\s*[:：]\s*",
        r"(?is)^answer\s*[:：]\s*",
        r"(?is)^the answer is\s*",
    ]
    for pattern in patterns:
        answer = re.sub(pattern, "", answer).strip()

    lines = [line.strip() for line in answer.splitlines() if line.strip()]
    if len(lines) > 1:
        answer = lines[-1]

    return answer.strip("`\"' \n\t")


def ensure_question_mark(question: str) -> str:
    question = (question or "").strip()
    if question and question[-1] != "?":
        question += "?"
    return question


def extract_golden_answers(example: Dict[str, Any]) -> List[str]:
    candidate_keys = [
        "golden_answers",
        "answers",
        "answer",
        "target",
    ]

    for key in candidate_keys:
        if key not in example:
            continue
        value = example[key]
        if value is None:
            continue
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(x) for x in value if str(x).strip()]

    return []


def load_flashrag_split(dataset_name: str, split: Optional[str] = None) -> Tuple[str, List[Dict[str, Any]]]:
    dataset = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", dataset_name)

    if split is not None:
        chosen_split = split
    elif "test" in dataset:
        chosen_split = "test"
    elif "dev" in dataset:
        chosen_split = "dev"
    else:
        chosen_split = "train"

    records = [dict(example) for example in dataset[chosen_split]]

    if dataset_name == "popqa":
        seen_questions = set()
        deduped_records = []
        for example in records:
            question = ensure_question_mark(example.get("question", ""))
            if question in seen_questions:
                continue
            seen_questions.add(question)
            example["question"] = question
            deduped_records.append(example)
        records = deduped_records
    else:
        for example in records:
            example["question"] = ensure_question_mark(example.get("question", ""))

    return chosen_split, records


def build_messages(question: str) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": BASE_SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


def call_deepseek_reasoner(
    question: str,
    api_url: str,
    model: str,
    api_key: str,
    timeout: int,
    max_retries: int,
    temperature: float,
    max_tokens: Optional[int],
) -> Tuple[str, Dict[str, Any]]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": build_messages(question),
        "temperature": temperature,
    }
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    last_err: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            response = requests.post(
                f"{api_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=timeout,
            )
            response.raise_for_status()
            data = response.json()
            message = data["choices"][0]["message"]
            content = strip_model_answer(message.get("content", ""))
            return content, data
        except Exception as exc:
            last_err = exc
            time.sleep(min(2 ** attempt, 8))

    raise last_err if last_err is not None else RuntimeError("Unknown API error")


def process_one_example(
    idx: int,
    example: Dict[str, Any],
    dataset_name: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    question = ensure_question_mark(example.get("question", ""))
    golden_answers = extract_golden_answers(example)

    if not question:
        return {
            "index": idx,
            "dataset": dataset_name,
            "question": "",
            "golden_answers": golden_answers,
            "predicted_answer": "",
            "em": 0.0,
            "f1": 0.0,
            "error": "Empty question",
        }

    try:
        predicted_answer, raw_response = call_deepseek_reasoner(
            question=question,
            api_url=args.api_url,
            model=args.model,
            api_key=args.api_key,
            timeout=args.timeout,
            max_retries=args.max_retries,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        error = None
    except Exception as exc:
        predicted_answer = ""
        raw_response = {}
        error = str(exc)

    em = float(exact_match_score(predicted_answer, golden_answers)) if golden_answers else 0.0
    f1 = float(best_f1_across_golds(predicted_answer, golden_answers)) if golden_answers else 0.0

    return {
        "index": idx,
        "dataset": dataset_name,
        "question": question,
        "golden_answers": golden_answers,
        "predicted_answer": predicted_answer,
        "em": em,
        "f1": f1,
        "error": error,
        "raw_response": raw_response if args.save_raw_response else None,
    }


def save_json(data: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def summarize_results(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not records:
        return {
            "count": 0,
            "em": 0.0,
            "f1": 0.0,
            "error_count": 0,
        }

    count = len(records)
    avg_em = sum(record["em"] for record in records) / count
    avg_f1 = sum(record["f1"] for record in records) / count
    error_count = sum(1 for record in records if record.get("error"))
    return {
        "count": count,
        "em": avg_em,
        "f1": avg_f1,
        "error_count": error_count,
    }


def run_single_dataset(
    dataset_name: str,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    chosen_split, examples = load_flashrag_split(dataset_name, split=args.split)

    if args.limit is not None:
        examples = examples[: args.limit]

    results: List[Optional[Dict[str, Any]]] = [None] * len(examples)
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_one_example, idx, example, dataset_name, args): idx
            for idx, example in enumerate(examples)
        }

        with tqdm(total=len(futures), desc=dataset_name) as pbar:
            for future in as_completed(futures):
                result = future.result()
                idx = result["index"]
                with lock:
                    results[idx] = result
                pbar.update(1)

    ordered_results = [result for result in results if result is not None]
    summary = summarize_results(ordered_results)
    summary["dataset"] = dataset_name
    summary["split"] = chosen_split

    output_path = os.path.join(
        args.output_dir,
        f"{sanitize_filename(dataset_name)}_predictions.json",
    )
    save_json(
        {
            "dataset": dataset_name,
            "split": chosen_split,
            "model": args.model,
            "api_url": args.api_url,
            "summary": summary,
            "results": ordered_results,
        },
        output_path,
    )

    return summary, ordered_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DeepSeek reasoner baseline on FlashRAG QA datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="Datasets to evaluate. Default: hotpotqa popqa musique bamboogle",
    )
    parser.add_argument(
        "--split",
        default=None,
        help="Dataset split to use. Default: test, then dev, then train if unavailable.",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to save predictions and summary JSON files.",
    )
    parser.add_argument(
        "--api_url",
        default=DEFAULT_API_URL,
        help="DeepSeek OpenAI-compatible API base URL.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Model name. Default: deepseek-reasoner",
    )
    parser.add_argument(
        "--api_key",
        default=os.environ.get("DEEPSEEK_API_KEY"),
        help="DeepSeek API key. Defaults to DEEPSEEK_API_KEY env var.",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of concurrent API workers.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max_retries",
        type=int,
        default=3,
        help="Number of retries after API failures.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=128,
        help="Maximum completion tokens.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of examples per dataset for debugging.",
    )
    parser.add_argument(
        "--save_raw_response",
        action="store_true",
        help="Whether to keep raw API responses in the saved predictions.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.api_key:
        raise ValueError("DeepSeek API key is required. Pass --api_key or set DEEPSEEK_API_KEY.")

    os.makedirs(args.output_dir, exist_ok=True)

    all_summaries = []
    all_results: List[Dict[str, Any]] = []

    for dataset_name in args.datasets:
        summary, results = run_single_dataset(dataset_name, args)
        all_summaries.append(summary)
        all_results.extend(results)
        print(
            f"[{dataset_name}] split={summary['split']} count={summary['count']} "
            f"EM={summary['em']:.4f} F1={summary['f1']:.4f} errors={summary['error_count']}"
        )

    overall = summarize_results(all_results)
    overall["datasets"] = args.datasets

    summary_path = os.path.join(args.output_dir, "summary.json")
    save_json(
        {
            "model": args.model,
            "api_url": args.api_url,
            "dataset_summaries": all_summaries,
            "overall": overall,
        },
        summary_path,
    )

    print(
        f"[overall] count={overall['count']} EM={overall['em']:.4f} "
        f"F1={overall['f1']:.4f} errors={overall['error_count']}"
    )
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
