#!/usr/bin/env python3
import argparse
import json
import re
import string
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


def normalize_answer(text: str) -> str:
    if text is None:
        return ""
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_answer(prediction) == normalize_answer(gold) else 0.0


def f1_score(prediction: str, gold: str) -> float:
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0

    pred_counts: dict[str, int] = {}
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


def postprocess_prediction(text: str) -> str:
    value = str(text or "")

    m = re.search(r"<answer>(.*?)</answer>", value, flags=re.IGNORECASE | re.DOTALL)
    if m:
        value = m.group(1)

    value = re.sub(r"<think>.*?</think>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<tool_call>.*?</tool_call>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<evidence>.*?</evidence>", " ", value, flags=re.IGNORECASE | re.DOTALL)
    value = re.sub(r"<[^>]+>", " ", value)

    for line in value.splitlines():
        line = line.strip()
        if line:
            return line
    return value.strip()


def extract_golds(row: dict[str, Any]) -> list[str]:
    golden_answers = row.get("golden_answers")
    if isinstance(golden_answers, list):
        vals = [str(x) for x in golden_answers if x is not None and str(x).strip()]
        if vals:
            return vals

    reward_model = row.get("reward_model")
    if isinstance(reward_model, dict):
        gt = reward_model.get("ground_truth")
        if isinstance(gt, dict):
            target = gt.get("target")
            if isinstance(target, list):
                vals = [str(x) for x in target if x is not None and str(x).strip()]
                if vals:
                    return vals
            if target is not None and str(target).strip():
                return [str(target)]

    return []


def build_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    prompt = row.get("prompt")
    if isinstance(prompt, list) and prompt:
        out = []
        for msg in prompt:
            if isinstance(msg, dict):
                out.append(
                    {
                        "role": str(msg.get("role", "user")),
                        "content": str(msg.get("content", "")),
                    }
                )
        if out:
            return out

    question = str(row.get("question", "")).strip()
    return [{"role": "user", "content": question}]


def generate_one(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: list[dict[str, str]],
    max_new_tokens: int,
) -> str:
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    gen_ids = outputs[0][inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True)
    return text.strip()


def evaluate_model(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    rows: list[dict[str, Any]],
    max_new_tokens: int,
    tag: str,
) -> list[dict[str, Any]]:
    preds: list[dict[str, Any]] = []
    for row in tqdm(rows, desc=f"Evaluating {tag}"):
        messages = build_messages(row)
        raw_pred = generate_one(model, tokenizer, messages, max_new_tokens=max_new_tokens)
        pred = postprocess_prediction(raw_pred)
        golds = extract_golds(row)

        if golds:
            em = max(exact_match(pred, g) for g in golds)
            f1 = max(f1_score(pred, g) for g in golds)
        else:
            em, f1 = 0.0, 0.0

        preds.append(
            {
                "id": str(row.get("id", "")),
                "question": str(row.get("question", "")),
                "golds": golds,
                "raw_prediction": raw_pred,
                "prediction": pred,
                "em": float(em),
                "f1": float(f1),
            }
        )
    return preds


def mean_metric(items: list[dict[str, Any]], key: str) -> float:
    if not items:
        return 0.0
    return float(sum(float(x.get(key, 0.0)) for x in items) / len(items))


def load_rows(parquet_path: Path, max_samples: int | None) -> list[dict[str, Any]]:
    df = pd.read_parquet(parquet_path)
    rows = df.to_dict(orient="records")
    if max_samples is not None and max_samples > 0:
        rows = rows[:max_samples]
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare base model vs LoRA checkpoint with EM/F1.")
    parser.add_argument("--data", type=Path, required=True, help="Evaluation parquet file path")
    parser.add_argument("--base-model", type=str, required=True, help="Base HF model path")
    parser.add_argument("--lora-adapter", type=str, required=True, help="LoRA adapter directory")
    parser.add_argument("--tokenizer", type=str, default=None, help="Tokenizer path, defaults to --base-model")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-samples", type=int, default=0, help="0 means all samples")
    parser.add_argument("--output", type=Path, required=True, help="Output JSON path")
    args = parser.parse_args()

    tokenizer_path = args.tokenizer or args.base_model
    max_samples = args.max_samples if args.max_samples > 0 else None

    rows = load_rows(args.data, max_samples=max_samples)
    if not rows:
        raise SystemExit("No evaluation samples loaded.")

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None

    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map=device_map,
    )
    base_model.eval()

    base_preds = evaluate_model(
        model=base_model,
        tokenizer=tokenizer,
        rows=rows,
        max_new_tokens=args.max_new_tokens,
        tag="base",
    )
    base_em = mean_metric(base_preds, "em")
    base_f1 = mean_metric(base_preds, "f1")

    lora_model = PeftModel.from_pretrained(base_model, args.lora_adapter)
    lora_model.eval()

    lora_preds = evaluate_model(
        model=lora_model,
        tokenizer=tokenizer,
        rows=rows,
        max_new_tokens=args.max_new_tokens,
        tag="step100_lora",
    )
    lora_em = mean_metric(lora_preds, "em")
    lora_f1 = mean_metric(lora_preds, "f1")

    samples = []
    for b, l in zip(base_preds, lora_preds, strict=True):
        samples.append(
            {
                "id": b["id"],
                "question": b["question"],
                "golds": b["golds"],
                "base_prediction": b["prediction"],
                "base_em": b["em"],
                "base_f1": b["f1"],
                "step100_prediction": l["prediction"],
                "step100_em": l["em"],
                "step100_f1": l["f1"],
                "delta_em": float(l["em"] - b["em"]),
                "delta_f1": float(l["f1"] - b["f1"]),
            }
        )

    result = {
        "config": {
            "data": str(args.data),
            "base_model": args.base_model,
            "lora_adapter": args.lora_adapter,
            "tokenizer": tokenizer_path,
            "max_new_tokens": args.max_new_tokens,
            "num_samples": len(rows),
        },
        "summary": {
            "base_em": base_em,
            "base_f1": base_f1,
            "step100_em": lora_em,
            "step100_f1": lora_f1,
            "delta_em": float(lora_em - base_em),
            "delta_f1": float(lora_f1 - base_f1),
        },
        "samples": samples,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("Saved to", args.output)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
