# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import os
import re
from typing import Any, Optional

import requests

from verl.utils.reward_score import search_r1_like_qa_em


def _extract_tool_stats(extra_info: dict[str, Any]) -> tuple[int, int]:
    extras = extra_info.get("extras", {}) if isinstance(extra_info, dict) else {}
    subagent_trajs = extras.get("search_subagent", {}) if isinstance(extras, dict) else {}
    total_calls = 0
    unique_queries = set()
    if isinstance(subagent_trajs, dict):
        for _, steps in subagent_trajs.items():
            if not isinstance(steps, list):
                continue
            total_calls += len(steps)
            for step in steps:
                q = step.get("query") if isinstance(step, dict) else None
                if isinstance(q, str) and q.strip():
                    unique_queries.add(re.sub(r"\s+", " ", q.strip().lower()))
    return total_calls, len(unique_queries)


def _judge_with_orchestrator_api(
    judge_api_url: str,
    question: str,
    answer: str,
    extra_info: dict[str, Any],
    timeout: float,
) -> Optional[float]:
    payload = {
        "question": question,
        "answer": answer,
        "trajectory": extra_info.get("extras", {}),
    }
    disable_backbone_proxy = str(os.environ.get("BACKBONE_API_NO_PROXY", "")).strip().lower() not in (
        "",
        "0",
        "false",
        "no",
    )
    with requests.Session() as session:
        if disable_backbone_proxy:
            session.trust_env = False
        resp = session.post(judge_api_url, json=payload, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    score = data.get("score")
    if score is None:
        return None
    return float(score)


def _extract_scalar_tool_reward(extra_info: dict[str, Any], reduction: str = "last") -> Optional[float]:
    """Extract the scalar final tool reward from rollout extra_info/tool trace."""
    tool_rewards = extra_info.get("tool_rewards")
    values: list[float] = []
    if isinstance(tool_rewards, (list, tuple)):
        for r in tool_rewards:
            try:
                values.append(float(r))
            except Exception:
                continue

    if not values:
        trace = extra_info.get("tool_trace")
        if isinstance(trace, (list, tuple)):
            for item in trace:
                if not isinstance(item, dict):
                    continue
                if "backbone_binary_judge" not in item:
                    continue
                try:
                    values.append(float(item["backbone_binary_judge"]))
                except Exception:
                    continue

    if not values:
        return None

    reduction = str(reduction).lower()
    if reduction == "mean":
        return float(sum(values) / len(values))
    if reduction == "max":
        return float(max(values))
    # default: last
    return float(values[-1])


def _coerce_judge_bit(value: Any, fallback: float = 0.0) -> float:
    try:
        return 1.0 if float(value) > 0.5 else 0.0
    except (TypeError, ValueError):
        return 1.0 if float(fallback) > 0.5 else 0.0


def _extract_tag_contents(text: str, tag: str) -> list[str]:
    return re.findall(rf"<{tag}>(.*?)</{tag}>", text or "", flags=re.DOTALL)


def _has_strict_answer_evidence(
    text: str,
    *,
    max_answer_chars: int = 256,
    max_evidence_chars: int = 768,
) -> bool:
    """Require exactly one concise answer block followed by one concise evidence block."""
    text = text or ""
    answers = _extract_tag_contents(text, "answer")
    evidences = _extract_tag_contents(text, "evidence")
    if len(answers) != 1 or len(evidences) != 1:
        return False

    answer = answers[0].strip()
    evidence = evidences[0].strip()
    if not answer or not evidence:
        return False
    if len(answer) > max_answer_chars or len(evidence) > max_evidence_chars:
        return False

    answer_open = text.find("<answer>")
    answer_close = text.find("</answer>", answer_open)
    evidence_open = text.find("<evidence>", answer_close)
    evidence_close = text.find("</evidence>", evidence_open)
    if min(answer_open, answer_close, evidence_open, evidence_close) < 0:
        return False
    if not (answer_open < answer_close < evidence_open < evidence_close):
        return False

    trailing = text[evidence_close + len("</evidence>") :].strip()
    if trailing:
        return False

    # Search blocks are allowed before the final answer, but not after the model
    # has begun answering.
    if "<search>" in text[answer_open:]:
        return False
    return True


def _extract_backbone_judge_details(extra_info: dict[str, Any]) -> dict[str, Any]:
    for key in (
        "final_backbone_binary_details",
        "backbone_binary_details",
        "final_binary_details",
    ):
        value = extra_info.get(key)
        if isinstance(value, dict):
            return value

    tool_trace = extra_info.get("tool_trace")
    if isinstance(tool_trace, (list, tuple)):
        for item in reversed(tool_trace):
            if isinstance(item, dict) and (
                "retrieval_effective" in item
                or "summary_reasonable" in item
                or "final_backbone_binary_details" in item
            ):
                nested = item.get("final_backbone_binary_details")
                return nested if isinstance(nested, dict) else item
    return {}


def _extract_dense_backbone_judge_reward(
    extra_info: dict[str, Any],
    *,
    binary_reduction: str,
    retrieval_effective_reward_weight: float,
    summary_reasonable_reward_weight: float,
) -> dict[str, float]:
    binary_score = _extract_scalar_tool_reward(extra_info, reduction=binary_reduction)
    details = _extract_backbone_judge_details(extra_info)
    if details:
        fallback = 0.0 if binary_score is None else float(binary_score)
        retrieval_effective = _coerce_judge_bit(details.get("retrieval_effective"), fallback)
        summary_reasonable = _coerce_judge_bit(details.get("summary_reasonable"), fallback)
        try:
            format_penalty = float(details.get("policy_format_penalty", extra_info.get("policy_format_penalty", 0.0)))
        except (TypeError, ValueError):
            format_penalty = 0.0
        retrieval_reward = float(retrieval_effective_reward_weight) * retrieval_effective
        summary_reward = float(summary_reasonable_reward_weight) * summary_reasonable
        dense_reward = retrieval_reward + summary_reward + format_penalty
        return {
            "score": float(dense_reward),
            "binary_reward": float(fallback),
            "retrieval_effective": float(retrieval_effective),
            "summary_reasonable": float(summary_reasonable),
            "policy_format_penalty": float(format_penalty),
            "retrieval_effective_reward": float(retrieval_reward),
            "summary_reasonable_reward": float(summary_reward),
            "dense_policy_reward": float(dense_reward),
        }

    if binary_score is not None:
        return {
            "score": float(binary_score),
            "binary_reward": float(binary_score),
            "dense_policy_reward": float(binary_score),
        }
    return {
        "score": 0.0,
        "binary_reward": 0.0,
        "dense_policy_reward": 0.0,
    }


def _extract_discrete_backbone_judge_reward(
    extra_info: dict[str, Any],
    solution_str: str,
    *,
    binary_reduction: str,
    format_invalid_reward: float,
    both_good_reward: float,
    summary_only_reward: float,
    retrieval_only_reward: float,
    both_bad_reward: float,
    max_output_chars: int,
    max_answer_chars: int,
    max_evidence_chars: int,
) -> dict[str, float]:
    binary_score = _extract_scalar_tool_reward(extra_info, reduction=binary_reduction)
    details = _extract_backbone_judge_details(extra_info)
    fallback = 0.0 if binary_score is None else float(binary_score)
    retrieval_effective = _coerce_judge_bit(details.get("retrieval_effective"), fallback) if details else fallback
    summary_reasonable = _coerce_judge_bit(details.get("summary_reasonable"), fallback) if details else fallback
    try:
        format_penalty = float(details.get("policy_format_penalty", extra_info.get("policy_format_penalty", 0.0)))
    except (AttributeError, TypeError, ValueError):
        format_penalty = 0.0

    strict_format_ok = _has_strict_answer_evidence(
        solution_str,
        max_answer_chars=int(max_answer_chars),
        max_evidence_chars=int(max_evidence_chars),
    )
    output_too_long = len(solution_str or "") > int(max_output_chars)
    format_invalid = bool(format_penalty < 0.0 or not strict_format_ok or output_too_long)

    if format_invalid:
        score = float(format_invalid_reward)
        reward_case = 0.0
    elif retrieval_effective > 0.5 and summary_reasonable > 0.5:
        score = float(both_good_reward)
        reward_case = 1.0
    elif retrieval_effective <= 0.5 and summary_reasonable > 0.5:
        score = float(summary_only_reward)
        reward_case = 2.0
    elif retrieval_effective > 0.5 and summary_reasonable <= 0.5:
        score = float(retrieval_only_reward)
        reward_case = 3.0
    else:
        score = float(both_bad_reward)
        reward_case = 4.0

    return {
        "score": float(score),
        "binary_reward": float(fallback),
        "retrieval_effective": float(retrieval_effective),
        "summary_reasonable": float(summary_reasonable),
        "policy_format_penalty": float(format_penalty),
        "strict_format_ok": float(strict_format_ok),
        "output_too_long": float(output_too_long),
        "format_invalid": float(format_invalid),
        "discrete_policy_reward": float(score),
        "reward_case": float(reward_case),
    }


def compute_score(
    data_source: Optional[str],
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
    *,
    trajectory_bonus_coef: float = 0.02,
    diversity_bonus_coef: float = 0.01,
    max_trajectory_bonus: float = 0.2,
    judge_api_url: Optional[str] = None,
    judge_timeout: float = 20.0,
    judge_weight: float = 0.2,
    reward_mode: str = "hybrid",
    binary_reduction: str = "last",
    retrieval_effective_reward_weight: float = 0.4,
    summary_reasonable_reward_weight: float = 0.6,
    format_invalid_reward: float = -0.5,
    both_good_reward: float = 1.0,
    summary_only_reward: float = 0.2,
    retrieval_only_reward: float = 0.0,
    both_bad_reward: float = -0.1,
    max_output_chars: int = 1500,
    max_answer_chars: int = 256,
    max_evidence_chars: int = 768,
    **kwargs,
) -> dict[str, float]:
    """SearchR1 reward with trajectory-aware shaping and optional orchestrator judging.

    - base: EM score
    - trajectory shaping: rewards productive multi-subagent search behavior
    - optional external judge: frozen orchestrator can grade full trajectory
    """
    extra_info = extra_info or {}
    mode = str(reward_mode).lower()

    if mode == "backbone_binary_only":
        binary_score = _extract_scalar_tool_reward(extra_info, reduction=binary_reduction)
        # Fallback to external judge endpoint if tool-level binary reward is unavailable.
        if binary_score is None and judge_api_url:
            question = extra_info.get("question", "")
            try:
                judged = _judge_with_orchestrator_api(
                    judge_api_url=judge_api_url,
                    question=question,
                    answer=solution_str,
                    extra_info=extra_info,
                    timeout=judge_timeout,
                )
                if judged is not None:
                    binary_score = float(judged)
            except Exception:
                binary_score = None
        final = 0.0 if binary_score is None else float(binary_score)
        return {
            "score": float(final),
            "binary_reward": float(final),
            "reward_mode": 1.0,
        }

    if mode in {"backbone_dense_judge", "backbone_dense_only"}:
        dense_result = _extract_dense_backbone_judge_reward(
            extra_info,
            binary_reduction=binary_reduction,
            retrieval_effective_reward_weight=retrieval_effective_reward_weight,
            summary_reasonable_reward_weight=summary_reasonable_reward_weight,
        )
        dense_result["reward_mode"] = 2.0
        return dense_result

    if mode in {"backbone_discrete_judge", "backbone_discrete_only"}:
        discrete_result = _extract_discrete_backbone_judge_reward(
            extra_info,
            solution_str,
            binary_reduction=binary_reduction,
            format_invalid_reward=format_invalid_reward,
            both_good_reward=both_good_reward,
            summary_only_reward=summary_only_reward,
            retrieval_only_reward=retrieval_only_reward,
            both_bad_reward=both_bad_reward,
            max_output_chars=max_output_chars,
            max_answer_chars=max_answer_chars,
            max_evidence_chars=max_evidence_chars,
        )
        discrete_result["reward_mode"] = 3.0
        return discrete_result

    base = float(search_r1_like_qa_em.compute_score(solution_str, ground_truth))
    total_calls, unique_queries = _extract_tool_stats(extra_info)

    trajectory_bonus = min(max_trajectory_bonus, total_calls * trajectory_bonus_coef)
    diversity_bonus = min(max_trajectory_bonus, unique_queries * diversity_bonus_coef)

    judge_score = 0.0
    if judge_api_url:
        question = extra_info.get("question", "")
        try:
            judge_score = _judge_with_orchestrator_api(
                judge_api_url=judge_api_url,
                question=question,
                answer=solution_str,
                extra_info=extra_info,
                timeout=judge_timeout,
            )
            if judge_score is None:
                judge_score = 0.0
        except Exception:
            judge_score = 0.0

    score = base + trajectory_bonus + diversity_bonus + judge_weight * float(judge_score)
    return {
        "score": float(score),
        "base_em": float(base),
        "trajectory_bonus": float(trajectory_bonus),
        "diversity_bonus": float(diversity_bonus),
        "judge_score": float(judge_score),
        "tool_calls": float(total_calls),
        "unique_queries": float(unique_queries),
    }
