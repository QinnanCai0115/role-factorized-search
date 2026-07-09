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
from typing import Any, Optional

import requests

from verl.utils.reward_score import search_r1_like_qa_em


def compute_score(
    data_source: Optional[str],
    solution_str: str,
    ground_truth: Any,
    extra_info: Optional[dict[str, Any]] = None,
    *,
    judge_api_url: Optional[str] = None,
    judge_timeout: float = 20.0,
    judge_weight: float = 0.2,
    **kwargs,
) -> dict[str, float]:
    """Trajectory score for GRPO group competition.

    Each rollout trajectory is scored independently; GRPO does group-relative competition
    via `rollout.n`.
    """
    base = float(search_r1_like_qa_em.compute_score(solution_str, ground_truth))
    judge_score = 0.0

    if judge_api_url:
        payload = {
            "question": (extra_info or {}).get("question", ""),
            "answer": solution_str,
            "extra_info": extra_info or {},
        }
        try:
            disable_backbone_proxy = str(os.environ.get("BACKBONE_API_NO_PROXY", "")).strip().lower() not in (
                "",
                "0",
                "false",
                "no",
            )
            with requests.Session() as session:
                if disable_backbone_proxy:
                    session.trust_env = False
                resp = session.post(judge_api_url, json=payload, timeout=judge_timeout)
            resp.raise_for_status()
            data = resp.json()
            judge_score = float(data.get("score", 0.0))
        except Exception:
            judge_score = 0.0

    score = base + judge_weight * judge_score
    return {
        "score": float(score),
        "base_em": float(base),
        "judge_score": float(judge_score),
    }
