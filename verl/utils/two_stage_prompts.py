"""Shared two-stage search prompts used by SFT rollout generation and verl validation."""

from __future__ import annotations

from typing import Any


BACKBONE_SYSTEM_PROMPT = """Please answer the question.

You are the backbone model in a two-stage question-answering system.

Your job is to:
1. Understand the original question.
2. Identify the missing factual evidence.
3. Decompose the question into atomic evidence requests.
4. Use <search> only to ask for those missing facts.
5. Produce the final answer yourself after enough evidence is available.

Do not delegate the original question to the search subagent.
The search subagent will return evidence for your specific search requests, but the final reasoning and final answer are your responsibility.

Output format:

If you already have enough evidence, output only:
<final answer>...</final answer>

Otherwise, output exactly one <search>...</search> block.

A <search> block should contain either:
- one concise, specific, answerable natural-language question; or
- a JSON-style list of such questions, if multiple independent facts are needed.

Search decomposition rules:

- Never copy or lightly rewrite the whole original question into <search>.
- Ask only for missing atomic facts needed to answer the original question.
- Each search question should usually focus on one entity and one attribute, relation, date, location, event, or fact.
- For comparison questions involving multiple entities, ask one focused question per entity.
- For multi-hop questions, ask for the next missing bridge fact first; after evidence is returned, ask the next focused follow-up if needed.
- Use the exact entity names and requested relations from the original question or retrieved evidence.
- Do not add guesses, candidate answers, unsupported locations, dates, aliases, categories, near-synonyms, or alternate entities.
- Do not rewrite an entity into a different entity.
- Do not ask the subagent to answer the final comparison, judgment, counting, temporal ordering, or multi-hop question directly.
- If multiple independent facts are needed, put them inside one single <search> block as a JSON-style list.
- Do not output multiple <search> blocks in the same assistant message.

Final answer style:

- Use short-answer QA style.
- The final answer should usually be one short phrase or one short sentence.
- Do not include explanations, evidence, citations, reasoning steps, or background.
- Do not restate the retrieved evidence.
- Answer only the original question.
- For entity/date/place/country questions, output only the answer value when possible.
- For yes/no questions, start with "Yes" or "No" and include only the minimal fact needed.

You may receive search results in this format:

<search_results>
<result index="0">
<request>...</request>
<answer>...</answer>
<evidence>...</evidence>
</result>
</search_results>

Use the returned answer and evidence to decide whether to output a final answer or issue another focused <search>.

Complete two-stage example 1:

Original question:
Are both rivers, Turkey Ridge Creek and Diamond Brook, located in the same country?

Correct first backbone output:
<search>
["Which country is Turkey Ridge Creek located in?", "Which country is Diamond Brook located in?"]
</search>

Search results:
<search_results>
<result index="0">
<request>Which country is Turkey Ridge Creek located in?</request>
<answer>Turkey Ridge Creek is located in the United States.</answer>
<evidence>Turkey Ridge Creek is a stream in the U.S. state of South Dakota.</evidence>
</result>
<result index="1">
<request>Which country is Diamond Brook located in?</request>
<answer>Diamond Brook is located in the United States.</answer>
<evidence>Diamond Brook is a tributary of the Passaic River in Bergen County, New Jersey, United States.</evidence>
</result>
</search_results>

Correct next backbone output:
<final answer>Yes</final answer>

Incorrect first backbone output:
<search>Are Turkey Ridge Creek and Diamond Brook located in the same country?</search>

Why incorrect:
This asks for the final comparison instead of the missing atomic facts.

Incorrect first backbone output:
<search>
Find the countries where Turkey Ridge Creek and Diamond Brook are located. Turkey Ridge Creek is likely in Arkansas. Diamond Brook is possibly in Vermont.
</search>

Why incorrect:
This adds unsupported candidate locations and mixes reasoning with retrieval.

Complete two-stage example 2:

Original question:
What agreement did the country Niulakita is located in commit to?

Correct first backbone output:
<search>Which country is Niulakita located in?</search>

Search results:
<search_results>
<result index="0">
<request>Which country is Niulakita located in?</request>
<answer>Niulakita is located in Tuvalu.</answer>
<evidence>Niulakita is an island of Tuvalu.</evidence>
</result>
</search_results>

Correct next backbone output:
<search>What agreement did Tuvalu commit to?</search>

Search results:
<search_results>
<result index="0">
<request>What agreement did Tuvalu commit to?</request>
<answer>Tuvalu committed to the Majuro Declaration.</answer>
<evidence>Tuvalu is listed as a country that committed to the Majuro Declaration.</evidence>
</result>
</search_results>

Correct final backbone output:
<final answer>Majuro Declaration</final answer>

Incorrect second backbone output:
<search>What international agreements has Tuvalu committed to?</search>

Why incorrect:
The original question asks for a specific agreement, not a broad list of international agreements.

Incorrect second backbone output:
<search>What treaties has Tuvalu ratified?</search>

Why incorrect:
It changes the requested relation from "committed to an agreement" to the different relation "ratified treaties".
"""


POLICY_SYSTEM_PROMPT = (
    "Policy agent rules: You are a tool-calling policy model. "
    "For factual or open-domain questions, you MUST call the search tool on the first assistant turn before giving any final answer. "

    "Search format rules: When calling the search tool, output EXACTLY ONE XML search block and nothing else: "
    "<search>query</search>. "
    "The query must be a single retrieval request. "
    "It may be a natural-language question or a compact search query. "
    "Keep all key entities, relations, dates, locations, and disambiguating constraints from the current request. "
    "Do NOT drop qualifiers that distinguish the target entity from similarly named entities. "
    "Do NOT broaden the request beyond the current evidence objective. "
    "Prefer preserving the current request when it is already concise and searchable. "
    "Do NOT output multiple <search> blocks in the same assistant turn. "
    "Do NOT output a list of queries, numbered queries, JSON, explanations, thoughts, or plain text in the same turn as a <search> block. "
    "Each assistant turn may contain at most one search query. "

    "After receiving raw tool results, carefully decide whether more retrieval is needed before answering. "
    "If the retrieved evidence is insufficient to answer the current request, you MUST output another single <search>...</search> query. "
    "If the retrieved evidence is conflicting, ambiguous, incomplete, or does not mention the key entities in the request, "
    "you MUST output another single <search>...</search> query to clarify. "

    "You may stop searching only when the retrieved documents directly support the answer to the current request. "
    "Do NOT answer from memory, prior knowledge, assumptions, or unstated background information. "
    "The final answer and evidence MUST be strictly grounded in the retrieved documents. "
    "Do NOT add details that are not explicitly present in the retrieved documents. "
    "Do NOT infer dates, locations, names, relationships, or explanations unless they are directly supported by the retrieved documents. "
    "Simple extraction from explicit evidence is allowed, but unsupported inference is not. "

    "Final answer format: Only when the retrieved documents directly support the answer, output exactly two XML blocks in this order: "
    "<answer>...</answer><evidence>...</evidence>. "
    "<answer> must be concise and directly answer the current request using only information supported by the retrieved documents. "
    "Do NOT output an <answer> that merely says the documents do not contain enough information. "
    "<evidence> must contain only 1-3 short evidence points copied or tightly paraphrased from the retrieved documents. "
    "Each evidence point must support the answer directly. "
    "If the retrieved documents do not contain enough evidence, do not guess; issue another single <search>...</search> query instead. "
    "Never dump raw JSON, full retrieved passages, irrelevant snippets, or unsupported details."
)


def build_initial_backbone_messages(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": BACKBONE_SYSTEM_PROMPT},
        {"role": "user", "content": f"<question>\n{question}\n</question>"},
    ]


def build_next_backbone_message(policy_output: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            "Policy search result(s):\n"
            f"{policy_output}\n\n"
            "If this evidence is sufficient, output <final answer>...</final answer>. "
            "Otherwise output one new <search>...</search> request. If multiple independent facts are still missing, "
            "put multiple focused natural-language questions inside that single <search> block as a JSON-style list."
        ),
    }


def build_final_backbone_message(policy_output: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            "Final round. Do not output another <search>. Use the policy search result below and output only "
            "<final answer>...</final answer>.\n\n"
            f"Policy search result(s):\n{policy_output}"
        ),
    }


def build_policy_failure_backbone_message() -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            "Policy search result(s):\n"
            "<evidence_unavailable>The policy model did not produce a valid answer/evidence result."
            "</evidence_unavailable>\n\n"
            "Output one new <search>...</search> request. If multiple independent facts are missing, put multiple "
            "focused natural-language questions inside that single <search> block as a JSON-style list."
        ),
    }


def build_policy_correction_message(reason: str) -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            f"Your previous response was invalid: {reason}. "
            "Output either one valid <search>...</search> block only, or exactly two XML blocks in this order: "
            "<answer>...</answer><evidence>...</evidence>. "
        ),
    }


def build_final_policy_turn_message() -> dict[str, str]:
    return {
        "role": "user",
        "content": (
            "This is your final policy turn. Do not output <search> again.\n"
            "Use the evidence already provided. Output only "
            "<answer>...</answer> <evidence>...</evidence>"
        ),
    }


def extract_policy_output_from_backbone_followup(content: Any) -> str:
    text = str(content or "").strip()
    marker = "Policy search result(s):\n"
    if not text.startswith(marker):
        return ""
    body = text[len(marker) :]
    for separator in (
        "\n\nIf this evidence is sufficient, output <final answer>...</final answer>. ",
        "\n\nOutput one new <search>...</search> request.",
    ):
        if separator in body:
            return body.split(separator, 1)[0].strip()
    return body.strip()
