"""Query rewriting for retrieval.

Two layers, on purpose:
- deterministic: expand insurance abbreviations and detect a named policy, so
  retrieval still improves when the LLM is unavailable or disabled;
- LLM: turn a follow-up question ("and its waiting period?") into standalone
  search queries, using one cheap small-model call.
"""

import json
import logging
import re

from insurance_policy_analyzer import llm, registry
from insurance_policy_analyzer.config import REASONING_EFFORT, REWRITE_MODEL
from insurance_policy_analyzer.prompts import REWRITE_PROMPT

log = logging.getLogger(__name__)

# Only whole-word insurance shorthands: they are what users actually type, and
# BM25 cannot match "PED" against "pre-existing disease" on its own.
ABBREVIATIONS = {
    "ped": "pre-existing disease",
    "pre existing": "pre-existing disease",
    "si": "sum insured",
    "copay": "co-payment",
    "co pay": "co-payment",
    "opd": "outpatient department",
    "icu": "intensive care unit",
    "day care": "daycare procedure",
    "tpa": "third party administrator",
}

MAX_QUERIES = 3
MAX_QUERY_CHARS = 300


def expand_query(question: str) -> str:
    """Append the long form of any insurance shorthand in the question."""
    low = question.lower()
    extra = []
    for short, long in ABBREVIATIONS.items():
        if long in low:
            continue
        if re.search(rf"\b{re.escape(short)}\b", low):
            extra.append(long)
    return " ".join([question, *extra]) if extra else question


def _fallback(question: str, known_ids: list[str]) -> dict:
    return {
        "queries": [expand_query(question)],
        "policy": registry.match_policy(question, known_ids),
        "source": "fallback",
    }


def parse_rewrite(raw: str, known_ids: list[str], question: str) -> dict:
    """Validate model output and fall back to the plain question on anything odd."""
    fallback = _fallback(question, known_ids)
    block = re.search(r"\{.*\}", raw or "", re.DOTALL)
    if not block:
        return fallback
    try:
        data = json.loads(block.group())
    except json.JSONDecodeError:
        return fallback
    if not isinstance(data, dict):
        return fallback

    queries: list[str] = []
    for value in data.get("queries") or []:
        if isinstance(value, str) and value.strip():
            queries.append(value.strip()[:MAX_QUERY_CHARS])
    # The original question always stays in the pool: rewriting is an addition,
    # not a replacement, so a bad rewrite cannot make retrieval worse.
    unique = list(dict.fromkeys([question.strip(), *queries]))
    policy = data.get("policy")
    if policy not in known_ids:
        policy = fallback["policy"]
    return {"queries": unique[:MAX_QUERIES], "policy": policy, "source": "llm"}


def rewrite(question: str, history: str = "", use_llm: bool = True) -> dict:
    """Return standalone search queries plus an explicit policy filter, if any."""
    known_ids = registry.policy_ids()
    if not use_llm:
        return _fallback(question, known_ids)

    user = (
        f"Conversation so far:\n{history}\n\nQuestion: {question}"
        if history
        else f"Question: {question}"
    )
    try:
        response = llm.chat(
            [
                {
                    "role": "system",
                    "content": REWRITE_PROMPT.format(policy_ids=", ".join(known_ids)),
                },
                {"role": "user", "content": user},
            ],
            model=REWRITE_MODEL,
            reasoning_effort=REASONING_EFFORT,
            response_format={"type": "json_object"},
        )
        return parse_rewrite(response.choices[0].message.content or "", known_ids, question)
    except Exception:
        # Retrieval must not fail because a query rewrite did.
        log.warning("query rewrite failed; using the raw question", exc_info=True)
        return _fallback(question, known_ids)
