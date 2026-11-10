"""End-to-end RAG: rewrite -> retrieve -> fuse -> rerank -> generate -> verify.

This is the only module the API and the UI need to call. Every step is visible
in the returned dict, so the UI can show the rewritten queries, the scores and
the grounding verdict instead of an opaque answer.
"""

import logging
import re
import time

from insurance_policy_analyzer import llm, registry
from insurance_policy_analyzer.citations import verify
from insurance_policy_analyzer.config import (
    MAX_CONTEXT_CHARS,
    REASONING_EFFORT,
    TOP_K,
)
from insurance_policy_analyzer.memory import Conversation
from insurance_policy_analyzer.prompts import REFUSAL, SYSTEM_PROMPT, answer_prompt
from insurance_policy_analyzer.rewrite import rewrite
from insurance_policy_analyzer.search import search_multi

log = logging.getLogger(__name__)


def page_range(hit: dict) -> str:
    start = hit.get("page_start")
    end = hit.get("page_end", start)
    if start is None:
        return "page unknown"
    return f"page {start}" if start == end else f"pages {start}-{end}"


def build_context(
    hits: list[dict], max_chars: int = MAX_CONTEXT_CHARS
) -> tuple[str, list[dict]]:
    """Number the sources and render the context block the model reads.

    Sources are numbered in the order the model sees them, so [2] in the answer
    always means sources[1]. Duplicate and over-budget chunks are dropped here,
    before numbering, so citation numbers never point at a missing source.
    """
    blocks: list[str] = []
    sources: list[dict] = []
    seen: set[str] = set()
    total = 0

    for hit in hits:
        text = (hit.get("text") or "").strip()
        if not text:
            continue
        key = re.sub(r"\s+", " ", text)[:160].lower()
        if key in seen:
            continue
        n = len(sources) + 1
        header = f"[{n}] {registry.label(hit['policy'])} | {hit['section']} | {page_range(hit)}"
        block = f"{header}\n{text}"
        if sources and total + len(block) > max_chars:
            break
        seen.add(key)
        sources.append(
            {
                "n": n,
                "chunk_id": hit.get("chunk_id"),
                "policy": hit["policy"],
                "policy_label": registry.label(hit["policy"]),
                "section": hit["section"],
                "page_start": hit.get("page_start"),
                "page_end": hit.get("page_end"),
                "text": text,
                "score": hit.get("rerank_score"),
                "retrieval": {k: hit[k] for k in ("distance", "bm25", "rrf") if k in hit},
            }
        )
        blocks.append(block)
        total += len(block)

    return "\n\n".join(blocks), sources


def _policy_from_history(history: list[dict] | None) -> str | None:
    """Carry the policy forward when a follow-up does not name one.

    Only when exactly one policy appears across the earlier questions, so a
    comparison or a switch to another policy does not pin the filter wrongly.
    """
    named = {
        registry.match_policy(turn["question"])
        for turn in history or []
        if turn.get("question")
    }
    named.discard(None)
    return named.pop() if len(named) == 1 else None


def _history_turns(history: list[dict] | None) -> str:
    convo = Conversation()
    for turn in history or []:
        if turn.get("question") and turn.get("answer"):
            convo.add(turn["question"], turn["answer"])
    return convo.transcript()


def ask(
    question: str,
    *,
    k: int = TOP_K,
    policy: str | None = None,
    history: list[dict] | None = None,
    mode: str = "hybrid",
    use_llm_rewrite: bool = True,
) -> dict:
    """Answer a question from the indexed policies, with citations and a grounding check."""
    if not question or not question.strip():
        raise ValueError("question is empty")
    started = time.perf_counter()

    transcript = _history_turns(history)
    rewritten = rewrite(question, history=transcript, use_llm=use_llm_rewrite)
    chosen = policy or rewritten["policy"] or _policy_from_history(history)
    hits = search_multi(rewritten["queries"], k=k, policy=chosen, mode=mode)
    context, sources = build_context(hits)
    retrieved_at = time.perf_counter()

    result = {
        "question": question,
        "answer": REFUSAL,
        "sources": sources,
        "grounding": verify(REFUSAL, []),
        "retrieval": {
            "queries": rewritten["queries"],
            "policy": chosen,
            "mode": mode,
            "rewrite_source": rewritten["source"],
        },
        "usage": {},
        "timings_ms": {"retrieval": round((retrieved_at - started) * 1000)},
    }
    if not sources:
        # Nothing retrieved: refusing is cheaper and more honest than asking the
        # model to answer from its own knowledge.
        result["timings_ms"]["total"] = result["timings_ms"]["retrieval"]
        return result

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": answer_prompt(question, context, transcript)},
    ]
    response = llm.chat(messages, reasoning_effort=REASONING_EFFORT, temperature=0)

    answer = (response.choices[0].message.content or "").strip() or REFUSAL
    result["answer"] = answer
    result["grounding"] = verify(answer, [source["text"] for source in sources])
    if response.usage:
        result["usage"] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    finished = time.perf_counter()
    result["timings_ms"]["generate"] = round((finished - retrieved_at) * 1000)
    result["timings_ms"]["total"] = round((finished - started) * 1000)
    log.debug("answered in %sms with %s sources", result["timings_ms"]["total"], len(sources))
    return result
