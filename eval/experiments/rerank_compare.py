"""Compare reranker configurations on the golden questions.

Usage:
    python eval/experiments/rerank_compare.py [pair_chars ...]

Retrieval (hybrid, 50 candidates) runs once per question; every variant then
reranks the same candidates, so the only thing that changes is the variant.
Retrieval metrics alone are measured by eval_retrieval.py.
"""

import sys

from sentence_transformers import CrossEncoder

from insurance_policy_analyzer.config import RERANK_MODEL
from insurance_policy_analyzer.eval_retrieval import (
    K,
    contains_all,
    haystack,
    load_chunks,
    load_questions,
)
from insurance_policy_analyzer.search import CANDIDATES, hybrid_search


def metrics(rows: list[dict]) -> dict:
    n = len(rows)
    return {
        **{
            f"hit@{k}": round(sum(1 for r in rows if r["rank"] and r["rank"] <= k) / n, 2)
            for k in (1, 3, 5, 10)
        },
        "mrr": round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 2),
    }


def rank_of(hits: list[dict], question: dict) -> int | None:
    return next(
        (
            i
            for i, h in enumerate(hits, 1)
            if h["policy"] == question["policy"] and contains_all(haystack(h), question["must_contain"])
        ),
        None,
    )


def main() -> None:
    pair_chars = [int(value) for value in sys.argv[1:]] or [1200, 2000, 3000]
    questions = [q for q in load_questions() if q["answerable"]]
    load_chunks()  # fail fast if the corpus is missing

    candidates = [(q, hybrid_search(q["question"], CANDIDATES)) for q in questions]
    print("hybrid     ", metrics([{"rank": rank_of(hits, q)} for q, hits in candidates]))

    model = CrossEncoder(RERANK_MODEL, max_length=512)
    for chars in pair_chars:
        rows = []
        for question, hits in candidates:
            pairs = [
                (question["question"], f"{h['section']}\n{h['text']}"[:chars]) for h in hits
            ]
            scores = model.predict(pairs)
            reranked = [
                hit for hit, _ in sorted(zip(hits, scores), key=lambda p: float(p[1]), reverse=True)[:K]
            ]
            rows.append({"rank": rank_of(reranked, question)})
        print(f"rerank {chars:5}", metrics(rows))


if __name__ == "__main__":
    main()
