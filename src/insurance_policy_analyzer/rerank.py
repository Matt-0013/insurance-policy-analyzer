"""Cross-encoder reranking.

Retrieval (vector + BM25) is fast but judges the query and the chunk separately.
A cross-encoder reads them together and scores the pair, which is much more
accurate but too slow to run over the whole corpus: run it on the top candidates.
"""

import logging
from functools import lru_cache

from sentence_transformers import CrossEncoder

from insurance_policy_analyzer.config import RERANK_MODEL

log = logging.getLogger(__name__)

# The model truncates at 512 tokens; keep the pair well inside that.
PAIR_CHARS = 1200


@lru_cache(maxsize=1)
def _model() -> CrossEncoder:
    return CrossEncoder(RERANK_MODEL, max_length=512)


def rerank(query: str, hits: list[dict], top_k: int = 5, scorer=None) -> list[dict]:
    """Reorder hits by cross-encoder score and keep the best top_k.

    scorer is injectable so tests can check the ordering logic without loading
    the model. If scoring fails the original order is kept rather than failing
    the whole request.
    """
    if not hits:
        return []
    pairs = [(query, f"{hit['section']}\n{hit['text']}"[:PAIR_CHARS]) for hit in hits]
    try:
        scores = scorer(pairs) if scorer is not None else _model().predict(pairs)
    except Exception:
        log.warning("reranking failed; keeping retrieval order", exc_info=True)
        return hits[:top_k]

    ranked = sorted(zip(hits, scores), key=lambda pair: float(pair[1]), reverse=True)
    return [{**hit, "rerank_score": round(float(score), 3)} for hit, score in ranked[:top_k]]
