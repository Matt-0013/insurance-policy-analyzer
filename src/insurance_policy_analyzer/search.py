import json
import sys
from functools import lru_cache

import bm25s
import chromadb
import torch
from sentence_transformers import SentenceTransformer

from insurance_policy_analyzer.config import (
    CHROMA_DIR,
    CHUNKS_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

MODES = ("vector", "keyword", "hybrid")
# Keep amounts like "2,50,000" as one token; \w+ covers everything else.
TOKEN_PATTERN = r"(?u)\d+(?:,\d+)+|\w+"
RRF_K = 60
CANDIDATES = 50


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, model_kwargs={"dtype": torch.float32})


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


@lru_cache(maxsize=1)
def _chunks() -> list[dict]:
    chunks = []
    for path in sorted(CHUNKS_DIR.glob("*.json")):
        chunks.extend(json.loads(path.read_text(encoding="utf-8")))
    return chunks


@lru_cache(maxsize=1)
def _bm25() -> bm25s.BM25:
    # Same content the embedding sees: section title + text.
    corpus = [f"{c['section']}\n{c['text']}" for c in _chunks()]
    tokens = bm25s.tokenize(corpus, token_pattern=TOKEN_PATTERN, stopwords=None, show_progress=False)
    retriever = bm25s.BM25()
    retriever.index(tokens, show_progress=False)
    return retriever


def vector_search(query: str, k: int = 5, policy: str | None = None) -> list[dict]:
    """Semantic search. Lower distance = closer match."""
    vector = _model().encode(query, normalize_embeddings=True).tolist()
    where = {"policy": policy} if policy else None
    result = _collection().query(
        query_embeddings=[vector],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"chunk_id": cid, "text": doc, **meta, "distance": round(dist, 3)}
        for cid, doc, meta, dist in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def keyword_search(query: str, k: int = 5, policy: str | None = None) -> list[dict]:
    """BM25 keyword search. Higher score = better match."""
    chunks = _chunks()
    tokens = bm25s.tokenize(
        query, token_pattern=TOKEN_PATTERN, stopwords=None, return_ids=False, show_progress=False
    )
    docs, scores = _bm25().retrieve(tokens, k=len(chunks), show_progress=False)
    hits = []
    for index, score in zip(docs[0], scores[0]):
        chunk = chunks[int(index)]
        if score <= 0 or (policy and chunk["policy"] != policy):
            continue
        hits.append({**chunk, "bm25": round(float(score), 2)})
        if len(hits) == k:
            break
    return hits


def rrf_fuse(rankings: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """Reciprocal Rank Fusion: score = sum of 1 / (k + rank) over all rankings."""
    scores: dict[str, float] = {}
    by_id: dict[str, dict] = {}
    for hits in rankings:
        for rank, hit in enumerate(hits, start=1):
            by_id.setdefault(hit["chunk_id"], hit)
            scores[hit["chunk_id"]] = scores.get(hit["chunk_id"], 0.0) + 1.0 / (k + rank)
    ordered = sorted(scores, key=scores.get, reverse=True)
    return [{**by_id[i], "rrf": round(scores[i], 4)} for i in ordered]


def hybrid_search(query: str, k: int = 5, policy: str | None = None) -> list[dict]:
    rankings = [
        vector_search(query, CANDIDATES, policy),
        keyword_search(query, CANDIDATES, policy),
    ]
    return rrf_fuse(rankings)[:k]


def search(query: str, k: int = 5, policy: str | None = None, mode: str = "hybrid") -> list[dict]:
    functions = {"vector": vector_search, "keyword": keyword_search, "hybrid": hybrid_search}
    return functions[mode](query, k, policy)


def main() -> None:
    args = sys.argv[1:]
    mode = args.pop(0) if args and args[0] in MODES else "hybrid"
    query = " ".join(args) or "maximum ambulance charges by road"
    for hit in search(query, mode=mode):
        print(f"{hit['policy']} | {hit['section']} | p.{hit['page_start']}")
        print("    " + hit["text"][:150].replace("\n", " "))


if __name__ == "__main__":
    main()

