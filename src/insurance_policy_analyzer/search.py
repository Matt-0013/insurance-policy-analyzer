import sys
from functools import lru_cache

import chromadb
import torch
from sentence_transformers import SentenceTransformer

from insurance_policy_analyzer.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL, model_kwargs={"dtype": torch.float32})


@lru_cache(maxsize=1)
def _collection():
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    return client.get_collection(COLLECTION_NAME)


def search(query: str, k: int = 5, policy: str | None = None) -> list[dict]:
    """Return the k closest chunks. Lower distance = closer match."""
    vector = _model().encode(query, normalize_embeddings=True).tolist()
    where = {"policy": policy} if policy else None
    result = _collection().query(
        query_embeddings=[vector],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {"text": doc, **meta, "distance": round(dist, 3)}
        for doc, meta, dist in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def main() -> None:
    query = " ".join(sys.argv[1:]) or "maximum ambulance charges by road"
    for hit in search(query):
        print(f"{hit['distance']}  {hit['policy']} | {hit['section']} | p.{hit['page_start']}")
        print("    " + hit["text"][:150].replace("\n", " "))


if __name__ == "__main__":
    main()