import json
import shutil
import time

import chromadb
import torch
from sentence_transformers import SentenceTransformer

from insurance_policy_analyzer.config import (
    CHROMA_DIR,
    CHUNKS_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
)

BATCH_SIZE = 32


def load_chunks() -> list[dict]:
    chunks = []
    for path in sorted(CHUNKS_DIR.glob("*.json")):
        chunks.extend(json.loads(path.read_text(encoding="utf-8")))
    return chunks


def embed_text(chunk: dict) -> str:
    """Prefix policy and section so each chunk keeps its context."""
    policy = chunk["policy"].replace("_", " ")
    return f"{policy} | {chunk['section']}\n{chunk['text']}"


def main() -> None:
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    # The index is derived data: rebuild from scratch so it always matches the chunks.
    shutil.rmtree(CHROMA_DIR, ignore_errors=True)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.create_collection(
        COLLECTION_NAME, configuration={"hnsw": {"space": "cosine"}}
    )

    # float32 was ~5x faster than bfloat16 on this CPU
    model = SentenceTransformer(EMBEDDING_MODEL, model_kwargs={"dtype": torch.float32})

    start = time.time()
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        vectors = model.encode([embed_text(c) for c in batch], normalize_embeddings=True)
        collection.add(
            ids=[c["chunk_id"] for c in batch],
            embeddings=vectors.tolist(),
            documents=[c["text"] for c in batch],
            metadatas=[
                {
                    "policy": c["policy"],
                    "section": c["section"],
                    "page_start": c["page_start"],
                    "page_end": c["page_end"],
                }
                for c in batch
            ],
        )
        print(f"  embedded {min(i + BATCH_SIZE, len(chunks))}/{len(chunks)}")
    print(f"Stored {collection.count()} chunks in {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()