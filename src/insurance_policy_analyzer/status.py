"""Is the index actually there?

A health check that returns "ok" while the collection is empty is worse than no
check at all: the orchestrator keeps the container in rotation, and the first
real /ask fails at request time with a stack trace instead of the caller being
told plainly that nothing is indexed.

The probe deliberately avoids importing chromadb, torch or the embedding model.
/health answers in milliseconds today because the heavy stack is imported inside
the endpoint; opening the store with stdlib sqlite3 keeps that property, and the
container can be probed without loading a 100 MB model.
"""

import json
import sqlite3

from insurance_policy_analyzer.config import CHROMA_DIR, CHUNKS_DIR


def _chunk_count() -> tuple[int, int]:
    """Total chunks and how many per-policy chunk files produced them."""
    files = sorted(CHUNKS_DIR.glob("*.json")) if CHUNKS_DIR.exists() else []
    chunks = 0
    for path in files:
        try:
            chunks += len(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return chunks, len(files)


def _vector_count() -> tuple[int | None, str]:
    """Vectors stored in the Chroma SQLite file, or None if it cannot be read."""
    sqlite_path = CHROMA_DIR / "chroma.sqlite3"
    if not sqlite_path.exists():
        return None, f"no chroma.sqlite3 in {CHROMA_DIR}"
    # mode=ro so a probe can never create, migrate or lock the live store.
    uri = f"file:{sqlite_path.as_posix()}?mode=ro"
    try:
        with sqlite3.connect(uri, uri=True, timeout=2) as conn:
            row = conn.execute("select count(*) from embeddings").fetchone()
    except sqlite3.Error as exc:
        return None, f"chroma.sqlite3 unreadable: {exc}"
    if not row:
        return None, "chroma.sqlite3 has no embeddings table"
    return int(row[0]), ""


def index_report() -> dict:
    """Report what is on disk and whether it is enough to answer a question."""
    vectors, detail = _vector_count()
    chunks, files = _chunk_count()

    if vectors is None:
        present, reason = False, detail
    elif vectors == 0:
        present, reason = False, "collection is empty: ingestion did not run"
    elif files == 0:
        # Vectors without chunk files can answer /ask but not /compare, and every
        # citation would be missing its section and page numbers.
        present, reason = False, f"{vectors} vectors but no chunk files in {CHUNKS_DIR}"
    elif chunks != vectors:
        present, reason = True, f"{vectors} vectors vs {chunks} chunks: stale index"
    else:
        present, reason = True, f"{vectors} vectors across {files} policies"

    return {
        "present": present,
        "vectors": vectors,
        "chunks": chunks,
        "chunk_files": files,
        "detail": reason,
    }
