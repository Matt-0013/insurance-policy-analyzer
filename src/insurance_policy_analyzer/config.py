import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.getenv("APP_DATA_DIR", PROJECT_ROOT / "data"))

RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHUNKS_DIR = DATA_DIR / "chunks"
CHROMA_DIR = DATA_DIR / "chroma"
POLICIES_FILE = DATA_DIR / "policies.json"

EMBEDDING_MODEL = "ibm-granite/granite-embedding-small-english-r2"
RERANK_MODEL = os.getenv("RERANK_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

COLLECTION_NAME = "policy_chunks"
GENERATION_MODEL = "openai/gpt-oss-120b"
REWRITE_MODEL = "openai/gpt-oss-20b"
REASONING_EFFORT = "low"

TOP_K = 5
MAX_CONTEXT_CHARS = 8000
