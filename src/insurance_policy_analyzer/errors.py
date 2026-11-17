"""Errors shared by modules that must not import each other.

`search` imports torch and chromadb, so `api` cannot import the exception it needs
from there without paying for the whole retrieval stack at import time. This module
imports nothing.
"""


class RetrievalUnavailable(RuntimeError):
    """The vector store cannot be opened, so the request cannot be served.

    Raised instead of letting a store-level failure surface as an opaque 500: the
    API maps it to 503 with the underlying reason, which is what a caller can act on.
    """
