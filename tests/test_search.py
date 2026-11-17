import re

import pytest

from insurance_policy_analyzer import search
from insurance_policy_analyzer.errors import RetrievalUnavailable
from insurance_policy_analyzer.search import TOKEN_PATTERN, rrf_fuse


def test_amounts_stay_one_token():
    assert re.findall(TOKEN_PATTERN, "Rs. 2,50,000 and 30-day") == ["Rs", "2,50,000", "and", "30", "day"]


def test_rrf_rewards_agreement_between_rankings():
    a = [{"chunk_id": "x"}, {"chunk_id": "y"}]
    b = [{"chunk_id": "y"}, {"chunk_id": "z"}]
    assert rrf_fuse([a, b])[0]["chunk_id"] == "y"


def test_a_store_failure_is_reported_as_unavailable_and_not_cached(monkeypatch):
    # Regression: a failed client build used to leave chromadb's per-path System
    # registered, so every later attempt failed inside _release_system with
    # "'RustBindingsAPI' object has no attribute 'bindings'" instead of the reason.
    attempts = []

    def unreadable(*args, **kwargs):
        attempts.append(1)
        raise RuntimeError("error returned from database: (code: 8) attempt to write a readonly database")

    monkeypatch.setattr(search.chromadb, "PersistentClient", unreadable)
    search._collection.cache_clear()
    try:
        for _ in range(2):
            with pytest.raises(RetrievalUnavailable) as raised:
                search._collection()
            assert "readonly database" in str(raised.value)
        # Not cached: the second attempt retries the store, so a transient cause can heal.
        assert len(attempts) == 2
    finally:
        search._collection.cache_clear()
