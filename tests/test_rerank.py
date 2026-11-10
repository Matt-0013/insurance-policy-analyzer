from insurance_policy_analyzer.rerank import rerank


def hits() -> list[dict]:
    return [
        {"chunk_id": "a", "section": "Room rent", "text": "text a"},
        {"chunk_id": "b", "section": "Room rent", "text": "text b"},
        {"chunk_id": "c", "section": "Room rent", "text": "text c"},
    ]


def test_rerank_orders_by_cross_encoder_score():
    ranked = rerank("q", hits(), top_k=2, scorer=lambda pairs: [0.1, 4.2, 2.0])
    assert [h["chunk_id"] for h in ranked] == ["b", "c"]
    assert ranked[0]["rerank_score"] == 4.2


def test_rerank_keeps_retrieval_order_when_scoring_fails():
    def boom(pairs):
        raise RuntimeError("model unavailable")

    assert [h["chunk_id"] for h in rerank("q", hits(), top_k=2, scorer=boom)] == ["a", "b"]


def test_rerank_of_nothing_is_nothing():
    assert rerank("q", []) == []
