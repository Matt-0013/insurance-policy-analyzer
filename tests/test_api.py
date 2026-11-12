"""API contract tests. The LLM and the corpus are faked: these test the
boundary (validation, status codes, response shape), not answer quality."""
import pytest
from fastapi.testclient import TestClient

from insurance_policy_analyzer import api

client = TestClient(api.app)


def fake_chunk(text: str) -> dict:
    return {
        "chunk_id": "c1",
        "policy": "test_policy",
        "section": "1. Benefits",
        "text": text,
        "page_start": 4,
        "page_end": 4,
    }


def fake_answer(question: str, **kwargs) -> dict:
    return {
        "question": question,
        "answer": "Room rent is capped at 1% of sum insured [1].",
        "sources": [
            {
                "n": 1,
                "chunk_id": "c1",
                "policy": "test_policy",
                "policy_label": "Test - Policy (policy wording)",
                "section": "1. Benefits",
                "page_start": 4,
                "page_end": 4,
                "text": "Room rent up to 1% of sum insured per day",
                "score": 0.91,
                "retrieval": {"rrf": 0.03},
            }
        ],
        "grounding": {"citations": [1], "invalid_citations": [], "unsupported_numbers": [], "grounded": True, "refused": False},
        "retrieval": {"queries": [question], "policy": None, "mode": "hybrid", "rewrite_source": "rules"},
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        "timings_ms": {"retrieval": 1, "generate": 2, "total": 3},
    }


def test_health_reports_modes_without_loading_models():
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "hybrid" in body["modes"]


def test_documents_lists_the_registered_policies():
    response = client.get("/documents")
    assert response.status_code == 200
    ids = {doc["id"] for doc in response.json()}
    assert "niva_bupa_senior_first_wording" in ids


def test_ask_passes_history_through_to_the_pipeline(monkeypatch):
    seen = {}

    def capture(question, **kwargs):
        seen.update(kwargs, question=question)
        return fake_answer(question, **kwargs)

    monkeypatch.setattr(api, "ask_pipeline", capture)
    response = client.post(
        "/ask",
        json={
            "question": "What is the room rent limit?",
            "k": 3,
            "history": [{"question": "Tell me about Niva", "answer": "It is a senior plan."}],
        },
    )
    assert response.status_code == 200
    assert response.json()["sources"][0]["page_start"] == 4
    assert seen["k"] == 3
    assert seen["history"][0]["question"] == "Tell me about Niva"


@pytest.mark.parametrize(
    "payload,status",
    [
        ({"question": "ok?", "mode": "semantic"}, 422),
        ({"question": "ok?", "policy": "not_a_policy"}, 404),
        ({"question": "hi"}, 422),
    ],
)
def test_ask_rejects_bad_requests(monkeypatch, payload, status):
    monkeypatch.setattr(api, "ask_pipeline", fake_answer)
    assert client.post("/ask", json=payload).status_code == status


def test_compare_returns_a_cell_per_field_and_policy(monkeypatch):
    monkeypatch.setattr(
        api, "load_policies", lambda: {"test_policy": [fake_chunk("Room rent up to 1% of sum insured per day")]}
    )
    monkeypatch.setattr(api, "to_markdown", lambda policies=None: "# table")
    response = client.post("/compare", json={"policies": ["test_policy"]})
    assert response.status_code == 200
    body = response.json()
    assert body["policies"] == ["test_policy"]
    assert len(body["cells"]) == len(api.FIELDS)
    assert body["cells"][0]["label"]


def test_compare_rejects_unknown_policy(monkeypatch):
    monkeypatch.setattr(api, "load_policies", lambda: {"test_policy": [fake_chunk("x")]})
    response = client.post("/compare", json={"policies": ["nope"]})
    assert response.status_code == 404
