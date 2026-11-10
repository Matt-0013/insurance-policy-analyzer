from insurance_policy_analyzer.pipeline import build_context, page_range

POLICY = "star_health_women_care_brochure"


def hit(chunk_id: str, text: str, page: int = 3) -> dict:
    return {
        "chunk_id": chunk_id,
        "policy": POLICY,
        "section": "Room rent",
        "page_start": page,
        "page_end": page,
        "text": text,
    }


def test_sources_are_numbered_in_prompt_order():
    context, sources = build_context([hit("a", "Room rent is capped at 1% of sum insured.")])
    assert context.startswith("[1] Star Health - Women Care (brochure) | Room rent | page 3")
    assert sources[0]["n"] == 1
    assert sources[0]["page_start"] == 3


def test_duplicate_text_is_dropped_before_numbering():
    text = "Room rent is capped at 1% of sum insured."
    context, sources = build_context([hit("a", text), hit("b", text)])
    assert len(sources) == 1
    assert "[2]" not in context


def test_budget_stops_the_context():
    hits = [hit(f"c{i}", f"chunk {i} " + "x" * 300) for i in range(5)]
    _, sources = build_context(hits, max_chars=700)
    assert len(sources) == 1


def test_page_range_collapses_equal_pages():
    assert page_range({"page_start": 4, "page_end": 4}) == "page 4"
    assert page_range({"page_start": 4, "page_end": 6}) == "pages 4-6"
