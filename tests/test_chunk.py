from insurance_policy_analyzer.chunk import chunk_document, split_long


def _table(rows: int) -> str:
    head = "|DESCRIPTION|TERMS|\n|---|---|\n"
    body = "".join(f"|Benefit {i} pays up to Rs. {i}000 per claim.<br>\n" for i in range(rows))
    return head + body


def test_split_long_repeats_table_header():
    parts = split_long(_table(60), max_chars=600)
    assert len(parts) > 1
    assert all(p.startswith("|DESCRIPTION|TERMS|") for p in parts)
    assert all(len(p) <= 600 for p in parts)


def test_split_long_keeps_every_row():
    joined = " ".join(split_long(_table(60), max_chars=600))
    assert all(f"Benefit {i} pays" in joined for i in range(60))


def test_short_text_is_not_split():
    assert split_long("Short clause.") == ["Short clause."]


def test_tiny_leftover_piece_is_dropped():
    lines = [f"Clause {i:02d} " + "a" * 95 for i in range(14)]
    text = "# 1. Preamble\n\n" + "\n".join(lines) + "\n" + "b" * 30
    chunks = chunk_document("demo", [{"page": 1, "text": text}])
    assert len(chunks) == 1
    assert chunks[0]["text"].startswith("Clause 00")


def test_definition_line_keeps_only_the_term():
    from insurance_policy_analyzer.chunk import heading_text

    assert heading_text("2.2.10 General Ward means a") == "2.2.10 General Ward"
    assert heading_text("14. Intensive Care Unit : means an identified") == "14. Intensive Care Unit"


def test_trailing_colon_is_removed():
    from insurance_policy_analyzer.chunk import heading_text

    assert heading_text("4.2. Specific Exclusions:") == "4.2. Specific Exclusions"
    assert heading_text("Just a normal sentence.") is None

def test_definition_heading_keeps_its_full_first_line():
       text = "2.2.10 General Ward means a\nroom with several beds, as defined in the policy wording."
       chunks = chunk_document("demo", [{"page": 1, "text": text}])
       assert chunks[0]["section"] == "2.2.10 General Ward"
       assert "General Ward means a" in chunks[0]["text"]


       