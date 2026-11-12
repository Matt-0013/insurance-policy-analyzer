"""The scoring table is the single source of truth, and the docs are generated
from it. A weight that drifts from its evidence is how a matcher starts lying.

Values are asserted explicitly: if a weight changes, this test fails and whoever
changed it has to bring the corpus evidence with them.
"""
from pathlib import Path

from insurance_policy_analyzer import compare

DOC = Path(__file__).resolve().parents[1] / "docs" / "SCORING.md"


def test_every_rule_and_window_is_wired_in():
    assert {rule.key for rule in compare.SCORING_RULES} == set(compare._WEIGHT)
    assert {window.key for window in compare.WINDOWS} == set(compare._WINDOW)


def test_weights_match_the_documented_values():
    assert compare._WEIGHT == {
        "clean": 2,
        "definition": 2,
        "scheduled": 1,
        "quantity": 1,
        "comma_list": -1,
        "section_code": 4,
        "coverage_cue": 5,
    }
    assert compare._WINDOW == {
        "preamble_chars": 250,
        "next_chars": 220,
        "tail_chars": 250,
        "code_radius": 150,
    }


def test_every_row_carries_corpus_evidence():
    for rule in compare.SCORING_RULES:
        assert len(rule.evidence) > 30, rule.key
    for window in compare.WINDOWS:
        assert len(window.evidence) > 30, window.key


def test_docs_scoring_table_is_generated_from_the_code():
    on_disk = DOC.read_text(encoding="utf-8").replace("\r\n", "\n").strip()
    assert on_disk == compare.rules_markdown().strip(), (
        "docs/SCORING.md is stale: run "
        "python -m insurance_policy_analyzer.compare --table > docs/SCORING.md"
    )
