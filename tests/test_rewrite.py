from insurance_policy_analyzer.registry import match_policy
from insurance_policy_analyzer.rewrite import expand_query, parse_rewrite

STAR = "star_health_women_care_brochure"
NIVA = "niva_bupa_senior_first_wording"
KNOWN = [STAR, NIVA]


def test_policy_alias_is_detected():
    assert match_policy("What is the co-pay in Niva Bupa Senior First?") == NIVA


def test_comparison_question_matches_no_single_policy():
    assert match_policy("Compare Niva Bupa and Star Health on room rent") is None


def test_question_without_a_policy_has_no_filter():
    assert match_policy("What is the claim settlement ratio?") is None


def test_abbreviations_are_expanded_for_keyword_search():
    assert "pre-existing disease" in expand_query("What is the PED waiting period?")


def test_expansion_is_idempotent():
    once = expand_query("PED waiting period")
    assert expand_query(once) == once


def test_parse_rewrite_keeps_the_original_question_first():
    raw = '{"queries": ["room rent limit star health"], "policy": "%s"}' % STAR
    out = parse_rewrite(raw, KNOWN, "room rent?")
    assert out["queries"] == ["room rent?", "room rent limit star health"]
    assert out["policy"] == STAR
    assert out["source"] == "llm"


def test_parse_rewrite_rejects_unknown_policy_id():
    out = parse_rewrite('{"queries": [], "policy": "made_up"}', KNOWN, "q")
    assert out["policy"] is None
    assert out["queries"] == ["q"]


def test_parse_rewrite_survives_fenced_json():
    assert parse_rewrite('```json\n{"queries": ["a"]}\n```', KNOWN, "q")["queries"] == ["q", "a"]


def test_parse_rewrite_falls_back_on_garbage():
    assert parse_rewrite("no json here", KNOWN, "q") == {
        "queries": ["q"],
        "policy": None,
        "source": "fallback",
    }
