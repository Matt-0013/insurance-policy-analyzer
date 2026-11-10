from insurance_policy_analyzer.citations import (
    extract_numbers,
    parse_citations,
    unsupported_numbers,
    verify,
)
from insurance_policy_analyzer.prompts import REFUSAL


def test_indian_grouping_and_lakh_are_the_same_amount():
    assert extract_numbers("Rs. 2,50,000")[0]["norm"] == "250000"
    assert extract_numbers("Rs. 2.5 lakh")[0]["norm"] == "250000"


def test_percent_is_not_confused_with_a_plain_number():
    kinds = [n["kind"] for n in extract_numbers("50% co-pay on 50 visits")]
    assert kinds == ["percent", "plain"]


def test_duration_is_detected_and_citation_markers_are_ignored():
    numbers = extract_numbers("waiting period of 24 months [1]")
    assert [(n["norm"], n["kind"]) for n in numbers] == [("24", "duration")]
    assert extract_numbers("as stated in [1] and [2]") == []


def test_parse_citations_sorts_and_dedupes():
    assert parse_citations("see [2] and [1][2]") == [1, 2]


def test_unsupported_amount_is_flagged():
    flagged = unsupported_numbers("The limit is Rs. 10,000 [1].", ["The limit is Rs. 5,000."])
    assert [f["raw"] for f in flagged] == ["10,000"]


def test_paraphrased_amount_is_supported():
    result = verify("Cover is Rs. 5 lakh [1].", ["Sum insured: 5,00,000"])
    assert result["unsupported_numbers"] == []
    assert result["grounded"] is True


def test_citation_to_missing_source_is_invalid():
    result = verify("The limit is Rs. 5,000 [3].", ["The limit is Rs. 5,000."])
    assert result["invalid_citations"] == [3]
    assert result["grounded"] is False


def test_answer_without_citations_is_not_grounded():
    assert verify("Room rent is capped.", ["Room rent is capped."])["grounded"] is False


def test_refusal_is_grounded():
    result = verify(REFUSAL, [])
    assert result["grounded"] is True
    assert result["refused"] is True
