import re

from insurance_policy_analyzer.search import TOKEN_PATTERN, rrf_fuse


def test_amounts_stay_one_token():
    assert re.findall(TOKEN_PATTERN, "Rs. 2,50,000 and 30-day") == ["Rs", "2,50,000", "and", "30", "day"]


def test_rrf_rewards_agreement_between_rankings():
    a = [{"chunk_id": "x"}, {"chunk_id": "y"}]
    b = [{"chunk_id": "y"}, {"chunk_id": "z"}]
    assert rrf_fuse([a, b])[0]["chunk_id"] == "y"
