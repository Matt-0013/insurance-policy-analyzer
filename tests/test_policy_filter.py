from insurance_policy_analyzer.pipeline import _policy_from_history

NIVA = "niva_bupa_senior_first_wording"
STAR = "star_health_women_care_brochure"


def test_policy_is_inherited_from_history():
    history = [
        {"question": "What is the co-payment in Niva Bupa Senior First?", "answer": "40%"},
        {"question": "And the waiting period?", "answer": "24 months"},
    ]
    assert _policy_from_history(history) == NIVA


def test_policy_is_not_inherited_when_history_names_several():
    history = [
        {"question": "What is the co-payment in Niva Bupa?", "answer": "40%"},
        {"question": "What about Star Health Women Care?", "answer": "30%"},
    ]
    assert _policy_from_history(history) is None


def test_policy_is_not_inherited_from_empty_history():
    assert _policy_from_history(None) is None
    assert _policy_from_history([]) is None
