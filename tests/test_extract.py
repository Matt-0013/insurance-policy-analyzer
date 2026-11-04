from insurance_policy_analyzer.extract import coverage


def test_coverage_ignores_markdown_symbols():
    assert coverage("|**Rs. 2,000**|<br>", "Rs. 2,000") == 1.0


def test_coverage_detects_missing_text():
    assert coverage("Rs", "Rs 2000 per hospitalization") < 0.2
