from insurance_policy_analyzer.clean import clean_text


def test_rebuilds_spaced_words():
    raw = "Pree x i s t i n g d i s e a s e s s h a l l \n\nbe modified"
    assert "Preexisting diseases shall" in clean_text(raw)


def test_rebuilds_coverage_sentence():
    raw = "of continuous c o v e r a g e h a s e l a p s e d a s mentioned in the Add-on Policy"
    assert "coverage has elapsed as mentioned in the Add-on Policy" in clean_text(raw)


def test_normal_text_is_unchanged():
    raw = "The Company shall pay the claim in Add-on cover."
    assert clean_text(raw) == raw