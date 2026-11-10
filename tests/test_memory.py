from insurance_policy_analyzer.memory import Conversation


def test_old_turns_are_dropped():
    convo = Conversation(max_turns=2)
    for i in range(3):
        convo.add(f"q{i}", f"a{i}")
    assert [turn["question"] for turn in convo.turns] == ["q1", "q2"]


def test_transcript_is_oldest_first_and_trims_long_answers():
    convo = Conversation()
    convo.add("q1", "a" * 1000)
    convo.add("q2", "b")
    text = convo.transcript()
    assert text.startswith("Q: q1")
    assert "..." in text
    assert text.index("q2") > text.index("q1")
