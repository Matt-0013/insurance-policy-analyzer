"""The committed report is produced by `render_report`, so these tests pin what it
must say.

The header is the part that silently drifts: a run of ten questions with two null
judge scores used to read "10 questions" and nothing else, which overstates the
sample each mean is computed over. These tests fail if the count, the sample size
or the judge's real name disappears.
"""
from insurance_policy_analyzer.eval_ragas import render_report

SUMMARY = {
    "mode": "hybrid",
    # Deliberately not the module default: the header must report the judge the run
    # used, not whichever model happens to be the constant.
    "judge_model": "openai/gpt-oss-120b",
    "sample_size": 10,
    "judge_failures": 2,
    "metrics": {
        "faithfulness": 0.843,
        "answer_relevancy": 0.957,
        "llm_context_precision_without_reference": None,
    },
    "deterministic_grounded_rate": 0.8,
    "refusal_rate_on_unanswerable": 1.0,
}


def header_of(report: str) -> str:
    return report.splitlines()[2]


def test_header_carries_the_sample_size_and_judge_failure_count():
    header = header_of(render_report(SUMMARY, "ragas_hybrid.json"))
    assert header.startswith("Generated ")
    assert "mode `hybrid`" in header
    assert "10 questions" in header
    assert "2 judge failures" in header


def test_header_names_the_judge_from_the_summary():
    assert "`openai/gpt-oss-120b`" in header_of(render_report(SUMMARY, "ragas_hybrid.json"))


def test_every_metric_is_rendered_and_a_null_is_labelled():
    report = render_report(SUMMARY, "ragas_hybrid.json")
    assert "| faithfulness | 0.843 |" in report
    assert "| answer_relevancy | 0.957 |" in report
    # A null is a judge failure, not a zero and not a missing row.
    assert "| llm_context_precision_without_reference | n/a |" in report
    assert "| grounded (deterministic, no judge) | 0.8 |" in report
    assert "| refused on unanswerable questions | 1.0 |" in report


def test_report_points_at_the_results_file_it_came_from():
    report = render_report(SUMMARY, "ragas_hybrid.json")
    assert "eval/results/ragas_hybrid.json" in report
    assert "Context recall is not reported" in report


def test_a_zero_failure_run_still_states_the_count():
    header = header_of(render_report({**SUMMARY, "judge_failures": 0}, "ragas_hybrid.json"))
    assert "0 judge failures" in header
