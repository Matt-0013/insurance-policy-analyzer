"""The committed report is produced by `render_report`, so these tests pin its text.

The header is the part that drifts silently: a run of ten questions with two null
judge scores once read "10 questions" and nothing else, overstating the sample each
mean rests on. Imported from `eval_report`, not `eval_ragas`, so this file does not
import torch to check a string.
"""
from insurance_policy_analyzer.eval_report import render_report

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

REPORT = render_report(SUMMARY, "ragas_hybrid.json")


def test_header_states_the_sample_size_the_failures_and_the_judge():
    header = REPORT.splitlines()[2]
    assert header.startswith("Generated ")
    assert "mode `hybrid`" in header
    assert "10 questions" in header
    assert "2 judge failures" in header
    assert "`openai/gpt-oss-120b`" in header


def test_a_zero_failure_run_still_states_the_count():
    header = render_report({**SUMMARY, "judge_failures": 0}, "ragas_hybrid.json").splitlines()[2]
    assert "0 judge failures" in header


def test_every_metric_is_rendered_and_a_null_is_labelled():
    assert "| faithfulness | 0.843 |" in REPORT
    assert "| answer_relevancy | 0.957 |" in REPORT
    # A null is a judge failure: not a zero, not a missing row.
    assert "| llm_context_precision_without_reference | n/a |" in REPORT
    assert "| grounded (deterministic, no judge) | 0.8 |" in REPORT
    assert "| refused on unanswerable questions | 1.0 |" in REPORT


def test_report_points_at_the_results_file_it_came_from():
    assert "eval/results/ragas_hybrid.json" in REPORT
    assert "Context recall is not reported" in REPORT
