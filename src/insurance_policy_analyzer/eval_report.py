"""Formatting for the committed RAGAS report.

Separate from `eval_ragas` because that module imports the pipeline and therefore
torch, sentence-transformers and chromadb; checking a header string should not cost
a model import. This module needs nothing but `datetime`.
"""

from datetime import date


def render_report(summary: dict, results_name: str) -> str:
    """Turn a summary into the committed markdown.

    The judge failure count belongs in the header, not only in the raw JSON: it
    says how many samples the means are missing, which a question count on its own
    hides (ten questions, two null scores, means over nine answers).
    """
    return "\n".join(
        [
            "# Answer-quality report",
            "",
            f"Generated {date.today().isoformat()} · mode `{summary['mode']}` · "
            f"judge `{summary['judge_model']}` · {summary['sample_size']} questions · "
            f"{summary['judge_failures']} judge failures",
            "",
            "| metric | score |",
            "| --- | --- |",
            *[
                f"| {name} | {value if value is not None else 'n/a'} |"
                for name, value in summary["metrics"].items()
            ],
            f"| grounded (deterministic, no judge) | {summary['deterministic_grounded_rate']} |",
            f"| refused on unanswerable questions | {summary['refusal_rate_on_unanswerable']} |",
            "",
            "A judge failure is a metric the judge returned null for: those samples are "
            "excluded from the mean, so a metric can be averaged over fewer answers than "
            "the question count.",
            "",
            "Context recall is not reported: it requires a reference answer per question. "
            f"Raw per-question scores: `eval/results/{results_name}`.",
            "",
        ]
    )
