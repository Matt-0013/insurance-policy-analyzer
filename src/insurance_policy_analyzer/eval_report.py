"""Formatting for the committed RAGAS report.

Deliberately separate from `eval_ragas`: that module imports the pipeline, which
imports torch, sentence-transformers and chromadb, so anything that only needs to
check the report's text would pay a multi-second import for it. This module
imports nothing heavier than `datetime`, and keeps the report's exact wording in
one place so a fresh run and a regeneration from a saved results file agree.
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
