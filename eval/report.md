# Answer-quality report

Generated 2026-11-13 · mode `hybrid` · judge `openai/gpt-oss-20b` · 10 questions · 2 judge failures

| metric | score |
| --- | --- |
| faithfulness | 0.843 |
| answer_relevancy | 0.957 |
| llm_context_precision_without_reference | 0.981 |
| grounded (deterministic, no judge) | 0.8 |
| refused on unanswerable questions | 1.0 |

A judge failure is a metric the judge returned null for: those samples are excluded from the mean, so a metric can be averaged over fewer answers than the question count.

Context recall is not reported: it requires a reference answer per question. Raw per-question scores: `eval/results/ragas_hybrid.json`.
