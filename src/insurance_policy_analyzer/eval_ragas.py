"""Answer-quality evaluation with RAGAS.

Measures what the retrieval eval cannot:

- faithfulness: is every claim in the answer supported by a retrieved clause?
- answer relevancy: does the answer address the question that was asked?
- context precision: did retrieval bring the clauses that matter?

Context recall is deliberately not reported. It needs a reference answer per
question, and inventing references after seeing the system's output would
measure the references, not the system.

The judge is Groq behind its OpenAI-compatible endpoint, so no OpenAI key is
needed. Everything is saved to eval/results/ so the README numbers are
reproducible instead of remembered.
"""
import argparse
import json
import os
import statistics
import sys

from dotenv import load_dotenv

from insurance_policy_analyzer.config import EMBEDDING_MODEL, PROJECT_ROOT
from insurance_policy_analyzer.eval_report import render_report
from insurance_policy_analyzer.pipeline import ask

QUESTIONS = PROJECT_ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
REPORT = PROJECT_ROOT / "eval" / "report.md"
JUDGE_MODEL = "openai/gpt-oss-20b"  # free tier caps at 8k tokens/min; the 120b judge 429s


def load_questions() -> list[dict]:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    lines = QUESTIONS.read_text(encoding="utf-8-sig").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def make_judge(model: str = JUDGE_MODEL):
    from openai import AsyncOpenAI
    from ragas.llms import llm_factory

    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise SystemExit("GROQ_API_KEY is not set")
    client = AsyncOpenAI(base_url="https://api.groq.com/openai/v1", api_key=key)
    # gpt-oss spends tokens reasoning before it emits JSON; without headroom the
    # judge fails with 'max completion tokens reached before generating a valid
    # document' instead of returning a score.
    return llm_factory(model, client=client, max_tokens=4096)


def make_embeddings():
    from langchain_community.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)


def collect(
    questions: list[dict], mode: str, limit: int | None, k: int
) -> tuple[list[dict], list[dict]]:
    """Answer every answerable question; record the refused unanswerable ones."""
    answerable = [q for q in questions if q["answerable"]]
    if limit:
        answerable = answerable[:limit]

    samples, refusals = [], []
    for question in answerable:
        result = ask(question["question"], policy=question["policy"], mode=mode, k=k)
        contexts = [source["text"] for source in result["sources"] if source["text"]]
        samples.append(
            {
                "question_id": question["id"],
                "user_input": question["question"],
                "response": result["answer"],
                "retrieved_contexts": contexts,
                "grounded": result["grounding"]["grounded"],
                "sources": len(contexts),
            }
        )
        print(f"answered {question['id']}: {len(contexts)} sources, grounded={samples[-1]['grounded']}")

    for question in [q for q in questions if not q["answerable"]]:
        result = ask(question["question"], mode=mode)
        refusals.append(
            {"question_id": question["id"], "refused": bool(result["grounding"]["refused"])}
        )
        print(f"refusal {question['id']}: {refusals[-1]['refused']}")

    return samples, refusals


def main() -> None:
    load_dotenv()
    parser = argparse.ArgumentParser(description="RAGAS answer-quality evaluation")
    parser.add_argument("--mode", default="hybrid")
    parser.add_argument("--limit", type=int, default=None, help="evaluate the first N answerable questions")
    parser.add_argument("--k", type=int, default=3, help="clauses per retrieval, kept small for judge token cost")
    parser.add_argument("--judge", default=JUDGE_MODEL)
    args = parser.parse_args()

    from ragas import EvaluationDataset, RunConfig, evaluate
    from ragas.metrics import Faithfulness, LLMContextPrecisionWithoutReference, ResponseRelevancy

    questions = load_questions()
    samples, refusals = collect(questions, args.mode, args.limit, args.k)
    if not samples:
        raise SystemExit("no answerable questions found")

    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": sample["user_input"],
                "response": sample["response"],
                "retrieved_contexts": sample["retrieved_contexts"],
            }
            for sample in samples
        ]
    )
    metrics = [
        Faithfulness(),
        # strictness=1 asks for one generation instead of three: same metric,
        # a third of the tokens on a rate-limited free tier.
        ResponseRelevancy(strictness=1),
        LLMContextPrecisionWithoutReference(),
    ]
    evaluation = evaluate(
        dataset,
        metrics=metrics,
        llm=make_judge(args.judge),
        embeddings=make_embeddings(),
        raise_exceptions=False,
        # One worker: concurrency is what triggers the 429s, not the volume.
        run_config=RunConfig(max_workers=1, timeout=180),
    )

    frame = evaluation.to_pandas()
    columns = [column for column in frame.columns if column not in {"user_input", "response", "retrieved_contexts"}]
    rows = []
    for index, sample in enumerate(samples):
        scores = {
            column: (None if frame[column].isna().iloc[index] else round(float(frame[column].iloc[index]), 3))
            for column in columns
        }
        rows.append(
            {
                "id": sample["question_id"],
                "sources": sample["sources"],
                "deterministic_grounded": sample["grounded"],
                **scores,
            }
        )

    def mean_of(column: str) -> float | None:
        values = [r[column] for r in rows if r.get(column) is not None]
        return round(statistics.fmean(values), 3) if values else None

    summary = {
        "mode": args.mode,
        "judge_model": args.judge,
        "embedding_model": EMBEDDING_MODEL,
        "sample_size": len(rows),
        "judge_failures": sum(1 for row in rows if any(row.get(c) is None for c in columns)),
        "metrics": {column: mean_of(column) for column in columns},
        "deterministic_grounded_rate": round(
            sum(1 for r in rows if r["deterministic_grounded"]) / len(rows), 2
        ),
        "refusal_rate_on_unanswerable": (
            round(sum(1 for r in refusals if r["refused"]) / len(refusals), 2) if refusals else None
        ),
        "notes": (
            "context recall not reported: it needs a reference answer per question. "
            "deterministic_grounded_rate comes from citations.py, not from the judge."
        ),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    output = RESULTS_DIR / f"ragas_{args.mode}.json"
    output.write_text(
        json.dumps({"summary": summary, "rows": rows, "refusals": refusals}, indent=2),
        encoding="utf-8",
    )

    REPORT.write_text(render_report(summary, output.name), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"\nwrote {output} and {REPORT}")


if __name__ == "__main__":
    main()
