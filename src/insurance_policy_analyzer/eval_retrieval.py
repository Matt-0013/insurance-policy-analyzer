import json
import re

from insurance_policy_analyzer.config import CHUNKS_DIR, PROJECT_ROOT
from insurance_policy_analyzer.search import search

QUESTIONS = PROJECT_ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = PROJECT_ROOT / "eval" / "results"
K = 10


def normalize(text: str) -> str:
    text = re.sub(r"<br>|\*", " ", text.lower())
    return re.sub(r"\s+", " ", text)


def contains_all(text: str, phrases: list[str]) -> bool:
    body = normalize(text)
    return all(normalize(p).strip() in body for p in phrases)


def haystack(item: dict) -> str:
       """Section title + text: the same content the embedding sees."""
       return f"{item['section']}\n{item['text']}"

def load_questions() -> list[dict]:
    lines = QUESTIONS.read_text(encoding="utf-8-sig").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def load_chunks() -> list[dict]:
    chunks = []
    for path in sorted(CHUNKS_DIR.glob("*.json")):
        chunks.extend(json.loads(path.read_text(encoding="utf-8")))
    return chunks


def main(label: str = "vector") -> None:
    questions = [q for q in load_questions() if q["answerable"]]
    chunks = load_chunks()
    rows, invalid = [], []
    for q in questions:
        phrases = q["must_contain"]
        in_data = any(c["policy"] == q["policy"] and contains_all(haystack(c), phrases) for c in chunks)
        if not in_data:
            invalid.append(q["id"])
            print(f"BAD TEST PHRASE  {q['id']}: no chunk of {q['policy']} contains {phrases}")
            continue
        hits = search(q["question"], k=K)
        rank = next(
            (i for i, h in enumerate(hits, 1) if h["policy"] == q["policy"] and contains_all(haystack(h), phrases)),
            None,
        )
        rows.append({"id": q["id"], "rank": rank})
        print(f"{'rank ' + str(rank) if rank else 'MISS':8} {q['id']}")

    n = len(rows)
    summary = {
        "label": label,
        "questions": n,
        "invalid": invalid,
        **{f"hit@{k}": round(sum(1 for r in rows if r["rank"] and r["rank"] <= k) / n, 2) for k in (1, 3, 5, 10)},
        "mrr": round(sum(1 / r["rank"] for r in rows if r["rank"]) / n, 2),
    }
    print(json.dumps(summary, indent=2))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / f"retrieval_{label}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

