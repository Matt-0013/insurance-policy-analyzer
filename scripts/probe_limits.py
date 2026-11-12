"""Where do the numeric limits live, and which pages lost their text?

A page that yields under ~250 characters across all its chunks is a page
whose content did not extract (scanned image, vector figure, or a table
the extractor could not linearise).
"""
import glob
import json
import re
import statistics
import sys

MIN_CHARS = 250

LIMITS = {
    "pct_of_sum_insured": r"\d\s*%\s*of\s*(?:the\s*)?(?:sum\s*insured|S\.?I\.?)",
    "per_day": r"per\s*day",
    "sublimit": r"\b(?:sub-?\s*)?limits?\b",
}

SHOW_LIMIT = ("pct_of_sum_insured", "per_day")


def load_records():
    records = []
    for path in sorted(glob.glob("data/chunks/*.json")):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise SystemExit(f"{path}: expected a JSON array")
        records.extend(data)
    return records


def window(text: str, start: int, end: int, pad: int = 65) -> str:
    left = max(0, start - pad)
    right = min(len(text), end + pad)
    piece = text[left:start] + "«" + text[start:end] + "»" + text[end:right]
    return " ".join(piece.split())[:200]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    records = load_records()

    by_policy = {}
    for record in records:
        by_policy.setdefault(record["policy"], []).append(record)

    for policy in sorted(by_policy):
        chunks = by_policy[policy]

        per_page = {}
        for chunk in chunks:
            span = max(chunk["page_end"] - chunk["page_start"] + 1, 1)
            share = len(chunk["text"]) // span
            for page in range(chunk["page_start"], chunk["page_end"] + 1):
                per_page[page] = per_page.get(page, 0) + share

        counts = [per_page[p] for p in sorted(per_page)]
        thin = [p for p in sorted(per_page) if per_page[p] < MIN_CHARS]
        print(f"\n=== {policy} ===")
        print(
            f"  pages {len(counts)}   total chars {sum(counts):,}   "
            f"median {int(statistics.median(counts)):,}"
        )
        print(f"  thin pages (<{MIN_CHARS}): {thin if thin else 'none'}")

        for name, pattern in LIMITS.items():
            regex = re.compile(pattern, re.IGNORECASE)
            hits = [c for c in chunks if regex.search(c["text"])]
            print(f"  {name.ljust(22)} {len(hits):>4}")
            if name in SHOW_LIMIT:
                for chunk in hits[:2]:
                    match = regex.search(chunk["text"])
                    print(f"      p{chunk['page_start']}: {window(chunk['text'], match.start(), match.end())}")


if __name__ == "__main__":
    main()