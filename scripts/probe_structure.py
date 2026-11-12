"""Structural audit: is the benefit and limit content actually in the text?

A missing phrase can mean the document says it differently, or that the
content never survived extraction. These counts separate the two.
"""
import glob
import json
import re
import sys

PATTERNS = {
    "hospitalisation_expenses": r"hospitali[sz]ation\s*expenses",
    "benefit": r"\bbenefits?\b",
    "schedule": r"\bschedule\b",
    "percent_of_sum_insured": r"\d\s*%\s*of\s*(?:the\s*)?sum\s*insured",
    "per_day": r"per\s*day",
    "irdai_excl_code": r"code\s*[-\u2013]\s*excl\s*\d+",
}

SHOW = ("hospitalisation_expenses", "percent_of_sum_insured", "irdai_excl_code")


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
    return " ".join(piece.split())[:220]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    records = load_records()

    by_policy = {}
    for record in records:
        by_policy.setdefault(record["policy"], []).append(record)

    for policy in sorted(by_policy):
        chunks = by_policy[policy]
        pages = sorted(
            {p for c in chunks for p in range(c["page_start"], c["page_end"] + 1)}
        )
        print(f"\n=== {policy} ===")
        print(f"  chunks {len(chunks)}   pages {pages[0]}..{pages[-1]}   distinct {len(pages)}")

        for name, pattern in PATTERNS.items():
            regex = re.compile(pattern, re.IGNORECASE)
            hits = [c for c in chunks if regex.search(c["text"])]
            print(f"  {name.ljust(24)} {len(hits):>4}")
            if name in SHOW and hits:
                first = hits[0]
                match = regex.search(first["text"])
                print(
                    f"      p{first['page_start']}: "
                    f"{window(first['text'], match.start(), match.end())}"
                )


if __name__ == "__main__":
    main()