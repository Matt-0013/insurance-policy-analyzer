"""Probe vocabulary gaps: where does each concept actually appear?

Prints a window around the match, not the start of the chunk: a chunk is
1000+ characters, so its opening words are usually unrelated to the hit.
"""
import glob
import json
import re
import sys

PROBES = {
    "room_rent": [
        r"room\s*,?\s*boarding",
        r"room\s*(rent|charges?)",
        r"room\s*(category|type|class)",
        r"hospital\s*accommodation",
        r"\bbed\s*charges\b",
        r"nursing\s*expenses",
    ],
    "co_payment": [
        r"co-?\s*pay",
        r"co-?\s*share",
        r"deductible",
        r"proportionate\s*deduction",
    ],
    "restoration": [
        r"recharge",
        r"\brestor(ation|ed|ing)\b",
        r"reinstat",
    ],
    "ped_code": [
        r"pre-?\s*existing\s*(disease|condition|ailment|illness)",
        r"code\s*[-\u2013]\s*excl\s*0?1(?!\d)",
    ],
}

POLICIES = [
    "care_health_care_advanced_terms",
    "icici_lombard_family_shield_wording",
    "niva_bupa_senior_first_wording",
    "star_health_women_care_brochure",
]


def load_records():
    records = []
    for path in sorted(glob.glob("data/chunks/*.json")):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise SystemExit(f"{path}: expected a JSON array")
        records.extend(data)
    return records


def window(text: str, start: int, end: int, pad: int = 70) -> str:
    left = max(0, start - pad)
    right = min(len(text), end + pad)
    piece = text[left:start] + "«" + text[start:end] + "»" + text[end:right]
    return " ".join(piece.split())[:230]


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    records = load_records()

    by_policy = {}
    for record in records:
        by_policy.setdefault(record["policy"], []).append(record)

    for concept, patterns in PROBES.items():
        print(f"\n=== {concept} ===")
        silent = []
        for policy in POLICIES:
            chunks = by_policy.get(policy, [])
            if not chunks:
                print(f"  {policy}: NO CHUNKS")
                continue
            print(f"  {policy}")
            any_hit = False
            for pattern in patterns:
                regex = re.compile(pattern, re.IGNORECASE)
                hits = [c for c in chunks if regex.search(c["text"])]
                label = pattern.ljust(36)
                if not hits:
                    print(f"    {label}   0")
                    continue
                any_hit = True
                first = hits[0]
                match = regex.search(first["text"])
                print(
                    f"    {label} {len(hits):>3}  p{first['page_start']}: "
                    f"{window(first['text'], match.start(), match.end())}"
                )
            if not any_hit:
                silent.append(policy)
        if silent:
            print(f"  !! no alias hit anything in: {', '.join(silent)}")


if __name__ == "__main__":
    main()