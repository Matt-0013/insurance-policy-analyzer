"""Count chunks matching each candidate comparison field, per policy.

Regex over chunk text rather than section titles: titles are truncated
labels, so they cannot be used as identifiers.
"""
import collections
import glob
import json
import re
import sys

FIELDS = {
    "sum_insured": r"sum\s*insured",
    "room_rent": r"room\s*rent|room\s*charges|room\s*category",
    "co_payment": r"co-?pay",
    "ped": r"pre-?existing",
    "specific_waiting": r"specified\s*waiting|specific\s*waiting|specified\s*disease",
    "initial_waiting": r"30[\s-]*day|initial\s*wait",
    "maternity": r"maternit",
    "day_care": r"day\s*care",
    "sub_limit": r"sub-?\s*limit",
    "restoration": r"restor",
    "pre_post_hosp": r"pre-?\s*hospitali|post-?\s*hospitali",
    "ambulance": r"ambulance",
}


def load_records():
    records = []
    for path in sorted(glob.glob("data/chunks/*.json")):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise SystemExit(f"{path}: expected a JSON array, got {type(data).__name__}")
        records.extend(data)
    return records


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    records = load_records()
    if not records:
        raise SystemExit("no chunks found")

    policies = sorted({r["policy"] for r in records})
    compiled = {name: re.compile(p, re.IGNORECASE) for name, p in FIELDS.items()}

    counts = collections.Counter()
    for record in records:
        for name, pattern in compiled.items():
            if pattern.search(record["text"]):
                counts[(name, record["policy"])] += 1

    print("field".ljust(18) + "".join(p[:22].rjust(24) for p in policies))
    print("-" * (18 + 24 * len(policies)))
    for name in FIELDS:
        print(name.ljust(18) + "".join(str(counts[(name, p)]).rjust(24) for p in policies))

    print("\nmissing in at least one policy:")
    for name in FIELDS:
        missing = [p for p in policies if counts[(name, p)] == 0]
        if missing:
            print(f"  {name}: {', '.join(missing)}")


if __name__ == "__main__":
    main()