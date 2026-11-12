"""Print chunk record shape and section titles per policy.

Read-only inspection used to ground the comparison schema in what the
documents actually say, instead of in an imagined list of fields.
"""
import collections
import glob
import json
import sys


def main() -> None:
    # Section titles can contain non-ASCII glyphs from the source PDFs;
    # without this a cp1252 console raises UnicodeEncodeError mid-report.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    records = []
    for path in sorted(glob.glob("data/chunks/*.json")):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise SystemExit(f"{path}: expected a JSON array, got {type(data).__name__}")
        records.extend(data)

    if not records:
        print("no chunks found - run the chunking step first")
        return

    print("chunk keys:", sorted(records[0].keys()))
    print("total chunks:", len(records))

    by_policy = collections.defaultdict(collections.Counter)
    for record in records:
        by_policy[record.get("policy")][record.get("section") or "(none)"] += 1

    for policy, sections in by_policy.items():
        print(f"\n### {policy} - {len(sections)} sections, {sum(sections.values())} chunks")
        for name, count in sections.most_common(50):
            print(f"  {count:3d}  {name}")


if __name__ == "__main__":
    main()