"""Turn chunk matches into comparison cells.

A field is mentioned many times in one document: in a definition, a benefit
clause, an exclusion list, a consumables price list. Taking the first mention
is what made earlier runs wrong, so candidates are scored and the state is
decided from the winner.
"""
import glob
import json
import re
import sys

from .schema import (
    EXCLUDED,
    EXCLUSION_MARKERS,
    FIELDS,
    NOT_STATED,
    SCHEDULED,
    STATED,
    Cell,
    FieldSpec,
)

SCHEDULED_MARKERS = (
    r"(?:specified|stated|mentioned)\s+in\s+the\s+(?:policy\s+)?(?:schedule|certificate)",
    r"as\s+per\s+the\s+(?:policy\s+)?(?:schedule|certificate)",
    r"(?:amount|details?)\s+(?:as\s+)?(?:stated|specified)\s+in\s+the",
)

# Only markers strong enough to govern the paragraph that follows them are
# allowed to leak across a chunk boundary. Broad markers (just "excluded")
# produced 30 false exclusions when applied to a whole neighbour chunk.
STRONG_EXCLUSION = (
    r"shall\s+not\s+be\s+liable",
    r"\bnot\s+covered\b",
)

# A section label only counts as evidence when it is an actual heading, not
# when it merely contains an IRDAI code: Care labels a waiting period
# "Code- Excl01", and that is not a carve-out.
EXCLUSION_SECTION = (
    r"^\W*(?:(?:\d+|[IVXLC]+)[.)])?\s*\**\s*exclusions?\b|what\s+is\s+not\s+covered"
)

SENTENCE_END = re.compile(r"(?<=[.!?])\s+|\n+")
CODE_MARKER = re.compile(r"code\s*[-\u2013]\s*excl\s*\d+", re.IGNORECASE)
DEFINITION = re.compile(r"\bmeans\b", re.IGNORECASE)
COVERAGE_CUE = re.compile(r"\bcover(?:ed|ing)?\b|\bbenefits?\b|\bwe\s+will\s+pay\b", re.IGNORECASE)
QUANTITY = re.compile(
    r"\d+\s*(?:%|percent)|\brs\.?\s*[\d,]+|\b\d+\s*(?:months?|days?|years?)\b",
    re.IGNORECASE,
)

PREAMBLE_CHARS = 250
TAIL_CHARS = 250
CODE_RADIUS = 150
CHUNKS_GLOB = "data/chunks/*.json"


def load_policies() -> dict[str, list[dict]]:
    """Chunks per policy, in file order, which is assumed to be document order."""
    by_policy: dict[str, list[dict]] = {}
    for path in sorted(glob.glob(CHUNKS_GLOB)):
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise SystemExit(f"{path}: expected a JSON array")
        for record in data:
            by_policy.setdefault(record["policy"], []).append(record)

    for policy, chunks in by_policy.items():
        pages = [c["page_start"] for c in chunks]
        if pages != sorted(pages):
            print(f"WARNING {policy}: chunks are not in page order", file=sys.stderr)
    return by_policy


def quote_around(text: str, start: int, end: int, pad: int = 60) -> str:
    left = max(0, start - pad)
    right = min(len(text), end + pad)
    return " ".join((text[left:start] + "\u00ab" + text[start:end] + "\u00bb" + text[end:right]).split())


def sentence_bounds(text: str, start: int) -> tuple[int, int]:
    """Bounds of the sentence containing offset `start`."""
    left = 0
    for sep in SENTENCE_END.finditer(text):
        if sep.end() <= start:
            left = sep.end()
        else:
            return left, sep.start()
    return left, len(text)


def _sentence(text: str, start: int) -> str:
    left, right = sentence_bounds(text, start)
    return text[left:right]


def _evidence(chunks: list[dict], index: int, match: re.Match) -> str:
    """Why this candidate is not a plain statement of cover. Empty means clean."""
    text = chunks[index]["text"]
    sentence = _sentence(text, match.start())

    near = text[max(0, match.start() - CODE_RADIUS):match.end() + CODE_RADIUS]
    if CODE_MARKER.search(near):
        return "code"
    if any(re.search(p, sentence, re.IGNORECASE) for p in EXCLUSION_MARKERS):
        return "sentence"

    preamble = text[max(0, match.start() - PREAMBLE_CHARS):match.start()]
    if any(re.search(p, preamble, re.IGNORECASE) for p in EXCLUSION_MARKERS):
        return "preamble"

    if index > 0:
        tail = chunks[index - 1]["text"][-TAIL_CHARS:]
        if any(re.search(p, tail, re.IGNORECASE) for p in STRONG_EXCLUSION):
            return "prev"

    section = chunks[index].get("section") or ""
    # Indian wordings name the exclusion in the heading itself: "XVIII. Maternity
    # Expenses (Code-Excl18)". The IRDAI code sits in the section label, not in
    # the body, so the body-only code check above never sees it.
    if CODE_MARKER.search(section):
        return "section-code"
    if re.search(EXCLUSION_SECTION, section, re.IGNORECASE):
        return "section"
    return ""


def _is_scheduled(chunk: dict) -> bool:
    return any(re.search(p, chunk["text"], re.IGNORECASE) for p in SCHEDULED_MARKERS)


def decide_state(evidence: str, scheduled: bool, has_quantity: bool) -> str:
    if evidence:
        return EXCLUDED
    if scheduled and not has_quantity:
        return SCHEDULED
    return STATED


def candidates_for(spec: FieldSpec, chunks: list[dict]) -> list[tuple]:
    """Every chunk mentioning the field, best evidence first."""
    regexes = [re.compile(p, re.IGNORECASE) for p in spec.aliases]
    found = []
    for index, chunk in enumerate(chunks):
        text = chunk["text"]
        match = next((m for m in (r.search(text) for r in regexes) if m), None)
        if match is None:
            continue

        sentence = _sentence(text, match.start())
        evidence = _evidence(chunks, index, match)
        score = 0
        if not evidence:
            score += 2
        if DEFINITION.search(sentence):
            score += 2
        if _is_scheduled(chunk):
            score += 1
        if QUANTITY.search(text):
            score += 1
        # A comma run is a list of things, not a clause that states a term.
        # A code in the section heading ("Maternity (Code-Excl18)") names the
        # document's verdict on this field; it outranks an incidental mention
        # inside some other exclusion list.
        if evidence == "section-code":
            score += 4
        # A clean sentence that explicitly grants cover ("providing cover for ...
        # maternity-related benefits") beats a coded exclusion heading: Star
        # Health sells maternity cover and still lists an Excl 18 for the base plan.
        if not evidence and COVERAGE_CUE.search(sentence):
            score += 5
        if sentence.count(",") >= 3:
            score -= 1
        found.append((score, -index, index, evidence, chunk, match))

    found.sort(key=lambda c: (c[0], c[1]), reverse=True)
    return found


def build_cell(spec: FieldSpec, policy: str, chunks: list[dict]) -> tuple[Cell, str, int, int]:
    found = candidates_for(spec, chunks)
    if not found:
        cell = Cell(spec.key, policy, NOT_STATED, aliases_tried=len(spec.aliases))
        return cell, "none", 0, 0

    score, _, index, evidence, chunk, match = found[0]
    state = decide_state(
        evidence, _is_scheduled(chunk), bool(QUANTITY.search(chunk["text"]))
    )
    cell = Cell(
        field_key=spec.key,
        policy=policy,
        state=state,
        quote=quote_around(chunk["text"], match.start(), match.end()),
        pages=(chunk["page_start"],),
        aliases_tried=len(spec.aliases),
    )
    return cell, evidence or "clean", score, len(found)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    by_policy = load_policies()
    policies = sorted(by_policy)
    tally = {STATED: 0, EXCLUDED: 0, SCHEDULED: 0, NOT_STATED: 0}

    for spec in FIELDS:
        print(f"\n=== {spec.key} ===")
        for policy in policies:
            cell, tag, score, total = build_cell(spec, policy, by_policy[policy])
            tally[cell.state] += 1
            flag = f"[{tag}:{score} of {total}]"
            if cell.state == NOT_STATED:
                print(f"  {policy:<40} {cell.state:<10} {flag:<18} ({len(spec.aliases)} aliases, 0 matches)")
            else:
                print(f"  {policy:<40} {cell.state:<10} {flag:<18} p{cell.pages[0]} {cell.quote[:90]}")

    print("\n" + "  ".join(f"{state}={count}" for state, count in tally.items()))
    print(f"total cells {sum(tally.values())} ({len(FIELDS)} fields x {len(policies)} policies)")


if __name__ == "__main__":
    main()
