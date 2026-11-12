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
from dataclasses import dataclass

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
# A coded heading that still grants cover after N months is a waiting period, not
# a carve-out: IRDAI Excl 01 (pre-existing disease) and the specified-disease code
# are waiting periods in almost every wording.
WAITING_PERIOD = re.compile(
    r"waiting\s+period|months?\s+of\s+continuous\s+coverage", re.IGNORECASE
)
QUANTITY = re.compile(
    r"\d+\s*(?:%|percent)|\brs\.?\s*[\d,]+|\b\d+\s*(?:months?|days?|years?)\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Scoring and scope, in one place. Every number below was tuned against the
# corpus audit in scripts/probe_*.py; docs/SCORING.md is generated from these
# two tables by `python -m insurance_policy_analyzer.compare --table`.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Rule:
    key: str
    weight: int
    what: str
    evidence: str


@dataclass(frozen=True)
class Window:
    key: str
    chars: int
    what: str
    evidence: str


SCORING_RULES: tuple[Rule, ...] = (
    Rule(
        "clean",
        2,
        "no exclusion evidence in the sentence, preamble, previous chunk or heading",
        "ICICI p13 maternity: the IRDAI code sits in the heading, so a body-only "
        "check scored a carve-out as a clean statement",
    ),
    Rule(
        "definition",
        2,
        'the owning sentence defines the term ("means")',
        'Niva p4 "Base Sum Insured means the amount stated in the Policy Schedule" '
        "beats an incidental mention elsewhere",
    ),
    Rule(
        "scheduled",
        1,
        "the chunk defers to the Policy Schedule or the Policy Certificate",
        'ICICI p7 "amount specified in the Policy Certificate"; all three wordings '
        "defer quantities, only the brochure prints them",
    ),
    Rule(
        "quantity",
        1,
        "the chunk carries an amount, percentage or duration",
        'Star p3 "Upto 1% of Sum Insured per day" against ICICI\'s 205,113 '
        "characters with no per-day figure",
    ),
    Rule(
        "comma_list",
        -1,
        "three or more commas: a list of things, not a clause",
        'Care p28 "Deductible, Non-payable claims, Co-payments, excess claim amount" '
        "outscored the real co-payment clause at p25 before this penalty existed",
    ),
    Rule(
        "section_code",
        4,
        "an IRDAI exclusion code in the section heading",
        'Niva p12 "XVIII. Maternity Expenses (Code-Excl18)"; ICICI p13 '
        '"3. Maternity (Code - Excl 18)"',
    ),
    Rule(
        "coverage_cue",
        5,
        "a clean sentence that explicitly grants cover",
        'Star p1 "providing cover ... comprehensive maternity-related benefits", in a '
        "brochure that also lists Excl 18 for the base indemnity",
    ),
)

WINDOWS: tuple[Window, ...] = (
    Window(
        "preamble_chars",
        250,
        "text before the match inside the same chunk",
        'ICICI p34 "No benefit payable in any respect of any insured person for: '
        'a. ... b. Pregnancy" - the header governs the list that follows it',
    ),
    Window(
        "next_chars",
        220,
        "text after the match, past the sentence end",
        'ICICI Excl 01 "until the expiry of / 36 months of continuous coverage" - a '
        "PDF line break splits the duration into the next sentence",
    ),
    Window(
        "tail_chars",
        250,
        "tail of the previous chunk",
        'ICICI p8 "We shall not be liable ... in respect of Prehospitalization '
        'Medical Expenses" - the governing line is one chunk above the item',
    ),
    Window(
        "code_radius",
        150,
        "characters either side of the match",
        'Star p15 "a medical practitioner as part of hospitalization claim or day '
        'care procedure. Code-Excl 14" - the code follows the sentence',
    ),
)

_WINDOW = {window.key: window.chars for window in WINDOWS}
PREAMBLE_CHARS = _WINDOW["preamble_chars"]
NEXT_CHARS = _WINDOW["next_chars"]
TAIL_CHARS = _WINDOW["tail_chars"]
CODE_RADIUS = _WINDOW["code_radius"]

_WEIGHT = {rule.key: rule.weight for rule in SCORING_RULES}

CHUNKS_GLOB = "data/chunks/*.json"


def rules_markdown() -> str:
    """The weights and windows as documentation. Generated, never hand-edited."""
    lines = [
        "# Comparison scoring: every number and the evidence behind it",
        "",
        "Generated by `python -m insurance_policy_analyzer.compare --table`.",
        "Do not edit by hand; edit the tables in `compare.py` instead.",
        "",
        "## Scoring weights",
        "",
        "| rule | weight | what it rewards | evidence in the corpus |",
        "| --- | --- | --- | --- |",
    ]
    for rule in SCORING_RULES:
        lines.append(f"| `{rule.key}` | {rule.weight:+d} | {rule.what} | {rule.evidence} |")
    lines += [
        "",
        "The state of a cell comes from the highest-scoring candidate, not from the",
        "first mention in document order. That single change is what turned both a",
        "run of 30 false exclusions and a run of 41 false statements into one",
        "consistent matrix.",
        "",
        "| state | decision |",
        "| --- | --- |",
        "| `excluded` | the winning candidate carries exclusion evidence |",
        "| `scheduled` | the winner defers to the schedule and states no number |",
        "| `stated` | otherwise |",
        "| `not stated` | no alias matched in a document proven complete |",
        "",
        "## Evidence windows",
        "",
        "| window | chars | what it reads | evidence in the corpus |",
        "| --- | --- | --- | --- |",
    ]
    for window in WINDOWS:
        lines.append(
            f"| `{window.key}` | {window.chars} | {window.what} | {window.evidence} |"
        )
    lines += [
        "",
        "Exclusion evidence is read narrowest first: the sentence owning the term,",
        "then these windows. A chunk-wide window (plus both neighbours) produced 30",
        "false exclusions; a sentence-only window produced 41 false statements. The",
        "ladder exists because the true scope of a clause sits between those two.",
        "",
    ]
    return "\n".join(lines)



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
        window = sentence + text[match.end():match.end() + NEXT_CHARS]
        if evidence == "section-code" and WAITING_PERIOD.search(window):
            # "Code-Excl01 ... until 36 months of continuous coverage" is cover
            # that starts later, so the cell is stated, not excluded.
            evidence = ""
        score = 0
        if not evidence:
            score += _WEIGHT["clean"]
        if DEFINITION.search(sentence):
            score += _WEIGHT["definition"]
        if _is_scheduled(chunk):
            score += _WEIGHT["scheduled"]
        if QUANTITY.search(text):
            score += _WEIGHT["quantity"]
        # A comma run is a list of things, not a clause that states a term.
        # A code in the section heading ("Maternity (Code-Excl18)") names the
        # document's verdict on this field; it outranks an incidental mention
        # inside some other exclusion list.
        if evidence == "section-code":
            score += _WEIGHT["section_code"]
        # A clean sentence that explicitly grants cover ("providing cover for ...
        # maternity-related benefits") beats a coded exclusion heading: Star
        # Health sells maternity cover and still lists an Excl 18 for the base plan.
        if not evidence and COVERAGE_CUE.search(sentence):
            score += _WEIGHT["coverage_cue"]
        if sentence.count(",") >= 3:
            score -= _WEIGHT["comma_list"]
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
    if "--table" in sys.argv:
        print(rules_markdown())
        return

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
