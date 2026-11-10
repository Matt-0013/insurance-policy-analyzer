"""Citation parsing and grounding checks for generated answers.

The generator is an LLM, so its output cannot be trusted on its own. These
checks are deterministic: every figure the answer states must appear in a
retrieved source, and every citation must point at a source that exists.
"""

import re

from insurance_policy_analyzer.prompts import REFUSAL

CITE = re.compile(r"\[(\d+)\]")

# A number, optional Indian multiplier or percent sign. The lookbehind keeps
# citation markers ([1]) and word-internal digits (chunk-0007) out.
NUMBER = re.compile(
    r"(?<![\w\[.])(\d+(?:,\d{2,3})*(?:\.\d+)?)\s*(lakhs?|crores?|%)?",
    re.IGNORECASE,
)

MULTIPLIERS = {"lakh": 100_000, "lakhs": 100_000, "crore": 10_000_000, "crores": 10_000_000}

# Figures in these categories must be traceable to a source. Plain counts
# ("3 policies") are informational and not flagged.
GROUNDED_KINDS = {"amount", "percent", "duration"}

DURATION_AFTER = re.compile(r"^[\s-]*(days?|months?|years?)", re.IGNORECASE)
CURRENCY_BEFORE = re.compile(r"(?:rs\.?|inr|₹)\s*$", re.IGNORECASE)


def normalize_number(digits: str, unit: str | None) -> str:
    """5,00,000 and 5 lakh both become 500000, so paraphrases still match."""
    value = float(digits.replace(",", ""))
    if unit and unit.lower() in MULTIPLIERS:
        value *= MULTIPLIERS[unit.lower()]
    suffix = "%" if unit == "%" else ""
    return f"{value:g}{suffix}"


def extract_numbers(text: str) -> list[dict]:
    """Every numeric token in the text with its normalised value and category."""
    found = []
    for match in NUMBER.finditer(text):
        digits, unit = match.group(1), match.group(2)
        tail = text[match.end() : match.end() + 12]
        before = text[max(0, match.start() - 6) : match.start()]
        if unit == "%" or "%" in text[match.end() : match.end() + 2]:
            kind = "percent"
        elif unit or "," in digits or CURRENCY_BEFORE.search(before):
            kind = "amount"
        elif DURATION_AFTER.match(tail):
            kind = "duration"
        else:
            kind = "plain"
        found.append(
            {
                "raw": match.group().strip(),
                "norm": normalize_number(digits, unit),
                "kind": kind,
            }
        )
    return found


def parse_citations(answer: str) -> list[int]:
    return sorted({int(n) for n in CITE.findall(answer)})


def unsupported_numbers(answer: str, sources: list[str]) -> list[dict]:
    """Figures the answer states that no source contains."""
    known = {n["norm"] for source in sources for n in extract_numbers(source)}
    flagged, seen = [], set()
    for number in extract_numbers(answer):
        if number["kind"] not in GROUNDED_KINDS or number["norm"] in known:
            continue
        if number["raw"] in seen:
            continue
        seen.add(number["raw"])
        flagged.append(number)
    return flagged


def verify(answer: str, sources: list[str], refusal: str = REFUSAL) -> dict:
    """Check citation numbers and figures against the sources.

    grounded is False when the answer cites a missing source, cites nothing, or
    states a figure that appears in no source.
    """
    if answer.strip() == refusal.strip():
        return {
            "citations": [],
            "invalid_citations": [],
            "unsupported_numbers": [],
            "grounded": True,
            "refused": True,
        }

    citations = parse_citations(answer)
    invalid = [n for n in citations if n < 1 or n > len(sources)]
    unsupported = unsupported_numbers(answer, sources)
    return {
        "citations": citations,
        "invalid_citations": invalid,
        "unsupported_numbers": unsupported,
        "grounded": bool(citations) and not invalid and not unsupported,
        "refused": False,
    }
