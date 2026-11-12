"""Comparison schema: fields, aliases, and what a cell is allowed to claim.

Every field here survived the corpus audit (scripts/check_field_coverage.py,
probe_concept.py, probe_structure.py, probe_limits.py). Fields the audit could
not support were dropped rather than guessed at.
"""
import sys
from dataclasses import dataclass

STATED = "stated"
EXCLUDED = "excluded"
SCHEDULED = "scheduled"
NOT_STATED = "not_stated"
STATES = (STATED, EXCLUDED, SCHEDULED, NOT_STATED)

# Phrases that turn a mention into a carve-out. A cell is never "stated"
# just because an alias matched: Indian wordings discuss exclusions using
# the same vocabulary as the benefit (maternity, obesity, dental).
EXCLUSION_MARKERS = (
    r"shall\s+not\s+be\s+liable",
    r"\bnot\s+covered\b",
    r"\bexcluded\b",
    r"code\s*[-\u2013]\s*excl\s*\d+",
    # ICICI writes the carve-out as "No benefit payable in any respect of any
    # insured person for: a. ... b. Pregnancy..." — the header governs the list.
    r"no\s+benefit\s+(?:is\s+)?payable",
    r"\bnot\s+payable\b",
)


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    aliases: tuple[str, ...]


@dataclass(frozen=True)
class Cell:
    """One field for one policy. Absence is a state, not an empty string."""

    field_key: str
    policy: str
    state: str
    value: str = ""
    quote: str = ""
    pages: tuple[int, ...] = ()
    aliases_tried: int = 0


FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("sum_insured", "Sum insured", (r"sum\s*insured",)),
    FieldSpec(
        "room_rent_limit",
        "Room rent limit",
        (
            r"room\s*rent",
            r"room\s*,?\s*boarding",
            r"room\s*(?:category|type)",
            r"nursing\s*expenses",
            r"general\s*ward",
            r"proportionate\s*deduction",
        ),
    ),
    FieldSpec("co_payment", "Co-payment", (r"co-?\s*pay",)),
    FieldSpec("deductible", "Deductible", (r"deductible",)),
    FieldSpec(
        "pre_existing_disease",
        "Pre-existing disease",
        (
            r"pre-?\s*existing\s*(?:disease|condition|ailment|illness)",
            r"code\s*[-\u2013]\s*excl\s*0?1(?!\d)",
        ),
    ),
    FieldSpec(
        "specific_waiting_period",
        "Specific disease waiting period",
        (
            r"specified\s*waiting\s*period",
            r"specific\s*waiting\s*period",
            r"specified\s*disease",
            r"time\s*bound\s*exclusion",
        ),
    ),
    FieldSpec(
        "initial_waiting_period",
        "Initial waiting period",
        (
            r"(?:initial|first)\s+waiting\s+period",
            r"waiting\s+period\s+of\s+30\s+days",
            r"30\s+days\s+from\s+the\s+(?:first\s+)?(?:policy\s+)?commencement",
            r"first\s+30\s+days\s+of\s+(?:the\s+)?(?:policy|cover)",
        ),
    ),
    FieldSpec(
        "maternity",
        "Maternity",
        (r"maternit", r"childbirth", r"pregnan", r"caesarean", r"ectopic"),
    ),
    FieldSpec("day_care", "Day care treatment", (r"day\s*care",)),
    FieldSpec(
        "pre_post_hospitalisation",
        "Pre/post hospitalisation",
        (r"pre-?\s*hospitali[sz]ation", r"post-?\s*hospitali[sz]ation"),
    ),
    FieldSpec(
        "ambulance",
        "Ambulance",
        (
            r"(?:road|air)\s+ambulance",
            r"ambulance\s+(?:charges|expenses|services|cover|benefit|fees)",
            r"expenses\s+to\s+reach\s+(?:the\s+)?hospital",
            r"(?:charges|expenses|cost)\s+(?:for|towards)\s+(?:an?\s+)?ambulance",
            r"\bambulance\b(?!\s*(?:collar|equipment|belt|binder|\d))",
        ),
    ),
    FieldSpec(
        "restoration",
        "Restoration of sum insured",
        (
            r"automatic\s*restoration",
            r"\brestored\s*sum\s*insured",
            r"restoration\s*of\s*(?:the\s*)?sum\s*insured",
        ),
    ),
)

FIELDS_BY_KEY = {spec.key: spec for spec in FIELDS}


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(f"{len(FIELDS)} fields   states: {', '.join(STATES)}\n")
    for spec in FIELDS:
        print(f"{spec.key:<26} {len(spec.aliases):>2} aliases   {spec.label}")
    print(f"\n{len(EXCLUSION_MARKERS)} exclusion markers")