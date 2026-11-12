"""Render the comparison matrix as markdown, one cell per field and policy.

A cell never prints a bare value: it prints why it should be believed (page
number) and what state the document is in when it does not state the term.
"""
import sys

from .compare import build_cell, load_policies
from .schema import (
    EXCLUDED,
    FIELDS,
    NOT_STATED,
    SCHEDULED,
    Cell,
)

SHORT_NAMES = {
    "care_health_care_advanced_terms": "Care Health",
    "icici_lombard_family_shield_wording": "ICICI Lombard",
    "niva_bupa_senior_first_wording": "Niva Bupa",
    "star_health_women_care_brochure": "Star Health",
}


def _short(policy: str) -> str:
    return SHORT_NAMES.get(policy, policy)


def _cell_text(cell: Cell, width: int = 90) -> str:
    if cell.state == NOT_STATED:
        return f"not stated ({cell.aliases_tried} aliases tried)"
    if cell.state == SCHEDULED:
        return f"set by the policy schedule (p{cell.pages[0]})"
    if cell.state == EXCLUDED:
        return f"excluded (p{cell.pages[0]})"
    quote = cell.quote.replace("|", "/")
    return f"{quote[:width]} (p{cell.pages[0]})"


def build_matrix(policies: list[str] | None = None) -> tuple[list[str], list[list[Cell]]]:
    by_policy = load_policies()
    chosen = sorted(policies or by_policy)
    missing = [policy for policy in chosen if policy not in by_policy]
    if missing:
        raise KeyError(", ".join(missing))
    rows = [
        [build_cell(spec, policy, by_policy[policy])[0] for policy in chosen]
        for spec in FIELDS
    ]
    return chosen, rows


def to_markdown(policies: list[str] | None = None) -> str:
    chosen, rows = build_matrix(policies)
    lines = [
        "| field | " + " | ".join(_short(p) for p in chosen) + " |",
        "| --- |" + " --- |" * len(chosen),
    ]
    for spec, cells in zip(FIELDS, rows):
        lines.append(
            "| " + spec.label + " | " + " | ".join(_cell_text(c) for c in cells) + " |"
        )
    lines.append("")
    lines.append(
        "*A cell reports what the document says, not what the product does in "
        "practice. `set by the policy schedule` means the wording defers the "
        "number to a schedule that is not part of this corpus.*"
    )
    return "\n".join(lines)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(to_markdown())


if __name__ == "__main__":
    main()
