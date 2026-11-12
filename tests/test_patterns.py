"""Every regex in the comparison engine must match a known example sentence.

A pattern once became literal backspace bytes (a non-raw string turned \\b into
\\x08), compiled without error, and matched nothing. No test failed; the cell just
came out wrong. These tests make that class of bug loud: every declared pattern
needs a sample, the sample must match, and no pattern may contain control
characters.
"""
import re

from insurance_policy_analyzer import compare, schema

CONTROL = re.compile(r"[\x00-\x1f]")


def pattern_text(value) -> str:
    return value.pattern if hasattr(value, "pattern") else value


def declared_patterns():
    for pattern in schema.EXCLUSION_MARKERS:
        yield "schema.EXCLUSION_MARKERS", pattern
    for spec in schema.FIELDS:
        for pattern in spec.aliases:
            yield f"field:{spec.key}", pattern
    for name in ("SCHEDULED_MARKERS", "STRONG_EXCLUSION"):
        for pattern in getattr(compare, name):
            yield f"compare.{name}", pattern
    for name in (
        "EXCLUSION_SECTION",
        "CODE_MARKER",
        "DEFINITION",
        "QUANTITY",
        "COVERAGE_CUE",
        "WAITING_PERIOD",
    ):
        yield f"compare.{name}", getattr(compare, name)


SAMPLES = {
    # schema.EXCLUSION_MARKERS
    r"shall\s+not\s+be\s+liable": "We shall not be liable to make any payment for this claim",
    r"\bnot\s+covered\b": "Maternity is not covered for the first two years",
    r"\bexcluded\b": "Expenses for cosmetic surgery are excluded",
    r"code\s*[-\u2013]\s*excl\s*\d+": "Obesity / Weight Control (Code \u2013 Excl 06)",
    r"no\s+benefit\s+(?:is\s+)?payable": "No benefit payable in any respect of any insured person for",
    r"\bnot\s+payable\b": "This benefit is not payable after the policy expires",
    # fields
    r"sum\s*insured": "up to the Sum Insured specified in the schedule",
    r"room\s*rent": "Room Rent means the amount charged by a hospital",
    r"room\s*,?\s*boarding": "Room, Boarding and Nursing Expenses",
    r"room\s*(?:category|type)": "Room Category shall be the most basic available",
    r"nursing\s*expenses": "Room, Boarding and Nursing Expenses",
    r"general\s*ward": "General Ward means a basic category of shared room",
    r"proportionate\s*deduction": "Proportionate deductions are not applied",
    r"co-?\s*pay": "the insured person will bear a specified co-payment on each claim",
    r"deductible": "subject to a deductible of Rs. 5,000 per event",
    r"pre-?\s*existing\s*(?:disease|condition|ailment|illness)": "pre-existing disease means any condition diagnosed before",
    r"code\s*[-\u2013]\s*excl\s*0?1(?!\d)": "Pre-Existing Diseases: Code-Excl01 applies",
    r"specified\s*waiting\s*period": "specified waiting period of 24 months applies",
    r"specific\s*waiting\s*period": "Specific Waiting Period means a period of exclusion",
    r"specified\s*disease": "specified disease/procedure waiting period",
    r"time\s*bound\s*exclusion": "time bound exclusions are listed in the annexure",
    r"(?:initial|first)\s+waiting\s+period": "an initial waiting period of 30 days applies",
    r"waiting\s+period\s+of\s+30\s+days": "a waiting period of 30 days from commencement",
    r"30\s+days\s+from\s+the\s+(?:first\s+)?(?:policy\s+)?commencement": "within 30 days from the first policy commencement date",
    r"first\s+30\s+days\s+of\s+(?:the\s+)?(?:policy|cover)": "expenses incurred in the first 30 days of cover",
    r"maternit": "Maternity Expenses are payable after the waiting period",
    r"childbirth": "Medical treatment expenses traceable to childbirth",
    r"pregnan": "except ectopic pregnancy, all pregnancy expenses are excluded",
    r"caesarean": "including caesarean sections incurred during hospitalization",
    r"ectopic": "except ectopic pregnancy",
    r"day\s*care": "Day Care Treatment in a Day Care Centre",
    r"pre-?\s*hospitali[sz]ation": "Pre-hospitalization Medical Expenses for 60 days",
    r"post-?\s*hospitali[sz]ation": "Post Hospitalization expenses for 90 days",
    r"(?:road|air)\s+ambulance": "Road Ambulance charges are payable",
    r"ambulance\s+(?:charges|expenses|services|cover|benefit|fees)": "Ambulance charges are payable up to Rs. 2,000",
    r"(?:charges|expenses|cost)\s+(?:for|towards)\s+(?:an?\s+)?ambulance": "expenses towards ambulance by road",
    r"expenses\s+to\s+reach\s+(?:the\s+)?hospital": "expenses to reach hospital in an emergency",
    r"\bambulance\b(?!\s*(?:collar|equipment|belt|binder|\d))": "a registered ambulance provider must be used",
    r"automatic\s*restoration": "Automatic Restoration of Sum Insured",
    r"\brestored\s*sum\s*insured": "Restored Sum Insured will not be taken into consideration",
    r"restoration\s*of\s*(?:the\s*)?sum\s*insured": "automatic restoration of the sum insured",
    # compare.SCHEDULED_MARKERS
    r"(?:specified|stated|mentioned)\s+in\s+the\s+(?:policy\s+)?(?:schedule|certificate)": "as specified in the Policy Schedule",
    r"as\s+per\s+the\s+(?:policy\s+)?(?:schedule|certificate)": "as per the policy certificate",
    r"(?:amount|details?)\s+(?:as\s+)?(?:stated|specified)\s+in\s+the": "the amount stated in the schedule",
    # compare.STRONG_EXCLUSION reuses two EXCLUSION_MARKERS patterns
    # compare singletons
    r"^\W*(?:(?:\d+|[IVXLC]+)[.)])?\s*\**\s*exclusions?\b|what\s+is\s+not\s+covered": "18. Exclusions",
    r"code\s*[-\u2013]\s*excl\s*\d+": "Maternity (Code \u2013 Excl 18)",
    r"\bmeans\b": "Room rent means the amount charged by a hospital",
    r"\d+\s*(?:%|percent)|\brs\.?\s*[\d,]+|\b\d+\s*(?:months?|days?|years?)\b": "Room rent up to 1% of Sum Insured per day",
    r"\bcover(?:ed|ing)?\b|\bbenefits?\b|\bwe\s+will\s+pay\b": "providing cover for comprehensive maternity-related benefits",
    r"waiting\s+period|months?\s+of\s+continuous\s+coverage": "until 36 months of continuous coverage, the disease is excluded",
}


def test_every_declared_pattern_has_a_sample_and_matches_it():
    control, missing, failures = [], [], []
    for owner, raw in declared_patterns():
        pattern = pattern_text(raw)
        if CONTROL.search(pattern):
            control.append(f"{owner}: control character in {pattern!r}")
        sample = SAMPLES.get(pattern)
        if sample is None:
            missing.append(f"{owner}: {pattern}")
            continue
        if not re.search(pattern, sample, re.IGNORECASE):
            failures.append(f"{owner} did not match {sample!r}: {pattern}")
    assert not control, control
    assert not missing, missing
    assert not failures, failures


def test_samples_are_sentences_not_fragments():
    for pattern, sample in SAMPLES.items():
        assert len(sample.split()) >= 2, pattern
