"""Tests for evidence-ranked cell selection.

Synthetic chunks, not the corpus: data/ is git-ignored, so the tests must not
depend on downloaded policies.
"""
from insurance_policy_analyzer.compare import candidates_for, decide_state
from insurance_policy_analyzer.schema import EXCLUDED, SCHEDULED, STATED, FieldSpec

SPEC = FieldSpec("test_field", "Test field", (r"room\s*rent",))
MATERNITY = FieldSpec("maternity", "Maternity", (r"maternit", r"pregnan", r"child\s*birth"))


def chunk(text: str, section: str = "1. Benefits", page: int = 1) -> dict:
    return {
        "chunk_id": f"c{page}",
        "policy": "test",
        "section": section,
        "text": text,
        "page_start": page,
        "page_end": page,
    }


def test_no_candidates_means_not_stated():
    assert candidates_for(SPEC, [chunk("nothing relevant here")]) == []


def test_definition_outranks_a_list_mention():
    chunks = [
        chunk("Charges towards room rent, nursing, surgeon, anaesthetist, medicines"),
        chunk("Room rent means the amount charged by a hospital towards room and boarding"),
    ]
    assert candidates_for(SPEC, chunks)[0][2] == 1


def test_irdai_code_near_the_term_marks_an_exclusion():
    chunks = [chunk("Room rent for cosmetic procedures (Code - Excl 14)")]
    assert candidates_for(SPEC, chunks)[0][3] == "code"


def test_governing_header_in_previous_chunk_marks_an_exclusion():
    chunks = [
        chunk(
            "Exclusions. We shall not be liable to make any payment for any claim "
            "under this policy in respect of"
        ),
        chunk("room rent where the insured is admitted for cosmetic treatment"),
    ]
    assert candidates_for(SPEC, chunks)[0][3] == "prev"


def test_irdai_code_in_the_section_label_marks_an_exclusion():
    """ICICI and Niva put the code in the heading: 'Maternity (Code-Excl18)'."""
    chunks = [
        chunk(
            "Medical treatment expenses traceable to childbirth except ectopic pregnancy",
            section="XVIII. Maternity Expenses (Code-Excl18)",
        )
    ]
    assert candidates_for(MATERNITY, chunks)[0][3] == "section-code"


def test_section_code_outranks_an_incidental_mention_elsewhere():
    """ICICI mentions pregnancy inside an unrelated exclusion list; the coded
    Maternity heading is still the document's answer about maternity."""
    chunks = [
        chunk(
            "Medical treatment expenses traceable to childbirth except ectopic pregnancy",
            section="XVIII. Maternity Expenses (Code-Excl18)",
            page=13,
        ),
        chunk(
            "Injury, Illness or Hospitalization resulting directly from or contributed "
            "to by pregnancy or childbirth",
            section="12. Payment of compensation in respect of death",
            page=35,
        ),
    ]
    best = candidates_for(MATERNITY, chunks)[0]
    assert best[2] == 0 and best[3] == "section-code"


def test_no_benefit_payable_header_governs_the_list_below_it():
    chunks = [
        chunk(
            "ii. No benefit payable in any respect of any insured person for: "
            "a. Congenital anomalies b. Pregnancy, child birth, miscarriage"
        )
    ]
    assert candidates_for(MATERNITY, chunks)[0][3]


def test_state_depends_on_delegation_and_quantity():
    assert decide_state("", scheduled=True, has_quantity=False) == SCHEDULED
    assert decide_state("", scheduled=True, has_quantity=True) == STATED
    assert decide_state("", scheduled=False, has_quantity=False) == STATED
    assert decide_state("sentence", scheduled=False, has_quantity=False) == EXCLUDED
