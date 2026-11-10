"""Policy metadata: display names, insurers and aliases used in filters and citations."""

import json
import re
from functools import lru_cache

from insurance_policy_analyzer.config import POLICIES_FILE

# Words that only describe the document type, not the product.
DOC_WORDS = {"wording", "terms", "conditions", "brochure", "policy", "document", "and"}


def _derive(policy_id: str) -> dict:
    words = [w for w in policy_id.split("_") if w not in DOC_WORDS]
    return {
        "insurer": words[0].replace("-", " ").title() if words else policy_id,
        "name": " ".join(words).title(),
        "doc_type": "document",
        "aliases": [],
    }


@lru_cache(maxsize=1)
def all_policies() -> dict[str, dict]:
    """Registry keyed by policy id, with a derived entry for unknown ids."""
    known = {}
    if POLICIES_FILE.exists():
        known = json.loads(POLICIES_FILE.read_text(encoding="utf-8"))
    return known


def info(policy_id: str) -> dict:
    return all_policies().get(policy_id) or _derive(policy_id)


def display_name(policy_id: str) -> str:
    meta = info(policy_id)
    return f"{meta['insurer']} - {meta['name']}"


def label(policy_id: str) -> str:
    """Full citation label, e.g. 'Niva Bupa - Senior First (policy wording)'."""
    meta = info(policy_id)
    return f"{meta['insurer']} - {meta['name']} ({meta['doc_type']})"


def policy_ids() -> list[str]:
    return sorted(all_policies())


def _phrase_in(text: str, phrase: str) -> bool:
    pattern = rf"(?<![a-z0-9]){re.escape(phrase.lower())}(?![a-z0-9])"
    return re.search(pattern, text) is not None


def search_query(text: str, policy_id: str) -> str:
    """Remove the policy name when retrieval is already filtered to that policy.

    Every chunk is embedded with its policy name prefixed, so leaving the name in
    a short query makes all chunks look equally similar and buries the terms that
    actually discriminate ("co-payment" works, "co-payment in Niva Bupa Senior
    First" returns definitions).
    """
    meta = info(policy_id)
    phrases = sorted({*meta["aliases"], meta["insurer"], meta["name"]}, key=len, reverse=True)
    result = text
    for phrase in phrases:
        if len(phrase) < 3:
            continue
        result = re.sub(
            rf"(?i)(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", " ", result
        )
    result = re.sub(r"\s+", " ", result).strip(" ?.,")
    return result or text


def match_policy(question: str, candidates: list[str] | None = None) -> str | None:
    """Return the policy named in the question, if exactly one is clearly named.

    A question naming two policies is a comparison, so it returns None on
    purpose: retrieval should then search every document.
    """
    text = re.sub(r"\s+", " ", question.lower())
    named = set()
    for policy_id in candidates or policy_ids():
        meta = info(policy_id)
        phrases = [*meta["aliases"], meta["insurer"], meta["name"], display_name(policy_id)]
        if any(len(p.strip()) >= 3 and _phrase_in(text, p.strip()) for p in phrases):
            named.add(policy_id)
    return named.pop() if len(named) == 1 else None
