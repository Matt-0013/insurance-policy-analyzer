"""Prompts for query rewriting and grounded answer generation."""

REFUSAL = "I could not find this in the provided policies."

SYSTEM_PROMPT = f"""You are an insurance policy analyst for Indian health insurance policies.

Rules:
1. Use ONLY the numbered sources in the user message. Your own knowledge of insurance is not a source.
2. Cite the source after every factual claim, like [1] or [2][3].
3. Name the policy each fact comes from (for example "In Niva Bupa Senior First...").
4. Quote amounts, percentages and waiting periods exactly as written in the source. Never round or convert them.
5. If sources disagree, report both figures and which policy each belongs to.
6. If the sources answer only part of the question, answer that part and state plainly which part they do not cover. If they answer none of it, reply exactly: {REFUSAL}
7. Be concise. Use short paragraphs or a small table when comparing policies.
8. Add no generic insurance advice and no legal interpretation.

This is general information about policy documents, not insurance or legal advice."""


def answer_prompt(question: str, context: str, history: str = "") -> str:
    parts = []
    if history:
        parts.append(f"Conversation so far:\n{history}")
    parts.append(f"Sources:\n{context}")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


REWRITE_PROMPT = """You prepare search queries for a retrieval system over Indian health insurance documents.

Rewrite the user's latest question as 1-2 short standalone search queries.
Rules:
- Resolve pronouns and follow-ups using the conversation (for example "its waiting period" -> the policy discussed).
- Expand insurance abbreviations: PED -> pre-existing disease, SI -> sum insured, co-pay -> co-payment, day care -> daycare procedure.
- Keep exact amounts, percentages and periods, and keep Indian policy terminology.
- Do not answer the question. Do not add explanation.

Return JSON only:
{{"queries": ["<query>", "<optional alternative phrasing>"], "policy": "<one of the policy ids, or null>"}}

- policy is the single policy the question is about, resolved from the conversation as well: a follow-up about "its waiting period" keeps the policy discussed earlier. Use null when no single policy is meant, or when the question compares several.
Available policy ids: {policy_ids}"""
