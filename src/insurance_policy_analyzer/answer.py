import os
import sys
from functools import lru_cache

from dotenv import load_dotenv
from groq import Groq

from insurance_policy_analyzer.config import GENERATION_MODEL, REASONING_EFFORT, TOP_K
from insurance_policy_analyzer.search import search

load_dotenv()

SYSTEM_PROMPT = """You answer questions about Indian health insurance policies.
Use ONLY the numbered sources provided. Cite the sources you use, like [1] or [2][3].
Say which policy each fact comes from.
If the sources do not contain the answer, reply exactly: I could not find this in the provided policies.
Never guess amounts, periods or limits.
This is general information, not insurance advice."""


@lru_cache(maxsize=1)
def _client() -> Groq:
    return Groq(api_key=os.environ["GROQ_API_KEY"])


def format_sources(hits: list[dict]) -> str:
    blocks = []
    for n, hit in enumerate(hits, start=1):
        policy = hit["policy"].replace("_", " ")
        blocks.append(f"[{n}] {policy} | {hit['section']} | page {hit['page_start']}\n{hit['text']}")
    return "\n\n".join(blocks)


def answer(question: str, k: int = TOP_K) -> dict:
    hits = search(question, k=k)
    response = _client().chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Sources:\n{format_sources(hits)}\n\nQuestion: {question}"},
        ],
        reasoning_effort=REASONING_EFFORT,
        temperature=0,
    )
    return {"answer": response.choices[0].message.content, "sources": hits, "usage": response.usage}


def main() -> None:
    question = " ".join(sys.argv[1:])
    result = answer(question)
    print(result["answer"])
    print("\nSources:")
    for n, hit in enumerate(result["sources"], start=1):
        print(f"  [{n}] {hit['policy']} | {hit['section']} | p.{hit['page_start']}")
    usage = result["usage"]
    print(f"\nTokens: prompt {usage.prompt_tokens}, completion {usage.completion_tokens}, total {usage.total_tokens}")


if __name__ == "__main__":
    main()