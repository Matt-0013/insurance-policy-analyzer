"""CLI: ask one question and print the grounded answer with its sources."""

import sys

from insurance_policy_analyzer.pipeline import ask


def answer(question: str, **kwargs) -> dict:
    """Backwards-compatible wrapper around pipeline.ask."""
    return ask(question, **kwargs)


def main() -> None:
    # Model output contains typographic dashes; the default Windows console codec
    # cannot encode them and would crash on print.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    question = " ".join(sys.argv[1:]).strip()
    if not question:
        print('usage: python -m insurance_policy_analyzer.answer "your question"')
        return

    result = answer(question)
    print(result["answer"])

    print("\nSources:")
    for source in result["sources"]:
        print(f"  [{source['n']}] {source['policy_label']} | {source['section']} | p.{source['page_start']}")

    grounding = result["grounding"]
    if not grounding["grounded"]:
        print("\nGrounding warning:")
        for number in grounding["unsupported_numbers"]:
            print(f"  figure not found in sources: {number['raw']} ({number['kind']})")
        for number in grounding["invalid_citations"]:
            print(f"  citation points at a missing source: [{number}]")
        if not grounding["citations"]:
            print("  answer cites no sources")

    if result["usage"]:
        usage = result["usage"]
        print(
            f"\nTokens: {usage['prompt_tokens']} prompt + "
            f"{usage['completion_tokens']} completion = {usage['total_tokens']}"
        )
    print(
        f"Timing: {result['timings_ms']['retrieval']} ms retrieval, "
        f"{result['timings_ms']['total']} ms total"
    )


if __name__ == "__main__":
    main()
