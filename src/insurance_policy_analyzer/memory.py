"""Conversation memory.

Follow-up questions ("and its waiting period?") only make sense with the earlier
turns, so the transcript is fed to both the query rewriter and the generator.
It is deliberately bounded: old turns are dropped instead of growing the prompt
(and the bill) without limit.
"""

from dataclasses import dataclass, field

MAX_ANSWER_CHARS = 600


@dataclass
class Conversation:
    max_turns: int = 3
    turns: list[dict] = field(default_factory=list)

    def add(self, question: str, answer: str) -> None:
        self.turns.append({"question": question, "answer": answer})
        del self.turns[: -self.max_turns]

    def transcript(self, max_chars: int = 2000) -> str:
        """Most recent turns that fit the budget, oldest first."""
        blocks = []
        used = 0
        for turn in reversed(self.turns):
            answer = turn["answer"].strip()
            if len(answer) > MAX_ANSWER_CHARS:
                answer = answer[:MAX_ANSWER_CHARS].rstrip() + "..."
            block = f"Q: {turn['question'].strip()}\nA: {answer}"
            if blocks and used + len(block) > max_chars:
                break
            blocks.append(block)
            used += len(block)
        return "\n\n".join(reversed(blocks))
