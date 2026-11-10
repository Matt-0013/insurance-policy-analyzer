"""Shared Groq chat helper.

Every LLM call in the project goes through here so model choice, temperature
and reasoning effort are set in one place (and can be swapped for tests).
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from groq import Groq

from insurance_policy_analyzer.config import GENERATION_MODEL, REASONING_EFFORT

load_dotenv()


@lru_cache(maxsize=1)
def client() -> Groq:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set; copy .env.example to .env and add your key")
    return Groq(api_key=key)


def chat(
    messages: list[dict],
    *,
    model: str = GENERATION_MODEL,
    temperature: float = 0.0,
    reasoning_effort: str | None = REASONING_EFFORT,
    response_format: dict | None = None,
):
    kwargs = {"model": model, "messages": messages, "temperature": temperature}
    # reasoning_effort only exists on the gpt-oss models.
    if reasoning_effort:
        kwargs["reasoning_effort"] = reasoning_effort
    if response_format:
        kwargs["response_format"] = response_format
    return client().chat.completions.create(**kwargs)
