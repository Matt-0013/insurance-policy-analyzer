import re

import wordninja

# 4+ single letters separated by spaces, optionally preceded by a short word
# fragment (e.g. "Pree x i s t i n g"). Normal prose does not match this.
SPACED_OUT = re.compile(
    r"(?<![A-Za-z])(?:[A-Za-z]{1,4} )?(?:[A-Za-z] ){3,}[A-Za-z](?![A-Za-z])"
)


def _rebuild(match: re.Match) -> str:
    letters = match.group().replace(" ", "")
    return " ".join(wordninja.split(letters))


def clean_text(text: str) -> str:
    """Rebuild words from letter-spaced text found in some insurer PDFs."""
    return SPACED_OUT.sub(_rebuild, text)