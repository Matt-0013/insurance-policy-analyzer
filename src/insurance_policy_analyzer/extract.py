import json
import re
from pathlib import Path

import pymupdf
import pymupdf4llm

from insurance_policy_analyzer.clean import clean_text
from insurance_policy_analyzer.config import PROCESSED_DIR, RAW_DIR

# Share of letters+digits that must survive vs plain PyMuPDF text.
MIN_COVERAGE = 0.8


def _alnum(text: str) -> int:
    text = re.sub(r"<br>|</?mark>", "", text)
    return len(re.sub(r"[^A-Za-z0-9]", "", text))


def coverage(text: str, reference: str) -> float:
    """How much of the reference text's letters and digits is present in text."""
    return _alnum(text) / max(1, _alnum(reference))


def _markdown_pages(pdf_path: Path, pages: list[int] | None, layout: bool) -> list[str]:
    """layout=False is classic mode (keeps table text complete);
    layout=True is PyMuPDF4LLM 2.x's default mode (better headings)."""
    pymupdf4llm.use_layout(layout)
    kwargs = {"page_chunks": True}
    if pages is not None:
        kwargs["pages"] = pages  # 0-based page indexes
    return [c["text"] for c in pymupdf4llm.to_markdown(str(pdf_path), **kwargs)]


def extract_pdf(pdf_path: Path) -> list[dict]:
    with pymupdf.open(pdf_path) as doc:
        plain = [page.get_text() for page in doc]

    texts = _markdown_pages(pdf_path, None, layout=False)
    methods = ["classic"] * len(texts)
    scores = [coverage(t, r) for t, r in zip(texts, plain)]

    weak = [i for i, s in enumerate(scores) if s < MIN_COVERAGE]
    if weak:
        for i, t in zip(weak, _markdown_pages(pdf_path, weak, layout=True)):
            score = coverage(t, plain[i])
            if score > scores[i]:
                texts[i], methods[i], scores[i] = t, "layout", score

    for i, s in enumerate(scores):
        if s < MIN_COVERAGE:  # last resort: keeps every word, loses structure
            texts[i], methods[i], scores[i] = plain[i], "plain", 1.0

    return [
        {"page": i + 1, "text": clean_text(t), "method": m, "coverage": round(s, 2)}
        for i, (t, m, s) in enumerate(zip(texts, methods, scores))
    ]


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
        pages = extract_pdf(pdf_path)
        out_file = PROCESSED_DIR / f"{pdf_path.stem}.json"
        out_file.write_text(json.dumps(pages, indent=2, ensure_ascii=False), encoding="utf-8")
        counts = {m: sum(1 for p in pages if p["method"] == m) for m in ("classic", "layout", "plain")}
        print(f"{pdf_path.name}: {len(pages)} pages, {counts}")


if __name__ == "__main__":
    main()
