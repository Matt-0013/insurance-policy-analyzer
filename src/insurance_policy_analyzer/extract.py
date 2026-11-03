import json
from pathlib import Path

import pymupdf4llm

from insurance_policy_analyzer.clean import clean_text

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")
MIN_CHARS = 50  # a page with less text than this is treated as a failed extraction


def _markdown_pages(pdf_path: Path, pages: list[int] | None, layout: bool) -> list[str]:
    """Extract markdown per page. layout=False is classic mode, which keeps
    table text complete; layout=True is PyMuPDF4LLM 2.x's default mode."""
    pymupdf4llm.use_layout(layout)
    kwargs = {"page_chunks": True}
    if pages is not None:
        kwargs["pages"] = pages  # 0-based page indexes
    return [c["text"] for c in pymupdf4llm.to_markdown(str(pdf_path), **kwargs)]


def extract_pdf(pdf_path: Path) -> list[dict]:
    """Classic mode first; re-extract nearly empty pages in layout mode."""
    texts = _markdown_pages(pdf_path, None, layout=False)
    weak = [i for i, t in enumerate(texts) if len(t.strip()) < MIN_CHARS]
    methods = ["classic"] * len(texts)
    if weak:
        for i, t in zip(weak, _markdown_pages(pdf_path, weak, layout=True)):
            texts[i] = t
            methods[i] = "layout"
    return [
        {"page": i + 1, "text": clean_text(t), "method": m}
        for i, (t, m) in enumerate(zip(texts, methods))
    ]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
        pages = extract_pdf(pdf_path)
        out_file = OUT_DIR / f"{pdf_path.stem}.json"
        out_file.write_text(json.dumps(pages, indent=2, ensure_ascii=False), encoding="utf-8")
        fallback = sum(1 for p in pages if p["method"] == "layout")
        empty = sum(1 for p in pages if len(p["text"].strip()) < MIN_CHARS)
        print(f"{pdf_path.name}: {len(pages)} pages, {fallback} via layout fallback, {empty} still empty")


if __name__ == "__main__":
    main()