import json
from pathlib import Path
from insurance_policy_analyzer.clean import clean_text
import pymupdf4llm

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")


def extract_pdf(pdf_path: Path) -> list[dict]:
    """Return one dict per page: {"page": 1-based number, "text": markdown}."""
    pages = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    # return [{"page": i, "text": p["text"]} for i, p in enumerate(pages, start=1)]
    return [{"page": i, "text": clean_text(p["text"])} for i, p in enumerate(pages, start=1)]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
        pages = extract_pdf(pdf_path)
        out_file = OUT_DIR / f"{pdf_path.stem}.json"
        out_file.write_text(json.dumps(pages, indent=2, ensure_ascii=False), encoding="utf-8")
        empty = sum(1 for p in pages if len(p["text"].strip()) < 50)
        print(f"{pdf_path.name}: {len(pages)} pages, {empty} nearly empty")


if __name__ == "__main__":
    main()