import json
import re
from pathlib import Path

IN_DIR = Path("data/processed")
OUT_DIR = Path("data/chunks")
MAX_CHARS = 1500
MIN_CHARS = 40


def clean_heading(line: str) -> str:
    line = re.sub(r"</?mark>", "", line)
    line = line.replace("*", "").lstrip("#")
    return re.sub(r"\s+", " ", line).strip()

NUMBERED = re.compile(r"^(?:\d+(?:\.\d+)+\.?|\d+\.|[IVX]+\.)\s+\S")
MAX_HEADING_CHARS = 80


def heading_text(line: str) -> str | None:
    """Return the cleaned title if this line is a section heading, else None."""
    line = line.strip()
    if line.startswith("#"):
        return clean_heading(line)
    plain = clean_heading(line)
    if NUMBERED.match(plain) and len(plain) <= MAX_HEADING_CHARS:
        return plain
    return None

def split_sections(pages: list[dict]) -> list[dict]:
    """Group paragraph blocks under the heading they appear after."""
    sections = []
    current = {"title": "Introduction", "blocks": []}
    for page in pages:
        for block in re.split(r"\n\s*\n", page["text"]):
            block = block.strip()
            if not block:
                continue
            lines = block.splitlines()
            title = heading_text(lines[0])
            if title is not None:
                sections.append(current)
                current = {"title": title, "blocks": []}
                block = "\n".join(lines[1:]).strip()
                if not block:
                    continue
            current["blocks"].append((page["page"], block))
    sections.append(current)
    return sections


def pack(blocks: list[tuple[int, str]], max_chars: int) -> list[list[tuple[int, str]]]:
    """Greedily group paragraph blocks into groups of up to max_chars."""
    groups, current, size = [], [], 0
    for page, text in blocks:
        if current and size + len(text) > max_chars:
            groups.append(current)
            current, size = [], 0
        current.append((page, text))
        size += len(text)
    if current:
        groups.append(current)
    return groups

def split_long(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    """Split an oversized chunk at line / <br> boundaries, repeating a table header."""
    if len(text) <= max_chars:
        return [text]
    lines = text.split("\n")
    header = ""
    if text.startswith("|") and len(lines) > 1 and lines[1].startswith("|---"):
        header = "\n".join(lines[:2]) + "\n"
        text = "\n".join(lines[2:])
    pieces = [p for p in re.split(r"(?<=<br>)|\n", text) if p.strip()]
    parts, current = [], ""
    for piece in pieces:
        if current and len(header) + len(current) + len(piece) > max_chars:
            parts.append(header + current)
            current = ""
        current += piece + "\n"
    if current.strip():
        parts.append(header + current)
    return [p.strip() for p in parts]

def chunk_document(policy: str, pages: list[dict]) -> list[dict]:
    chunks = []
    for section in split_sections(pages):
        for group in pack(section["blocks"], MAX_CHARS):
            text = "\n\n".join(t for _, t in group)
            text = re.sub(r"</?mark>", "", text).strip()
            if len(text) < MIN_CHARS:
                continue
            for part in split_long(text):
                chunks.append(
                    {
                        "chunk_id": f"{policy}-{len(chunks):04d}",
                        "policy": policy,
                        "section": section["title"],
                        "page_start": group[0][0],
                        "page_end": group[-1][0],
                        "text": part,
                    }
                )
    return chunks


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(IN_DIR.glob("*.json")):
        pages = json.loads(path.read_text(encoding="utf-8"))
        chunks = chunk_document(path.stem, pages)
        (OUT_DIR / path.name).write_text(
            json.dumps(chunks, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        sizes = [len(c["text"]) for c in chunks]
        print(
            f"{path.stem}: {len(chunks)} chunks, "
            f"min {min(sizes)}, avg {sum(sizes) // len(sizes)}, max {max(sizes)} chars"
        )


if __name__ == "__main__":
    main()