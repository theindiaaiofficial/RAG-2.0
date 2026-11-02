from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SUPPORTED_EXTENSIONS = {".txt", ".md", ".pdf", ".docx", ".csv", ".json"}


@dataclass(frozen=True)
class ParsedSection:
    text: str
    page: int | None = None
    title: str | None = None


def _nonempty(text: str) -> str:
    return "\n".join(line.rstrip() for line in text.replace("\x00", "").splitlines()).strip()


def parse_file(path: Path) -> list[ParsedSection]:
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext or '(no extension)'}")

    if ext in {".txt", ".md"}:
        text = _nonempty(path.read_text(encoding="utf-8-sig"))
        return [ParsedSection(text=text, title=path.stem)] if text else []

    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        sections: list[ParsedSection] = []
        for index, page in enumerate(reader.pages, start=1):
            text = _nonempty(page.extract_text() or "")
            if text:
                sections.append(ParsedSection(text=text, page=index, title=path.stem))
        return sections

    if ext == ".docx":
        from docx import Document

        doc = Document(str(path))
        lines: list[str] = []
        for paragraph in doc.paragraphs:
            text = paragraph.text.strip()
            if text:
                style = paragraph.style.name if paragraph.style else ""
                if style.lower().startswith("heading"):
                    lines.append(f"\n{ text }\n")
                else:
                    lines.append(text)
        for table in doc.tables:
            for row in table.rows:
                cells = [cell.text.strip().replace("\n", " / ") for cell in row.cells]
                if any(cells):
                    lines.append(" | ".join(cells))
        text = _nonempty("\n".join(lines))
        return [ParsedSection(text=text, title=path.stem)] if text else []

    if ext == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle))
        if not rows:
            return []
        header = rows[0]
        sections: list[ParsedSection] = []
        for row_index, row in enumerate(rows[1:], start=2):
            pairs = []
            for col_index, value in enumerate(row):
                key = header[col_index] if col_index < len(header) and header[col_index] else f"column_{col_index + 1}"
                if value.strip():
                    pairs.append(f"{key}: {value.strip()}")
            if pairs:
                sections.append(ParsedSection(text="\n".join(pairs), title=f"{path.stem} — row {row_index}"))
        return sections

    if ext == ".json":
        payload: Any = json.loads(path.read_text(encoding="utf-8-sig"))
        # Preserve readable key/value structure; avoid turning nested JSON into an opaque token.
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        return [ParsedSection(text=_nonempty(text), title=path.stem)] if text.strip() else []

    raise ValueError(f"Unsupported file type: {ext}")
