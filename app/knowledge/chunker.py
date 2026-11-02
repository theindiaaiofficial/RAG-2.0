from __future__ import annotations

import re
from dataclasses import dataclass

from app.knowledge.parsers import ParsedSection


@dataclass(frozen=True)
class TextChunk:
    text: str
    index: int
    page: int | None
    title: str | None


def _split_long(text: str, size: int, overlap: int) -> list[str]:
    pieces: list[str] = []
    cursor = 0
    while cursor < len(text):
        end = min(len(text), cursor + size)
        if end < len(text):
            boundary = max(text.rfind("\n", cursor, end), text.rfind(". ", cursor, end), text.rfind(" ", cursor, end))
            if boundary > cursor + int(size * 0.65):
                end = boundary + (1 if text[boundary] == "\n" else 2 if text[boundary:boundary + 2] == ". " else 0)
        piece = text[cursor:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        cursor = max(cursor + 1, end - overlap)
    return pieces


def chunk_sections(sections: list[ParsedSection], size: int, overlap: int) -> list[TextChunk]:
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("Invalid chunk size/overlap configuration")
    output: list[TextChunk] = []
    for section in sections:
        text = section.text.strip()
        if not text:
            continue
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        assembled = ""
        units: list[str] = []
        for paragraph in paragraphs or [text]:
            if len(paragraph) > size:
                if assembled:
                    units.append(assembled)
                    assembled = ""
                units.extend(_split_long(paragraph, size, overlap))
            elif not assembled:
                assembled = paragraph
            elif len(assembled) + len(paragraph) + 2 <= size:
                assembled += "\n\n" + paragraph
            else:
                units.append(assembled)
                assembled = paragraph
        if assembled:
            units.append(assembled)
        for unit in units:
            output.append(TextChunk(text=unit, index=len(output), page=section.page, title=section.title))
    return output
