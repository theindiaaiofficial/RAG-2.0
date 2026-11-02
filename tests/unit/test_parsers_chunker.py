from __future__ import annotations

import json

import pytest
from docx import Document
from reportlab.pdfgen import canvas

from app.knowledge.chunker import chunk_sections
from app.knowledge.parsers import parse_file


def test_parse_txt_and_markdown(tmp_path):
    txt = tmp_path / "notes.txt"
    txt.write_text("Service hours are 9 to 5.", encoding="utf-8")
    md = tmp_path / "policy.md"
    md.write_text("# Refunds\n\nCustomers may request refunds.", encoding="utf-8")
    assert "9 to 5" in parse_file(txt)[0].text
    assert "# Refunds" in parse_file(md)[0].text


def test_parse_csv_rows(tmp_path):
    path = tmp_path / "services.csv"
    path.write_text("service,price\nWeb,100\nAI,250\n", encoding="utf-8")
    sections = parse_file(path)
    assert len(sections) == 2
    assert "service: Web" in sections[0].text
    assert "price: 250" in sections[1].text


def test_parse_json_preserves_keys(tmp_path):
    path = tmp_path / "facts.json"
    path.write_text(json.dumps({"company": {"name": "North Star"}}), encoding="utf-8")
    assert '"name": "North Star"' in parse_file(path)[0].text


def test_parse_docx_paragraphs_and_tables(tmp_path):
    path = tmp_path / "brief.docx"
    doc = Document()
    doc.add_heading("Services", level=1)
    doc.add_paragraph("We build websites.")
    table = doc.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "Support"
    table.rows[0].cells[1].text = "Email"
    doc.save(path)
    text = parse_file(path)[0].text
    assert "Services" in text and "We build websites" in text and "Support | Email" in text


def test_parse_pdf_captures_page_number(tmp_path):
    path = tmp_path / "one-page.pdf"
    file = path.open("wb")
    pdf = canvas.Canvas(file)
    pdf.drawString(72, 720, "North Star support hours: 9 AM to 5 PM.")
    pdf.save()
    file.close()
    sections = parse_file(path)
    assert sections and sections[0].page == 1
    assert "support hours" in sections[0].text


def test_empty_and_corrupt_files(tmp_path):
    empty = tmp_path / "empty.txt"
    empty.write_text("\n ", encoding="utf-8")
    corrupt = tmp_path / "broken.pdf"
    corrupt.write_text("not a pdf", encoding="utf-8")
    assert parse_file(empty) == []
    with pytest.raises(Exception):
        parse_file(corrupt)


def test_unsupported_extension_rejected(tmp_path):
    path = tmp_path / "facts.exe"
    path.write_text("ignored")
    with pytest.raises(ValueError):
        parse_file(path)


def test_chunking_respects_limit_and_overlap():
    from app.knowledge.parsers import ParsedSection

    text = "First paragraph has some text.\n\n" + ("Second paragraph continues with meaningful words. " * 16)
    chunks = chunk_sections([ParsedSection(text=text, page=4, title="Heading")], size=120, overlap=20)
    assert len(chunks) > 1
    assert all(len(item.text) <= 120 for item in chunks)
    assert all(item.page == 4 and item.title == "Heading" for item in chunks)
    assert [item.index for item in chunks] == list(range(len(chunks)))


def test_chunking_rejects_invalid_overlap():
    from app.knowledge.parsers import ParsedSection

    with pytest.raises(ValueError):
        chunk_sections([ParsedSection(text="test")], size=100, overlap=100)
