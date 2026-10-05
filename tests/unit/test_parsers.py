from __future__ import annotations

import io
import json

import pytest

from rag_service.core.errors import ValidationFailedError
from rag_service.core.registry import PARSERS


def parse(mime: str, data: str | bytes):  # type: ignore[no-untyped-def]
    raw = data.encode() if isinstance(data, str) else data
    return PARSERS.create(mime).parse(raw)


def test_markdown_title() -> None:
    doc = parse("text/markdown", "# Delta RAD80\n\nBody")
    assert doc.title == "Delta RAD80"


def test_html_keeps_structure_and_drops_scripts() -> None:
    html = (
        "<html><head><title>Fan page</title><script>evil()</script></head><body>"
        "<h1>Fans</h1><p>Quiet &amp; efficient.</p><ul><li>80 CFM</li></ul><nav>menu</nav></body></html>"
    )
    doc = parse("text/html", html)
    assert doc.title == "Fan page"
    assert "# Fans" in doc.full_text
    assert "Quiet & efficient." in doc.full_text
    assert "- 80 CFM" in doc.full_text
    assert "evil" not in doc.full_text and "menu" not in doc.full_text


def test_json_array_becomes_sections() -> None:
    doc = parse(
        "application/json",
        json.dumps([{"name": "Fan", "specs": {"cfm": 80}}, {"name": "Blender", "tags": ["a", "b"]}]),
    )
    assert len(doc.sections) == 2
    assert "specs.cfm: 80" in doc.sections[0].text
    assert "tags: a, b" in doc.sections[1].text


def test_csv_rows_are_sections() -> None:
    doc = parse("text/csv", "title,brand,cfm\nFan A,Delta,80\nFan B,Broan,110\n")
    assert [s.metadata["row"] for s in doc.sections] == [1, 2]
    assert "brand: Delta" in doc.sections[0].text


def test_invalid_json_is_a_validation_error() -> None:
    with pytest.raises(ValidationFailedError):
        parse("application/json", "{not json")


def test_pdf_pages_carry_page_numbers() -> None:
    pypdf = pytest.importorskip("pypdf")
    writer = pypdf.PdfWriter()
    writer.add_blank_page(width=200, height=200)
    buf = io.BytesIO()
    writer.write(buf)
    doc = parse("application/pdf", buf.getvalue())
    assert doc.metadata["page_count"] == 1  # blank page → no text sections, but parsed
