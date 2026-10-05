"""Dependency-free parsers: plain text, Markdown, HTML, JSON and CSV."""

from __future__ import annotations

import csv
import io
import json
import re
from html.parser import HTMLParser
from typing import Any

from rag_service.core.errors import ValidationFailedError
from rag_service.core.registry import PARSERS
from rag_service.domain.models import ParsedDocument, Section

_MD_TITLE = re.compile(r"^\s{0,3}#\s+(.+?)\s*#*\s*$", re.MULTILINE)


def decode_text(data: bytes) -> str:
    """Decode bytes as UTF-8 (BOM aware) or UTF-16 when a UTF-16 BOM is present."""
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        return data.decode("utf-16", errors="replace")
    return data.decode("utf-8-sig", errors="replace")


class PlainTextParser:
    def parse(self, data: bytes, *, filename: str | None = None) -> ParsedDocument:
        return ParsedDocument(sections=[Section(text=decode_text(data))], title=None)


class MarkdownParser:
    def parse(self, data: bytes, *, filename: str | None = None) -> ParsedDocument:
        text = decode_text(data)
        match = _MD_TITLE.search(text)
        return ParsedDocument(
            sections=[Section(text=text)],
            title=match.group(1).strip() if match else None,
            metadata={"format": "markdown"},
        )


class _HTMLToMarkdown(HTMLParser):
    """Extracts readable text, converting headings/lists to Markdown so structure survives."""

    SKIP = {"script", "style", "noscript", "template", "svg", "head", "iframe", "nav", "footer"}
    BLOCK = {
        "p",
        "div",
        "section",
        "article",
        "main",
        "header",
        "br",
        "tr",
        "table",
        "blockquote",
        "pre",
        "ul",
        "ol",
    }
    HEADINGS = {"h1": 1, "h2": 2, "h3": 3, "h4": 4, "h5": 5, "h6": 6}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title: str | None = None
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        if tag in self.SKIP:
            self._skip_depth += 1
            return
        if tag in self.HEADINGS:
            self.parts.append("\n\n" + "#" * self.HEADINGS[tag] + " ")
        elif tag == "li":
            self.parts.append("\n- ")
        elif tag in ("td", "th"):
            self.parts.append(" | ")
        elif tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if tag in self.SKIP:
            self._skip_depth = max(0, self._skip_depth - 1)
            return
        if tag in self.HEADINGS or tag in self.BLOCK:
            self.parts.append("\n\n")

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self.title = (self.title or "") + data.strip()
            return
        if self._skip_depth == 0:
            self.parts.append(data)


class HtmlParser:
    def parse(self, data: bytes, *, filename: str | None = None) -> ParsedDocument:
        extractor = _HTMLToMarkdown()
        extractor.feed(decode_text(data))
        extractor.close()
        text = re.sub(r"[ \t]+", " ", "".join(extractor.parts))
        text = re.sub(r"\n\s*\n\s*(\n\s*)+", "\n\n", text).strip()
        title = extractor.title
        if not title:
            match = _MD_TITLE.search(text)
            title = match.group(1).strip() if match else None
        return ParsedDocument(sections=[Section(text=text)], title=title, metadata={"format": "markdown"})


def _flatten(value: Any, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        lines: list[str] = []
        for k, v in value.items():
            lines.extend(_flatten(v, f"{prefix}{k}." if isinstance(v, dict) else f"{prefix}{k}"))
        return lines
    if isinstance(value, list):
        if all(not isinstance(v, dict | list) for v in value):
            return [f"{prefix.rstrip('.')}: {', '.join(str(v) for v in value)}"]
        return [line for v in value for line in _flatten(v, prefix)]
    return [f"{prefix.rstrip('.')}: {value}"] if prefix else [str(value)]


class JSONParser:
    """Objects become ``key: value`` lines; a top-level array becomes one section per element."""

    def parse(self, data: bytes, *, filename: str | None = None) -> ParsedDocument:
        try:
            doc = json.loads(decode_text(data))
        except json.JSONDecodeError as exc:
            raise ValidationFailedError(f"Invalid JSON document: {exc}") from exc
        items = doc if isinstance(doc, list) else [doc]
        sections = [
            Section(text="\n".join(_flatten(item)), metadata={"item": i} if len(items) > 1 else {})
            for i, item in enumerate(items)
        ]
        title = None
        if isinstance(doc, dict):
            title = next((str(doc[k]) for k in ("title", "name") if isinstance(doc.get(k), str | int)), None)
        return ParsedDocument(sections=[s for s in sections if s.text.strip()], title=title)


class CSVParser:
    """Each row becomes its own section (``column: value`` lines) so rows are retrieved independently."""

    def parse(self, data: bytes, *, filename: str | None = None) -> ParsedDocument:
        text = decode_text(data)
        try:
            dialect = csv.Sniffer().sniff(text[:4096], delimiters=",;\t|")
        except csv.Error:
            dialect = csv.excel
        reader = csv.DictReader(io.StringIO(text), dialect=dialect)
        sections = []
        for row_number, row in enumerate(reader, start=1):
            lines = [f"{k}: {v}" for k, v in row.items() if k and v not in (None, "")]
            if lines:
                sections.append(Section(text="\n".join(lines), metadata={"row": row_number}))
        return ParsedDocument(sections=sections, title=None, metadata={"columns": reader.fieldnames or []})


@PARSERS.register("text/plain", aliases=("text/x-rst", "text/x-log"), description="UTF-8 text")
def _text() -> PlainTextParser:
    return PlainTextParser()


@PARSERS.register("text/markdown", aliases=("text/x-markdown",), description="Markdown (title from first H1)")
def _markdown() -> MarkdownParser:
    return MarkdownParser()


@PARSERS.register("text/html", aliases=("application/xhtml+xml",), description="HTML → Markdown-structured text")
def _html() -> HtmlParser:
    return HtmlParser()


@PARSERS.register("application/json", description="JSON objects / arrays flattened to key: value lines")
def _json() -> JSONParser:
    return JSONParser()


@PARSERS.register("text/csv", description="CSV, one section per row")
def _csv() -> CSVParser:
    return CSVParser()
