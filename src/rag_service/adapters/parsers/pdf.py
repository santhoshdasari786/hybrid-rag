"""PDF parser (optional extra ``pdf``). One section per page so citations carry page numbers."""

from __future__ import annotations

import io

from rag_service.core.errors import DependencyMissingError, ValidationFailedError
from rag_service.core.registry import PARSERS
from rag_service.domain.models import ParsedDocument, Section


class PDFParser:
    def __init__(self) -> None:
        try:
            import pypdf  # noqa: F401, PLC0415 - optional dependency
        except ImportError as exc:
            raise DependencyMissingError.for_extra("pypdf", "pdf") from exc

    def parse(self, data: bytes, *, filename: str | None = None) -> ParsedDocument:
        from pypdf import PdfReader  # noqa: PLC0415
        from pypdf.errors import PdfReadError  # noqa: PLC0415

        try:
            reader = PdfReader(io.BytesIO(data))
            sections = [
                Section(text=page.extract_text() or "", metadata={"page": number})
                for number, page in enumerate(reader.pages, start=1)
            ]
        except PdfReadError as exc:
            raise ValidationFailedError(f"Unreadable PDF: {exc}") from exc
        title = None
        if reader.metadata and reader.metadata.title:
            title = str(reader.metadata.title).strip() or None
        return ParsedDocument(
            sections=[s for s in sections if s.text.strip()],
            title=title,
            metadata={"page_count": len(reader.pages)},
        )


@PARSERS.register("application/pdf", description="PDF text extraction per page (pypdf)")
def _pdf() -> PDFParser:
    return PDFParser()
