"""Document parsers, registered by MIME type in ``PARSERS``."""

from rag_service.adapters.parsers import builtin, pdf

__all__ = ["builtin", "pdf"]
