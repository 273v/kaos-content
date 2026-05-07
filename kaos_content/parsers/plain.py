"""Parse plain text into a :class:`ContentDocument`.

The simplest parser in the stack — splits on blank lines, wraps each
non-empty block as a paragraph, and attaches an optional source
reference. Exists because WS-3.7 (multi-format benchmark) needs to feed
``.txt`` fixtures through the same Corpus Protocol path as PDF/HTML/DOCX
sources, and the WS-1 grounding corpus bypassed that by handing raw
``dict[str, str]`` payloads to RAG instead of building a Corpus.

Design notes:

- **No heading detection.** Plain text has no reliable heading markers;
  callers that want structure should use markdown. Everything becomes a
  top-level paragraph.
- **Blank-line splitting only.** Single newlines within a paragraph
  are preserved as-is (markdown's "hard line break" behavior would be
  wrong here because we have no inline formatting to anchor it).
- **Optional source.** When passed, every paragraph inherits its
  provenance via the builder, so downstream ``ContentDocumentCorpus``
  and ``Corpus.from_documents`` thread ``doc_uri`` correctly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_content.builders.builder import DocumentBuilder

if TYPE_CHECKING:
    from kaos_content.model.attr import SourceRef
    from kaos_content.model.document import ContentDocument


def parse_plain_text(
    text: str,
    *,
    source: SourceRef | None = None,
) -> ContentDocument:
    """Split ``text`` on blank lines, wrap each block as a paragraph.

    Args:
        text: Plain-text input. May be UTF-8 — no normalization applied.
        source: Optional :class:`SourceRef` propagated to
            ``document.metadata.source`` so multi-document corpora can
            thread ``doc_uri`` without an explicit ``doc_uris``
            constructor argument to ``ContentDocumentCorpus``.

    Returns:
        A :class:`ContentDocument` with one :class:`Paragraph` per
        non-empty block. Empty input returns a document with an empty
        body.
    """
    from kaos_content.model.attr import SourceRef as _SourceRef

    builder = DocumentBuilder()
    if source is not None:
        # Two-pronged source wiring: ``set_source`` attaches provenance to
        # every block; ``set_metadata(source=...)`` populates
        # ``document.metadata.source`` so ``ContentDocumentCorpus`` picks up
        # the URI without an explicit ``doc_uris`` kwarg. ``extract_pdf``
        # uses the same dual-call pattern (kaos-pdf/extract.py:284-285).
        mime = source.mime_type or "text/plain"
        builder.set_source(uri=source.uri, mime_type=mime)
        builder.set_metadata(source=_SourceRef(uri=source.uri, mime_type=mime))
    for block in text.split("\n\n"):
        stripped = block.strip()
        if stripped:
            builder.paragraph(stripped)
    return builder.build()


__all__ = ["parse_plain_text"]
