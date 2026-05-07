"""Unit tests for :func:`kaos_content.parsers.plain.parse_plain_text`.

Exists to feed WS-3.7's multi-format benchmark — `.txt` fixtures need to
flow through the same ``ContentDocument`` / ``ContentDocumentCorpus`` /
``Corpus.from_documents`` / ``RAG.query`` pipeline as PDF/HTML/DOCX.
"""

from __future__ import annotations

import pytest

from kaos_content.corpus import ContentDocumentCorpus
from kaos_content.model.attr import SourceRef
from kaos_content.model.document import ContentDocument
from kaos_content.parsers import parse_plain_text


@pytest.mark.unit
class TestParsePlainText:
    def test_empty_input_produces_empty_document(self) -> None:
        doc = parse_plain_text("")
        assert isinstance(doc, ContentDocument)
        assert len(doc.body) == 0

    def test_single_paragraph(self) -> None:
        doc = parse_plain_text("A single paragraph of text.")
        assert len(doc.body) == 1

    def test_blank_line_splits_paragraphs(self) -> None:
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        doc = parse_plain_text(text)
        assert len(doc.body) == 3

    def test_single_newlines_preserved_within_paragraph(self) -> None:
        """A single newline is NOT a paragraph break — matches common
        plain-text semantics where wrapping is visual, not structural."""
        text = "Line one\nstill same paragraph.\n\nNew paragraph here."
        doc = parse_plain_text(text)
        assert len(doc.body) == 2

    def test_whitespace_only_blocks_dropped(self) -> None:
        text = "Real block.\n\n   \n\nAnother real block."
        doc = parse_plain_text(text)
        assert len(doc.body) == 2

    def test_source_ref_is_set_on_metadata(self) -> None:
        source = SourceRef(uri="file:///tmp/example.txt", mime_type="text/plain")
        doc = parse_plain_text("hello", source=source)
        assert doc.metadata.source is not None
        assert doc.metadata.source.uri == "file:///tmp/example.txt"
        assert doc.metadata.source.mime_type == "text/plain"

    def test_source_none_leaves_metadata_source_unset(self) -> None:
        doc = parse_plain_text("hello")
        assert doc.metadata.source is None

    def test_default_mime_type_when_source_has_none(self) -> None:
        """When the SourceRef omits mime_type, the builder defaults to text/plain."""
        source = SourceRef(uri="file:///tmp/example.txt")
        doc = parse_plain_text("hi", source=source)
        assert doc.metadata.source is not None
        assert doc.metadata.source.mime_type == "text/plain"


@pytest.mark.unit
class TestParsePlainTextInCorpusPipeline:
    """Confirms the helper composes cleanly with the Corpus Protocol —
    the exact path WS-3.7 uses."""

    def test_parse_plain_then_corpus_threads_doc_uri(self) -> None:
        source = SourceRef(uri="doc:example-plain", mime_type="text/plain")
        doc = parse_plain_text("first.\n\nsecond.\n\nthird.", source=source)
        corpus = ContentDocumentCorpus([doc])
        passages = list(corpus.iter_passages())
        assert corpus.size == 3
        assert all(p.doc_uri == "doc:example-plain" for p in passages)
        assert [p.text for p in passages] == ["first.", "second.", "third."]

    def test_multiple_plain_docs_combine_in_corpus(self) -> None:
        a = parse_plain_text(
            "alpha one.\n\nalpha two.",
            source=SourceRef(uri="doc:a", mime_type="text/plain"),
        )
        b = parse_plain_text(
            "beta one.",
            source=SourceRef(uri="doc:b", mime_type="text/plain"),
        )
        corpus = ContentDocumentCorpus([a, b])
        assert corpus.size == 3
        rows = [(p.row, p.doc_uri) for p in corpus.iter_passages()]
        assert rows == [(0, "doc:a"), (1, "doc:a"), (2, "doc:b")]
