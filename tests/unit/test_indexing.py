"""Tests for SearchableDocument — pre-built indexed search over ContentDocument.

Verifies that SearchableDocument caches the index and that search results
carry char_start/char_end at sentence level and AST addresses at both levels.
"""

from __future__ import annotations

import importlib.util

import pytest

from kaos_content.model.attr import Provenance, SourceRef
from kaos_content.model.blocks import Heading, Paragraph
from kaos_content.model.document import ContentDocument
from kaos_content.model.inlines import Text
from kaos_content.model.metadata import DocumentMetadata
from kaos_content.search import SearchResults

_has_nlp = importlib.util.find_spec("kaos_nlp_core") is not None

_source = SourceRef(uri="test.pdf")


def _prov(page: int) -> Provenance:
    return Provenance(source=_source, page=page)


def _para(text: str, page: int) -> Paragraph:
    return Paragraph(children=(Text(value=text),), provenance=_prov(page))


def _heading(text: str, depth: int, page: int) -> Heading:
    return Heading(children=(Text(value=text),), depth=depth, provenance=_prov(page))


@pytest.fixture()
def multi_section_doc() -> ContentDocument:
    """A document with two sections across two pages."""
    return ContentDocument(
        metadata=DocumentMetadata(title="Test Document"),
        body=(
            _heading("Introduction", 1, 1),
            _para(
                "This contract governs the sale of commercial real estate "
                "between the buyer and seller parties.",
                1,
            ),
            _para(
                "The closing date shall be no later than thirty days after "
                "the execution of this agreement.",
                1,
            ),
            _heading("Remedies", 1, 2),
            _para(
                "In the event of a material breach, the non-breaching party "
                "may pursue all available legal remedies including damages.",
                2,
            ),
            _para(
                "Specific performance may be granted by the court when "
                "monetary damages are inadequate to compensate the injured party.",
                2,
            ),
        ),
    )


@pytest.mark.skipif(not _has_nlp, reason="kaos-nlp-core not installed")
class TestSearchableDocumentParagraph:
    """Paragraph-level SearchableDocument tests."""

    def test_basic_search(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="paragraph")
        results = sdoc.search("breach remedies")
        assert isinstance(results, SearchResults)
        assert results.total_matches > 0
        assert "breach" in results.results[0].text.lower()

    def test_block_ref_present(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="paragraph")
        results = sdoc.search("contract sale")
        for r in results.results:
            assert r.block_ref.startswith("#/body/")

    def test_page_present(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="paragraph")
        results = sdoc.search("specific performance")
        perf = [r for r in results.results if "specific performance" in r.text.lower()]
        assert perf
        assert perf[0].page == 2

    def test_section_ref_and_title(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="paragraph")
        results = sdoc.search("breach damages")
        breach = [r for r in results.results if "breach" in r.text.lower()]
        assert breach
        assert breach[0].section_ref is not None
        assert breach[0].section_title == "Remedies"

    def test_no_char_offsets_at_paragraph_level(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="paragraph")
        results = sdoc.search("contract")
        for r in results.results:
            assert r.char_start is None
            assert r.char_end is None

    def test_reuse_index(self, multi_section_doc: ContentDocument) -> None:
        """Multiple queries should use the same pre-built index."""
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="paragraph")
        r1 = sdoc.search("contract")
        r2 = sdoc.search("breach")
        r3 = sdoc.search("court")
        assert r1.total_matches > 0
        assert r2.total_matches > 0
        assert r3.total_matches > 0

    def test_empty_query_raises(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="paragraph")
        with pytest.raises(ValueError, match="empty"):
            sdoc.search("")

    def test_top_k_limits(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="paragraph")
        results = sdoc.search("the", top_k=2)
        assert len(results.results) <= 2

    def test_preview_length(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="paragraph")
        results = sdoc.search("contract", preview_length=30)
        for r in results.results:
            assert len(r.text) <= 34  # 30 + len("...")

    def test_properties(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="paragraph")
        assert sdoc.document is multi_section_doc
        assert sdoc.level == "paragraph"
        assert sdoc.view is not None
        assert len(sdoc.units) > 0


@pytest.mark.skipif(not _has_nlp, reason="kaos-nlp-core not installed")
class TestSearchableDocumentSentence:
    """Sentence-level SearchableDocument tests — verifies char offsets."""

    def test_basic_sentence_search(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="sentence")
        results = sdoc.search("closing date")
        assert results.total_matches > 0
        assert "closing" in results.results[0].text.lower()

    def test_char_offsets_populated(self, multi_section_doc: ContentDocument) -> None:
        """Sentence-level results must carry char_start and char_end."""
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="sentence")
        results = sdoc.search("breach")
        breach = [r for r in results.results if "breach" in r.text.lower()]
        assert breach
        r = breach[0]
        assert r.char_start is not None
        assert r.char_end is not None
        assert r.char_start >= 0
        assert r.char_end > r.char_start

    def test_char_offsets_correct(self, multi_section_doc: ContentDocument) -> None:
        """char_start/char_end should slice the paragraph text to the sentence."""
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="sentence")
        results = sdoc.search("breach")

        # Find the paragraph that contains the breach sentence
        for r in results.results:
            if r.char_start is None:
                continue
            # Look up the paragraph via the view
            para = next(
                (p for p in sdoc.view.paragraphs if p.block_ref == r.block_ref),
                None,
            )
            if para is None:
                continue
            # The char offsets should slice the paragraph text to produce
            # text that matches (or is contained in) the result text
            sliced = para.text[r.char_start : r.char_end]
            # The result text may be truncated by preview_length, so check
            # that the slice starts the same way
            assert sliced.startswith(r.text[: min(len(r.text), 20)])

    def test_sentence_block_ref(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="sentence")
        results = sdoc.search("damages")
        for r in results.results:
            assert r.block_ref.startswith("#/body/")

    def test_sentence_page(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="sentence")
        results = sdoc.search("court monetary")
        court = [r for r in results.results if "court" in r.text.lower()]
        assert court
        assert court[0].page == 2

    def test_sentence_section_title(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="sentence")
        results = sdoc.search("breach")
        breach = [r for r in results.results if "breach" in r.text.lower()]
        assert breach
        assert breach[0].section_title == "Remedies"

    async def test_search_corpus_threads_sentence_passage_uri(
        self, multi_section_doc: ContentDocument
    ) -> None:
        from kaos_content.indexing import SearchableDocument
        from kaos_content.model.metadata import DocumentMetadata
        from kaos_content.search import search_corpus

        doc = multi_section_doc.model_copy(
            update={"metadata": DocumentMetadata(title="Test Document", source=_source)}
        )
        sdoc = SearchableDocument(doc, level="sentence")

        results = await search_corpus([sdoc], "breach", top_k=2)

        assert results
        assert results[0].metadata["passage_uri"].startswith("test.pdf#c")


@pytest.mark.skipif(not _has_nlp, reason="kaos-nlp-core not installed")
class TestSearchDocumentCharOffsets:
    """Test that search_document also threads char offsets at sentence level."""

    def test_sentence_char_offsets_via_search_document(
        self, multi_section_doc: ContentDocument
    ) -> None:
        from kaos_content.search import search_document

        results = search_document(multi_section_doc, "breach", level="sentence")
        breach = [r for r in results.results if "breach" in r.text.lower()]
        assert breach
        r = breach[0]
        assert r.char_start is not None
        assert r.char_end is not None
        assert r.char_end > r.char_start

    def test_paragraph_no_char_offsets_via_search_document(
        self, multi_section_doc: ContentDocument
    ) -> None:
        from kaos_content.search import search_document

        results = search_document(multi_section_doc, "contract", level="paragraph")
        for r in results.results:
            assert r.char_start is None
            assert r.char_end is None


@pytest.mark.skipif(not _has_nlp, reason="kaos-nlp-core not installed")
class TestSearchableDocumentEmpty:
    """Edge cases: empty documents."""

    def test_empty_document(self) -> None:
        from kaos_content.indexing import SearchableDocument

        doc = ContentDocument(metadata=DocumentMetadata(), body=())
        sdoc = SearchableDocument(doc, level="paragraph")
        results = sdoc.search("anything")
        assert results.total_matches == 0
        assert results.results == []

    def test_empty_document_sentence(self) -> None:
        from kaos_content.indexing import SearchableDocument

        doc = ContentDocument(metadata=DocumentMetadata(), body=())
        sdoc = SearchableDocument(doc, level="sentence")
        results = sdoc.search("anything")
        assert results.total_matches == 0
        assert results.results == []
