"""Tests for search_document — BM25 and TF search on ContentDocument.

Verifies that AST addresses (block_ref, page, section_ref) survive the
round-trip through kaos-nlp-core's DocumentCollection/Searcher pipeline.
"""

from __future__ import annotations

import importlib.util

import pytest

from kaos_content.model.attr import Provenance, SourceRef
from kaos_content.model.blocks import Heading, Paragraph
from kaos_content.model.document import ContentDocument
from kaos_content.model.inlines import Text
from kaos_content.model.metadata import DocumentMetadata
from kaos_content.search import SearchResults, search_document

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


@pytest.fixture()
def nested_section_doc() -> ContentDocument:
    """A document with h1 → h2 → h3 nesting."""
    return ContentDocument(
        metadata=DocumentMetadata(title="Nested Doc"),
        body=(
            _heading("Chapter 1", 1, 1),
            _para("Chapter 1 intro paragraph.", 1),
            _heading("Section 1.1", 2, 1),
            _para("Section 1.1 intro paragraph.", 1),
            _heading("Subsection 1.1.1", 3, 1),
            _para(
                "Subsection 1.1.1 contains the unique sentinel "
                "alpaca-cardamom-zircon for searches.",
                1,
            ),
            _heading("Chapter 2", 1, 2),
            _para("Chapter 2 standalone paragraph.", 2),
        ),
    )


class TestSearchDocumentTF:
    """TF fallback search (works without kaos-nlp-core)."""

    def test_basic_query(self, multi_section_doc: ContentDocument) -> None:
        # Force TF path by searching paragraph level (works without nlp)
        # We test TF by monkeypatching out kaos_nlp_core later, but for now
        # just test the public API
        results = search_document(multi_section_doc, "breach remedies")
        assert isinstance(results, SearchResults)
        assert results.total_matches > 0
        assert results.query == "breach remedies"

    def test_block_ref_present(self, multi_section_doc: ContentDocument) -> None:
        results = search_document(multi_section_doc, "contract")
        for r in results.results:
            assert r.block_ref.startswith("#/body/")

    def test_page_present(self, multi_section_doc: ContentDocument) -> None:
        results = search_document(multi_section_doc, "breach")
        # "breach" is on page 2
        breach_results = [r for r in results.results if "breach" in r.text.lower()]
        assert breach_results
        assert breach_results[0].page == 2

    def test_section_ref_present(self, multi_section_doc: ContentDocument) -> None:
        results = search_document(multi_section_doc, "breach")
        breach_results = [r for r in results.results if "breach" in r.text.lower()]
        assert breach_results
        # Should be under "Remedies" section
        assert breach_results[0].section_ref is not None
        assert breach_results[0].section_title == "Remedies"

    def test_heading_path_empty_for_top_level_section(
        self, multi_section_doc: ContentDocument
    ) -> None:
        # Both sections are h1 — no ancestors. heading_path must be ().
        results = search_document(multi_section_doc, "breach")
        breach_results = [r for r in results.results if "breach" in r.text.lower()]
        assert breach_results
        assert breach_results[0].heading_path == ()

    def test_heading_path_nested(self, nested_section_doc: ContentDocument) -> None:
        # Subsection 1.1.1 → ancestors are ["Chapter 1", "Section 1.1"].
        # section_title remains "Subsection 1.1.1".
        results = search_document(nested_section_doc, "alpaca-cardamom-zircon")
        assert results.results
        hit = results.results[0]
        assert hit.section_title == "Subsection 1.1.1"
        assert hit.heading_path == ("Chapter 1", "Section 1.1")

    def test_empty_query_raises(self, multi_section_doc: ContentDocument) -> None:
        with pytest.raises(ValueError, match="empty"):
            search_document(multi_section_doc, "")

    def test_top_k_limits(self, multi_section_doc: ContentDocument) -> None:
        results = search_document(multi_section_doc, "the", top_k=2)
        assert len(results.results) <= 2

    def test_preview_length(self, multi_section_doc: ContentDocument) -> None:
        results = search_document(multi_section_doc, "contract", preview_length=30)
        for r in results.results:
            # Text should be truncated + "..."
            assert len(r.text) <= 34  # 30 + len("...")


@pytest.mark.skipif(not _has_nlp, reason="kaos-nlp-core not installed")
class TestSearchDocumentBM25:
    """BM25 search via kaos-nlp-core with AST address preservation."""

    def test_paragraph_search(self, multi_section_doc: ContentDocument) -> None:
        results = search_document(multi_section_doc, "breach remedies", level="paragraph")
        assert results.total_matches > 0
        # The paragraph about breach should rank highest
        assert "breach" in results.results[0].text.lower()

    def test_block_ref_roundtrip(self, multi_section_doc: ContentDocument) -> None:
        """block_ref must survive the DocumentCollection round-trip."""
        results = search_document(multi_section_doc, "contract sale", level="paragraph")
        assert results.total_matches > 0
        for r in results.results:
            assert r.block_ref.startswith("#/body/"), f"bad block_ref: {r.block_ref}"

    def test_page_roundtrip(self, multi_section_doc: ContentDocument) -> None:
        """Page numbers must survive the DocumentCollection round-trip."""
        results = search_document(multi_section_doc, "specific performance", level="paragraph")
        perf_results = [r for r in results.results if "specific performance" in r.text.lower()]
        assert perf_results
        assert perf_results[0].page == 2

    def test_section_ref_roundtrip(self, multi_section_doc: ContentDocument) -> None:
        """Section refs and titles must survive the round-trip."""
        results = search_document(multi_section_doc, "breach damages", level="paragraph")
        breach_results = [r for r in results.results if "breach" in r.text.lower()]
        assert breach_results
        assert breach_results[0].section_ref is not None
        assert breach_results[0].section_title == "Remedies"

    def test_heading_path_bm25_nested(self, nested_section_doc: ContentDocument) -> None:
        """T4c: heading_path breadcrumb survives the BM25 round-trip."""
        results = search_document(nested_section_doc, "alpaca-cardamom-zircon", level="paragraph")
        assert results.results
        hit = results.results[0]
        assert hit.section_title == "Subsection 1.1.1"
        assert hit.heading_path == ("Chapter 1", "Section 1.1")

    def test_heading_path_sentence_level(self, nested_section_doc: ContentDocument) -> None:
        """heading_path also propagates through sentence-level BM25."""
        results = search_document(nested_section_doc, "alpaca-cardamom-zircon", level="sentence")
        assert results.results
        hit = results.results[0]
        assert hit.heading_path == ("Chapter 1", "Section 1.1")

    def test_sentence_search(self, multi_section_doc: ContentDocument) -> None:
        results = search_document(multi_section_doc, "closing date", level="sentence")
        assert results.total_matches > 0
        # Sentence-level should return individual sentences, not full paragraphs
        top = results.results[0]
        assert "closing" in top.text.lower()

    def test_sentence_block_ref(self, multi_section_doc: ContentDocument) -> None:
        """Sentence results carry the containing paragraph's block_ref."""
        results = search_document(multi_section_doc, "damages", level="sentence")
        for r in results.results:
            assert r.block_ref.startswith("#/body/"), f"bad block_ref: {r.block_ref}"

    def test_sentence_page(self, multi_section_doc: ContentDocument) -> None:
        results = search_document(multi_section_doc, "court monetary", level="sentence")
        court_results = [r for r in results.results if "court" in r.text.lower()]
        assert court_results
        assert court_results[0].page == 2

    def test_corpus_wide_idf(self, multi_section_doc: ContentDocument) -> None:
        """BM25 should use corpus-wide IDF, not per-paragraph IDF.

        "the" appears in all paragraphs → low IDF → low score.
        "breach" appears in one paragraph → high IDF → high score.
        Searching for "breach the" should rank the breach paragraph
        above paragraphs that only contain "the".
        """
        results = search_document(multi_section_doc, "breach the", level="paragraph")
        assert results.total_matches > 0
        # Breach paragraph should be first (high IDF for "breach")
        assert "breach" in results.results[0].text.lower()

    def test_has_more_flag(self, multi_section_doc: ContentDocument) -> None:
        results = search_document(multi_section_doc, "the", top_k=1, level="paragraph")
        if results.total_matches > 1:
            assert results.has_more

    def test_preview_length_bm25(self, multi_section_doc: ContentDocument) -> None:
        results = search_document(
            multi_section_doc, "contract", level="paragraph", preview_length=20
        )
        for r in results.results:
            assert len(r.text) <= 24  # 20 + len("...")
