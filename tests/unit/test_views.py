"""Tests for DocumentView: dynamic hierarchical document views."""

from __future__ import annotations

import pytest

from kaos_content import (
    BlockQuote,
    ContentDocument,
    DocumentBuilder,
    DocumentMetadata,
    Heading,
    Paragraph,
    Text,
)
from kaos_content.model.attr import Provenance
from kaos_content.views import DocumentView

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _p(text: str, *, page: int | None = None) -> Paragraph:
    """Build a paragraph with optional page provenance."""
    prov = None
    if page is not None:
        prov = Provenance(page=page, extractor="test")
    return Paragraph(children=(Text(value=text),), provenance=prov)


def _h(depth: int, text: str, *, page: int | None = None) -> Heading:
    """Build a heading with optional page provenance."""
    prov = None
    if page is not None:
        prov = Provenance(page=page, extractor="test")
    return Heading(depth=depth, children=(Text(value=text),), provenance=prov)


def _doc(*blocks, title: str | None = None) -> ContentDocument:
    return ContentDocument(
        metadata=DocumentMetadata(title=title),
        body=blocks,
    )


class MockSegmenter:
    """Mock sentence segmenter matching PunktTokenizer API."""

    def tokenize_spans(self, text: str) -> list[tuple[int, int]]:
        """Split on '. ' boundaries."""
        spans = []
        start = 0
        while start < len(text):
            end = text.find(". ", start)
            if end == -1:
                spans.append((start, len(text)))
                break
            spans.append((start, end + 1))  # include the period
            start = end + 2
        return spans


# ---------------------------------------------------------------------------
# Page views
# ---------------------------------------------------------------------------


class TestPageViews:
    def test_has_pages_with_provenance(self) -> None:
        doc = _doc(_p("Hello", page=1), _p("World", page=2))
        view = DocumentView(doc)
        assert view.has_pages
        assert view.page_count == 2

    def test_has_pages_without_provenance(self) -> None:
        doc = _doc(_p("Hello"), _p("World"))
        view = DocumentView(doc)
        assert not view.has_pages

    def test_page_grouping(self) -> None:
        doc = _doc(
            _p("Para 1 on page 1", page=1),
            _p("Para 2 on page 1", page=1),
            _p("Para 3 on page 2", page=2),
        )
        view = DocumentView(doc)
        assert view.page_count == 2

        p1 = view.page(1)
        assert len(p1.blocks) == 2
        assert p1.page_number == 1

        p2 = view.page(2)
        assert len(p2.blocks) == 1

    def test_page_block_refs(self) -> None:
        doc = _doc(_p("A", page=1), _p("B", page=1), _p("C", page=2))
        view = DocumentView(doc)
        p1 = view.page(1)
        assert p1.block_refs == ("#/body/0", "#/body/1")

    def test_page_not_found_raises(self) -> None:
        doc = _doc(_p("A", page=1))
        view = DocumentView(doc)
        with pytest.raises(KeyError, match="Page 99"):
            view.page(99)

    def test_blocks_without_page_assigned_to_last(self) -> None:
        """Blocks without provenance.page inherit the last seen page."""
        doc = _doc(_p("On page 1", page=1), _p("No page"), _p("On page 2", page=2))
        view = DocumentView(doc)
        p1 = view.page(1)
        assert len(p1.blocks) == 2  # First block + orphan

    def test_page_as_markdown(self) -> None:
        doc = _doc(_p("Hello world", page=1), _p("Goodbye", page=2))
        view = DocumentView(doc)
        md = view.page_as_markdown(1)
        assert "Hello world" in md
        assert "Goodbye" not in md

    def test_empty_document_no_pages(self) -> None:
        doc = _doc()
        view = DocumentView(doc)
        assert not view.has_pages
        assert view.page_count == 0


# ---------------------------------------------------------------------------
# Section views
# ---------------------------------------------------------------------------


class TestSectionViews:
    def test_has_sections_with_headings(self) -> None:
        doc = _doc(_h(1, "Title"), _p("Content"))
        view = DocumentView(doc)
        assert view.has_sections
        assert len(view.sections) == 1

    def test_has_sections_without_headings(self) -> None:
        doc = _doc(_p("Just paragraphs"), _p("More text"))
        view = DocumentView(doc)
        assert not view.has_sections
        # Still creates one "everything" section
        assert len(view.sections) == 1
        assert view.sections[0].depth == 0
        assert view.sections[0].heading_ref is None

    def test_preamble_before_first_heading(self) -> None:
        doc = _doc(_p("Preamble"), _h(1, "Section 1"), _p("Content"))
        view = DocumentView(doc)
        assert len(view.sections) == 2

        preamble = view.sections[0]
        assert preamble.depth == 0
        assert preamble.heading_ref is None
        assert len(preamble.blocks) == 1

        section = view.sections[1]
        assert section.depth == 1
        assert section.heading_text == "Section 1"

    def test_nested_sections(self) -> None:
        doc = _doc(
            _h(1, "Chapter 1"),
            _p("Intro"),
            _h(2, "Section 1.1"),
            _p("Detail"),
            _h(2, "Section 1.2"),
            _p("More detail"),
            _h(1, "Chapter 2"),
            _p("Another chapter"),
        )
        view = DocumentView(doc)

        # Two top-level sections (Chapter 1, Chapter 2)
        assert len(view.sections) == 2

        ch1 = view.sections[0]
        assert ch1.heading_text == "Chapter 1"
        assert ch1.depth == 1
        assert len(ch1.subsections) == 2
        assert ch1.subsections[0].heading_text == "Section 1.1"
        assert ch1.subsections[1].heading_text == "Section 1.2"

        ch2 = view.sections[1]
        assert ch2.heading_text == "Chapter 2"
        assert len(ch2.subsections) == 0

    def test_flat_sections(self) -> None:
        doc = _doc(
            _h(1, "A"),
            _h(2, "A.1"),
            _h(3, "A.1.1"),
            _h(1, "B"),
        )
        view = DocumentView(doc)
        flat = view.flat_sections
        texts = [s.heading_text for s in flat]
        assert texts == ["A", "A.1", "A.1.1", "B"]

    def test_section_by_ref(self) -> None:
        doc = _doc(_h(1, "Title"), _p("Content"))
        view = DocumentView(doc)
        sv = view.section_by_ref("#/body/0")
        assert sv is not None
        assert sv.heading_text == "Title"

    def test_section_by_ref_not_found(self) -> None:
        doc = _doc(_h(1, "Title"))
        view = DocumentView(doc)
        assert view.section_by_ref("#/body/99") is None

    def test_section_as_markdown(self) -> None:
        doc = _doc(
            _h(1, "Introduction"),
            _p("Hello world"),
            _h(1, "Conclusion"),
            _p("Goodbye"),
        )
        view = DocumentView(doc)
        md = view.section_as_markdown("#/body/0")
        assert "Introduction" in md
        assert "Hello world" in md
        assert "Conclusion" not in md

    def test_section_as_markdown_with_subsections(self) -> None:
        doc = _doc(
            _h(1, "Chapter"),
            _p("Intro"),
            _h(2, "Sub"),
            _p("Detail"),
        )
        view = DocumentView(doc)
        md = view.section_as_markdown("#/body/0")
        # Should include subsection content
        assert "Chapter" in md
        assert "Sub" in md
        assert "Detail" in md

    def test_section_page_range(self) -> None:
        doc = _doc(
            _h(1, "Section", page=3),
            _p("A", page=3),
            _p("B", page=4),
            _p("C", page=5),
        )
        view = DocumentView(doc)
        sv = view.sections[0]
        assert sv.page_range == (3, 5)

    def test_section_page_range_none_without_provenance(self) -> None:
        doc = _doc(_h(1, "Section"), _p("Content"))
        view = DocumentView(doc)
        assert view.sections[0].page_range is None


# ---------------------------------------------------------------------------
# block_path (structural breadcrumbs)
# ---------------------------------------------------------------------------


class TestBlockPath:
    """``DocumentView.block_path`` exposes the chain of enclosing heading
    texts for any block. Empty tuple means "no enclosing heading" (the
    contract that downstream agents must not invent section identifiers).
    """

    def test_single_top_level_section(self) -> None:
        # Heading text mirrors the NDA case: agent must cite "11. GOVERNING LAW"
        # verbatim — including the leading numbering token — not just "GOVERNING LAW".
        doc = _doc(_h(1, "11. GOVERNING LAW"), _p("Delaware law applies."))
        view = DocumentView(doc)
        assert view.block_path("#/body/1") == ("11. GOVERNING LAW",)

    def test_heading_itself_returns_own_section_path(self) -> None:
        doc = _doc(_h(1, "Section A"), _p("text"))
        view = DocumentView(doc)
        # The heading ref belongs to its own section.
        assert view.block_path("#/body/0") == ("Section A",)

    def test_nested_sections(self) -> None:
        doc = _doc(
            _h(1, "Chapter 1"),
            _h(2, "Section 1.1"),
            _h(3, "Subsection 1.1.1"),
            _p("Body here."),
        )
        view = DocumentView(doc)
        assert view.block_path("#/body/3") == (
            "Chapter 1",
            "Section 1.1",
            "Subsection 1.1.1",
        )

    def test_descendant_ref_uses_containing_top_level_block(self) -> None:
        doc = _doc(
            _h(1, "Section A"),
            BlockQuote(children=(_p("Nested quoted text."),)),
        )
        view = DocumentView(doc)
        assert view.block_path("#/body/1/children/0") == ("Section A",)

    def test_preamble_returns_empty(self) -> None:
        doc = _doc(_p("Preamble paragraph"), _h(1, "Body"), _p("real content"))
        view = DocumentView(doc)
        assert view.block_path("#/body/0") == ()

    def test_no_headings_returns_empty(self) -> None:
        doc = _doc(_p("only paragraph"))
        view = DocumentView(doc)
        assert view.block_path("#/body/0") == ()

    def test_unknown_ref_returns_empty(self) -> None:
        doc = _doc(_h(1, "Title"), _p("Content"))
        view = DocumentView(doc)
        # No KeyError — empty tuple is the explicit contract.
        assert view.block_path("#/body/99") == ()
        assert view.block_path("#/notabody/0") == ()

    def test_idempotent_across_calls(self) -> None:
        doc = _doc(_h(1, "A"), _h(2, "A.1"), _p("text"))
        view = DocumentView(doc)
        first = view.block_path("#/body/2")
        second = view.block_path("#/body/2")
        # Two calls produce identical, equal tuples — the cache is sound.
        assert first == second == ("A", "A.1")


# ---------------------------------------------------------------------------
# Paragraph views
# ---------------------------------------------------------------------------


class TestParagraphViews:
    def test_basic_paragraphs(self) -> None:
        doc = _doc(_p("First"), _p("Second"), _p("Third"))
        view = DocumentView(doc)
        assert len(view.paragraphs) == 3
        assert view.paragraphs[0].text == "First"
        assert view.paragraphs[2].text == "Third"

    def test_paragraph_refs(self) -> None:
        doc = _doc(_p("A"), _h(1, "H"), _p("B"))
        view = DocumentView(doc)
        assert len(view.paragraphs) == 2
        assert view.paragraphs[0].block_ref == "#/body/0"
        assert view.paragraphs[1].block_ref == "#/body/2"

    def test_paragraph_page_context(self) -> None:
        doc = _doc(_p("A", page=3), _p("B", page=4))
        view = DocumentView(doc)
        assert view.paragraphs[0].page == 3
        assert view.paragraphs[1].page == 4

    def test_paragraph_section_context(self) -> None:
        doc = _doc(_h(1, "Intro"), _p("Content"), _h(1, "End"), _p("Bye"))
        view = DocumentView(doc)
        assert view.paragraphs[0].section_ref == "#/body/0"
        assert view.paragraphs[1].section_ref == "#/body/2"

    def test_headings_not_in_paragraphs(self) -> None:
        doc = _doc(_h(1, "Title"), _p("Text"))
        view = DocumentView(doc)
        assert len(view.paragraphs) == 1

    def test_empty_doc_no_paragraphs(self) -> None:
        doc = _doc()
        view = DocumentView(doc)
        assert len(view.paragraphs) == 0


# ---------------------------------------------------------------------------
# Sentence views
# ---------------------------------------------------------------------------


class TestSentenceViews:
    def test_has_sentences_without_segmenter(self) -> None:
        doc = _doc(_p("Hello. World."))
        view = DocumentView(doc)
        assert not view.has_sentences
        assert view.sentences == ()

    def test_has_sentences_with_segmenter(self) -> None:
        doc = _doc(_p("Hello. World."))
        view = DocumentView(doc, sentence_segmenter=MockSegmenter())
        assert view.has_sentences
        assert len(view.sentences) >= 1

    def test_sentence_text(self) -> None:
        doc = _doc(_p("First sentence. Second sentence."))
        view = DocumentView(doc, sentence_segmenter=MockSegmenter())
        texts = [s.text for s in view.sentences]
        assert "First sentence." in texts
        assert "Second sentence." in texts

    def test_sentence_offsets(self) -> None:
        doc = _doc(_p("Hello. World."))
        view = DocumentView(doc, sentence_segmenter=MockSegmenter())
        for sent in view.sentences:
            # Round-trip: offset slicing produces the sentence text
            para = next(p for p in view.paragraphs if p.block_ref == sent.paragraph_ref)
            assert para.text[sent.start : sent.end] == sent.text

    def test_sentence_inherits_page_and_section(self) -> None:
        doc = _doc(_h(1, "S1", page=2), _p("Hello. World.", page=2))
        view = DocumentView(doc, sentence_segmenter=MockSegmenter())
        for sent in view.sentences:
            assert sent.page == 2
            assert sent.section_ref == "#/body/0"

    def test_sentences_for_paragraph(self) -> None:
        doc = _doc(_p("A. B."), _p("C. D."))
        view = DocumentView(doc, sentence_segmenter=MockSegmenter())
        sents_0 = view.sentences_for_paragraph("#/body/0")
        sents_1 = view.sentences_for_paragraph("#/body/1")
        assert len(sents_0) >= 1
        assert len(sents_1) >= 1
        assert all(s.paragraph_ref == "#/body/0" for s in sents_0)

    def test_empty_paragraph_no_sentences(self) -> None:
        doc = _doc(_p(""))
        view = DocumentView(doc, sentence_segmenter=MockSegmenter())
        assert len(view.sentences) == 0


# ---------------------------------------------------------------------------
# Cross-view consistency
# ---------------------------------------------------------------------------


class TestCrossViewConsistency:
    def test_page_section_crossref(self) -> None:
        """Section refs in page view should point to valid sections."""
        doc = _doc(
            _h(1, "S1", page=1),
            _p("A", page=1),
            _h(1, "S2", page=2),
            _p("B", page=2),
        )
        view = DocumentView(doc)
        for pv in view.pages:
            for sref in pv.section_refs:
                sv = view.section_by_ref(sref)
                assert sv is not None, f"Section ref {sref} not found"

    def test_paragraph_section_ref_valid(self) -> None:
        """Paragraph.section_ref should point to a valid section."""
        doc = _doc(_h(1, "S1"), _p("A"), _h(1, "S2"), _p("B"))
        view = DocumentView(doc)
        for pv in view.paragraphs:
            if pv.section_ref is not None:
                sv = view.section_by_ref(pv.section_ref)
                assert sv is not None

    def test_total_blocks_consistent(self) -> None:
        """Sum of blocks across pages should equal total body blocks."""
        doc = _doc(
            _p("A", page=1),
            _h(1, "H", page=1),
            _p("B", page=2),
            _p("C", page=2),
        )
        view = DocumentView(doc)
        total_page_blocks = sum(len(p.blocks) for p in view.pages)
        assert total_page_blocks == len(doc.body)


# ---------------------------------------------------------------------------
# Integration with parsed documents
# ---------------------------------------------------------------------------


class TestWithParsedDocuments:
    def test_from_parsed_markdown(self) -> None:
        from kaos_content import parse_markdown

        md = """\
# Introduction

This is the intro.

## Background

Some background.

# Methods

We used these methods.
"""
        doc = parse_markdown(md)
        view = DocumentView(doc)

        assert view.has_sections
        assert len(view.sections) >= 2  # Intro + Methods at top level
        assert len(view.flat_sections) >= 3  # Intro + Background + Methods
        assert len(view.paragraphs) >= 3

    def test_from_builder(self) -> None:
        doc = (
            DocumentBuilder(title="Test")
            .heading(1, "Part 1")
            .paragraph("Content 1")
            .heading(2, "Sub 1.1")
            .paragraph("Detail")
            .heading(1, "Part 2")
            .paragraph("Content 2")
            .build()
        )
        view = DocumentView(doc)
        assert view.has_sections
        assert len(view.sections) == 2  # Part 1, Part 2
        assert view.sections[0].subsections[0].heading_text == "Sub 1.1"

    def test_repr(self) -> None:
        doc = _doc(_h(1, "H", page=1), _p("A", page=1))
        view = DocumentView(doc)
        r = repr(view)
        assert "blocks=2" in r
        assert "pages=" in r
        assert "sections=" in r
