"""Unit tests for kaos_content.units — paragraph and sentence enumeration.

Tests iter_paragraph_units and iter_sentence_units against synthetic
ContentDocuments, verifying the dataclass fields, edge cases, and
consistency between ContentDocument and DocumentView inputs.
"""

from __future__ import annotations

import pytest

from kaos_content import ContentDocument, DocumentMetadata, Heading, Paragraph, Text
from kaos_content.model.attr import Provenance
from kaos_content.units import (
    ParagraphUnit,
    SentenceUnit,
    iter_paragraph_units,
    iter_sentence_units,
)
from kaos_content.views import DocumentView

pytestmark = pytest.mark.unit


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
# iter_paragraph_units
# ---------------------------------------------------------------------------


class TestIterParagraphUnits:
    def test_basic_paragraphs(self) -> None:
        doc = _doc(_p("First paragraph."), _p("Second paragraph."), _p("Third paragraph."))
        units = iter_paragraph_units(doc)
        assert len(units) == 3
        for i, u in enumerate(units):
            assert isinstance(u, ParagraphUnit)
            assert u.row == i
            assert u.text
            assert u.block_ref

    def test_dense_row_indices(self) -> None:
        doc = _doc(*[_p(f"Paragraph {i}.") for i in range(10)])
        units = iter_paragraph_units(doc)
        assert len(units) == 10
        for i, u in enumerate(units):
            assert u.row == i

    def test_skips_empty_paragraphs(self) -> None:
        doc = _doc(_p("Real text."), _p(""), _p("   "), _p("More text."))
        units = iter_paragraph_units(doc)
        assert len(units) == 2
        assert units[0].text == "Real text."
        assert units[1].text == "More text."

    def test_skips_whitespace_only(self) -> None:
        doc = _doc(_p("  \t\n  "), _p("Actual content."))
        units = iter_paragraph_units(doc)
        assert len(units) == 1
        assert units[0].row == 0
        assert units[0].text == "Actual content."

    def test_page_provenance(self) -> None:
        doc = _doc(_p("Page 1 text.", page=1), _p("Page 2 text.", page=2))
        units = iter_paragraph_units(doc)
        assert units[0].page == 1
        assert units[1].page == 2

    def test_no_page_provenance(self) -> None:
        doc = _doc(_p("No page info."))
        units = iter_paragraph_units(doc)
        assert units[0].page is None

    def test_block_ref_format(self) -> None:
        doc = _doc(_p("First."), _p("Second."))
        units = iter_paragraph_units(doc)
        # block_refs should be JSON pointer paths like #/body/0, #/body/1
        for u in units:
            assert u.block_ref.startswith("#/body/")

    def test_section_ref_and_title(self) -> None:
        doc = _doc(_h(1, "Introduction"), _p("Intro text."), _h(1, "Methods"), _p("Method text."))
        units = iter_paragraph_units(doc)
        assert len(units) == 2
        assert units[0].section_ref is not None
        assert units[0].section_title == "Introduction"
        assert units[1].section_ref is not None
        assert units[1].section_title == "Methods"

    def test_no_section(self) -> None:
        doc = _doc(_p("No heading above."))
        units = iter_paragraph_units(doc)
        assert units[0].section_ref is None
        assert units[0].section_title is None

    def test_empty_document(self) -> None:
        doc = _doc()
        units = iter_paragraph_units(doc)
        assert units == []

    def test_document_with_only_empty_paragraphs(self) -> None:
        doc = _doc(_p(""), _p("   "))
        units = iter_paragraph_units(doc)
        assert units == []

    def test_accepts_document_view(self) -> None:
        doc = _doc(_p("Hello."), _p("World."))
        view = DocumentView(doc)
        units = iter_paragraph_units(view)
        assert len(units) == 2

    def test_view_matches_document(self) -> None:
        """Units from a DocumentView should match units from a ContentDocument."""
        doc = _doc(
            _h(1, "S1", page=1),
            _p("A.", page=1),
            _p("B.", page=1),
            _h(1, "S2", page=2),
            _p("C.", page=2),
        )
        from_doc = iter_paragraph_units(doc)
        from_view = iter_paragraph_units(DocumentView(doc))
        assert len(from_doc) == len(from_view)
        for d, v in zip(from_doc, from_view, strict=True):
            assert d.row == v.row
            assert d.text == v.text
            assert d.block_ref == v.block_ref
            assert d.page == v.page
            assert d.section_ref == v.section_ref
            assert d.section_title == v.section_title


# ---------------------------------------------------------------------------
# iter_sentence_units
# ---------------------------------------------------------------------------


class TestIterSentenceUnits:
    def test_basic_sentences(self) -> None:
        doc = _doc(_p("First sentence. Second sentence."))
        view = DocumentView(doc, sentence_segmenter=MockSegmenter())
        units = iter_sentence_units(view)
        assert len(units) >= 2
        for i, u in enumerate(units):
            assert isinstance(u, SentenceUnit)
            assert u.row == i
            assert u.text
            assert u.block_ref

    def test_dense_row_indices(self) -> None:
        doc = _doc(_p("A. B. C."), _p("D. E."))
        view = DocumentView(doc, sentence_segmenter=MockSegmenter())
        units = iter_sentence_units(view)
        for i, u in enumerate(units):
            assert u.row == i

    def test_char_offsets(self) -> None:
        doc = _doc(_p("Hello. World."))
        view = DocumentView(doc, sentence_segmenter=MockSegmenter())
        units = iter_sentence_units(view)
        for u in units:
            assert u.char_start >= 0
            assert u.char_end > u.char_start

    def test_block_ref_is_paragraph_ref(self) -> None:
        """SentenceUnit.block_ref should be the containing paragraph's block_ref."""
        doc = _doc(_p("A. B."), _p("C. D."))
        view = DocumentView(doc, sentence_segmenter=MockSegmenter())
        units = iter_sentence_units(view)
        para_units = iter_paragraph_units(view)
        para_refs = {u.block_ref for u in para_units}
        for u in units:
            assert u.block_ref in para_refs

    def test_page_provenance(self) -> None:
        doc = _doc(_p("Hello. World.", page=3))
        view = DocumentView(doc, sentence_segmenter=MockSegmenter())
        units = iter_sentence_units(view)
        for u in units:
            assert u.page == 3

    def test_section_info(self) -> None:
        doc = _doc(_h(1, "Title"), _p("A. B."))
        view = DocumentView(doc, sentence_segmenter=MockSegmenter())
        units = iter_sentence_units(view)
        for u in units:
            assert u.section_ref is not None
            assert u.section_title == "Title"

    def test_raises_without_segmenter(self) -> None:
        doc = _doc(_p("Hello. World."))
        with pytest.raises(RuntimeError, match="sentence segmenter"):
            iter_sentence_units(doc)

    def test_raises_without_segmenter_on_view(self) -> None:
        doc = _doc(_p("Hello. World."))
        view = DocumentView(doc)  # no segmenter
        with pytest.raises(RuntimeError, match="sentence segmenter"):
            iter_sentence_units(view)

    def test_empty_document(self) -> None:
        doc = _doc()
        view = DocumentView(doc, sentence_segmenter=MockSegmenter())
        units = iter_sentence_units(view)
        assert units == []

    def test_empty_paragraphs_skipped(self) -> None:
        doc = _doc(_p(""), _p("Real. Text."))
        view = DocumentView(doc, sentence_segmenter=MockSegmenter())
        units = iter_sentence_units(view)
        assert len(units) >= 1
        # All units should have non-empty text
        for u in units:
            assert u.text.strip()

    def test_view_matches_document_with_segmenter(self) -> None:
        """Passing a ContentDocument with a segmenter-aware DocumentView
        should give consistent results."""
        doc = _doc(_p("Hello. World.", page=1))
        segmenter = MockSegmenter()
        view = DocumentView(doc, sentence_segmenter=segmenter)
        units = iter_sentence_units(view)
        assert len(units) >= 2
        # All have consistent block_ref pointing to #/body/0
        for u in units:
            assert u.block_ref == "#/body/0"


# ---------------------------------------------------------------------------
# Frozen dataclass properties
# ---------------------------------------------------------------------------


class TestUnitDataclasses:
    def test_paragraph_unit_is_frozen(self) -> None:
        unit = ParagraphUnit(
            row=0,
            text="Hello.",
            block_ref="#/body/0",
            page=1,
            section_ref=None,
            section_title=None,
        )
        with pytest.raises(AttributeError):
            unit.row = 1

    def test_sentence_unit_is_frozen(self) -> None:
        unit = SentenceUnit(
            row=0,
            text="Hello.",
            block_ref="#/body/0",
            page=1,
            section_ref=None,
            section_title=None,
            char_start=0,
            char_end=6,
        )
        with pytest.raises(AttributeError):
            unit.row = 1
