"""Unit tests for kaos_content.tokens (K9)."""

from __future__ import annotations

from kaos_content.model.document import ContentDocument
from kaos_content.shortcuts import heading, paragraph
from kaos_content.tokens import (
    document_token_frequency,
    paragraph_token_frequency,
    section_token_frequency,
)
from kaos_content.units import iter_paragraph_units
from kaos_content.views import DocumentView


def _nda_doc() -> ContentDocument:
    return ContentDocument(
        body=(
            heading(1, "Mutual Non-Disclosure Agreement"),
            paragraph("This Agreement is dated January 1, 2026."),
            paragraph("Confidential Information includes business plans."),
            paragraph("The Agreement governs both parties."),
        ),
    )


class TestDocumentTokenFrequency:
    def test_basic_counts(self) -> None:
        freq = document_token_frequency(_nda_doc())
        assert freq["agreement"] == 3  # heading + 2 paragraphs
        assert freq["confidential"] == 1
        assert freq["information"] == 1

    def test_empty_doc_returns_empty_dict(self) -> None:
        assert document_token_frequency(ContentDocument(body=())) == {}

    def test_accepts_view_or_document(self) -> None:
        doc = _nda_doc()
        view = DocumentView(doc)
        from_doc = document_token_frequency(doc)
        from_view = document_token_frequency(view)
        assert from_doc == from_view

    def test_lowercase_aggregation(self) -> None:
        """'Agreement' / 'agreement' / 'AGREEMENT' all aggregate."""
        doc = ContentDocument(
            body=(paragraph("Agreement AGREEMENT agreement"),),
        )
        freq = document_token_frequency(doc)
        assert freq.get("agreement") == 3

    def test_stopwords_not_filtered(self) -> None:
        """tokens.py is the raw histogram. Stopword filtering is a
        caller concern (e.g. K1 summary builder)."""
        doc = ContentDocument(body=(paragraph("the the the agreement"),))
        freq = document_token_frequency(doc)
        assert freq.get("the") == 3
        assert freq.get("agreement") == 1


class TestSectionTokenFrequency:
    def test_counts_section_only_not_subsections(self) -> None:
        """section_token_frequency() returns counts for the section's
        own blocks; recursion across subsections is the caller's
        responsibility."""
        doc = ContentDocument(
            body=(
                heading(1, "Top"),
                paragraph("alpha alpha"),
                heading(2, "Nested"),
                paragraph("beta beta"),
            ),
        )
        view = DocumentView(doc)
        sections = view.flat_sections
        # Find the top-level section
        top = next(s for s in sections if s.heading_text == "Top")
        freq = section_token_frequency(top)
        # Top-level section owns "alpha alpha" only (the heading itself
        # is part of the section block list).
        assert freq.get("alpha") == 2
        assert freq.get("beta", 0) == 0


class TestParagraphTokenFrequency:
    def test_counts_paragraph_text(self) -> None:
        doc = _nda_doc()
        units = iter_paragraph_units(doc)
        # Pick the second paragraph: "Confidential Information includes business plans."
        target = next(u for u in units if "Confidential" in u.text)
        freq = paragraph_token_frequency(target)
        assert freq["confidential"] == 1
        assert freq["information"] == 1
        assert freq["business"] == 1
        assert freq["plans"] == 1

    def test_empty_text_returns_empty_dict(self) -> None:
        # Construct a unit-like object with empty text via the
        # ParagraphUnit dataclass — easier than creating a real
        # empty paragraph.
        from kaos_content.units import ParagraphUnit

        unit = ParagraphUnit(
            row=0,
            text="",
            block_ref="",
            page=None,
            section_ref=None,
            section_title=None,
        )
        assert paragraph_token_frequency(unit) == {}


class TestCrossEntrypointConsistency:
    def test_doc_equals_section_sum_for_single_section(self) -> None:
        """For a doc with one top-level section, document_token_frequency
        and section_token_frequency should produce identical totals."""
        doc = ContentDocument(
            body=(
                heading(1, "Only Section"),
                paragraph("alpha beta gamma"),
                paragraph("alpha delta"),
            ),
        )
        view = DocumentView(doc)
        sections = view.flat_sections
        only_section = next(s for s in sections if s.heading_text == "Only Section")
        doc_freq = document_token_frequency(doc)
        sec_freq = section_token_frequency(only_section)
        # The section frequency includes the heading text too via
        # serialize_text — so it should equal the doc frequency.
        assert doc_freq == sec_freq
