"""Unit tests for kaos_content.views.entity_filters (K2)."""

from __future__ import annotations

import pytest

from kaos_content.model.document import ContentDocument
from kaos_content.shortcuts import heading, paragraph
from kaos_content.views import DocumentView
from kaos_content.views.entity_filters import (
    ENTITY_TYPES,
    EntityMatch,
    ParagraphEntityHit,
    SentenceEntityHit,
    iter_paragraphs_with_entity,
    iter_sentences_with_entity,
    paragraphs_with_dates,
    paragraphs_with_durations,
    paragraphs_with_money,
    sentences_with_dates,
    sentences_with_durations,
    sentences_with_money,
    sentences_with_numbers,
    sentences_with_percents,
)


def _segmenter():
    from kaos_nlp_core._defaults import get_default_punkt_tokenizer

    return get_default_punkt_tokenizer()


def _doc_with_entities() -> ContentDocument:
    """Document with one paragraph per entity-type test case."""
    return ContentDocument(
        body=(
            heading(1, "Test Document"),
            paragraph("This Agreement is dated January 1, 2026 by both parties."),
            paragraph("The cap is $100,000 per occurrence and the floor is $5,000."),
            paragraph("Interest accrues at 7.5% per annum."),
            paragraph("The notice period is 30 days; the term runs 12 months."),
            paragraph("Section 4.2 governs liability."),
            paragraph("This paragraph has no entities at all, just plain prose."),
        ),
    )


def _view(doc: ContentDocument | None = None) -> DocumentView:
    return DocumentView(doc or _doc_with_entities(), sentence_segmenter=_segmenter())


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


class TestEntityMatchValueType:
    def test_construction(self) -> None:
        m = EntityMatch(
            entity_type="dates",
            text="January 1, 2026",
            value=None,
            start=12,
            end=27,
        )
        assert m.entity_type == "dates"
        assert m.text == "January 1, 2026"
        assert m.start == 12
        assert m.end == 27

    def test_frozen(self) -> None:
        m = EntityMatch(entity_type="dates", text="x", value=None, start=0, end=1)
        with pytest.raises((AttributeError, TypeError)):
            m.start = 5  # ty: ignore[invalid-assignment]


class TestSentenceEntityHit:
    def test_construction(self) -> None:
        view = _view()
        sentences = view.sentences
        m = EntityMatch(entity_type="dates", text="x", value=None, start=0, end=1)
        hit = SentenceEntityHit(sentence=sentences[0], matches=(m,))
        assert hit.matches == (m,)
        assert hit.sentence == sentences[0]


# ---------------------------------------------------------------------------
# ENTITY_TYPES
# ---------------------------------------------------------------------------


class TestEntityTypes:
    def test_known_types(self) -> None:
        for t in ("dates", "money", "percents", "durations", "numbers"):
            assert t in ENTITY_TYPES

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown entity_type"):
            list(iter_sentences_with_entity(_view(), "nonsense"))


# ---------------------------------------------------------------------------
# Sentence-level filters
# ---------------------------------------------------------------------------


class TestSentenceFilters:
    def test_sentences_with_dates(self) -> None:
        hits = sentences_with_dates(_view())
        assert len(hits) == 1
        assert "January 1, 2026" in hits[0].sentence.text
        assert hits[0].matches[0].entity_type == "dates"

    def test_sentences_with_money_multiple_per_sentence(self) -> None:
        """A single sentence with two $ amounts produces one hit with two matches.

        Note: the extractor sometimes captures trailing punctuation (e.g.
        ``"$5,000."`` when followed by a sentence-final period). We assert
        on substring containment rather than exact equality so the test
        tracks extractor-precision regressions rather than punctuation
        quirks.
        """
        hits = sentences_with_money(_view())
        assert len(hits) == 1
        amounts = [m.text for m in hits[0].matches]
        assert any("100,000" in a for a in amounts)
        assert any("5,000" in a for a in amounts)
        # The typed values must be exact (the extractor parses to Decimal
        # regardless of trailing punctuation in the matched text).
        from decimal import Decimal

        values = {m.value.amount for m in hits[0].matches}
        assert Decimal("100000") in values
        assert Decimal("5000") in values

    def test_sentences_with_percents(self) -> None:
        hits = sentences_with_percents(_view())
        assert len(hits) == 1
        assert "7.5%" in hits[0].matches[0].text

    def test_sentences_with_durations(self) -> None:
        hits = sentences_with_durations(_view())
        assert len(hits) >= 1
        # Should catch "30 days" and "12 months"
        texts = {m.text for h in hits for m in h.matches}
        assert any("days" in t for t in texts)
        assert any("months" in t for t in texts)

    def test_sentences_with_numbers(self) -> None:
        hits = sentences_with_numbers(_view())
        # Lots of numbers — every paragraph except the no-entity one
        # has at least a number (the dates, money, percent, durations
        # all parse as numbers too).
        assert len(hits) >= 3

    def test_no_matches_returns_empty(self) -> None:
        """A document with no dates produces no hits."""
        bare = ContentDocument(body=(paragraph("plain text only, no entities."),))
        view = _view(bare)
        assert sentences_with_dates(view) == ()
        assert sentences_with_money(view) == ()

    def test_match_offsets_are_within_sentence(self) -> None:
        """EntityMatch.start/end must index into the SENTENCE text,
        not the paragraph or document. Asserts the contract that
        ``sentence.text[match.start:match.end] == match.text``."""
        for hit in sentences_with_dates(_view()):
            for m in hit.matches:
                assert hit.sentence.text[m.start : m.end] == m.text

    def test_requires_segmenter(self) -> None:
        """sentence-level filters require a sentence segmenter."""
        no_seg_view = DocumentView(_doc_with_entities(), sentence_segmenter=None)
        with pytest.raises(RuntimeError, match="sentence segmenter"):
            list(iter_sentences_with_entity(no_seg_view, "dates"))


# ---------------------------------------------------------------------------
# Paragraph-level filters
# ---------------------------------------------------------------------------


class TestParagraphFilters:
    def test_paragraphs_with_dates(self) -> None:
        hits = paragraphs_with_dates(_view())
        assert len(hits) == 1
        assert "January" in hits[0].paragraph.text

    def test_paragraphs_with_money(self) -> None:
        hits = paragraphs_with_money(_view())
        assert len(hits) == 1
        assert len(hits[0].matches) == 2

    def test_paragraphs_with_durations(self) -> None:
        hits = paragraphs_with_durations(_view())
        assert len(hits) >= 1

    def test_no_segmenter_needed(self) -> None:
        """Paragraph filters work without a sentence segmenter."""
        no_seg_view = DocumentView(_doc_with_entities(), sentence_segmenter=None)
        hits = paragraphs_with_dates(no_seg_view)
        assert len(hits) == 1

    def test_paragraph_text_contains_match(self) -> None:
        """Paragraph offsets must index into the paragraph's own text."""
        for hit in paragraphs_with_money(_view()):
            for m in hit.matches:
                assert hit.paragraph.text[m.start : m.end] == m.text


# ---------------------------------------------------------------------------
# Generic iterator API
# ---------------------------------------------------------------------------


class TestGenericIterators:
    def test_iter_sentences_with_entity_yields_lazily(self) -> None:
        """The iter_* variant returns an iterator (not a tuple) so callers
        can break early or short-circuit."""
        gen = iter_sentences_with_entity(_view(), "dates")
        # We don't strictly require it to be a generator, just iterable.
        first = next(iter(gen))
        assert isinstance(first, SentenceEntityHit)

    def test_iter_paragraphs_with_entity_returns_paragraph_hits(self) -> None:
        gen = iter_paragraphs_with_entity(_view(), "money")
        first = next(iter(gen))
        assert isinstance(first, ParagraphEntityHit)


# ---------------------------------------------------------------------------
# DocumentView method wrappers
# ---------------------------------------------------------------------------


class TestDocumentViewMethods:
    def test_view_sentences_with_entity_method(self) -> None:
        """DocumentView.sentences_with_entity() mirrors the free function."""
        view = _view()
        free = sentences_with_dates(view)
        method = view.sentences_with_entity("dates")
        # Free function returns Tuple[SentenceEntityHit, ...]; method too.
        assert len(free) == len(method)
        assert free[0].sentence == method[0].sentence

    def test_view_paragraphs_with_entity_method(self) -> None:
        view = _view()
        free = paragraphs_with_money(view)
        method = view.paragraphs_with_entity("money")
        assert len(free) == len(method)


# ---------------------------------------------------------------------------
# Cross-extractor sanity
# ---------------------------------------------------------------------------


class TestCrossExtractor:
    def test_money_extracts_decimal_amount(self) -> None:
        """The typed value field must carry the numeric value, not just
        the text. This is the contract that distinguishes the filter
        from a regex grep."""
        from decimal import Decimal

        for hit in sentences_with_money(_view()):
            for m in hit.matches:
                # MoneyMatch has .amount (Decimal) and .currency (str)
                assert hasattr(m.value, "amount")
                assert isinstance(m.value.amount, Decimal)

    def test_date_extracts_datetime(self) -> None:
        from datetime import datetime

        hits = sentences_with_dates(_view())
        assert len(hits) == 1
        d = hits[0].matches[0].value
        assert isinstance(d, datetime)
        assert d.year == 2026
