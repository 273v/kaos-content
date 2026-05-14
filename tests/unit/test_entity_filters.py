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


# ---------------------------------------------------------------------------
# Salience (PA9)
#
# K3 originally returned hits in document order — "top 3 dates" yielded
# the first 3 dates mentioned, often boilerplate ("the year 2026 …").
# PA9 adds a salience score combining match-density + document position
# + sentence length so top-K selection picks the load-bearing hits.
# ---------------------------------------------------------------------------


def _nda_like_doc() -> ContentDocument:
    """An NDA-shaped document with multiple irrelevant early dates and
    a load-bearing "effective + termination" sentence in the middle.

    Mirrors the real failure mode the user described: boilerplate
    references to "the year 2026" early in the document outrank the
    actual term clause when ranking by position.
    """
    from kaos_content.shortcuts import heading, paragraph

    return ContentDocument(
        body=(
            heading(1, "Mutual Non-Disclosure Agreement"),
            # Boilerplate paragraph 1 — single throwaway date.
            paragraph(
                "WHEREAS the parties have been discussing matters since "
                "January 1, 2025, they wish to formalise the terms below."
            ),
            # Boilerplate paragraph 2 — another single year-reference.
            paragraph("In the year 2026, the parties continue to confer."),
            # The MIDDLE paragraph carries TWO dates and is the term clause.
            paragraph(
                "This Agreement is effective as of March 15, 2026 and expires on March 15, 2027."
            ),
            # Body filler — single date.
            paragraph("Notice was given on April 1, 2026 of certain matters."),
            heading(2, "Signatures"),
            # Short signature-block sentence at the end — high position score.
            paragraph("Dated: April 22, 2026."),
        ),
    )


class TestSalience:
    def test_salience_is_in_unit_interval(self) -> None:
        """Every hit's salience must land in [0.0, 1.0] — the public contract."""
        for hit in sentences_with_dates(_view(_nda_like_doc())):
            assert 0.0 <= hit.salience <= 1.0

    def test_salience_picks_dense_clause_over_early_boilerplate(self) -> None:
        """The top-by-salience date sentence is the multi-date term clause,
        NOT the first date sentence in the document.

        This is the regression PA9 exists to prevent: document-order
        selection returns the first dates mentioned (often boilerplate),
        salience-order surfaces the load-bearing clause.
        """
        hits = sentences_with_dates(_view(_nda_like_doc()))
        # In doc order the first hit is the WHEREAS boilerplate.
        by_position = list(hits)
        assert "January 1, 2025" in by_position[0].sentence.text
        # By salience the first hit is the dense term clause.
        by_salience = sorted(hits, key=lambda h: (-h.salience, 0))
        top_text = by_salience[0].sentence.text
        assert "March 15, 2026" in top_text or "expires on March 15, 2027" in top_text

    def test_topk_by_salience_differs_from_topk_by_position(self) -> None:
        """Top-3 by salience must NOT equal top-3 by position on this doc.

        The fixture is constructed so the count signal alone (two distinct
        dates in the term clause vs one in each boilerplate sentence)
        forces the term clause out of position-order's top window.
        """
        hits = list(sentences_with_dates(_view(_nda_like_doc())))
        top3_position = [h.sentence.text for h in hits[:3]]
        top3_salience = [h.sentence.text for h in sorted(hits, key=lambda h: (-h.salience, 0))[:3]]
        assert top3_position != top3_salience

    def test_empty_matches_yields_zero_salience(self) -> None:
        """Hits with no matches get salience 0.0 — sanity guard for the
        default value on the dataclass. The filter skips empty hits in
        practice, but the contract holds when callers construct hits
        directly (e.g. in tests)."""
        hit = SentenceEntityHit(
            sentence=_view(_nda_like_doc()).sentences[0],
            matches=(),
        )
        assert hit.salience == 0.0

    def test_position_score_boosts_heading_adjacent_paragraph(self) -> None:
        """A heading-adjacent paragraph with the same match count + length
        as a mid-body paragraph scores higher on salience. Confirms the
        heading-proximity bump described in the salience docstring."""
        from kaos_content.shortcuts import heading as _h
        from kaos_content.shortcuts import paragraph as _p

        text_with_date = (
            "The notice was served on January 1, 2026 in compliance with "
            "the agreement terms then in effect."
        )
        # Doc 1: heading then the dated paragraph (heading-adjacent).
        doc1 = ContentDocument(body=(_h(1, "Section"), _p(text_with_date), _p("body filler")))
        # Doc 2: dated paragraph not adjacent to any heading.
        doc2 = ContentDocument(body=(_p("body filler"), _p(text_with_date), _p("body filler")))
        hits1 = sentences_with_dates(_view(doc1))
        hits2 = sentences_with_dates(_view(doc2))
        assert hits1 and hits2
        assert hits1[0].salience > hits2[0].salience

    def test_length_score_penalises_very_long_sentences(self) -> None:
        """A pathologically long sentence loses points on the length axis.

        Same date in both docs; only the surrounding prose changes.
        """
        from kaos_content.shortcuts import paragraph as _p

        short = "Dated: January 1, 2026."
        long_filler = " ".join(["this is filler prose"] * 50)  # >400 chars
        very_long = f"On January 1, 2026 {long_filler} — by both parties."
        doc_short = ContentDocument(body=(_p(short),))
        doc_long = ContentDocument(body=(_p(very_long),))
        hits_short = sentences_with_dates(_view(doc_short))
        hits_long = sentences_with_dates(_view(doc_long))
        assert hits_short and hits_long
        # very_long is >400 chars -> length score 0; short is <40 -> ramp.
        # Both should still score, but short > long by virtue of the length
        # axis alone (count + position factors are identical).
        assert hits_short[0].salience > hits_long[0].salience
