"""Unit tests for kaos_content.summarize (K1).

Deterministic — exercises the build_document_summary() pipeline
without an LLM. Live test against real NDA documents lives in
``tests/integration/test_summary_real_ndas.py``.
"""

from __future__ import annotations

import pytest

from kaos_content.model.document import ContentDocument
from kaos_content.model.summary import DocumentSummary, NGramFrequency
from kaos_content.shortcuts import heading, paragraph
from kaos_content.summarize import (
    DEFAULT_BOTTOM_K,
    DEFAULT_TOP_K,
    build_document_summary,
)
from kaos_content.summarize.stopwords import ENGLISH_STOPWORDS


def _nda_like_doc() -> ContentDocument:
    """A small NDA-flavored document used by several tests."""
    return ContentDocument(
        body=(
            heading(1, "Mutual Non-Disclosure Agreement"),
            paragraph(
                "This Agreement is entered into as of January 1, 2026 "
                "between Acme Corp and Beta Inc."
            ),
            paragraph(
                "Confidential Information includes business plans, financial "
                "projections, and customer lists."
            ),
            paragraph("The term is twenty-four (24) months from the Effective Date."),
            paragraph("Either party may terminate with 30 days written notice."),
            paragraph("The cap on liability is $100,000 per occurrence."),
            paragraph("Both parties shall hold the information in strict confidence."),
            paragraph("Indemnification carve-outs apply for gross negligence."),
        ),
    )


# ---------------------------------------------------------------------------
# Value type
# ---------------------------------------------------------------------------


class TestDocumentSummaryValueType:
    def test_defaults(self) -> None:
        s = DocumentSummary()
        assert s.head_tokens == ""
        assert s.top_ngrams == ()
        assert s.bottom_ngrams == ()
        assert s.char_length == 0
        assert s.sentence_count == 0
        assert s.paragraph_count == 0
        assert s.entity_counts == {}
        assert s.schema_version == 1

    def test_frozen(self) -> None:
        s = DocumentSummary(head_tokens="hello")
        # Pydantic frozen — reassignment raises ValidationError.
        with pytest.raises(Exception):  # noqa: B017 — Pydantic ValidationError
            s.head_tokens = "different"  # type: ignore[misc]

    def test_round_trip_json(self) -> None:
        original = DocumentSummary(
            head_tokens="Mutual Non-Disclosure Agreement",
            top_ngrams=(
                NGramFrequency(ngram="agreement", count=4),
                NGramFrequency(ngram="information", count=3),
            ),
            char_length=120,
            sentence_count=3,
            paragraph_count=3,
            entity_counts={"dates": 1, "money": 2},
        )
        round_tripped = DocumentSummary.model_validate_json(original.model_dump_json())
        assert round_tripped == original


# ---------------------------------------------------------------------------
# Builder — basic shape
# ---------------------------------------------------------------------------


class TestBuildSummaryBasic:
    def test_empty_document(self) -> None:
        s = build_document_summary(ContentDocument(body=()))
        assert s.paragraph_count == 0
        assert s.sentence_count == 0
        assert s.char_length == 0
        assert s.head_tokens == ""
        assert s.top_ngrams == ()
        assert s.bottom_ngrams == ()
        # entity_counts still has the four keys (just zero values)
        assert set(s.entity_counts) == {"dates", "money", "percents", "durations"}
        assert all(v == 0 for v in s.entity_counts.values())

    def test_counts_match_iters(self) -> None:
        doc = _nda_like_doc()
        s = build_document_summary(doc)
        # 7 body paragraphs (the heading is not a paragraph)
        assert s.paragraph_count == 7
        # Each paragraph is one sentence
        assert s.sentence_count == 7
        # char_length is the sum of paragraph text lengths
        assert s.char_length > 0
        assert s.char_length < 10_000  # sanity

    def test_head_tokens_starts_with_heading(self) -> None:
        """When the doc begins with a heading, head_tokens should
        preserve it verbatim — that's the signal a human reader uses
        to identify the document type at a glance."""
        s = build_document_summary(_nda_like_doc())
        assert "Mutual Non-Disclosure Agreement" in s.head_tokens

    def test_head_tokens_respects_target(self) -> None:
        """A small head_token_target should produce a short head_tokens."""
        s = build_document_summary(_nda_like_doc(), head_token_target=10)
        # Within a small slack of 10 tokens — the slice is to the end
        # of the Nth token, so up to one whitespace-trim worth of slop.
        from kaos_nlp_core.tokenizer import Tokenizer

        tok = Tokenizer(lowercase=True)
        n = len(tok.tokenize(s.head_tokens))
        assert n <= 15  # the slice ends at the 10th token

    def test_short_doc_head_is_full_body(self) -> None:
        """When the document is shorter than the head target, the
        full text is returned."""
        small = ContentDocument(body=(paragraph("Hello world."),))
        s = build_document_summary(small, head_token_target=500)
        assert "Hello world" in s.head_tokens


# ---------------------------------------------------------------------------
# Builder — n-gram quality
# ---------------------------------------------------------------------------


class TestNGramQuality:
    def test_top_ngrams_sorted_descending(self) -> None:
        s = build_document_summary(_nda_like_doc())
        if len(s.top_ngrams) >= 2:
            counts = [ng.count for ng in s.top_ngrams]
            assert counts == sorted(counts, reverse=True)

    def test_bottom_ngrams_respect_min_count(self) -> None:
        """bottom_ngrams excludes singletons by default (min_count >= 2)."""
        s = build_document_summary(_nda_like_doc())
        for ng in s.bottom_ngrams:
            assert ng.count >= 2

    def test_bottom_singletons_allowed_when_min_relaxed(self) -> None:
        """Passing min_bottom_count=1 surfaces singletons too."""
        s = build_document_summary(_nda_like_doc(), min_bottom_count=1)
        # Most n-grams in the small NDA are singletons; the bottom list
        # should be non-empty.
        assert len(s.bottom_ngrams) > 0

    def test_top_k_caps_result(self) -> None:
        s = build_document_summary(_nda_like_doc(), top_k=3)
        assert len(s.top_ngrams) <= 3

    def test_bottom_k_caps_result(self) -> None:
        s = build_document_summary(_nda_like_doc(), bottom_k=2, min_bottom_count=1)
        assert len(s.bottom_ngrams) <= 2

    def test_ngrams_dont_span_block_boundaries(self) -> None:
        """The trailing word of a heading should not concatenate with the
        leading word of the following paragraph into a bigram. We had
        this bug during development; this test guards against
        regression."""
        doc = ContentDocument(
            body=(
                heading(1, "AGREEMENT"),
                paragraph("AGREEMENT body text."),
            ),
        )
        s = build_document_summary(doc, ngram_max=3, min_bottom_count=1)
        # "agreement agreement" must NOT appear in any ngram.
        all_ngrams = [ng.ngram for ng in (*s.top_ngrams, *s.bottom_ngrams)]
        assert "agreement agreement" not in all_ngrams

    def test_stopwords_excluded(self) -> None:
        """No top n-gram should be a stopword."""
        s = build_document_summary(_nda_like_doc())
        for ng in s.top_ngrams:
            # Multi-word n-grams can contain a stopword as a middle
            # word in principle; we only filter at the *token* stage,
            # so the first and last tokens of any multi-word n-gram
            # should not be stopwords (a stronger guarantee than
            # nothing, weaker than full stopword exclusion).
            tokens = ng.ngram.split()
            assert tokens[0] not in ENGLISH_STOPWORDS
            assert tokens[-1] not in ENGLISH_STOPWORDS

    def test_ngram_max_controls_length(self) -> None:
        s = build_document_summary(_nda_like_doc(), ngram_max=1)
        for ng in s.top_ngrams:
            assert " " not in ng.ngram  # all unigrams


# ---------------------------------------------------------------------------
# Builder — entity counts
# ---------------------------------------------------------------------------


class TestEntityCounts:
    def test_entity_counts_populated_by_default(self) -> None:
        s = build_document_summary(_nda_like_doc())
        # The NDA has 1 date, 1 money, 2 durations, 0 percents.
        assert s.entity_counts["dates"] >= 1
        assert s.entity_counts["money"] >= 1
        assert s.entity_counts["durations"] >= 1

    def test_entity_counts_skipped_when_disabled(self) -> None:
        s = build_document_summary(_nda_like_doc(), with_entities=False)
        assert s.entity_counts == {}


# ---------------------------------------------------------------------------
# Builder — determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_same_doc_same_summary(self) -> None:
        """Building the summary twice over the same doc must yield
        an identical DocumentSummary. This is the contract that
        lets us cache summaries on disk and treat them as content-
        addressable."""
        doc = _nda_like_doc()
        a = build_document_summary(doc)
        b = build_document_summary(doc)
        assert a == b


# ---------------------------------------------------------------------------
# ContentDocument integration
# ---------------------------------------------------------------------------


class TestContentDocumentIntegration:
    def test_summary_field_defaults_none(self) -> None:
        doc = ContentDocument()
        assert doc.summary is None

    def test_summary_can_be_attached(self) -> None:
        """ContentDocument is frozen, but you can pass summary at
        construction. ``model_copy(update={...})`` is the canonical
        way to add a summary to an already-constructed document."""
        base = _nda_like_doc()
        summary = build_document_summary(base)
        with_summary = base.model_copy(update={"summary": summary})
        assert with_summary.summary == summary

    def test_summary_survives_json_round_trip(self) -> None:
        base = _nda_like_doc()
        summary = build_document_summary(base)
        with_summary = base.model_copy(update={"summary": summary})
        round_tripped = ContentDocument.model_validate_json(with_summary.model_dump_json())
        assert round_tripped.summary is not None
        assert round_tripped.summary == summary


# ---------------------------------------------------------------------------
# Stopword list
# ---------------------------------------------------------------------------


class TestStopwords:
    def test_contains_common_english(self) -> None:
        for w in ("the", "a", "an", "of", "and", "or", "to", "is", "are"):
            assert w in ENGLISH_STOPWORDS

    def test_does_not_contain_domain_terms(self) -> None:
        """Domain-specific terms like "agreement" or "party" are NOT
        stopwords — they're the topical signal we want to surface."""
        for w in (
            "agreement",
            "party",
            "confidential",
            "information",
            "merger",
            "contract",
            "company",
        ):
            assert w not in ENGLISH_STOPWORDS


# ---------------------------------------------------------------------------
# Configuration knobs
# ---------------------------------------------------------------------------


class TestKnobs:
    def test_defaults_match_module_constants(self) -> None:
        s = build_document_summary(_nda_like_doc())
        assert len(s.top_ngrams) <= DEFAULT_TOP_K
        assert len(s.bottom_ngrams) <= DEFAULT_BOTTOM_K

    def test_custom_stopwords(self) -> None:
        """Passing a custom stopword list overrides the default."""
        # Make "agreement" a stopword for this build. It should
        # disappear from top_ngrams entirely.
        custom = ENGLISH_STOPWORDS | {"agreement"}
        s = build_document_summary(_nda_like_doc(), stopwords=custom)
        ngrams = [ng.ngram for ng in s.top_ngrams]
        # No bare "agreement" unigram, and no multi-word n-gram starts
        # or ends with "agreement".
        for ng in ngrams:
            tokens = ng.split()
            assert tokens[0] != "agreement"
            assert tokens[-1] != "agreement"
