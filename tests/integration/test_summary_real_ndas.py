"""Integration tests for kaos_content.summarize against real NDA files.

These tests parse a set of real Mutual NDA samples and assert that the
resulting summaries surface document-type signal (e.g.
"confidential information" in top n-grams, dates >= 1 in entity_counts).
The point is to prove the summary is *useful* — top n-grams identify
the document type, entity_counts match what a human would tally —
not just structurally well-formed.

The NDA fixture corpus is private and not redistributable. Point
``KAOS_CONTENT_NDA_DIR`` at a directory containing ``MNDA*.docx`` files
to run the suite; without that env var these tests skip cleanly so
external contributors and CI lanes without the fixture are unaffected.

No LLM. Uses kaos-office's DOCX parser, which is a required dev
dependency for this test.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from kaos_content.model.document import ContentDocument
from kaos_content.summarize import build_document_summary

# Private fixture corpus location. The env var takes precedence so
# external contributors can stage the corpus anywhere; absent that, the
# suite skips. We intentionally do NOT hard-code a maintainer-local
# default path — the release checklist mandates internal-path scrubs.
NDA_DIR_ENV = "KAOS_CONTENT_NDA_DIR"
NDA_DIR = Path(os.environ[NDA_DIR_ENV]) if os.environ.get(NDA_DIR_ENV) else None

requires_nda_fixtures = pytest.mark.skipif(
    NDA_DIR is None or not NDA_DIR.exists() or not any(NDA_DIR.glob("MNDA*.docx")),
    reason=f"NDA fixtures missing: set {NDA_DIR_ENV} to a directory containing MNDA*.docx",
)


def _parse(docx_path: Path) -> ContentDocument:
    """Parse one NDA docx into a ContentDocument."""
    from kaos_office import parse_docx

    return parse_docx(str(docx_path))


def _nda_paths() -> list[Path]:
    """MNDA docx fixtures discovered under ``KAOS_CONTENT_NDA_DIR``."""
    if NDA_DIR is None or not NDA_DIR.exists():
        return []
    return sorted(NDA_DIR.glob("MNDA*.docx"))


# ---------------------------------------------------------------------------
# Per-document tests
# ---------------------------------------------------------------------------


@requires_nda_fixtures
class TestSummaryOnRealNDAs:
    """Each NDA gets one parametrised pass; assertions hold for every doc."""

    @pytest.mark.parametrize("nda_path", _nda_paths(), ids=lambda p: p.name)
    def test_summary_built_with_real_signal(self, nda_path: Path) -> None:
        doc = _parse(nda_path)
        s = build_document_summary(doc)

        # Structural: the summary must not be empty.
        assert s.char_length > 0, f"empty doc? {nda_path.name}"
        assert s.paragraph_count > 0
        assert s.sentence_count > 0
        assert len(s.top_ngrams) > 0
        assert s.head_tokens

        # Topical signal: every MNDA mentions confidentiality. The
        # phrase "confidential information" is one of the most
        # frequent bigrams in any NDA — assert it appears in either
        # the top n-grams or the head_tokens.
        top_ngram_texts = {ng.ngram for ng in s.top_ngrams}
        topical_signal_present = (
            "confidential" in top_ngram_texts
            or "confidential information" in top_ngram_texts
            or "confidential information" in s.head_tokens.lower()
        )
        assert topical_signal_present, (
            f"{nda_path.name}: no 'confidential' signal found. "
            f"top_ngrams={top_ngram_texts!r}, head[:100]={s.head_tokens[:100]!r}"
        )

        # Entity signal: every NDA has dates (effective date,
        # termination date, signing dates), durations (term length,
        # notice period), and parties.
        assert s.entity_counts["dates"] >= 1, (
            f"{nda_path.name}: expected dates >= 1, got {s.entity_counts}"
        )

    @pytest.mark.parametrize("nda_path", _nda_paths(), ids=lambda p: p.name)
    def test_summary_is_deterministic(self, nda_path: Path) -> None:
        """Re-building the summary must produce identical output. This is
        the contract that lets us cache summaries to disk by content hash."""
        doc = _parse(nda_path)
        a = build_document_summary(doc)
        b = build_document_summary(doc)
        assert a == b


# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------


@requires_nda_fixtures
class TestSummaryPerformance:
    """Summary building must be fast enough to run on corpus scale.

    Target: <100 ms per typical NDA (~5-10 pages). Allow generous
    headroom (2x) so the test doesn't get noisy.
    """

    def test_each_nda_under_200ms(self) -> None:
        slow: list[tuple[str, float]] = []
        for nda_path in _nda_paths():
            doc = _parse(nda_path)
            t0 = time.perf_counter()
            build_document_summary(doc)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            if elapsed_ms > 200:
                slow.append((nda_path.name, elapsed_ms))
        assert not slow, f"slow summary builds: {slow}"


# ---------------------------------------------------------------------------
# Cross-document signal (corpus-scale triage)
# ---------------------------------------------------------------------------


@requires_nda_fixtures
class TestCorpusSignal:
    """When summaries are computed for an entire corpus, distinct
    documents should produce distinguishable bottom_ngrams (the
    rare-recurring-terms signature). That's the property that
    enables "find the doc that mentions X" queries to work on
    summaries alone."""

    def test_bottom_ngrams_differ_across_documents(self) -> None:
        summaries = {
            nda_path.name: build_document_summary(_parse(nda_path)) for nda_path in _nda_paths()
        }
        # Take each summary's bottom-ngram set. Different documents
        # should produce non-identical bottom-ngram fingerprints.
        signatures = {
            name: frozenset(ng.ngram for ng in s.bottom_ngrams) for name, s in summaries.items()
        }
        unique_signatures = set(signatures.values())
        # We have 5 NDAs and at least 2 distinct bottom-ngram sets is
        # the minimum useful result. In practice we expect close to 5.
        assert len(unique_signatures) >= 2, (
            f"All NDAs produced identical bottom_ngrams — summary lacks "
            f"discriminative power. signatures={signatures!r}"
        )

    def test_head_tokens_distinguish_documents(self) -> None:
        """The first ~500 tokens of any two NDAs should differ
        (parties, dates, opening recitals are unique)."""
        heads = {
            nda_path.name: build_document_summary(_parse(nda_path)).head_tokens
            for nda_path in _nda_paths()
        }
        unique_heads = set(heads.values())
        assert len(unique_heads) == len(heads), (
            f"two NDAs produced identical head_tokens — head slice is "
            f"too short to distinguish documents. heads keys={list(heads)}"
        )
