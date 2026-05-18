"""Integration tests for entity filters against real NDA docx files (K2).

Every Mutual NDA in the sample corpus contains dates (effective date,
signature date, sometimes termination date), durations (the term length,
notice periods), and parties. We assert each filter type produces
non-trivial results on every NDA, and that the matched text in the
result actually appears in the document.

No LLM. Uses kaos-office's DOCX parser.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from kaos_content.model.document import ContentDocument
from kaos_content.views import DocumentView
from kaos_content.views.entity_filters import (
    paragraphs_with_dates,
    sentences_with_dates,
    sentences_with_durations,
    sentences_with_money,
)

# Private NDA fixture corpus. Set ``KAOS_CONTENT_NDA_DIR`` to point at
# a directory containing ``MNDA*.docx`` to run the suite; without it the
# tests skip cleanly. We do NOT hard-code a maintainer-local default —
# the release checklist mandates internal-path scrubs.
NDA_DIR_ENV = "KAOS_CONTENT_NDA_DIR"
NDA_DIR = Path(os.environ[NDA_DIR_ENV]) if os.environ.get(NDA_DIR_ENV) else None

requires_nda_fixtures = pytest.mark.skipif(
    NDA_DIR is None or not NDA_DIR.exists() or not any(NDA_DIR.glob("MNDA*.docx")),
    reason=f"NDA fixtures missing: set {NDA_DIR_ENV} to a directory containing MNDA*.docx",
)


def _parse(path: Path) -> ContentDocument:
    from kaos_office import parse_docx

    return parse_docx(str(path))


def _view(doc: ContentDocument) -> DocumentView:
    from kaos_nlp_core._defaults import get_default_punkt_tokenizer

    return DocumentView(doc, sentence_segmenter=get_default_punkt_tokenizer())


def _nda_paths() -> list[Path]:
    if NDA_DIR is None or not NDA_DIR.exists():
        return []
    return sorted(NDA_DIR.glob("MNDA*.docx"))


@requires_nda_fixtures
class TestEntityFiltersOnNDAs:
    @pytest.mark.parametrize("nda_path", _nda_paths(), ids=lambda p: p.name)
    def test_dates_present(self, nda_path: Path) -> None:
        """Every NDA has at least one date (effective date / signing date)."""
        view = _view(_parse(nda_path))
        hits = sentences_with_dates(view)
        assert len(hits) >= 1, f"{nda_path.name}: no date hits — extractor regression?"
        # Every matched text must actually appear in the sentence.
        for hit in hits:
            for m in hit.matches:
                assert hit.sentence.text[m.start : m.end] == m.text

    @pytest.mark.parametrize("nda_path", _nda_paths(), ids=lambda p: p.name)
    def test_durations_present(self, nda_path: Path) -> None:
        """Every NDA has at least one duration (term length, notice period)."""
        view = _view(_parse(nda_path))
        hits = sentences_with_durations(view)
        assert len(hits) >= 1, f"{nda_path.name}: no duration hits — extractor regression?"

    @pytest.mark.parametrize("nda_path", _nda_paths(), ids=lambda p: p.name)
    def test_paragraph_and_sentence_filters_agree(self, nda_path: Path) -> None:
        """If a paragraph has a date match, at least one of its sentences
        must also be flagged. The two filter granularities should be
        consistent on the same document."""
        view = _view(_parse(nda_path))
        para_hits = paragraphs_with_dates(view)
        sent_hits = sentences_with_dates(view)
        paragraph_refs_with_dates = {h.paragraph.block_ref for h in para_hits}
        # Every paragraph that has dates should have at least one
        # sentence-level hit pointing into it.
        for ref in paragraph_refs_with_dates:
            matching_sentences = [s for s in sent_hits if s.sentence.paragraph_ref == ref]
            assert matching_sentences, (
                f"{nda_path.name}: paragraph {ref!r} has dates at "
                f"paragraph level but no sentence-level hits — "
                f"the two filters disagree."
            )


@requires_nda_fixtures
class TestEntityFilterPerformance:
    """Filters must be fast enough to run on every doc during triage."""

    def test_each_nda_filter_pass_under_500ms(self) -> None:
        import time as _time

        slow: list[tuple[str, float]] = []
        for nda_path in _nda_paths():
            doc = _parse(nda_path)
            view = _view(doc)
            t0 = _time.perf_counter()
            sentences_with_dates(view)
            sentences_with_money(view)
            sentences_with_durations(view)
            elapsed_ms = (_time.perf_counter() - t0) * 1000
            if elapsed_ms > 500:
                slow.append((nda_path.name, elapsed_ms))
        assert not slow, f"slow filter passes: {slow}"


@requires_nda_fixtures
class TestEntityValueTypes:
    """The typed value field carries real domain types, not strings."""

    def test_dates_are_datetime(self) -> None:
        from datetime import datetime

        for nda_path in _nda_paths()[:1]:  # one is enough
            view = _view(_parse(nda_path))
            hits = sentences_with_dates(view)
            assert hits  # smoke
            seen_datetime = any(isinstance(m.value, datetime) for h in hits for m in h.matches)
            assert seen_datetime, f"{nda_path.name}: no datetime values found in date hits"
