"""Canonical paragraph and sentence enumeration over a ContentDocument.

Single source of truth for "what counts as a row" in any pipeline that
iterates AST-grounded units of a document. Used by:

- ``kaos_content.search`` (planned refactor) — BM25 search rows
- ``kaos_ml_core.corpus`` — feature matrix rows

Both call this exact same enumeration so a row index in either pipeline
maps deterministically to the same ``block_ref``. The behavior mirrors
the ``_paragraphs_to_records`` helper currently in ``search.py`` so the
existing BM25 path's tests continue to pass after the planned refactor.

The two unit dataclasses are intentionally minimal — they carry only the
fields shared by every downstream consumer. Modules that need to attach
extra context (e.g. ``kaos_ml_core.CorpusUnit`` adds ``doc_uri`` and a
global row index) wrap these.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaos_content.model.document import ContentDocument
    from kaos_content.views import DocumentView

    DocOrView = ContentDocument | DocumentView


@dataclass(frozen=True, slots=True)
class ParagraphUnit:
    """A non-empty paragraph with its AST address and provenance."""

    row: int
    text: str
    block_ref: str
    page: int | None
    section_ref: str | None
    section_title: str | None
    confidence: float | None = None
    """N6: source-level extraction confidence (e.g. OCR confidence
    on scanned PDFs). ``None`` when the extractor didn't report a
    score (born-digital docs). In ``[0.0, 1.0]`` when set."""


@dataclass(frozen=True, slots=True)
class SentenceUnit:
    """A non-empty sentence with the containing paragraph's block_ref."""

    row: int
    text: str
    block_ref: str
    """The block_ref of the containing paragraph (mirrors SentenceView.paragraph_ref)."""
    page: int | None
    section_ref: str | None
    section_title: str | None
    char_start: int
    char_end: int


def iter_paragraph_units(
    document: ContentDocument | DocumentView,
) -> list[ParagraphUnit]:
    """Enumerate non-empty paragraphs as ParagraphUnits.

    Mirrors the loop in ``kaos_content.search._paragraphs_to_records``:
    skips empty / whitespace-only paragraphs, resolves section titles
    once per ``section_ref`` via a small cache, assigns dense row indices
    starting at 0.

    Accepts either a ``ContentDocument`` or an existing ``DocumentView``
    so callers that already constructed a view (e.g.
    ``kaos_content.search``) avoid the cost of building a second one.
    """
    from kaos_content.views import DocumentView

    view = document if isinstance(document, DocumentView) else DocumentView(document)
    units: list[ParagraphUnit] = []
    section_titles: dict[str, str | None] = {}
    row = 0
    for pv in view.paragraphs:
        if not pv.text or not pv.text.strip():
            continue
        if pv.section_ref is not None and pv.section_ref not in section_titles:
            sec = view.section_by_ref(pv.section_ref)
            section_titles[pv.section_ref] = sec.heading_text if sec is not None else None
        units.append(
            ParagraphUnit(
                row=row,
                text=pv.text,
                block_ref=pv.block_ref,
                page=pv.page,
                section_ref=pv.section_ref,
                section_title=(section_titles.get(pv.section_ref) if pv.section_ref else None),
                confidence=pv.confidence,
            )
        )
        row += 1
    return units


def iter_sentence_units(
    document: ContentDocument | DocumentView,
) -> list[SentenceUnit]:
    """Enumerate sentences within non-empty paragraphs as SentenceUnits.

    Requires a sentence segmenter on the DocumentView (the standard
    kaos-content sentence pipeline, which uses kaos-nlp-core's Punkt
    tokenizer when the ``[nlp]`` extra is installed). Each sentence
    carries the ``block_ref`` of its containing paragraph.

    Accepts either a ``ContentDocument`` or an existing ``DocumentView``.

    Raises:
        RuntimeError: If the DocumentView has no segmenter wired up.
    """
    from kaos_content.views import DocumentView

    view = document if isinstance(document, DocumentView) else DocumentView(document)
    if not view.has_sentences:
        msg = (
            "DocumentView has no sentence segmenter; cannot enumerate sentences. "
            "Fix: install kaos-content[nlp] which pulls in kaos-nlp-core's Punkt tokenizer, "
            "or pass a segmenter explicitly when constructing the DocumentView. "
            "Alternative: use iter_paragraph_units() for paragraph-level grounding."
        )
        raise RuntimeError(msg)

    units: list[SentenceUnit] = []
    section_titles: dict[str, str | None] = {}
    row = 0
    for sv in view.sentences:
        text = sv.text
        if not text or not text.strip():
            continue
        if sv.section_ref is not None and sv.section_ref not in section_titles:
            sec = view.section_by_ref(sv.section_ref)
            section_titles[sv.section_ref] = sec.heading_text if sec is not None else None
        units.append(
            SentenceUnit(
                row=row,
                text=text,
                block_ref=sv.paragraph_ref,
                page=sv.page,
                section_ref=sv.section_ref,
                section_title=(section_titles.get(sv.section_ref) if sv.section_ref else None),
                char_start=sv.start,
                char_end=sv.end,
            )
        )
        row += 1
    return units


__all__ = [
    "ParagraphUnit",
    "SentenceUnit",
    "iter_paragraph_units",
    "iter_sentence_units",
]
