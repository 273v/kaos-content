"""Typed-entity sentence/paragraph filter primitives.

K2 of docs/design/findings-entities-summary.md. Given a
:class:`DocumentView`, surface every sentence or paragraph that
contains at least one match of a target entity type (date, money,
percent, duration, number), each paired with the typed matches it
contains.

Implemented as free functions over a :class:`DocumentView` rather
than methods on the view itself. Two reasons:

1. The view class lives in ``kaos_content.views.document_view``,
   uses ``__slots__``, and is hot-path code. Each new filter method
   bloats that class. Free functions keep the surface tidy and let
   the entity-filter implementation live next to the
   :class:`EntityMatch` / :class:`EntityFilterHit` value types
   they return.
2. Several thin wrapper methods on :class:`DocumentView` (e.g.
   ``view.sentences_with_dates()``) delegate to these functions —
   gives the caller a discoverable surface without duplicating
   logic.

The matches are aligned to the **unit's own coordinate frame** (a
sentence's ``text`` field, or a paragraph's ``text`` field), not the
document's. Callers that need document-level char spans can compose
with :attr:`SentenceView.paragraph_ref` plus
:attr:`SentenceView.start` to recover them.

Performance: pure regex / dictionary lookups via kaos-nlp-core's
``Alpha*Extractor`` family. ~1-10 ms per sentence on typical NDA
text, single-pass.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kaos_content.views.document_view import DocumentView
    from kaos_content.views.models import ParagraphView, SentenceView


# ---------------------------------------------------------------------------
# Value types
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class EntityMatch:
    """One extracted entity span, expressed in the containing unit's coords.

    The ``value`` field carries the *typed* extracted value — a
    :class:`datetime.datetime` for dates, a
    :class:`~kaos_nlp_core.extract.alpha.money.MoneyMatch` for money
    (with ``amount: Decimal`` and ``currency: str``), a
    :class:`~kaos_nlp_core.extract.alpha.duration.DurationMatch` for
    durations, a :class:`decimal.Decimal` for percents and numbers.
    Consumers that only want the matched text use ``text``; consumers
    that need the typed value (for analytics, comparison, sorting)
    use ``value``.
    """

    entity_type: str
    """Lower-cased entity type name (``"dates"``, ``"money"``,
    ``"percents"``, ``"durations"``, ``"numbers"``). Matches the
    keys used in :attr:`DocumentSummary.entity_counts`."""

    text: str
    """The matched text slice from the containing unit, exactly as
    it appeared (preserves case)."""

    value: Any
    """The typed extracted value (see class docstring for per-type
    shape). ``Any`` because the typed shapes vary per extractor and
    we don't want to leak kaos-nlp-core's internal types into the
    public surface of kaos-content."""

    start: int
    """Character offset of the match in the containing unit's
    ``text`` (sentence text for sentence filters, paragraph text
    for paragraph filters). 0-based, inclusive."""

    end: int
    """Exclusive end offset in the containing unit's text."""


@dataclass(frozen=True, slots=True)
class SentenceEntityHit:
    """A sentence that contains at least one match of a target entity type.

    Returned by the ``sentences_with_*`` filter functions. Pairs the
    :class:`SentenceView` (with its AST anchor — block_ref, page,
    section_ref, etc.) with the matches found inside its ``text``.
    """

    sentence: SentenceView
    matches: tuple[EntityMatch, ...]


@dataclass(frozen=True, slots=True)
class ParagraphEntityHit:
    """A paragraph that contains at least one match of a target entity type.

    Returned by the ``paragraphs_with_*`` filter functions.
    """

    paragraph: ParagraphView
    matches: tuple[EntityMatch, ...]


# ---------------------------------------------------------------------------
# Extractor registry
# ---------------------------------------------------------------------------
#
# Map from entity type name → (extractor factory, AlphaSpan → typed value
# adapter). The adapter is the identity for most types; AlphaSpan.value
# is already the typed payload from the extractor.

ENTITY_TYPES: tuple[str, ...] = ("dates", "money", "percents", "durations", "numbers")
"""The set of entity types supported by the filter functions. Matches
the keys used in :attr:`DocumentSummary.entity_counts` (minus
``"parties"``, which doesn't have a single bundled extractor today)."""


def _get_extractor(entity_type: str) -> Any:
    """Lazy import + construct an extractor for one entity type.

    Lazy so that ``[nlp]`` is only required when entity filters are
    actually called, not at module import time.
    """
    from kaos_nlp_core.extract.alpha import (
        AlphaDateExtractor,
        AlphaDurationExtractor,
        AlphaMoneyExtractor,
        AlphaNumberExtractor,
        AlphaPercentExtractor,
    )

    factories: dict[str, type] = {
        "dates": AlphaDateExtractor,
        "money": AlphaMoneyExtractor,
        "percents": AlphaPercentExtractor,
        "durations": AlphaDurationExtractor,
        "numbers": AlphaNumberExtractor,
    }
    if entity_type not in factories:
        raise ValueError(
            f"Unknown entity_type {entity_type!r}. Known: {sorted(factories)}. "
            "Fix: pass one of the supported types or extend ENTITY_TYPES "
            "with a new extractor."
        )
    return factories[entity_type]()


# ---------------------------------------------------------------------------
# Public filter functions
# ---------------------------------------------------------------------------


def iter_sentences_with_entity(
    view: DocumentView,
    entity_type: str,
) -> Iterator[SentenceEntityHit]:
    """Yield each sentence containing at least one match of ``entity_type``.

    Args:
        view: A sentence-aware :class:`DocumentView`. Must have a
            sentence segmenter configured (see ``DocumentView.has_sentences``).
        entity_type: One of :data:`ENTITY_TYPES`.

    Yields:
        :class:`SentenceEntityHit` for each sentence whose text
        contains >=1 match. Sentences with zero matches are skipped.

    Raises:
        RuntimeError: If ``view`` lacks a sentence segmenter.
        ValueError: If ``entity_type`` is not in :data:`ENTITY_TYPES`.
    """
    if not view.has_sentences:
        msg = (
            "DocumentView has no sentence segmenter; cannot enumerate "
            "sentence-level entity hits. Fix: construct the view with "
            "sentence_segmenter=get_default_punkt_tokenizer() (install "
            "kaos-content[nlp])."
        )
        raise RuntimeError(msg)

    extractor = _get_extractor(entity_type)
    for sentence in view.sentences:
        matches = _matches_in(extractor, entity_type, sentence.text)
        if matches:
            yield SentenceEntityHit(sentence=sentence, matches=matches)


def iter_paragraphs_with_entity(
    view: DocumentView,
    entity_type: str,
) -> Iterator[ParagraphEntityHit]:
    """Yield each paragraph containing at least one match of ``entity_type``.

    No sentence segmenter required — paragraphs are always available
    on a :class:`DocumentView`.
    """
    extractor = _get_extractor(entity_type)
    for paragraph in view.paragraphs:
        matches = _matches_in(extractor, entity_type, paragraph.text)
        if matches:
            yield ParagraphEntityHit(paragraph=paragraph, matches=matches)


# Per-type convenience wrappers. These exist purely for ergonomics —
# IDE autocomplete and intent signaling. All delegate to
# iter_sentences_with_entity / iter_paragraphs_with_entity above.


def sentences_with_dates(view: DocumentView) -> tuple[SentenceEntityHit, ...]:
    """Sentences containing >=1 date match."""
    return tuple(iter_sentences_with_entity(view, "dates"))


def sentences_with_money(view: DocumentView) -> tuple[SentenceEntityHit, ...]:
    """Sentences containing >=1 money match."""
    return tuple(iter_sentences_with_entity(view, "money"))


def sentences_with_percents(view: DocumentView) -> tuple[SentenceEntityHit, ...]:
    """Sentences containing >=1 percentage match."""
    return tuple(iter_sentences_with_entity(view, "percents"))


def sentences_with_durations(view: DocumentView) -> tuple[SentenceEntityHit, ...]:
    """Sentences containing >=1 duration match."""
    return tuple(iter_sentences_with_entity(view, "durations"))


def sentences_with_numbers(view: DocumentView) -> tuple[SentenceEntityHit, ...]:
    """Sentences containing >=1 numeric match."""
    return tuple(iter_sentences_with_entity(view, "numbers"))


def paragraphs_with_dates(view: DocumentView) -> tuple[ParagraphEntityHit, ...]:
    """Paragraphs containing >=1 date match."""
    return tuple(iter_paragraphs_with_entity(view, "dates"))


def paragraphs_with_money(view: DocumentView) -> tuple[ParagraphEntityHit, ...]:
    """Paragraphs containing >=1 money match."""
    return tuple(iter_paragraphs_with_entity(view, "money"))


def paragraphs_with_percents(view: DocumentView) -> tuple[ParagraphEntityHit, ...]:
    """Paragraphs containing >=1 percentage match."""
    return tuple(iter_paragraphs_with_entity(view, "percents"))


def paragraphs_with_durations(view: DocumentView) -> tuple[ParagraphEntityHit, ...]:
    """Paragraphs containing >=1 duration match."""
    return tuple(iter_paragraphs_with_entity(view, "durations"))


def paragraphs_with_numbers(view: DocumentView) -> tuple[ParagraphEntityHit, ...]:
    """Paragraphs containing >=1 numeric match."""
    return tuple(iter_paragraphs_with_entity(view, "numbers"))


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _matches_in(extractor: Any, entity_type: str, text: str) -> tuple[EntityMatch, ...]:
    """Run the extractor over ``text`` and wrap each AlphaSpan as an EntityMatch."""
    if not text:
        return ()
    matches: list[EntityMatch] = []
    try:
        spans = extractor.extract_spans(text)
    except Exception:
        # Defensive: a single bad sentence shouldn't kill the whole
        # filter pass. The downside is silent miss; the upside is
        # documents with one weird sentence still triage cleanly.
        return ()
    for span in spans:
        matched_text = text[span.start : span.end]
        matches.append(
            EntityMatch(
                entity_type=entity_type,
                text=matched_text,
                value=span.value,
                start=span.start,
                end=span.end,
            )
        )
    return tuple(matches)


__all__ = [
    "ENTITY_TYPES",
    "EntityMatch",
    "ParagraphEntityHit",
    "SentenceEntityHit",
    "iter_paragraphs_with_entity",
    "iter_sentences_with_entity",
    "paragraphs_with_dates",
    "paragraphs_with_durations",
    "paragraphs_with_money",
    "paragraphs_with_numbers",
    "paragraphs_with_percents",
    "sentences_with_dates",
    "sentences_with_durations",
    "sentences_with_money",
    "sentences_with_numbers",
    "sentences_with_percents",
]
