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
    salience: float = 0.0
    """Salience score in ``[0.0, 1.0]``.

    Combines three signals so top-K selection picks the load-bearing
    sentence (effective date, signature line, term length) rather than
    the first date mentioned (often a year-of boilerplate):

    1. **Count** (weight 0.6) — distinct typed values in the sentence;
       saturates at 3. The dominant signal: a sentence with two dates
       ("effective ... expires ...") is almost always more important
       than a sentence with one passing date reference.
    2. **Position** (weight 0.2) — back-of-document signature blocks
       and heading-adjacent paragraphs score above the mid-body floor.
       No blanket "front of document" bonus — the very first sentence
       is typically a WHEREAS preamble, not a term clause.
    3. **Length** (weight 0.2) — triangular peak around 40-200 chars;
       very short (< 20) or very long (> 400) get 0. Signature stubs
       and run-on boilerplate are both penalised.

    ``0.0`` for hits with no matches (which the filter normally skips,
    but the constructor default keeps the dataclass usable on its own).
    Higher = more important; the MCP wrapper sorts by this descending,
    with document position ascending as tiebreaker.
    """


@dataclass(frozen=True, slots=True)
class ParagraphEntityHit:
    """A paragraph that contains at least one match of a target entity type.

    Returned by the ``paragraphs_with_*`` filter functions.
    """

    paragraph: ParagraphView
    matches: tuple[EntityMatch, ...]
    salience: float = 0.0
    """Salience score in ``[0.0, 1.0]``. See :class:`SentenceEntityHit`
    for the formula. Paragraph-level scores use the same three signals
    over paragraph text rather than sentence text."""


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
        Hits carry a precomputed ``salience`` score; the iterator
        order is still document order (PA9: the MCP tool sorts).

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
    all_sentences = view.sentences
    total = len(all_sentences)
    # Block_refs that are the first paragraph after a heading — used
    # for the "heading-adjacent" position boost. We precompute on the
    # paragraph view since paragraphs carry section_ref.
    heading_proximate_refs = _heading_proximate_paragraph_refs(view)
    for idx, sentence in enumerate(all_sentences):
        matches = _matches_in(extractor, entity_type, sentence.text)
        if not matches:
            continue
        salience = _compute_salience(
            text=sentence.text,
            matches=matches,
            position_index=idx,
            position_total=total,
            heading_proximate=sentence.paragraph_ref in heading_proximate_refs,
        )
        yield SentenceEntityHit(sentence=sentence, matches=matches, salience=salience)


def iter_paragraphs_with_entity(
    view: DocumentView,
    entity_type: str,
) -> Iterator[ParagraphEntityHit]:
    """Yield each paragraph containing at least one match of ``entity_type``.

    No sentence segmenter required — paragraphs are always available
    on a :class:`DocumentView`. Hits carry a precomputed ``salience``
    score (PA9); document order is preserved in the iteration sequence.
    """
    extractor = _get_extractor(entity_type)
    all_paragraphs = view.paragraphs
    total = len(all_paragraphs)
    heading_proximate_refs = _heading_proximate_paragraph_refs(view)
    for idx, paragraph in enumerate(all_paragraphs):
        matches = _matches_in(extractor, entity_type, paragraph.text)
        if not matches:
            continue
        salience = _compute_salience(
            text=paragraph.text,
            matches=matches,
            position_index=idx,
            position_total=total,
            heading_proximate=paragraph.block_ref in heading_proximate_refs,
        )
        yield ParagraphEntityHit(paragraph=paragraph, matches=matches, salience=salience)


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


# Salience component weights. Tunable but stable; documented in the
# SentenceEntityHit docstring. Sum == 1.0 so the result is already in
# [0, 1] given each component score is in [0, 1].
#
# Count weight dominates: density of distinct typed values is the
# strongest signal that a sentence is a "term clause" rather than a
# passing date reference. Position and length round out the score so
# the count signal doesn't crown noisy duplicates.
_W_COUNT: float = 0.6
_W_POSITION: float = 0.2
_W_LENGTH: float = 0.2

# Length-score band: triangular peak in [LEN_LOW, LEN_HIGH] chars.
# Below LEN_MIN: penalty grows linearly to 0. Above LEN_MAX: same on
# the other side. Tuned for legal-prose text after eyeballing NDA
# sentence-length distributions:
#   - signature lines ("Dated: ____") cluster <30 chars
#   - body sentences cluster 80-200 chars
#   - boilerplate "By signing this agreement, the parties ..." can
#     run 400+ chars
_LEN_MIN: int = 20
_LEN_LOW: int = 40
_LEN_HIGH: int = 200
_LEN_MAX: int = 400

# Position-score: signature blocks (last ~10% of document) get a boost
# on top of the mid-body floor; the front of the document gets no
# blanket bonus (the heading-proximity signal handles "first paragraph
# after the title" cases).
_BACK_RATIO: float = 0.1
_POSITION_FLOOR: float = 0.25
_HEADING_PROXIMITY_BUMP: float = 0.3


def _compute_salience(
    *,
    text: str,
    matches: tuple[EntityMatch, ...],
    position_index: int,
    position_total: int,
    heading_proximate: bool,
) -> float:
    """Combine count + position + length signals into a [0, 1] salience score.

    See :class:`SentenceEntityHit` for the formula's rationale. Pure
    function of the inputs — no side effects, deterministic, cheap
    (constant time given match count is bounded by sentence length).
    """
    if not matches:
        return 0.0

    # 1. Count signal. Distinct typed values is a better proxy than raw
    # match count: "$5,000 and $5,000" should not double-score over
    # "$5,000". Fall back to the matched text when value isn't hashable
    # (the typed extractors return dataclasses, datetime, Decimal —
    # all hashable in practice).
    distinct: set[Any] = set()
    for m in matches:
        try:
            distinct.add(m.value if m.value is not None else m.text)
        except TypeError:
            # Unhashable typed value — degrade to text-level dedup.
            distinct.add(m.text)
    count_score = min(1.0, len(distinct) / 3.0)

    # 2. Position signal. The back-of-document ramp (signature block)
    # plus a heading-proximity bump. We intentionally do NOT add a
    # "front of document" boost — the very first sentence is often
    # a WHEREAS preamble or year-of boilerplate, not load-bearing.
    # Heading-proximity is the structural signal for "front matter
    # that matters" (first paragraph after the title carries the
    # "as of ___" date in most NDAs).
    if position_total <= 1:
        position_score = 1.0
    else:
        pos = position_index / position_total
        back = max(0.0, (pos - (1.0 - _BACK_RATIO)) / _BACK_RATIO) if _BACK_RATIO > 0 else 0.0
        position_score = max(_POSITION_FLOOR, back)
    if heading_proximate:
        position_score = min(1.0, position_score + _HEADING_PROXIMITY_BUMP)

    # 3. Length signal — triangular peak. Very short or very long
    # sentences are penalised; the sweet spot is the legal-prose band.
    n = len(text)
    if n <= _LEN_MIN or n >= _LEN_MAX:
        length_score = 0.0
    elif _LEN_LOW <= n <= _LEN_HIGH:
        length_score = 1.0
    elif n < _LEN_LOW:
        # Ramp up from 0 (at LEN_MIN) to 1 (at LEN_LOW).
        length_score = (n - _LEN_MIN) / (_LEN_LOW - _LEN_MIN)
    else:
        # Ramp down from 1 (at LEN_HIGH) to 0 (at LEN_MAX).
        length_score = (_LEN_MAX - n) / (_LEN_MAX - _LEN_HIGH)

    score = _W_COUNT * count_score + _W_POSITION * position_score + _W_LENGTH * length_score
    # Numerical safety: clamp into [0, 1]. With sum-of-weights == 1
    # and each component in [0, 1], score is already in [0, 1] modulo
    # float drift; clamp keeps the public contract strict.
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _heading_proximate_paragraph_refs(view: DocumentView) -> frozenset[str]:
    """Block-refs for paragraphs that sit directly after a heading.

    "Directly after" = the first paragraph encountered after a heading
    block within the document's flat body order. These tend to be
    high-salience (e.g. "This Agreement is entered into as of ___" as
    the first paragraph after the title). Cheap to compute on demand
    and cached implicitly by the view's paragraph cache.
    """
    from kaos_content.model.blocks import Heading

    proximate: set[str] = set()
    saw_heading = False
    # Walk the flat body; only first-level paragraphs get tagged. The
    # rare nested paragraph (inside a BlockQuote etc.) is skipped — the
    # heading-proximity signal is meant for top-level structure.
    for i, block in enumerate(view.document.body):
        if isinstance(block, Heading):
            saw_heading = True
            continue
        if block.node_type == "paragraph":
            if saw_heading:
                proximate.add(f"#/body/{i}")
            saw_heading = False
    return frozenset(proximate)


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
