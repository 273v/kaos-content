"""Token-frequency primitives (K9).

Free functions that count tokens over a :class:`ContentDocument`,
:class:`DocumentView`, :class:`SectionView`, or
:class:`ParagraphUnit`. Kelvin populated these per-document /
per-section / per-sentence at load time and then ignored them; we
provide them as a clean utility that composes with K1's summary
builder (top n-grams) and K6's FindingsAgent (token-match selectors).

Design notes:

- **Rust-backed.** Both tokenisation AND counting go through
  :func:`kaos_nlp_core.vocabulary.token_frequency`, which uses the
  Rust ``FrequencyVocabulary`` for accumulation. We never round-trip
  individual tokens through Python — the FFI boundary is crossed once
  per call, not once per token. (See ``feedback_rust_first.md``.)
- **Why a free function, not a method.** ``DocumentView`` uses
  ``__slots__``; ``SectionView`` is frozen Pydantic. Adding
  ``@cached_property`` to either is awkward (slots need an explicit
  cache slot; frozen Pydantic needs ``computed_field`` + side
  storage). A free function is simpler and side-effect-free, and
  callers who need caching can stash the returned dict on whatever
  carrier they're using.
- **Tokenisation is lowercase.** Matches how K1's summary builder
  and the rest of the BM25 pipeline tokenise — frequencies computed
  here align with frequencies used downstream.
- **Stopwords are NOT removed.** This function returns the raw
  count, so callers can decide whether to filter. K1's summary
  builder filters at n-gram time; consumers that want the
  bag-of-words histogram (e.g. for visualisation) want the unfiltered
  view.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from kaos_content.model.document import ContentDocument
    from kaos_content.units import ParagraphUnit, SentenceUnit
    from kaos_content.views import DocumentView
    from kaos_content.views.models import SectionView


def document_token_frequency(doc_or_view: ContentDocument | DocumentView) -> dict[str, int]:
    """Lower-cased token-frequency histogram for the whole document.

    Accepts either a raw :class:`ContentDocument` or a pre-built
    :class:`DocumentView`. Returns a fresh dict — callers who want
    to mutate or share can hold the result; we don't cache because
    the cost is dominated by the lazy view construction (already
    cached on DocumentView).
    """
    from kaos_content.serializers import serialize_text
    from kaos_content.views import DocumentView

    doc = doc_or_view.document if isinstance(doc_or_view, DocumentView) else doc_or_view
    return _count_tokens(serialize_text(doc))


def section_token_frequency(section: SectionView) -> dict[str, int]:
    """Lower-cased token-frequency histogram for one section.

    Walks the section's ``blocks`` (direct content) only — does NOT
    descend into ``subsections``. Callers who want the recursive
    count must accumulate themselves over ``section.subsections``.
    The non-recursive default matches how
    ``kaos_content.units.iter_paragraph_units`` flattens — each
    paragraph counts once, in the section that owns it.
    """
    from kaos_content.model.document import ContentDocument
    from kaos_content.serializers import serialize_text

    # Wrap the section blocks in a throwaway document so the
    # serializer's per-block walk produces canonical text.
    text = serialize_text(ContentDocument(body=section.blocks))
    return _count_tokens(text)


def paragraph_token_frequency(unit: ParagraphUnit | SentenceUnit) -> dict[str, int]:
    """Lower-cased token-frequency histogram for one ParagraphUnit / SentenceUnit.

    Accepts either iterator unit type since both carry a ``text``
    attribute. Per-unit counts are rarely useful on their own but
    compose nicely when you want section- or chunk-level rollups
    (sum the per-paragraph dicts).
    """
    return _count_tokens(unit.text)


def _count_tokens(text: str) -> dict[str, int]:
    """Tokenize + count via the Rust-backed
    :func:`kaos_nlp_core.vocabulary.token_frequency`.

    Single FFI boundary crossing per call; counting happens in the
    Rust ``FrequencyVocabulary``, not Python. Centralised so all
    four public entry points produce identical results for identical
    inputs.
    """
    if not text:
        return {}
    from kaos_nlp_core.vocabulary import token_frequency

    result = token_frequency(text, lowercase=True)
    return {tc.text: tc.count for tc in result.counts}


__all__ = [
    "document_token_frequency",
    "paragraph_token_frequency",
    "section_token_frequency",
]
