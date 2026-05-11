"""Deterministic, no-LLM document summarization for corpus-scale triage.

The public surface is :func:`build_document_summary`. Given a
:class:`~kaos_content.model.document.ContentDocument` (or a
:class:`~kaos_content.views.DocumentView`), returns a populated
:class:`~kaos_content.model.summary.DocumentSummary`.

Performance target: <100 ms for a 50-page document, <1 s for a
500-page contract on a single core. We meet this by:

- Streaming through the body via the canonical ``iter_paragraph_units``
  (no full-tree walks).
- Tokenising once via ``kaos_nlp_core.tokenizer.Tokenizer`` (Rust-backed).
- Counting n-grams in a single pass with a plain ``Counter`` — no
  intermediate lists, no second sort beyond the top-k selection.
- Skipping stopwords inline rather than post-filtering.

The summary is *deterministic*: same input → same output. This lets
us safely cache it on disk (via the artifact store) and detect
drift across schema versions.

Example::

    from kaos_content.summarize import build_document_summary
    summary = build_document_summary(content_document)
    print(summary.head_tokens[:200])
    for ng in summary.top_ngrams[:5]:
        print(f"  {ng.count:>3}  {ng.ngram}")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from kaos_content.model.summary import DocumentSummary, NGramFrequency
from kaos_content.summarize.stopwords import ENGLISH_STOPWORDS
from kaos_content.units import iter_paragraph_units, iter_sentence_units

if TYPE_CHECKING:
    from collections.abc import Iterable

    from kaos_content.model.document import ContentDocument
    from kaos_content.views import DocumentView

    DocOrView = ContentDocument | DocumentView


# Tunable defaults. These are bundled here so callers don't need to
# know the magic numbers; override per call if a workflow demands.
DEFAULT_HEAD_TOKEN_TARGET = 500
DEFAULT_TOP_K = 50
DEFAULT_BOTTOM_K = 50
DEFAULT_MIN_BOTTOM_COUNT = 2
DEFAULT_NGRAM_MAX = 3


def build_document_summary(
    document: DocOrView,
    *,
    head_token_target: int = DEFAULT_HEAD_TOKEN_TARGET,
    top_k: int = DEFAULT_TOP_K,
    bottom_k: int = DEFAULT_BOTTOM_K,
    min_bottom_count: int = DEFAULT_MIN_BOTTOM_COUNT,
    ngram_max: int = DEFAULT_NGRAM_MAX,
    stopwords: frozenset[str] | None = None,
    with_entities: bool = True,
) -> DocumentSummary:
    """Build a deterministic summary for ``document``.

    Args:
        document: A :class:`ContentDocument` or :class:`DocumentView`.
        head_token_target: Approximate token count for ``head_tokens``.
            We stop accumulating paragraph text once the token count
            crosses this threshold; the returned string contains
            whole paragraphs so the actual count may slightly exceed.
        top_k: Number of high-frequency n-grams to retain.
        bottom_k: Number of low-frequency-but-recurring n-grams to
            retain.
        min_bottom_count: Minimum occurrence count for an n-gram to
            qualify as "bottom" rather than a singleton. Singletons
            are noisy; we want rare *recurring* terms.
        ngram_max: Maximum n-gram length (1 → unigrams only, 2 → uni+bi,
            3 → uni+bi+tri). Cost is roughly linear in this value.
        stopwords: Override the default English stopword list. Useful
            for non-English documents or domain-specific tuning.
            ``None`` uses :data:`ENGLISH_STOPWORDS`.
        with_entities: When ``True`` (default), run the typed-entity
            extractors over each sentence to populate ``entity_counts``.
            Set to ``False`` for the pure lexical signal at lower cost.

    Returns:
        A populated :class:`DocumentSummary`.
    """
    from kaos_nlp_core._defaults import get_default_punkt_tokenizer
    from kaos_nlp_core.tokenizer import Tokenizer

    from kaos_content.serializers import serialize_text
    from kaos_content.views import DocumentView

    sw = stopwords if stopwords is not None else ENGLISH_STOPWORDS

    # Sentence-aware view so ``iter_sentence_units`` succeeds and the
    # downstream entity-count pass has sentence-level granularity.
    # Callers that pre-built their own view get to keep it — we don't
    # second-guess their wiring.
    if isinstance(document, DocumentView):
        view: DocumentView = document
        doc = view.document
    else:
        view = DocumentView(document, sentence_segmenter=get_default_punkt_tokenizer())
        doc = document
    paragraphs = iter_paragraph_units(view)
    sentences = iter_sentence_units(view)

    # ---- head_tokens ---------------------------------------------------
    # ``serialize_text`` walks headings + paragraphs + lists + table
    # cells uniformly. We tokenise that and truncate to the head-token
    # target. Result preserves capitalization and punctuation exactly
    # as the serializer produced it (slice by char offset on the
    # original string, not on the lowercased tokens).
    tokenizer = Tokenizer(lowercase=True)
    full_text = serialize_text(doc)
    head_spans = tokenizer.tokenize(full_text)
    if len(head_spans) > head_token_target:
        head_end = head_spans[head_token_target - 1].end
        head_tokens = full_text[:head_end].rstrip()
    else:
        head_tokens = full_text.strip()

    # ---- char_length / sentence_count / paragraph_count ---------------
    char_length = sum(len(p.text) for p in paragraphs)
    paragraph_count = len(paragraphs)
    sentence_count = len(sentences)

    # ---- n-gram counts -----------------------------------------------
    # Tokenise per text-bearing block (heading, paragraph, etc.) so
    # n-grams cannot span block boundaries. Without this, the
    # adjacent-block bigrams ("agreement agreement" where the trailing
    # word of a heading meets the leading word of the following
    # paragraph) inflate the noise floor and pollute top_ngrams.
    #
    # Counting is via the Rust-backed FrequencyVocabulary — single
    # FFI hop per insert rather than building intermediate Python
    # lists. Stopwords are dropped *before* n-gram windowing so we
    # don't propagate noise (cf. "of the" appearing 800x in any
    # English document).
    from kaos_nlp_core.structures import FrequencyVocabulary

    ngram_vocab = FrequencyVocabulary()
    for block_text in _iter_text_blocks(doc):
        tokens = tokenizer.tokenize(block_text)
        filtered = [t.text for t in tokens if _is_topical(t.text, sw)]
        for n in range(1, ngram_max + 1):
            for gram in _slide(filtered, n):
                ngram_vocab.insert(gram)
    # Materialize into a plain dict for the downstream selection
    # helpers (top/bottom k). FrequencyVocabulary has top_n() but no
    # bottom_n; centralising the conversion keeps both selections
    # using the same data shape.
    ngram_counts = dict(ngram_vocab.top_n(ngram_vocab.__len__()))

    top_ngrams = _select_top_ngrams(ngram_counts, top_k)
    bottom_ngrams = _select_bottom_ngrams(ngram_counts, bottom_k, min_bottom_count)

    # ---- entity_counts -----------------------------------------------
    entity_counts: dict[str, int] = {}
    if with_entities:
        entity_counts = _compute_entity_counts(sentences)

    return DocumentSummary(
        head_tokens=head_tokens,
        top_ngrams=top_ngrams,
        bottom_ngrams=bottom_ngrams,
        char_length=char_length,
        sentence_count=sentence_count,
        paragraph_count=paragraph_count,
        entity_counts=entity_counts,
        schema_version=1,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _iter_text_blocks(doc: ContentDocument) -> Iterable[str]:
    """Yield the plain-text of each text-bearing top-level block.

    Walks ``doc.body`` and yields the rendered text of each Heading,
    Paragraph, ListItem, CodeBlock, BlockQuote, and Table cell. Each
    yielded string is one "atom" for n-gram purposes — n-grams will
    not span across yields.

    Implementation note: we lean on ``serialize_text`` for the
    per-block rendering rather than re-implementing the AST → string
    walk. The per-block call cost is dominated by string assembly
    which is microseconds-scale.
    """
    from kaos_content.model.document import ContentDocument as _CD
    from kaos_content.serializers import serialize_text

    for block in doc.body:
        # Wrap the lone block in a fresh ContentDocument so the
        # serializer has a root. We don't care about provenance or
        # annotations on this throwaway doc.
        text = serialize_text(_CD(body=(block,))).strip()
        if text:
            yield text


def _is_topical(token: str, stopwords: frozenset[str]) -> bool:
    """Decide whether a token is topical signal vs noise.

    Drops: stopwords, single-character tokens (almost always
    punctuation noise after tokenisation), purely numeric tokens
    (numbers are captured by ``entity_counts``, not n-grams), and
    tokens that don't contain a letter (defensive against tokeniser
    edge cases like ``"---"``).
    """
    if not token:
        return False
    lower = token.lower()
    if lower in stopwords:
        return False
    if len(lower) < 2:
        return False
    return any(c.isalpha() for c in lower)


def _slide(tokens: list[str], n: int) -> Iterable[str]:
    """Yield space-joined n-grams of length ``n`` from ``tokens``."""
    if n <= 0 or n > len(tokens):
        return
    for i in range(len(tokens) - n + 1):
        yield " ".join(tokens[i : i + n])


def _select_top_ngrams(counts: dict[str, int], k: int) -> tuple[NGramFrequency, ...]:
    """Top-k by descending count, ties broken by ngram lexicographic order.

    Re-sorts by (count_desc, ngram_asc) for deterministic output —
    ``FrequencyVocabulary.top_n`` already sorts by count desc but
    its tie-break is insertion order, which would make summary
    output non-deterministic across runs.
    """
    if not counts or k <= 0:
        return ()
    candidates = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return tuple(NGramFrequency(ngram=ng, count=c) for ng, c in candidates[:k])


def _select_bottom_ngrams(
    counts: dict[str, int], k: int, min_count: int
) -> tuple[NGramFrequency, ...]:
    """Bottom-k recurring n-grams (count >= min_count), ascending count."""
    if not counts or k <= 0:
        return ()
    recurring = [(ng, c) for ng, c in counts.items() if c >= min_count]
    recurring.sort(key=lambda kv: (kv[1], kv[0]))
    return tuple(NGramFrequency(ngram=ng, count=c) for ng, c in recurring[:k])


def _compute_entity_counts(sentences) -> dict[str, int]:  # type: ignore[no-untyped-def]
    """Count *sentences* containing at least one match of each entity type.

    Sentence-level granularity (vs. raw match count) matches the
    Kelvin pattern: "show me sentences with dates" is the agent's
    actual unit of retrieval. A single sentence with 3 dates counts
    as one, not three.

    Entity extractors live in ``kaos-nlp-core.extract.alpha``. We
    keep the keys lower-snake-cased and stable so downstream
    consumers (entity_counts["money"]) don't need to know which
    extractor backed them.
    """
    from kaos_nlp_core.extract.alpha import (
        AlphaDateExtractor,
        AlphaDurationExtractor,
        AlphaMoneyExtractor,
        AlphaPercentExtractor,
    )

    extractors: dict[str, object] = {
        "dates": AlphaDateExtractor(),
        "money": AlphaMoneyExtractor(),
        "percents": AlphaPercentExtractor(),
        "durations": AlphaDurationExtractor(),
    }

    counts: dict[str, int] = dict.fromkeys(extractors, 0)
    for s in sentences:
        for name, extractor in extractors.items():
            # All Alpha extractors expose ``extract_spans(text) -> iterable``.
            try:
                matches = list(extractor.extract_spans(s.text))  # ty: ignore[unresolved-attribute]
            except Exception:
                # Defensive: extractor errors don't break summary
                # construction. The downside is silent miss; the
                # upside is no test passing here ever crashes a
                # summary build.
                continue
            if matches:
                counts[name] += 1
    return counts


__all__ = [
    "DEFAULT_BOTTOM_K",
    "DEFAULT_HEAD_TOKEN_TARGET",
    "DEFAULT_MIN_BOTTOM_COUNT",
    "DEFAULT_NGRAM_MAX",
    "DEFAULT_TOP_K",
    "build_document_summary",
]
