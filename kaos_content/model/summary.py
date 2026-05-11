"""Document summary value types.

A :class:`DocumentSummary` is a cheap, deterministic, zero-LLM
fingerprint of a :class:`~kaos_content.model.document.ContentDocument`.
It exists so corpus-scale workflows (10,000-document upload,
"narrow to the relevant 50" UX) can search and triage without
re-indexing the full body of every document on every query.

The shape is opinionated:

- ``head_tokens``: the first ~500 tokens verbatim. Captures the
  structural opening of the document — for legal contracts this is
  the title, parties, effective date, and recitals; for financial
  filings it's the cover page; for memos it's the To/From/Re block.
  This is what a human reads to decide "is this the document I'm
  looking for" in <10 seconds.
- ``top_ngrams``: the 50 most-frequent n-grams (1- to N-word) after
  stopword removal. Identifies *thematic* content — what the
  document is *about*. A merger agreement's top n-grams will
  include "merger consideration", "company stock", "effective time".
- ``bottom_ngrams``: the 50 least-frequent n-grams (appearing >= 2
  times) after stopword removal. Identifies *distinctive* content —
  what makes *this* document different. A licence agreement among
  100 NDAs will surface here as an outlier.
- ``entity_counts``: a small histogram of how many of each typed
  entity (dates, money amounts, percentages, durations, parties)
  the document contains. Composes with the entity-filter tools (K2)
  so an agent can triage "find the contracts with >10 money
  mentions" without re-scanning.

All fields are JSON-serialisable. The whole summary fits in ~1-2 KB
when serialised — three orders of magnitude smaller than a typical
50-page contract.

See ``docs/design/findings-entities-summary.md`` for the design
rationale and the corpus-scale triage workflow this enables.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class NGramFrequency(BaseModel):
    """One n-gram with its observed count in the document."""

    model_config = ConfigDict(frozen=True)

    ngram: str
    """The n-gram itself, lower-cased, space-joined (e.g. ``"effective date"``)."""

    count: int
    """Raw occurrence count in the document. Strictly positive."""


class DocumentSummary(BaseModel):
    """Cheap, deterministic, no-LLM summary for corpus-scale triage.

    Built by :func:`kaos_content.summarize.build_document_summary`.
    All fields are computable in O(document_size) at parse time
    without external services or model loads.

    The summary is *attached* to the :class:`ContentDocument` via the
    optional ``summary`` field, but it's a separately-constructed
    value — the AST does not own its computation, which lets older
    serialised documents (no ``summary`` field) round-trip cleanly.

    Schema version is included so future shape changes (e.g. adding
    an ``embedding`` field) can be detected and rebuilt lazily.
    """

    model_config = ConfigDict(frozen=True)

    head_tokens: str = ""
    """First ~500 tokens of the document verbatim. Captures the
    structural opening — contract type, parties, effective date,
    recitals."""

    top_ngrams: tuple[NGramFrequency, ...] = ()
    """Top n-grams (1- to N-word) by raw frequency after stopword
    removal. Identifies thematic content. Ordered by descending
    count; ties broken by lexicographic order on ``ngram``."""

    bottom_ngrams: tuple[NGramFrequency, ...] = ()
    """Rare-but-recurring n-grams (count >= 2) after stopword
    removal. Captures distinctive vocabulary — terms-of-art and
    unusual clauses that don't appear in most peer documents.
    Ordered by ascending count, then ngram."""

    char_length: int = 0
    """Total character count across all paragraph text."""

    sentence_count: int = 0
    """Number of non-empty sentences (per ``iter_sentence_units``)."""

    paragraph_count: int = 0
    """Number of non-empty paragraphs (per ``iter_paragraph_units``)."""

    entity_counts: dict[str, int] = Field(default_factory=dict)
    """Histogram of typed-entity occurrences. Keys are entity-type
    names (e.g. ``"dates"``, ``"money"``, ``"percents"``,
    ``"durations"``, ``"parties"``); values are the count of
    *sentences* containing at least one match of that type.

    Composes with the entity-filter tools (K2). Empty dict means
    extraction wasn't run or no matches were found — the builder
    accepts ``with_entities=False`` to skip the entity-extraction
    pass when only the lexical signal is needed.
    """

    schema_version: int = 1
    """Schema version of this summary value type. Bump on shape
    changes. Consumers should check this and rebuild summaries with
    older versions when convenient."""
