"""Corpus Protocol — formal contract for AST-grounded passage iterables.

Anchors the WS-3 "corpus abstraction" from
``docs/design/fundamentals-roadmap.md`` §WS-3 and the audit at
``docs/design/corpus-actual-state.md``. The audit found that
``kaos_ml_core.Corpus`` already implements this shape under the names
``__iter__ / __len__ / unit`` and that ``kaos_content.SearchableDocument``
also exposes ``.units``. This module adds a formal
``@runtime_checkable typing.Protocol`` so consumers (RAG, ResearchAgent,
future CorpusIndex) can declare a real isinstance contract instead of
the string-name ``_is_corpus`` duck-type currently in
``kaos_llm_core.programs.rag._is_corpus``.

The Protocol declarations here are **pure contracts** — no implementation
lives in kaos-content beyond the tiny ``ContentDocumentCorpus`` wrapper
that delegates to ``iter_paragraph_units``. The canonical heavyweight
implementation stays in ``kaos_ml_core.Corpus``, which now satisfies
this Protocol via alias methods (see ``kaos_ml_core.corpus``).

## Design notes

- ``Passage`` is a **structural Protocol**, not a class. The concrete
  passage types are ``kaos_ml_core.CorpusUnit`` (full-featured, with
  ``doc_uri``) and ``kaos_content.units.ParagraphUnit`` (lightweight,
  per-document). Both satisfy the Protocol's required attributes
  (``row``, ``text``, ``block_ref``, ``page``, ``section_ref``,
  ``section_title``). Consumers that need ``doc_uri`` check for it with
  ``getattr``; it is optional at the Protocol level.
- ``Corpus.size`` is a property, not a method, so ``len(corpus)``
  implementations can fan out to both. The Protocol exposes ``size``
  because a Protocol cannot declare ``__len__`` in a way that satisfies
  isinstance() reliably across Python versions; concrete impls are
  expected to provide both.
- Additions to the Protocol require updating this docstring AND the
  alias methods on every known implementation (``kaos_ml_core.Corpus``,
  ``ContentDocumentCorpus``) in the same PR.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from kaos_content.model.document import ContentDocument


@runtime_checkable
class Passage(Protocol):
    """Structural contract for one unit of text inside a ``Corpus``.

    Implementations: ``kaos_ml_core.CorpusUnit`` (canonical),
    ``kaos_content.units.ParagraphUnit``, ``kaos_content.units.SentenceUnit``.

    All implementations are frozen dataclasses — Passage is read-only by
    contract.
    """

    row: int
    """Dense row index into the containing Corpus (0..size-1)."""

    text: str
    """The passage text."""

    block_ref: str
    """JSON pointer into the source AST (e.g. ``#/body/12``)."""

    doc_uri: str
    """Source document URI for the passage."""

    page: int | None
    """1-indexed source page, or None."""

    section_ref: str | None
    """Containing section ref, or None."""

    section_title: str | None
    """Resolved heading text of the containing section, or None."""


@runtime_checkable
class Corpus(Protocol):
    """Structural contract for an AST-grounded passage iterable.

    Implementations:
    - ``kaos_ml_core.Corpus`` — canonical heavyweight impl with embedding
      cache, retriever factory, bidirectional row↔block_ref maps.
    - ``kaos_content.corpus.ContentDocumentCorpus`` — thin wrapper for
      one or more ``ContentDocument`` instances, delegates to
      ``iter_paragraph_units``.

    Consumers:
    - ``kaos_llm_core.programs.rag.RAG`` uses ``isinstance(obj, Corpus)``
      (replacing the legacy ``_is_corpus`` duck-type).
    - Future ``CorpusIndex`` (WS-3.4) bundles a Corpus + retrievers +
      VFS-persistable manifest.
    - ``kaos_agents.patterns.research.ResearchAgent`` (WS-3.6) accepts a
      Corpus as an explicit ctor parameter.
    """

    def iter_passages(self) -> Iterator[Passage]:
        """Yield every passage in row order, one-shot or repeatable.

        Implementations MAY be single-shot (generators). Consumers that
        need to iterate multiple times should materialize via
        ``tuple(corpus.iter_passages())``.
        """
        ...

    def get_passage(self, row: int) -> Passage:
        """Return the passage at the given dense row index.

        Raises ``IndexError`` / ``KeyError`` if the row is out of range;
        concrete impls choose the exception type.
        """
        ...

    @property
    def size(self) -> int:
        """Number of passages in the corpus."""
        ...


@dataclass(frozen=True, slots=True)
class ContentPassage:
    """Passage yielded by :class:`ContentDocumentCorpus`.

    Mirrors :class:`kaos_ml_core.CorpusUnit`'s shape (carries ``doc_uri``)
    so downstream consumers that dispatch on ``unit.doc_uri + block_ref``
    (e.g. :func:`kaos_nlp_core.retrieval.protocol.corpus_unit_passage_uri`)
    work without a special case. Satisfies the :class:`Passage` Protocol.
    """

    row: int
    text: str
    block_ref: str
    doc_uri: str
    page: int | None
    section_ref: str | None
    section_title: str | None
    confidence: float | None = None
    """N6: source-level extraction confidence (e.g. Tesseract OCR
    confidence on scanned PDFs). Threaded through from the underlying
    block's Provenance.confidence. ``None`` for born-digital text or
    when the extractor didn't report a score."""


class ContentDocumentCorpus:
    """Thin ``Corpus`` over one or more ``ContentDocument`` instances.

    Delegates to :func:`kaos_content.units.iter_paragraph_units` so
    passage enumeration matches every other kaos-content consumer
    (``search``, ``chunking``). Row indices are dense across the
    concatenation of per-document passage lists in the order docs are
    passed to the constructor.

    Each passage carries a ``doc_uri`` drawn from
    ``document.metadata.source.uri`` when present, or from the parallel
    ``doc_uris`` constructor argument, or ``doc:anon-<index>`` as a last
    resort. Downstream consumers that inspect ``unit.doc_uri`` (e.g.
    ``kaos_nlp_core.retrieval.protocol.corpus_unit_passage_uri``) work
    without a special case.

    This is the lightweight alternative to ``kaos_ml_core.Corpus`` for
    callers that do NOT need embedding caches or retriever factories
    (e.g. the WS-1 grounding-calibration harness can adopt this without
    pulling in kaos-ml-core).

    Example::

        from kaos_content.corpus import ContentDocumentCorpus

        corpus = ContentDocumentCorpus([doc_a, doc_b])
        assert corpus.size == sum(len(iter_paragraph_units(d)) for d in (doc_a, doc_b))
    """

    __slots__ = ("_passages",)

    def __init__(
        self,
        documents: Sequence[ContentDocument],
        *,
        doc_uris: Sequence[str] | None = None,
    ) -> None:
        from kaos_content.units import iter_paragraph_units

        if doc_uris is not None and len(doc_uris) != len(documents):
            msg = (
                f"doc_uris has length {len(doc_uris)}, documents has "
                f"length {len(documents)}; they must match. "
                "Fix: pass one doc_uri per document, or omit doc_uris and "
                "rely on document.metadata.source.uri."
            )
            raise ValueError(msg)

        passages: list[ContentPassage] = []
        row = 0
        for i, doc in enumerate(documents):
            if doc_uris is not None:
                uri = doc_uris[i]
            elif doc.metadata.source is not None:
                uri = doc.metadata.source.uri
            else:
                uri = f"doc:anon-{i}"
            for unit in iter_paragraph_units(doc):
                passages.append(
                    ContentPassage(
                        row=row,
                        text=unit.text,
                        block_ref=unit.block_ref,
                        doc_uri=uri,
                        page=unit.page,
                        section_ref=unit.section_ref,
                        section_title=unit.section_title,
                        confidence=unit.confidence,
                    )
                )
                row += 1
        self._passages: tuple[ContentPassage, ...] = tuple(passages)

    def iter_passages(self) -> Iterator[Passage]:
        return iter(self._passages)  # ty: ignore[invalid-return-type]

    def get_passage(self, row: int) -> Passage:
        if row < 0 or row >= len(self._passages):
            msg = f"row index {row} out of range [0, {len(self._passages)})"
            raise IndexError(msg)
        return self._passages[row]  # ty: ignore[invalid-return-type]

    @property
    def size(self) -> int:
        return len(self._passages)

    def __len__(self) -> int:
        return len(self._passages)

    def __iter__(self) -> Iterator[Passage]:
        return iter(self._passages)  # ty: ignore[invalid-return-type]


__all__ = ["ContentDocumentCorpus", "ContentPassage", "Corpus", "Passage"]
