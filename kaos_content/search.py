"""Search within a ContentDocument using BM25, embeddings, or hybrid retrieval.

Paragraph-level and sentence-level search grounded to the kaos-content
AST (block_refs via DocumentView). Uses kaos-nlp-core BM25 when
available, falls back to simple term frequency scoring.

This is the canonical search implementation for all KAOS extraction
modules (kaos-pdf, kaos-web, kaos-office). All produce ContentDocument
and use this shared search.

**Retrieval modes** (P6 hybrid retrieval):

* ``"bm25"`` (default): pure BM25 via kaos-nlp-core (or TF fallback).
* ``"embeddings"``: pure dense retrieval via ``kaos-nlp-transformers``.
  The embedding model is selectable via the ``model_id`` argument; it
  defaults to ``KaosNLPTransformersSettings.default_model``.
* ``"hybrid"``: BM25 picks ``rerank_candidate_k`` candidates, embedding
  cosine similarity reranks them, return top ``rerank_top_k``.

Both ``"embeddings"`` and ``"hybrid"`` require the optional
``kaos-nlp-transformers`` package. They raise ``ImportError`` with a
clear install hint when it is missing. ``"bm25"`` keeps working without
that dep.

**AST address preservation**: search results always carry ``block_ref``
(JSON pointer into the AST), ``page`` (from provenance), and
``section_ref`` (containing heading). When using kaos-nlp-core, these
are threaded through ``DocumentCollection.external_id`` (block_ref) and
``metadata`` (page, section_ref) so that corpus-wide BM25/IDF is
computed correctly across a single index, not per-paragraph.

Usage::

    from kaos_content.search import search_document

    doc = extract_pdf("report.pdf")  # or html_to_document(html)
    results = search_document(doc, "fair use", level="sentence")
    # Hybrid (BM25 → embedding rerank):
    results = search_document(doc, "non-compete", retrieval="hybrid")
"""

from __future__ import annotations

import functools
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from kaos_content.model.document import ContentDocument
from kaos_content.model.tabular import TabularDocument
from kaos_content.views import DocumentView

if TYPE_CHECKING:
    from kaos_nlp_core.retrieval.protocol import RetrievalResult
    from kaos_nlp_core.search import SearchHit

    from kaos_content.indexing import SearchableDocument


RetrievalMode = Literal["bm25", "embeddings", "hybrid"]
"""Retrieval mode for :func:`search_document` and :class:`SearchableDocument`.

* ``"bm25"`` — lexical BM25 via kaos-nlp-core (TF fallback when missing).
* ``"embeddings"`` — dense retrieval via kaos-nlp-transformers.
* ``"hybrid"`` — BM25 candidate set, embedding rerank.
"""

# Default candidate pool size for hybrid retrieval. BM25 returns this many
# top hits, the embedding model reranks them, and we trim to ``rerank_top_k``.
# 50 is the sweet spot in legal-document benchmarks: large enough that the
# semantic reranker can recover from BM25's lexical blind spots, small enough
# that embedding cost stays bounded (~50 * 1ms ≈ 50ms on bge-small-en-v1.5).
HYBRID_DEFAULT_CANDIDATE_K = 50


@dataclass(frozen=True, slots=True)
class SearchResult:
    """A single search result with context."""

    text: str
    """Matching text (may be truncated to preview_length)."""

    score: float
    """Relevance score (higher is better)."""

    block_ref: str
    """JSON pointer ref of the containing block (e.g., '#/body/5')."""

    page: int | None
    """1-indexed page number from provenance, or None."""

    section_ref: str | None
    """Heading ref of the containing section, or None."""

    section_title: str | None
    """Text of the section heading, or None."""

    char_start: int | None = None
    """Character offset start within the containing paragraph (sentence-level only)."""

    char_end: int | None = None
    """Character offset end within the containing paragraph (sentence-level only)."""

    heading_path: tuple[str, ...] = ()
    """Ancestor heading texts (shallowest-first), excluding ``section_title``.

    For a hit inside section ``"Section 1.1"`` of chapter ``"Chapter 1"``,
    this is ``("Chapter 1",)`` — ``section_title`` continues to carry the
    immediate section. Empty when the hit is in document preamble or when
    the section has no ancestors."""

    doc_index: int | None = None
    """Index into ``SearchableCorpus.documents`` for the source document.

    Populated only by :class:`SearchableCorpus.search`; ``None`` from
    :func:`search_document` and :class:`SearchableDocument.search` (per-
    document searches don't have a corpus axis). Backwards-compatible
    additive field — existing single-doc callers see no behavioral
    change."""

    doc_uri: str | None = None
    """URI of the source document (e.g. ``metadata.source.uri``).

    Populated only by :class:`SearchableCorpus.search`. ``None`` from
    single-document searches. Useful when downstream tooling wants the
    human-readable / URL-shaped handle without round-tripping through
    ``SearchableCorpus.documents[doc_index]``."""


@dataclass(frozen=True, slots=True)
class SearchResults:
    """Search results with pagination metadata."""

    results: list[SearchResult]
    """Matching results, ordered by score descending."""

    total_matches: int
    """Total number of segments that matched (before top_k truncation)."""

    has_more: bool
    """True if total_matches > len(results)."""

    query: str
    """The original query string."""


def search_document(
    document: ContentDocument,
    query: str,
    *,
    top_k: int = 10,
    preview_length: int = 200,
    level: Literal["paragraph", "sentence"] = "paragraph",
    retrieval: RetrievalMode = "bm25",
    rerank_top_k: int = 10,
    rerank_candidate_k: int = HYBRID_DEFAULT_CANDIDATE_K,
    model_id: str | None = None,
) -> SearchResults:
    """Search within a ContentDocument by text query.

    Uses BM25 via kaos-nlp-core when available, falls back to simple
    term frequency scoring. Searches per-paragraph via DocumentView
    so results carry proper block_refs from the AST.

    Args:
        document: The ContentDocument to search.
        query: Search query text (must not be empty).
        top_k: Maximum number of results to return (used for ``"bm25"``
            and ``"embeddings"``). Hybrid mode uses ``rerank_top_k``.
        preview_length: Maximum characters in result text. 0 = full text.
        level: Search granularity — ``"paragraph"`` or ``"sentence"``.
            Sentence-level search requires kaos-nlp-core.
        retrieval: Retrieval mode — ``"bm25"`` (default, lexical),
            ``"embeddings"`` (dense, semantic), or ``"hybrid"`` (BM25
            candidate set + embedding rerank).
        rerank_top_k: How many results to keep after embedding rerank
            in ``"hybrid"`` mode. Ignored for non-hybrid modes.
        rerank_candidate_k: How many BM25 candidates to feed into the
            embedding reranker for ``"hybrid"`` mode. Larger values
            improve recall at the cost of more embedding inference.
        model_id: HF Hub model id for the embedding model used by
            ``"embeddings"`` and ``"hybrid"`` modes (e.g.
            ``"intfloat/e5-large-v2"``). ``None`` selects the
            kaos-nlp-transformers default
            (``KaosNLPTransformersSettings.default_model``). Ignored
            for ``"bm25"``.

    Returns:
        SearchResults with matching results and pagination metadata.

    Raises:
        ValueError: If query is empty.
        ImportError: If level="sentence" and kaos-nlp-core is not installed,
            or if ``retrieval in {"embeddings", "hybrid"}`` and the
            ``kaos-nlp-transformers`` extra is missing. The error message
            includes the install hint for ``pip install kaos-nlp-transformers``.
    """
    if not query or not query.strip():
        msg = "Query must not be empty"
        raise ValueError(msg)

    if retrieval not in ("bm25", "embeddings", "hybrid"):
        msg = (
            f"Unknown retrieval mode {retrieval!r}. "
            "Fix: choose one of 'bm25' (lexical, default), 'embeddings' "
            "(dense), or 'hybrid' (BM25 + embedding rerank). "
            "Alternative: omit the argument to use the default 'bm25'."
        )
        raise ValueError(msg)

    if retrieval in ("embeddings", "hybrid"):
        # Eager-check the optional dep so we fail fast with a clear hint
        # rather than deep inside the BM25 path or numpy import.
        _ensure_transformers_available()

    if retrieval == "embeddings":
        return _search_embeddings(
            document,
            query,
            top_k=top_k,
            preview_length=preview_length,
            level=level,
            model_id=model_id,
        )

    if retrieval == "hybrid":
        return _search_hybrid(
            document,
            query,
            top_k=rerank_top_k,
            preview_length=preview_length,
            level=level,
            candidate_k=rerank_candidate_k,
            model_id=model_id,
        )

    # retrieval == "bm25" — preserve original behavior exactly.
    try:
        from kaos_nlp_core.search import Searcher  # noqa: F401

        return _search_bm25(
            document, query, top_k=top_k, preview_length=preview_length, level=level
        )
    except ImportError:
        pass

    if level == "sentence":
        msg = "Sentence-level search requires kaos-nlp-core."
        raise ImportError(msg)
    return _search_tf(document, query, top_k=top_k, preview_length=preview_length)


def _ensure_transformers_available() -> None:
    """Raise a clean ImportError if kaos-nlp-transformers is not installed.

    Used by retrieval modes that depend on the dense embedding backend.
    """
    try:
        import kaos_nlp_transformers  # type: ignore[import-not-found]  # noqa: F401
    except ImportError as exc:
        msg = (
            "Embedding-based retrieval requires the optional "
            "'kaos-nlp-transformers' package. "
            "Fix: `pip install kaos-content[transformers]` (or "
            "`uv add kaos-content[transformers]`, or "
            "`pip install kaos-nlp-transformers>=0.2.0a2` directly). "
            "Alternative: use retrieval='bm25' for lexical-only search "
            "with no extra dependencies."
        )
        raise ImportError(msg) from exc


# Module-level model cache. Loading a kaos-nlp-transformers ``EmbeddingModel``
# is cheap on the second call (the Rust ``ort`` backend caches the
# session) but the Python-side wrapper still re-resolves the registry,
# probes capabilities, and hits ``hf-hub`` for revision pinning each
# time. KNT-601 audit finding H-3: a single hybrid query previously
# called ``load()`` four times. lru_cache(maxsize=4) keyed on model_id
# (None ⇒ default) returns the same wrapper across the per-text /
# per-query call sites.
@functools.lru_cache(maxsize=4)
def _get_embedding_model(model_id: str | None) -> Any:
    """Cached ``EmbeddingModel`` keyed on ``model_id``.

    ``None`` resolves to the kaos-nlp-transformers default
    (``KaosNLPTransformersSettings.default_model``). Caller is
    responsible for guarding the import — ``_ensure_transformers_available``
    has already run by the time we get here.
    """
    from kaos_nlp_transformers import EmbeddingModel

    return EmbeddingModel.load(model_id=model_id) if model_id else EmbeddingModel.load()


# ─── BM25 search via kaos-nlp-core ──────────────────────────────────────────


def _search_bm25(
    document: ContentDocument,
    query: str,
    *,
    top_k: int,
    preview_length: int,
    level: Literal["paragraph", "sentence"],
) -> SearchResults:
    """BM25 search via kaos-nlp-core, grounded to AST block_refs.

    Builds a single InvertedIndex over all paragraphs (or sentences) so
    IDF is corpus-wide. AST addresses (block_ref, page, section_ref) are
    threaded through DocumentCollection.external_id and metadata, then
    mapped back onto SearchResult after retrieval.
    """
    if level == "sentence":
        return _search_bm25_sentences(document, query, top_k=top_k, preview_length=preview_length)
    return _search_bm25_paragraphs(document, query, top_k=top_k, preview_length=preview_length)


def _paragraphs_to_records(
    view: DocumentView,
) -> tuple[list[dict[str, Any]], dict[str, str | None]]:
    """Convert paragraph views to search records, preserving AST addresses.

    Delegates to ``kaos_content.units.iter_paragraph_units`` for the
    enumeration so BM25 search and ``kaos_ml_core.Corpus`` share exactly
    one source of truth for "what counts as a paragraph row." The dict
    record shape matches what ``Searcher.from_documents`` expects via
    ``id_field`` / ``external_id_field`` / ``metadata_fields``.

    Returns:
        (records, section_titles) — records carry block_ref as external_id
        and page/section_ref as metadata; section_titles is a cache of
        resolved heading text keyed by section_ref. Note: row indices in
        the returned records are dense (0..N-1, no gaps for skipped empty
        paragraphs) — Searcher only requires uniqueness, not contiguity
        with view.paragraphs positions.
    """
    from kaos_content.units import iter_paragraph_units

    units = iter_paragraph_units(view)

    section_titles: dict[str, str | None] = {}
    records: list[dict[str, Any]] = []
    for u in units:
        if u.section_ref is not None and u.section_ref not in section_titles:
            section_titles[u.section_ref] = u.section_title
        records.append(
            {
                "id": u.row,
                "text": u.text,
                "block_ref": u.block_ref,
                "page": u.page,
                "section_ref": u.section_ref,
            }
        )

    return records, section_titles


def _results_to_search_results(
    hits: list[SearchHit],
    section_titles: dict[str, str | None],
    preview_length: int,
    query: str,
    top_k: int,
    heading_paths: dict[str, tuple[str, ...]] | None = None,
) -> SearchResults:
    """Convert Searcher results back to SearchResults with AST addresses."""
    scored: list[SearchResult] = []
    for hit in hits:
        text = hit.text
        if preview_length > 0 and len(text) > preview_length:
            text = text[:preview_length] + "..."

        sec_ref = hit.metadata.get("section_ref")
        path: tuple[str, ...] = ()
        if heading_paths is not None and sec_ref is not None:
            path = heading_paths.get(sec_ref, ())
        scored.append(
            SearchResult(
                text=text,
                score=hit.score,
                block_ref=hit.external_id or "",
                page=hit.metadata.get("page"),
                section_ref=sec_ref,
                section_title=section_titles.get(sec_ref) if sec_ref else None,
                char_start=hit.metadata.get("char_start"),
                char_end=hit.metadata.get("char_end"),
                heading_path=path,
            )
        )

    total = len(scored)
    results = scored[:top_k]
    return SearchResults(
        results=results,
        total_matches=total,
        has_more=total > top_k,
        query=query,
    )


def _heading_path_index(view: DocumentView) -> dict[str, tuple[str, ...]]:
    """Build a {section_ref → ancestor_heading_texts} index from the
    section tree. Ancestors are ordered shallowest-first and exclude
    the section itself. Preamble (no heading_ref) is omitted.
    """
    out: dict[str, tuple[str, ...]] = {}

    def _walk(sections: tuple[Any, ...], ancestors: tuple[str, ...]) -> None:
        for sec in sections:
            if sec.heading_ref is None:
                # Preamble — recurse without recording (subsections inherit
                # the empty ancestor chain).
                _walk(sec.subsections, ancestors)
                continue
            out[sec.heading_ref] = ancestors
            new_ancestors = (*ancestors, sec.heading_text)
            _walk(sec.subsections, new_ancestors)

    _walk(view.sections, ())
    return out


def _search_bm25_paragraphs(
    document: ContentDocument,
    query: str,
    *,
    top_k: int,
    preview_length: int,
) -> SearchResults:
    """BM25 paragraph search with corpus-wide IDF, AST-grounded.

    Builds one index over all paragraphs, queries once. AST block_refs
    round-trip through external_id; page/section_ref through metadata.
    """
    from kaos_nlp_core.search import Searcher

    view = DocumentView(document)
    records, section_titles = _paragraphs_to_records(view)

    if not records:
        return SearchResults(results=[], total_matches=0, has_more=False, query=query)

    searcher = Searcher.from_documents(
        records,
        external_id_field="block_ref",
        metadata_fields=["page", "section_ref"],
    )
    # Retrieve more than top_k so total_matches is accurate
    hits = searcher.search(query, top_k=len(records))
    heading_paths = _heading_path_index(view)
    return _results_to_search_results(
        hits, section_titles, preview_length, query, top_k, heading_paths
    )


def _search_bm25_sentences(
    document: ContentDocument,
    query: str,
    *,
    top_k: int,
    preview_length: int,
) -> SearchResults:
    """BM25 sentence search with corpus-wide IDF, AST-grounded.

    Segments each paragraph into sentences, builds one index over all
    sentences across the document. Each sentence carries the block_ref
    of its containing paragraph as external_id.
    """
    from kaos_nlp_core.search import Searcher
    from kaos_nlp_core.segmentation import segment_sentences

    view = DocumentView(document)
    records: list[dict[str, Any]] = []
    section_titles: dict[str, str | None] = {}
    sent_id = 0

    for pv in view.paragraphs:
        if not pv.text or not pv.text.strip():
            continue

        if pv.section_ref is not None and pv.section_ref not in section_titles:
            section_titles[pv.section_ref] = _resolve_section(view, pv.section_ref)

        sentences = segment_sentences(pv.text)
        for sent in sentences:
            if not sent.text.strip():
                continue
            records.append(
                {
                    "id": sent_id,
                    "text": sent.text,
                    "block_ref": pv.block_ref,
                    "page": pv.page,
                    "section_ref": pv.section_ref,
                    "char_start": sent.start,
                    "char_end": sent.end,
                }
            )
            sent_id += 1

    if not records:
        return SearchResults(results=[], total_matches=0, has_more=False, query=query)

    searcher = Searcher.from_documents(
        records,
        external_id_field="block_ref",
        metadata_fields=["page", "section_ref", "char_start", "char_end"],
    )
    hits = searcher.search(query, top_k=len(records))
    heading_paths = _heading_path_index(view)
    return _results_to_search_results(
        hits, section_titles, preview_length, query, top_k, heading_paths
    )


# ─── Embedding / hybrid search ──────────────────────────────────────────────


def _build_search_records(
    document: ContentDocument,
    *,
    level: Literal["paragraph", "sentence"],
) -> tuple[
    DocumentView,
    list[dict[str, Any]],
    dict[str, str | None],
    dict[str, tuple[str, ...]],
]:
    """Build the flat record list for either retrieval level.

    Returns ``(view, records, section_titles, heading_paths)``. Records
    follow the same shape used by ``_search_bm25_*`` so we can feed them
    to either the lexical Searcher or the embedding model without a
    second enumeration of the AST.
    """
    view = DocumentView(document)
    records: list[dict[str, Any]] = []
    section_titles: dict[str, str | None] = {}

    if level == "sentence":
        from kaos_nlp_core.segmentation import segment_sentences

        sent_id = 0
        for pv in view.paragraphs:
            if not pv.text or not pv.text.strip():
                continue
            if pv.section_ref is not None and pv.section_ref not in section_titles:
                section_titles[pv.section_ref] = _resolve_section(view, pv.section_ref)
            for sent in segment_sentences(pv.text):
                if not sent.text.strip():
                    continue
                records.append(
                    {
                        "id": sent_id,
                        "text": sent.text,
                        "block_ref": pv.block_ref,
                        "page": pv.page,
                        "section_ref": pv.section_ref,
                        "char_start": sent.start,
                        "char_end": sent.end,
                    }
                )
                sent_id += 1
    else:
        from kaos_content.units import iter_paragraph_units

        for u in iter_paragraph_units(view):
            if u.section_ref is not None and u.section_ref not in section_titles:
                section_titles[u.section_ref] = u.section_title
            records.append(
                {
                    "id": u.row,
                    "text": u.text,
                    "block_ref": u.block_ref,
                    "page": u.page,
                    "section_ref": u.section_ref,
                }
            )

    heading_paths = _heading_path_index(view)
    return view, records, section_titles, heading_paths


def _records_to_search_results(
    *,
    records: list[dict[str, Any]],
    indices: list[int],
    scores: list[float],
    section_titles: dict[Any, str | None],
    heading_paths: dict[Any, tuple[str, ...]],
    preview_length: int,
    query: str,
    top_k: int,
    doc_uris: list[str] | None = None,
) -> SearchResults:
    """Materialize SearchResults from row indices + scores.

    Shared between the embeddings and hybrid paths — both compute a
    ranked list of record indices and need the same provenance fan-out.
    Records may carry an optional ``doc_index`` field (populated by
    :class:`SearchableCorpus`); when present, the result's ``doc_index``
    + ``doc_uri`` fields are filled. ``section_titles`` and
    ``heading_paths`` are keyed by ``section_ref`` for single-document
    callers and by ``(doc_index, section_ref)`` for the corpus path —
    the dict's actual key shape is opaque to this helper.
    """
    scored: list[SearchResult] = []
    for idx, score in zip(indices, scores, strict=True):
        rec = records[idx]
        text = rec["text"]
        if preview_length > 0 and len(text) > preview_length:
            text = text[:preview_length] + "..."
        sec_ref = rec.get("section_ref")
        doc_index = rec.get("doc_index")
        # Section-title and heading-path lookup keys differ between
        # single-doc (sec_ref) and corpus (doc_index, sec_ref) callers.
        # Accept either; the dict knows its own key shape.
        section_key = (doc_index, sec_ref) if doc_index is not None else sec_ref
        title = section_titles.get(section_key) if sec_ref else None
        path = heading_paths.get(section_key, ()) if sec_ref else ()
        doc_uri = (
            doc_uris[doc_index]
            if doc_uris is not None and doc_index is not None and 0 <= doc_index < len(doc_uris)
            else None
        )
        scored.append(
            SearchResult(
                text=text,
                score=float(score),
                block_ref=rec.get("block_ref", ""),
                page=rec.get("page"),
                section_ref=sec_ref,
                section_title=title,
                char_start=rec.get("char_start"),
                char_end=rec.get("char_end"),
                heading_path=path,
                doc_index=doc_index,
                doc_uri=doc_uri,
            )
        )

    total = len(scored)
    results = scored[:top_k]
    return SearchResults(
        results=results,
        total_matches=total,
        has_more=total > top_k,
        query=query,
    )


def _embed_texts(texts: Iterable[str], *, model_id: str | None = None) -> Any:
    """Return an L2-normalized (N, dim) numpy array of dense embeddings.

    Uses the kaos-nlp-transformers model selected by ``model_id`` — None
    resolves to the package default (``KaosNLPTransformersSettings.default_model``,
    currently ``BAAI/bge-small-en-v1.5``). Caller is responsible for
    guarding the import — ``_ensure_transformers_available`` has already
    run by the time we get here.
    """
    import numpy as np

    text_list = list(texts)
    model = _get_embedding_model(model_id)
    if not text_list:
        # Empty input — return a (0, dim) array. Use getattr so a future
        # attribute rename in kaos-nlp-transformers (`dim` → `dimension`,
        # `embed_dim`, ...) doesn't crash this call site; the caller
        # short-circuits on empty records anyway, so a (0, 0) shape is
        # fine if the dim probe ever fails.
        dim = int(getattr(model, "dim", 0) or 0)
        return np.zeros((0, dim), dtype=np.float32)
    vecs = model.embed(text_list)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return (vecs / norms).astype(np.float32)


def _embed_query(query: str, *, model_id: str | None = None) -> Any:
    """Return an L2-normalized (dim,) numpy vector for ``query``.

    Single-vector front to :func:`_embed_texts` so the normalization
    (and zero-norm guard, and dtype coercion) only lives in one place.
    """
    return _embed_texts([query], model_id=model_id)[0]


def _search_embeddings(
    document: ContentDocument,
    query: str,
    *,
    top_k: int,
    preview_length: int,
    level: Literal["paragraph", "sentence"],
    model_id: str | None = None,
) -> SearchResults:
    """Pure dense retrieval via cosine similarity over passage embeddings.

    Embeds every paragraph (or sentence) once, then ranks by cosine
    similarity to the query. For repeated queries against the same
    document, prefer :class:`SearchableDocument` which caches the
    embedding matrix.
    """
    import numpy as np

    _, records, section_titles, heading_paths = _build_search_records(document, level=level)

    if not records:
        return SearchResults(results=[], total_matches=0, has_more=False, query=query)

    texts = [r["text"] for r in records]
    doc_vecs = _embed_texts(texts, model_id=model_id)
    q_vec = _embed_query(query, model_id=model_id)

    sims = doc_vecs @ q_vec
    n = len(records)
    effective_k = min(top_k, n)
    if effective_k <= 0:
        return SearchResults(results=[], total_matches=0, has_more=False, query=query)

    if effective_k < n:
        idx_top = np.argpartition(-sims, effective_k)[:effective_k]
        idx_top = idx_top[np.argsort(-sims[idx_top])]
    else:
        idx_top = np.argsort(-sims)

    indices = [int(i) for i in idx_top]
    scores = [float(sims[i]) for i in idx_top]
    return _records_to_search_results(
        records=records,
        indices=indices,
        scores=scores,
        section_titles=section_titles,
        heading_paths=heading_paths,
        preview_length=preview_length,
        query=query,
        top_k=top_k,
    )


def _search_hybrid(
    document: ContentDocument,
    query: str,
    *,
    top_k: int,
    preview_length: int,
    level: Literal["paragraph", "sentence"],
    candidate_k: int,
    model_id: str | None = None,
) -> SearchResults:
    """BM25 candidate set + embedding rerank.

    BM25 is free at query time and narrows the candidate pool; the
    embedding model is then used to rescore those candidates. This is
    the principled fix for over-eager small-chunk BM25 hits — the
    embedding cosine similarity demotes spurious lexical matches that
    don't actually answer the query.

    Falls back to TF if kaos-nlp-core is missing (still does the
    embedding rerank step on the TF candidates).
    """
    import numpy as np

    _view, records, section_titles, heading_paths = _build_search_records(document, level=level)

    if not records:
        return SearchResults(results=[], total_matches=0, has_more=False, query=query)

    # Candidate selection — prefer BM25, fall back to TF.
    candidate_indices: list[int]
    try:
        from kaos_nlp_core.search import Searcher

        searcher = Searcher.from_documents(
            records,
            external_id_field="block_ref",
            metadata_fields=["page", "section_ref", "char_start", "char_end"]
            if level == "sentence"
            else ["page", "section_ref"],
        )
        # Searcher.search returns hits keyed by SearchHit.doc_id (which is
        # the record["id"]). Records here are dense (id == row index).
        hits = searcher.search(query, top_k=min(candidate_k, len(records)))
        candidate_indices = [int(h.doc_id) for h in hits]
    except ImportError:
        if level == "sentence":
            msg = "Hybrid sentence-level retrieval requires kaos-nlp-core (BM25)."
            raise ImportError(msg) from None
        # TF fallback — score each paragraph by simple term frequency.
        tf_results = _search_tf(
            document, query, top_k=min(candidate_k, len(records)), preview_length=0
        )
        # Map TF results back to record indices via block_ref.
        block_to_idx = {r.get("block_ref"): i for i, r in enumerate(records)}
        candidate_indices = []
        for r in tf_results.results:
            i = block_to_idx.get(r.block_ref)
            if i is not None:
                candidate_indices.append(i)

    if not candidate_indices:
        return SearchResults(results=[], total_matches=0, has_more=False, query=query)

    # Embedding rerank over the candidate pool.
    cand_texts = [records[i]["text"] for i in candidate_indices]
    cand_vecs = _embed_texts(cand_texts, model_id=model_id)
    q_vec = _embed_query(query, model_id=model_id)
    sims = cand_vecs @ q_vec

    n = len(candidate_indices)
    k = min(top_k, n)
    if k <= 0:
        return SearchResults(results=[], total_matches=0, has_more=False, query=query)

    if k < n:
        local_top = np.argpartition(-sims, k)[:k]
        local_top = local_top[np.argsort(-sims[local_top])]
    else:
        local_top = np.argsort(-sims)

    indices = [int(candidate_indices[i]) for i in local_top]
    scores = [float(sims[i]) for i in local_top]
    return _records_to_search_results(
        records=records,
        indices=indices,
        scores=scores,
        section_titles=section_titles,
        heading_paths=heading_paths,
        preview_length=preview_length,
        query=query,
        top_k=top_k,
    )


# ─── TF fallback search ─────────────────────────────────────────────────────


def _search_tf(
    document: ContentDocument,
    query: str,
    *,
    top_k: int,
    preview_length: int,
) -> SearchResults:
    """Simple term frequency search (fallback without kaos-nlp-core)."""
    view = DocumentView(document)
    heading_paths = _heading_path_index(view)
    query_lower = query.lower()
    query_words = [w for w in query_lower.split() if len(w) > 2]
    all_scored: list[SearchResult] = []

    for pv in view.paragraphs:
        text = pv.text
        if not text or not text.strip():
            continue

        text_lower = text.lower()
        score = float(text_lower.count(query_lower))
        if score <= 0 and query_words:
            score = float(sum(text_lower.count(w) for w in query_words))
        if score <= 0:
            continue

        display = text
        if preview_length > 0 and len(display) > preview_length:
            display = display[:preview_length] + "..."

        all_scored.append(
            SearchResult(
                text=display,
                score=score,
                block_ref=pv.block_ref,
                page=pv.page,
                section_ref=pv.section_ref,
                section_title=_resolve_section(view, pv.section_ref),
                heading_path=heading_paths.get(pv.section_ref, ()) if pv.section_ref else (),
            )
        )

    all_scored.sort(key=lambda r: r.score, reverse=True)
    total = len(all_scored)
    results = all_scored[:top_k]
    return SearchResults(
        results=results,
        total_matches=total,
        has_more=total > len(results),
        query=query,
    )


def _resolve_section(view: DocumentView, section_ref: str | None) -> str | None:
    """Resolve section heading text from a section ref."""
    if not section_ref:
        return None
    sec = view.section_by_ref(section_ref)
    return sec.heading_text if sec is not None else None


# ─── Tabular search ────────────────────────────────────────────────────────


def search_tabular(
    document: TabularDocument,
    query: str,
    *,
    table_name: str | None = None,
    column: str | None = None,
    top_k: int = 10,
) -> SearchResults:
    """Search within a TabularDocument by text query.

    Performs case-insensitive substring matching across cell values.
    Can be scoped to a specific table and/or column. Results are
    scored by exact match (2.0) > substring match (1.0), with
    ties broken by row order.

    Args:
        document: The TabularDocument to search.
        query: Search query text (must not be empty).
        table_name: Restrict search to a specific table name.
        column: Restrict search to a specific column name.
        top_k: Maximum number of results to return.

    Returns:
        SearchResults with matching cell values and table/column refs.

    Raises:
        ValueError: If query is empty.
    """
    if not query or not query.strip():
        msg = "Query must not be empty"
        raise ValueError(msg)

    query_lower = query.lower().strip()
    all_scored: list[SearchResult] = []

    tables = document.tables
    if table_name is not None:
        tables = tuple(t for t in tables if t.name == table_name)

    for table in tables:
        col_indices: list[int] = list(range(len(table.columns)))
        if column is not None:
            col_indices = [i for i, c in enumerate(table.columns) if c.name == column]

        for row_idx, row in enumerate(table.rows):
            for col_idx in col_indices:
                if col_idx >= len(row):
                    continue
                val = row[col_idx]
                if val is None:
                    continue
                val_str = str(val)
                val_lower = val_str.lower()
                if query_lower not in val_lower:
                    continue

                score = 2.0 if val_lower == query_lower else 1.0
                col_name = table.columns[col_idx].name if col_idx < len(table.columns) else "?"

                all_scored.append(
                    SearchResult(
                        text=val_str,
                        score=score,
                        block_ref=f"#/tables/{table.name}/rows/{row_idx}/{col_name}",
                        page=None,
                        section_ref=table.name,
                        section_title=f"{table.name}.{col_name}",
                    )
                )

    all_scored.sort(key=lambda r: r.score, reverse=True)
    total = len(all_scored)
    results = all_scored[:top_k]
    return SearchResults(
        results=results,
        total_matches=total,
        has_more=total > len(results),
        query=query,
    )


# ─── Multi-document corpus search ──────────────────────────────────────────


async def search_corpus(
    documents: list[SearchableDocument] | dict[str, str],
    query: str,
    *,
    top_k: int = 10,
    level: Literal["paragraph", "sentence"] = "paragraph",
    retrieval: RetrievalMode = "bm25",
    rerank_top_k: int = 10,
    model_id: str | None = None,
) -> list[RetrievalResult]:
    """Search across multiple documents, returning globally ranked results.

    Accepts either a list of :class:`SearchableDocument` instances (pre-built
    indexes, efficient for repeated queries) or a plain ``dict[str, str]``
    mapping ``uri -> text`` (convenience path that builds temporary indexes).

    Results are globally ranked by score across all documents and returned as
    ``RetrievalResult`` objects with provenance (doc_id=uri, char_start,
    char_end, page).

    Args:
        documents: Either pre-indexed SearchableDocument instances or a
            ``{uri: text}`` dict for convenience.
        query: The search query text.
        top_k: Maximum number of results to return (globally ranked).
        level: Search granularity -- ``"paragraph"`` or ``"sentence"``.
        retrieval: Retrieval mode (see :func:`search_document`).
        rerank_top_k: Hybrid mode only — see :func:`search_document`.
        model_id: HF Hub embedding model id for ``"embeddings"`` and
            ``"hybrid"`` modes. When ``documents`` is a list of
            ``SearchableDocument`` instances and the caller wants the
            same model used inside each one, the ``SearchableDocument``s
            should be constructed with the same ``model_id``. The
            argument here only affects the dict-mode synthetic
            documents this function builds. ``None`` selects the
            kaos-nlp-transformers default.

    Returns:
        Globally ranked list of ``RetrievalResult`` with provenance.

    Raises:
        ValueError: If query is empty or documents is empty.
    """
    import asyncio

    from kaos_nlp_core.retrieval.protocol import RetrievalResult as _RetrievalResult

    if not query or not query.strip():
        msg = "Query must not be empty"
        raise ValueError(msg)

    if not documents:
        return []

    # Normalize to list of (uri, SearchableDocument) pairs
    indexed_docs: list[tuple[str, Any]] = []
    if isinstance(documents, dict):
        # Build temporary SearchableDocuments from plain text.
        from kaos_content.builders import DocumentBuilder
        from kaos_content.indexing import SearchableDocument as _SearchableDocument
        from kaos_content.model.attr import SourceRef
        from kaos_content.model.metadata import DocumentMetadata

        for uri, text in documents.items():
            builder = DocumentBuilder()
            builder.paragraph(text)
            doc = builder.build()
            # Replace the metadata with one carrying the URI (frozen model)
            doc = doc.model_copy(
                update={
                    "metadata": DocumentMetadata(
                        source=SourceRef(uri=uri),
                    ),
                }
            )
            sdoc = _SearchableDocument(doc, level=level, retrieval=retrieval, model_id=model_id)
            indexed_docs.append((uri, sdoc))
    else:
        for sdoc in documents:
            source = sdoc.document.metadata.source
            uri = source.uri if source is not None else f"doc:{id(sdoc)}"
            indexed_docs.append((uri, sdoc))

    # Search each document concurrently. The Rust ort backend in
    # kaos-nlp-transformers releases the GIL during ``embed()`` (KNT-601
    # / KNT-602) so a thread-pool offload genuinely parallelizes the
    # embedding work rather than serializing on the event loop.
    def _run_one(sdoc: Any) -> SearchResults:
        return sdoc.search(query, top_k=top_k, preview_length=0, rerank_top_k=rerank_top_k)

    per_doc = await asyncio.gather(
        *(asyncio.to_thread(_run_one, sdoc) for _uri, sdoc in indexed_docs)
    )

    all_results: list[_RetrievalResult] = []
    for (uri, sdoc), search_results in zip(indexed_docs, per_doc, strict=True):
        for sr in search_results.results:
            passage_uri = _searchable_passage_uri(
                doc_uri=uri,
                text=sr.text,
                block_ref=sr.block_ref,
                char_start=sr.char_start,
                level=sdoc.level,
            )
            all_results.append(
                _RetrievalResult(
                    text=sr.text,
                    score=sr.score,
                    doc_id=uri,
                    metadata={
                        "block_ref": sr.block_ref,
                        "passage_uri": passage_uri,
                        "section_ref": sr.section_ref,
                        "section_title": sr.section_title,
                        "heading_path": list(sr.heading_path),
                    },
                    char_start=sr.char_start,
                    char_end=sr.char_end,
                    page=sr.page,
                )
            )

    # Global ranking by score, descending
    all_results.sort(key=lambda r: r.score, reverse=True)
    return all_results[:top_k]


def _searchable_passage_uri(
    *,
    doc_uri: str,
    text: str,
    block_ref: str | None,
    char_start: int | None = None,
    level: str | None = None,
) -> str:
    """Build a stable passage URI for SearchableDocument corpus search."""
    if level == "sentence" and char_start is not None:
        return f"{doc_uri}#c{char_start}"
    if block_ref and block_ref != "#/body/0" and "#" not in doc_uri:
        return f"{doc_uri}{block_ref}"
    if char_start is not None:
        return f"{doc_uri}#c{char_start}"

    import hashlib

    text_hash = hashlib.sha256(text.encode()).hexdigest()[:8]
    return f"{doc_uri}#h{text_hash}"
