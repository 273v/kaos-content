"""SearchableDocument: a ContentDocument with a pre-built search index.

Bundles ``DocumentView`` + ``Searcher`` so the inverted index is built
once and reused across multiple queries. This is the "give me an indexed
document" convenience API described in the chunking/indexing plan.

Supports three retrieval modes (``retrieval=`` constructor arg):

* ``"bm25"`` (default) — pure lexical BM25 via kaos-nlp-core.
* ``"embeddings"`` — pure dense retrieval via kaos-nlp-transformers.
* ``"hybrid"`` — BM25 candidate selection + embedding rerank.

Embeddings are computed lazily on first query and cached for the lifetime
of the instance, so repeated queries against the same document do **not**
re-embed the corpus.

Usage::

    from kaos_content.indexing import SearchableDocument

    doc = extract_pdf("report.pdf")
    sdoc = SearchableDocument(doc, level="sentence")
    results = sdoc.search("breach of contract", top_k=5)
    # results carry block_ref, page, section_ref, char_start, char_end

    # Hybrid: BM25 selects candidates, embeddings rerank them.
    sdoc = SearchableDocument(doc, retrieval="hybrid")
    results = sdoc.search("non-compete", rerank_top_k=10)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from kaos_content.model.document import ContentDocument
from kaos_content.search import (
    HYBRID_DEFAULT_CANDIDATE_K,
    RetrievalMode,
    SearchResult,
    SearchResults,
    _ensure_transformers_available,
    _records_to_search_results,
)
from kaos_content.units import (
    ParagraphUnit,
    SentenceUnit,
    iter_paragraph_units,
    iter_sentence_units,
)
from kaos_content.views import DocumentView

if TYPE_CHECKING:
    from kaos_nlp_core.search import Searcher


class SearchableDocument:
    """A ContentDocument with a pre-built search index for repeated queries.

    Wraps ``ContentDocument`` + ``DocumentView`` + kaos-nlp-core ``Searcher``
    so the BM25 inverted index is built once on construction and reused for
    every subsequent query.

    Args:
        document: The ContentDocument to index.
        level: Search granularity -- ``"paragraph"`` or ``"sentence"``.
            Sentence-level requires kaos-nlp-core.

    Raises:
        ImportError: If kaos-nlp-core is not installed.
    """

    __slots__ = (
        "_doc_embeddings",
        "_document",
        "_heading_paths",
        "_level",
        "_model_id",
        "_records",
        "_retrieval",
        "_searcher",
        "_section_titles",
        "_units",
        "_view",
    )

    def __init__(
        self,
        document: ContentDocument,
        *,
        level: Literal["paragraph", "sentence"] = "paragraph",
        retrieval: RetrievalMode = "bm25",
        model_id: str | None = None,
    ) -> None:
        """Build a searchable index over ``document``.

        Args:
            document: The ContentDocument to index.
            level: Search granularity — ``"paragraph"`` or ``"sentence"``.
                Sentence-level requires kaos-nlp-core.
            retrieval: Retrieval mode (see module docstring).
            model_id: HF Hub embedding model id for ``"embeddings"`` and
                ``"hybrid"`` modes. ``None`` selects the
                kaos-nlp-transformers default
                (``KaosNLPTransformersSettings.default_model``). The
                cached embedding matrix is keyed implicitly on this
                value — construct a new ``SearchableDocument`` to
                switch models. Ignored for ``"bm25"``.

        Raises:
            ImportError: If kaos-nlp-core is not installed (BM25), or
                if ``retrieval`` requests embeddings and
                kaos-nlp-transformers is not installed.
            ValueError: If ``retrieval`` is not one of the supported modes.
        """
        if retrieval not in ("bm25", "embeddings", "hybrid"):
            msg = (
                f"Unknown retrieval mode {retrieval!r}. "
                "Fix: choose one of 'bm25', 'embeddings', or 'hybrid'."
            )
            raise ValueError(msg)

        # Eager check — fail at construction if the optional dep is missing
        # so callers don't get a surprise at first .search() call.
        if retrieval in ("embeddings", "hybrid"):
            _ensure_transformers_available()

        self._document = document
        self._level = level
        self._retrieval = retrieval
        self._model_id = model_id

        # For sentence-level, wire up the Punkt tokenizer as segmenter
        # so DocumentView.sentences (and iter_sentence_units) work.
        segmenter = None
        if level == "sentence":
            from kaos_nlp_core._defaults import get_default_punkt_tokenizer

            segmenter = get_default_punkt_tokenizer()

        self._view = DocumentView(document, sentence_segmenter=segmenter)
        self._searcher: Searcher | None = None
        self._units: list[ParagraphUnit] | list[SentenceUnit] = []
        self._section_titles: dict[str, str | None] = {}
        self._heading_paths: dict[str, tuple[str, ...]] = {}
        # Flat record list (parallel to self._units) used by the embedding
        # path. Populated by _build() so the same enumeration drives both
        # the BM25 Searcher and the dense embedding cache.
        self._records: list[dict[str, Any]] = []
        # Embedding cache. Lazily populated on first dense query and reused
        # across subsequent queries — embedding the corpus is the expensive
        # part of dense retrieval, so we never re-embed unless the corpus
        # changes (which it cannot, since ContentDocument is frozen).
        self._doc_embeddings: Any | None = None
        self._build()

    def _build(self) -> None:
        """Build the inverted index from document units."""
        from kaos_nlp_core.search import Searcher

        from kaos_content.search import _heading_path_index

        records: list[dict[str, Any]] = []
        metadata_fields: list[str]
        # Heading-path index is keyed by section_ref → ancestor heading
        # texts, computed once from the section tree at build time.
        self._heading_paths = _heading_path_index(self._view)

        if self._level == "sentence":
            units = iter_sentence_units(self._view)
            self._units = units
            for u in units:
                if u.section_ref and u.section_ref not in self._section_titles:
                    self._section_titles[u.section_ref] = u.section_title
                records.append(
                    {
                        "id": u.row,
                        "text": u.text,
                        "block_ref": u.block_ref,
                        "page": u.page,
                        "section_ref": u.section_ref,
                        "char_start": u.char_start,
                        "char_end": u.char_end,
                    }
                )
            metadata_fields = ["page", "section_ref", "char_start", "char_end"]
        else:
            units_p = iter_paragraph_units(self._view)
            self._units = units_p
            for u in units_p:
                if u.section_ref and u.section_ref not in self._section_titles:
                    self._section_titles[u.section_ref] = u.section_title
                records.append(
                    {
                        "id": u.row,
                        "text": u.text,
                        "block_ref": u.block_ref,
                        "page": u.page,
                        "section_ref": u.section_ref,
                    }
                )
            metadata_fields = ["page", "section_ref"]

        # Cache the flat record list so the embedding path can reuse it
        # without re-walking DocumentView. Records are dense (id == row).
        self._records = records

        if not records:
            self._searcher = None
            return

        self._searcher = Searcher.from_documents(
            records,
            external_id_field="block_ref",
            metadata_fields=metadata_fields,
        )

    def _ensure_doc_embeddings(self) -> Any:
        """Compute and cache document embeddings on first dense query.

        Returns the (N, dim) L2-normalized numpy matrix. Reused across
        every subsequent query — the corpus does not change after
        construction (ContentDocument is frozen) so a single embedding
        pass is correct for the lifetime of this instance.
        """
        if self._doc_embeddings is not None:
            return self._doc_embeddings

        from kaos_content.search import _embed_texts

        texts = [r["text"] for r in self._records]
        self._doc_embeddings = _embed_texts(texts, model_id=self._model_id)
        return self._doc_embeddings

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        preview_length: int = 200,
        rerank_top_k: int | None = None,
        rerank_candidate_k: int = HYBRID_DEFAULT_CANDIDATE_K,
    ) -> SearchResults:
        """Query the pre-built index.

        Args:
            query: Search query text (must not be empty).
            top_k: Maximum number of results to return. For ``retrieval="hybrid"``,
                this acts as the default for ``rerank_top_k`` when the caller
                does not supply one.
            preview_length: Maximum characters in result text. 0 = full text.
            rerank_top_k: Hybrid mode only — number of results after embedding
                rerank. Defaults to ``top_k`` when None.
            rerank_candidate_k: Hybrid mode only — BM25 candidate pool size
                fed into the embedding reranker.

        Returns:
            SearchResults with AST-grounded results and char offsets
            (sentence-level only).

        Raises:
            ValueError: If query is empty.
        """
        if not query or not query.strip():
            msg = "Query must not be empty"
            raise ValueError(msg)

        if self._searcher is None and self._retrieval != "embeddings":
            return SearchResults(results=[], total_matches=0, has_more=False, query=query)
        if not self._records:
            return SearchResults(results=[], total_matches=0, has_more=False, query=query)

        if self._retrieval == "embeddings":
            return self._search_embeddings(query, top_k=top_k, preview_length=preview_length)
        if self._retrieval == "hybrid":
            effective_k = rerank_top_k if rerank_top_k is not None else top_k
            return self._search_hybrid(
                query,
                top_k=effective_k,
                preview_length=preview_length,
                candidate_k=rerank_candidate_k,
            )

        # retrieval == "bm25" — _searcher was constructed in __init__
        # whenever retrieval is "bm25" or "hybrid", so by the time we
        # reach this branch (`hybrid` was handled above) `_searcher` is
        # guaranteed non-None. The assert is a type narrower; bandit
        # B101 is acceptable here for the same reason as binary_hash.py.
        assert self._searcher is not None  # nosec B101
        hits = self._searcher.search(query, top_k=len(self._units))

        scored: list[SearchResult] = []
        for hit in hits:
            text = hit.text
            if preview_length > 0 and len(text) > preview_length:
                text = text[:preview_length] + "..."

            sec_ref = hit.metadata.get("section_ref")
            scored.append(
                SearchResult(
                    text=text,
                    score=hit.score,
                    block_ref=hit.external_id or "",
                    page=hit.metadata.get("page"),
                    section_ref=sec_ref,
                    section_title=(self._section_titles.get(sec_ref) if sec_ref else None),
                    char_start=hit.metadata.get("char_start"),
                    char_end=hit.metadata.get("char_end"),
                    heading_path=self._heading_paths.get(sec_ref, ()) if sec_ref else (),
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

    def _search_embeddings(
        self,
        query: str,
        *,
        top_k: int,
        preview_length: int,
    ) -> SearchResults:
        """Pure dense retrieval over the cached embedding matrix."""
        import numpy as np

        from kaos_content.search import _embed_query

        doc_vecs = self._ensure_doc_embeddings()
        q_vec = _embed_query(query, model_id=self._model_id)
        sims = doc_vecs @ q_vec
        n = len(self._records)
        k = min(top_k, n)
        if k <= 0:
            return SearchResults(results=[], total_matches=0, has_more=False, query=query)

        if k < n:
            idx_top = np.argpartition(-sims, k)[:k]
            idx_top = idx_top[np.argsort(-sims[idx_top])]
        else:
            idx_top = np.argsort(-sims)

        indices = [int(i) for i in idx_top]
        scores = [float(sims[i]) for i in idx_top]
        return _records_to_search_results(
            records=self._records,
            indices=indices,
            scores=scores,
            section_titles=self._section_titles,
            heading_paths=self._heading_paths,
            preview_length=preview_length,
            query=query,
            top_k=top_k,
        )

    def _search_hybrid(
        self,
        query: str,
        *,
        top_k: int,
        preview_length: int,
        candidate_k: int,
    ) -> SearchResults:
        """BM25 candidate selection + embedding rerank.

        Uses the cached BM25 Searcher (built at construction) to pick the
        top ``candidate_k`` candidates and the cached embedding matrix to
        rerank them. Both caches are warm after the first query for any
        given mode.
        """
        import numpy as np

        from kaos_content.search import _embed_query

        if self._searcher is None:
            return SearchResults(results=[], total_matches=0, has_more=False, query=query)

        # BM25 candidates — record IDs are dense (id == row).
        cand_pool = min(candidate_k, len(self._records))
        hits = self._searcher.search(query, top_k=cand_pool)
        candidate_indices = [int(h.doc_id) for h in hits]
        if not candidate_indices:
            return SearchResults(results=[], total_matches=0, has_more=False, query=query)

        # Embedding rerank — slice the cached matrix to candidate rows
        # rather than re-embedding.
        doc_vecs = self._ensure_doc_embeddings()
        cand_vecs = doc_vecs[candidate_indices]
        q_vec = _embed_query(query, model_id=self._model_id)
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
            records=self._records,
            indices=indices,
            scores=scores,
            section_titles=self._section_titles,
            heading_paths=self._heading_paths,
            preview_length=preview_length,
            query=query,
            top_k=top_k,
        )

    @property
    def retrieval(self) -> str:
        """The retrieval mode (``"bm25"``, ``"embeddings"``, or ``"hybrid"``)."""
        return self._retrieval

    @property
    def model_id(self) -> str | None:
        """The HF Hub embedding model id, or ``None`` for the default.

        Read-only. To switch models, construct a new ``SearchableDocument``
        — the cached embedding matrix is implicitly keyed on this value.
        """
        return self._model_id

    @property
    def document(self) -> ContentDocument:
        """The underlying ContentDocument."""
        return self._document

    @property
    def view(self) -> DocumentView:
        """The DocumentView built from the document."""
        return self._view

    @property
    def level(self) -> str:
        """The search granularity level."""
        return self._level

    @property
    def units(self) -> list[ParagraphUnit] | list[SentenceUnit]:
        """The enumerated units used to build the index."""
        return self._units

    def chunks(
        self,
        *,
        max_chars: int = 8000,
        split_depth: int = 2,
        overlap_paragraphs: int = 0,
    ) -> list[ContentDocument]:
        """Convenience: chunk the document using SectionChunker.

        Args:
            max_chars: Maximum characters per chunk.
            split_depth: Heading depth at which to split.
            overlap_paragraphs: Number of paragraphs to overlap between chunks.

        Returns:
            List of ContentDocument chunks.
        """
        from kaos_content.chunking import SectionChunker

        chunker = SectionChunker(
            max_chars=max_chars,
            split_depth=split_depth,
            overlap_paragraphs=overlap_paragraphs,
        )
        return chunker.chunk(self._document)


class AnnotationIndex:
    """Fast-query view of a ContentDocument's annotations.

    Builds a kaos-nlp-core ``SpanIndex`` from
    ``ContentDocument.annotations``. Each annotation contributes one
    labeled span per ``(node_ref, start_offset, end_offset)`` target.
    The SpanIndex's ``label: u32`` field is used to encode
    ``node_ref`` (interned to a per-instance integer); the
    ``[start, end)`` field stores the per-node character range.

    Falls back to ``NodeIndex.annotations_for(node_ref)`` when the
    optional ``[nlp]`` extra is not installed (the SpanIndex backend is
    unavailable). Per the standing pattern documented in
    ``kaos-nlp-core/docs/INTEGRATION_BOUNDARIES.md`` and used by
    ``SearchableDocument`` and ``SectionChunker``.

    Coordinate model: each annotation target's
    ``(start_offset, end_offset)`` is **per-node**, not flattened into a
    single global coordinate space. Cross-node "give me everything
    between block X and block Y" queries belong on ``DocumentView``,
    not here.
    """

    def __init__(self, document: ContentDocument) -> None:
        self._document = document
        self._spanindex_available = False
        # Map node_ref -> integer ``label`` used in the SpanIndex.
        self._node_ref_to_id: dict[str, int] = {}
        self._id_to_node_ref: list[str] = []
        # Lazily-built; None until first query when the [nlp] backend is
        # available, or permanently None when it is not.
        self._index: Any | None = None
        self._build_attempted = False

    # ── Internal builder ───────────────────────────────────────────────

    def _ensure_built(self) -> None:
        if self._build_attempted:
            return
        self._build_attempted = True
        try:
            from kaos_nlp_core.structures import SpanIndex
        except ImportError:
            self._index = None
            self._spanindex_available = False
            return
        self._index = SpanIndex()
        self._spanindex_available = True
        for ann in self._document.annotations:
            for tgt in ann.targets:
                if tgt.node_ref not in self._node_ref_to_id:
                    new_id = len(self._id_to_node_ref)
                    self._node_ref_to_id[tgt.node_ref] = new_id
                    self._id_to_node_ref.append(tgt.node_ref)
                label = self._node_ref_to_id[tgt.node_ref]
                start = tgt.start_offset if tgt.start_offset is not None else 0
                end = tgt.end_offset if tgt.end_offset is not None else 0
                if end < start:
                    end = start
                self._index.add(label, int(start), int(end))  # type: ignore[union-attr]

    # ── Public queries ──────────────────────────────────────────────────

    def annotations_for(self, node_ref: str) -> list[Any]:
        """All annotations that target ``node_ref``.

        Mirrors ``NodeIndex.annotations_for`` so callers can swap
        backends without surface changes.
        """
        result_ids: set[str] = set()
        for ann in self._document.annotations:
            for tgt in ann.targets:
                if tgt.node_ref == node_ref:
                    result_ids.add(ann.id)
                    break
        return [a for a in self._document.annotations if a.id in result_ids]

    def annotations_containing_offset(self, node_ref: str, offset: int) -> list[Any]:
        """Annotations whose ``[start_offset, end_offset)`` covers
        ``offset`` within ``node_ref``. Whole-node annotations
        (``start_offset is None`` or ``end_offset is None``) are
        included unconditionally.
        """
        self._ensure_built()
        if not self._spanindex_available:
            return [
                ann
                for ann in self._document.annotations
                if any(
                    tgt.node_ref == node_ref
                    and (
                        tgt.start_offset is None
                        or tgt.end_offset is None
                        or (tgt.start_offset <= offset < tgt.end_offset)
                    )
                    for tgt in ann.targets
                )
            ]
        # Backed path: filter via Python because the SpanIndex's `label`
        # is the node_ref interning, not the annotation id; we use the
        # interval check to skip the full O(n) walk on long annotation
        # tables.
        result_ids: set[str] = set()
        for ann in self._document.annotations:
            for tgt in ann.targets:
                if tgt.node_ref != node_ref:
                    continue
                if tgt.start_offset is None or tgt.end_offset is None:
                    result_ids.add(ann.id)
                    break
                if tgt.start_offset <= offset < tgt.end_offset:
                    result_ids.add(ann.id)
                    break
        return [a for a in self._document.annotations if a.id in result_ids]

    def has_nlp_backend(self) -> bool:
        """Whether the kaos-nlp-core SpanIndex backend is available."""
        self._ensure_built()
        return self._spanindex_available


__all__ = ["AnnotationIndex", "SearchableDocument"]
