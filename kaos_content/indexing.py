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

import logging
from collections.abc import Sequence
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

_logger = logging.getLogger(__name__)


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
        max_tokens: int | None = None,
        model_id: str | None = None,
    ) -> list[ContentDocument]:
        """Convenience: chunk the document using SectionChunker.

        Args:
            max_chars: Maximum characters per chunk.
            split_depth: Heading depth at which to split.
            overlap_paragraphs: Number of paragraphs to overlap between chunks.
            max_tokens: Optional embedding-model token budget per chunk.
                When set, chunks that exceed this token count under the
                selected model's tokenizer are further split (sentence
                boundaries within paragraphs, block boundaries otherwise).
                Requires kaos-nlp-transformers. ``None`` skips the
                token check (cheap char-only path).
            model_id: HF Hub embedding model id used to count tokens
                when ``max_tokens`` is set. ``None`` resolves to
                ``self.model_id`` (the model the index was built with),
                so chunker output and dense retrieval stay aligned by
                default.

        Returns:
            List of ContentDocument chunks.
        """
        from kaos_content.chunking import SectionChunker

        chunker = SectionChunker(
            max_chars=max_chars,
            split_depth=split_depth,
            overlap_paragraphs=overlap_paragraphs,
            max_tokens=max_tokens,
            model_id=model_id if model_id is not None else self._model_id,
        )
        return chunker.chunk(self._document)


class SearchableCorpus:
    """N ContentDocuments with one shared search index.

    The corpus-level analog of :class:`SearchableDocument`. Builds one
    BM25 inverted index over the concatenated row stream of every
    document so IDF is corpus-wide, and shares one embedding matrix
    across all dense queries. Returned :class:`SearchResult` objects
    carry the new ``doc_index`` / ``doc_uri`` fields so callers can
    fan out to the source document.

    Three retrieval modes mirror :class:`SearchableDocument`:

    * ``"bm25"`` (default) — pure lexical BM25 via kaos-nlp-core.
    * ``"embeddings"`` — pure dense retrieval via kaos-nlp-transformers.
    * ``"hybrid"`` — corpus-wide BM25 candidate selection + embedding
      rerank.

    BM25 index is built eagerly at construction; the embedding matrix
    is built lazily on first dense query and reused thereafter (the
    corpus is conceptually immutable after construction). For dense /
    hybrid corpora exceeding ``max_embed_rows`` rows, the first dense
    query raises ``ValueError`` with an actionable hint rather than
    silently allocating multi-gigabyte arrays.

    See ``docs/SEARCHABLE_CORPUS.md`` for the design rationale; this
    class implements the v0.1 API described there.
    """

    __slots__ = (
        "_doc_embeddings",
        "_doc_offsets",
        "_doc_uris",
        "_documents",
        "_heading_paths",
        "_level",
        "_max_embed_rows",
        "_model_id",
        "_records",
        "_reranker_model_id",
        "_retrieval",
        "_searcher",
        "_section_titles",
        "_units",
        "_views",
    )

    def __init__(
        self,
        documents: Sequence[ContentDocument],
        *,
        level: Literal["paragraph", "sentence"] = "paragraph",
        retrieval: RetrievalMode = "bm25",
        doc_uris: Sequence[str] | None = None,
        model_id: str | None = None,
        reranker_model_id: str | None = None,
        max_embed_rows: int = 200_000,
    ) -> None:
        """Build a corpus-wide searchable index.

        Args:
            documents: ContentDocuments to index. Order is preserved as
                the ``doc_index`` axis (``corpus.documents[i]``).
            level: ``"paragraph"`` or ``"sentence"`` — the row
                granularity for the shared index. Sentence-level
                requires kaos-nlp-core for the segmenter.
            retrieval: See :class:`SearchableDocument` — same three
                modes.
            doc_uris: Per-document URIs for ``SearchResult.doc_uri``.
                Length must equal ``len(documents)``. ``None`` falls
                back to ``document.metadata.source.uri`` per document,
                or the synthetic ``"doc:anon-N"`` when neither is set.
            model_id: HF Hub embedding model id. ``None`` selects the
                kaos-nlp-transformers default.
            reranker_model_id: HF Hub cross-encoder reranker id. NOTE:
                this class does not call the reranker itself — the
                actual rerank step lives in upstream consumers. Stored
                for downstream access via the
                :attr:`reranker_model_id` property.
            max_embed_rows: Hard cap on the dense matrix size. The
                first dense query raises ``ValueError`` when the
                corpus exceeds this many rows. Default 200_000
                (~300 MB at dim=384). Raise explicitly for larger
                corpora.

        Raises:
            ValueError: If ``retrieval`` is unknown, or ``doc_uris``
                length does not match.
            ImportError: If ``retrieval`` requires kaos-nlp-transformers
                and it is not installed.
        """
        if retrieval not in ("bm25", "embeddings", "hybrid"):
            msg = (
                f"Unknown retrieval mode {retrieval!r}. "
                "Fix: choose one of 'bm25', 'embeddings', or 'hybrid'."
            )
            raise ValueError(msg)
        if retrieval in ("embeddings", "hybrid"):
            _ensure_transformers_available()

        docs = tuple(documents)
        if doc_uris is not None and len(doc_uris) != len(docs):
            msg = (
                f"doc_uris length ({len(doc_uris)}) does not match documents "
                f"length ({len(docs)}). Fix: pass one URI per document, or "
                f"omit doc_uris to fall back to metadata.source.uri."
            )
            raise ValueError(msg)

        self._documents = docs
        self._level = level
        self._retrieval = retrieval
        self._model_id = model_id
        self._reranker_model_id = reranker_model_id
        self._max_embed_rows = max_embed_rows

        # Resolve URIs: explicit override > metadata.source.uri > synthetic.
        resolved_uris: list[str] = []
        for i, doc in enumerate(docs):
            if doc_uris is not None:
                resolved_uris.append(str(doc_uris[i]))
                continue
            src = doc.metadata.source if doc.metadata is not None else None
            if src is not None and src.uri:
                resolved_uris.append(src.uri)
            else:
                resolved_uris.append(f"doc:anon-{i}")
        self._doc_uris = tuple(resolved_uris)

        # Detect & log doc_uri collisions (design §6 — disambiguation
        # falls back to doc_index).
        seen: dict[str, int] = {}
        for i, uri in enumerate(self._doc_uris):
            if uri in seen:
                _logger.info(
                    "SearchableCorpus: doc_uri collision %r at indices %d and %d; "
                    "results disambiguate by doc_index",
                    uri,
                    seen[uri],
                    i,
                )
                break
            seen[uri] = i

        # Sentence-level segmenter wiring (mirror SearchableDocument).
        segmenter = None
        if level == "sentence":
            from kaos_nlp_core._defaults import get_default_punkt_tokenizer

            segmenter = get_default_punkt_tokenizer()
        self._views: tuple[DocumentView, ...] = tuple(
            DocumentView(d, sentence_segmenter=segmenter) for d in docs
        )

        self._units: list[Any] = []
        self._records: list[dict[str, Any]] = []
        self._doc_offsets: list[int] = [0]
        # Section-title and heading-path keys are (doc_idx, section_ref)
        # tuples — different documents can have the same section_ref.
        self._section_titles: dict[tuple[int, str], str | None] = {}
        self._heading_paths: dict[tuple[int, str], tuple[str, ...]] = {}
        self._searcher: Searcher | None = None
        self._doc_embeddings: Any | None = None

        self._build()

    def _build(self) -> None:
        """Walk every document and build one shared BM25 index."""
        from kaos_nlp_core.search import Searcher

        from kaos_content.search import _heading_path_index

        global_row = 0
        metadata_fields: list[str]
        for doc_idx, view in enumerate(self._views):
            if self._level == "sentence":
                units = iter_sentence_units(view)
            else:
                units = iter_paragraph_units(view)

            heading_paths = _heading_path_index(view)
            for u in units:
                if u.section_ref:
                    key = (doc_idx, u.section_ref)
                    self._section_titles.setdefault(key, u.section_title)
                    if key not in self._heading_paths:
                        self._heading_paths[key] = heading_paths.get(u.section_ref, ())
                rec: dict[str, Any] = {
                    "id": global_row,
                    "text": u.text,
                    "block_ref": u.block_ref,
                    "page": u.page,
                    "section_ref": u.section_ref,
                    "doc_index": doc_idx,
                }
                if isinstance(u, SentenceUnit):
                    rec["char_start"] = u.char_start
                    rec["char_end"] = u.char_end
                self._records.append(rec)
                self._units.append(u)
                global_row += 1
            self._doc_offsets.append(global_row)

        if self._level == "sentence":
            metadata_fields = ["page", "section_ref", "doc_index", "char_start", "char_end"]
        else:
            metadata_fields = ["page", "section_ref", "doc_index"]

        if not self._records:
            self._searcher = None
            return

        self._searcher = Searcher.from_documents(
            self._records,
            external_id_field="block_ref",
            metadata_fields=metadata_fields,
        )

    def _ensure_doc_embeddings(self) -> Any:
        """Compute and cache the corpus-wide embedding matrix lazily.

        Honors ``max_embed_rows``: a corpus that exceeds the cap raises
        ``ValueError`` rather than silently allocating a multi-GB array.
        """
        if self._doc_embeddings is not None:
            return self._doc_embeddings

        n_total = len(self._records)
        if n_total > self._max_embed_rows:
            msg = (
                f"Corpus has {n_total:,} rows; refusing to build a "
                f"({n_total:,}, dim) embedding matrix above the configured "
                f"max_embed_rows cap of {self._max_embed_rows:,}. "
                "Fix: raise max_embed_rows explicitly, use retrieval='bm25', "
                "or chunk the corpus and merge results upstream."
            )
            raise ValueError(msg)

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
        """Query the corpus.

        Args:
            query: Search query text (must not be empty).
            top_k: Maximum results to return. For ``"hybrid"`` this is
                the cap before rerank; ``rerank_top_k`` (defaulting to
                ``top_k``) is the cap after rerank.
            preview_length: Maximum characters in result text. 0 = full.
            rerank_top_k: Hybrid only — cap after rerank. Defaults to
                ``top_k``.
            rerank_candidate_k: Hybrid only — BM25 candidate-pool size
                fed into the embedding reranker.

        Raises:
            ValueError: If ``query`` is empty.
        """
        if not query or not query.strip():
            msg = "Query must not be empty"
            raise ValueError(msg)

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

        # bm25
        assert self._searcher is not None  # narrowed by `if not self._records` above  # nosec B101
        hits = self._searcher.search(query, top_k=len(self._records))
        scored: list[SearchResult] = []
        doc_uris_list = list(self._doc_uris)
        for hit in hits:
            text = hit.text
            if preview_length > 0 and len(text) > preview_length:
                text = text[:preview_length] + "..."
            sec_ref = hit.metadata.get("section_ref")
            doc_index = hit.metadata.get("doc_index")
            title: str | None = None
            path: tuple[str, ...] = ()
            if sec_ref and doc_index is not None:
                section_key: tuple[int, str] = (doc_index, sec_ref)
                title = self._section_titles.get(section_key)
                path = self._heading_paths.get(section_key, ())
            doc_uri = (
                doc_uris_list[doc_index]
                if doc_index is not None and 0 <= doc_index < len(doc_uris_list)
                else None
            )
            scored.append(
                SearchResult(
                    text=text,
                    score=hit.score,
                    block_ref=hit.external_id or "",
                    page=hit.metadata.get("page"),
                    section_ref=sec_ref,
                    section_title=title,
                    char_start=hit.metadata.get("char_start"),
                    char_end=hit.metadata.get("char_end"),
                    heading_path=path,
                    doc_index=doc_index,
                    doc_uri=doc_uri,
                )
            )

        total = len(scored)
        return SearchResults(
            results=scored[:top_k],
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
        """Pure dense retrieval over the cached corpus matrix."""
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
            doc_uris=list(self._doc_uris),
        )

    def _search_hybrid(
        self,
        query: str,
        *,
        top_k: int,
        preview_length: int,
        candidate_k: int,
    ) -> SearchResults:
        """Corpus-wide BM25 candidate set + embedding rerank."""
        import numpy as np

        from kaos_content.search import _embed_query

        if self._searcher is None:
            return SearchResults(results=[], total_matches=0, has_more=False, query=query)

        cand_pool = min(candidate_k, len(self._records))
        hits = self._searcher.search(query, top_k=cand_pool)
        candidate_indices = [int(h.doc_id) for h in hits]
        if not candidate_indices:
            return SearchResults(results=[], total_matches=0, has_more=False, query=query)

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
            doc_uris=list(self._doc_uris),
        )

    # ── Read-only properties ────────────────────────────────────────────

    @property
    def documents(self) -> tuple[ContentDocument, ...]:
        """The underlying ContentDocument tuple, in build-order."""
        return self._documents

    @property
    def num_documents(self) -> int:
        """Number of documents in the corpus."""
        return len(self._documents)

    @property
    def size(self) -> int:
        """Total number of indexed rows (paragraphs or sentences)."""
        return len(self._records)

    @property
    def doc_uris(self) -> tuple[str, ...]:
        """Per-document URIs, parallel to :attr:`documents`."""
        return self._doc_uris

    @property
    def retrieval(self) -> str:
        """The retrieval mode (``"bm25"``, ``"embeddings"``, or ``"hybrid"``)."""
        return self._retrieval

    @property
    def level(self) -> str:
        """Search granularity (``"paragraph"`` or ``"sentence"``)."""
        return self._level

    @property
    def model_id(self) -> str | None:
        """HF Hub embedding model id, or ``None`` for the default."""
        return self._model_id

    @property
    def reranker_model_id(self) -> str | None:
        """HF Hub reranker model id, or ``None``.

        Stored for downstream consumers; this class does not invoke a
        reranker itself.
        """
        return self._reranker_model_id

    def doc_for_row(self, row: int) -> tuple[int, ContentDocument]:
        """Map global row index back to ``(doc_index, document)``.

        Raises:
            IndexError: If ``row`` is out of range.
        """
        if row < 0 or row >= len(self._records):
            msg = f"row {row} out of range [0, {len(self._records)})"
            raise IndexError(msg)
        # Binary search would be O(log N); the offset list is short.
        # Linear is fine for any realistic corpus size.
        for doc_idx in range(self.num_documents):
            if self._doc_offsets[doc_idx] <= row < self._doc_offsets[doc_idx + 1]:
                return doc_idx, self._documents[doc_idx]
        # Unreachable given the bounds check above; included for ty.
        msg = f"row {row} unmapped"
        raise IndexError(msg)


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
