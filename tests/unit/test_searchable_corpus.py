"""Tests for SearchableCorpus — corpus-wide search over multiple documents.

KNT-601 P6.4. Verifies the design described in
``docs/SEARCHABLE_CORPUS.md``: corpus-wide BM25 with shared IDF,
``doc_index`` / ``doc_uri`` provenance on every result, the lazy
embedding matrix, and the ``max_embed_rows`` guardrail.

Embedding-mode tests patch ``kaos_content.search._get_embedding_model``
with a deterministic fake so they run whether or not
kaos-nlp-transformers is installed.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Iterable

import pytest

from kaos_content.model.attr import Provenance, SourceRef
from kaos_content.model.blocks import Heading, Paragraph
from kaos_content.model.document import ContentDocument
from kaos_content.model.inlines import Text
from kaos_content.model.metadata import DocumentMetadata

_has_nlp = importlib.util.find_spec("kaos_nlp_core") is not None


def _src(uri: str | None) -> SourceRef:
    return SourceRef(uri=uri or "")


def _prov(uri: str | None, page: int) -> Provenance:
    return Provenance(source=_src(uri), page=page)


def _para(text: str, *, uri: str | None = None, page: int = 1) -> Paragraph:
    return Paragraph(children=(Text(value=text),), provenance=_prov(uri, page))


def _heading(text: str, *, depth: int = 1, uri: str | None = None, page: int = 1) -> Heading:
    return Heading(children=(Text(value=text),), depth=depth, provenance=_prov(uri, page))


def _doc(uri: str | None, paragraphs: list[str]) -> ContentDocument:
    src = SourceRef(uri=uri) if uri else None
    return ContentDocument(
        metadata=DocumentMetadata(source=src),
        body=tuple(_para(p, uri=uri) for p in paragraphs),
    )


@pytest.fixture()
def small_corpus() -> list[ContentDocument]:
    """Three small documents — a contract, a recipe, and a court opinion."""
    return [
        _doc(
            "doc1.pdf",
            [
                "The seller hereby grants the buyer an exclusive license.",
                "All disputes arising under this contract shall be arbitrated.",
            ],
        ),
        _doc(
            "doc2.pdf",
            [
                "Combine flour, sugar, and butter in a large mixing bowl.",
                "Bake at 350 degrees Fahrenheit for thirty minutes.",
            ],
        ),
        _doc(
            "doc3.pdf",
            [
                "The court held that the defendant breached the contract.",
                "Damages were awarded in the amount of one hundred thousand dollars.",
            ],
        ),
    ]


# ─── Construction ──────────────────────────────────────────────────────────


@pytest.mark.skipif(not _has_nlp, reason="kaos-nlp-core not installed")
class TestConstruction:
    def test_doc_offsets_dense_and_correct(
        self,
        small_corpus: list[ContentDocument],
    ) -> None:
        from kaos_content.indexing import SearchableCorpus

        corpus = SearchableCorpus(small_corpus)
        # Each doc has 2 paragraphs ⇒ size 6, offsets [0, 2, 4, 6].
        assert corpus.size == 6
        assert corpus._doc_offsets == [0, 2, 4, 6]

    def test_size_matches_sum_of_units(
        self,
        small_corpus: list[ContentDocument],
    ) -> None:
        from kaos_content.indexing import SearchableCorpus
        from kaos_content.units import iter_paragraph_units
        from kaos_content.views import DocumentView

        corpus = SearchableCorpus(small_corpus)
        expected = sum(len(iter_paragraph_units(DocumentView(d))) for d in small_corpus)
        assert corpus.size == expected

    def test_doc_uris_default_to_metadata_source(
        self,
        small_corpus: list[ContentDocument],
    ) -> None:
        from kaos_content.indexing import SearchableCorpus

        corpus = SearchableCorpus(small_corpus)
        assert corpus.doc_uris == ("doc1.pdf", "doc2.pdf", "doc3.pdf")

    def test_doc_uris_fallback_to_anon(self) -> None:
        from kaos_content.indexing import SearchableCorpus

        # Doc with no source.uri set.
        bare = ContentDocument(
            metadata=DocumentMetadata(),
            body=(_para("just some text"),),
        )
        corpus = SearchableCorpus([bare])
        assert corpus.doc_uris == ("doc:anon-0",)

    def test_doc_uris_explicit_override(
        self,
        small_corpus: list[ContentDocument],
    ) -> None:
        from kaos_content.indexing import SearchableCorpus

        corpus = SearchableCorpus(small_corpus, doc_uris=("a", "b", "c"))
        assert corpus.doc_uris == ("a", "b", "c")

    def test_invalid_doc_uris_length_raises(
        self,
        small_corpus: list[ContentDocument],
    ) -> None:
        from kaos_content.indexing import SearchableCorpus

        with pytest.raises(ValueError, match="doc_uris length"):
            SearchableCorpus(small_corpus, doc_uris=("only-one",))

    def test_invalid_retrieval_raises(
        self,
        small_corpus: list[ContentDocument],
    ) -> None:
        from kaos_content.indexing import SearchableCorpus

        with pytest.raises(ValueError, match="Unknown retrieval mode"):
            SearchableCorpus(small_corpus, retrieval="bogus")  # ty: ignore[invalid-argument-type]

    def test_num_documents(self, small_corpus: list[ContentDocument]) -> None:
        from kaos_content.indexing import SearchableCorpus

        corpus = SearchableCorpus(small_corpus)
        assert corpus.num_documents == 3
        assert len(corpus.documents) == 3


# ─── BM25 corpus-wide IDF ──────────────────────────────────────────────────


@pytest.mark.skipif(not _has_nlp, reason="kaos-nlp-core not installed")
class TestBm25CorpusWide:
    def test_every_result_has_doc_index(self, small_corpus: list[ContentDocument]) -> None:
        from kaos_content.indexing import SearchableCorpus

        corpus = SearchableCorpus(small_corpus)
        results = corpus.search("contract")
        assert results.results
        for r in results.results:
            assert r.doc_index is not None
            assert 0 <= r.doc_index < corpus.num_documents

    def test_doc_uri_matches_doc_index(self, small_corpus: list[ContentDocument]) -> None:
        from kaos_content.indexing import SearchableCorpus

        corpus = SearchableCorpus(small_corpus)
        results = corpus.search("contract")
        for r in results.results:
            assert r.doc_index is not None
            assert r.doc_uri == corpus.doc_uris[r.doc_index]

    def test_results_can_come_from_multiple_docs(
        self,
        small_corpus: list[ContentDocument],
    ) -> None:
        """`contract` appears in doc1 AND doc3 — both should hit."""
        from kaos_content.indexing import SearchableCorpus

        corpus = SearchableCorpus(small_corpus)
        results = corpus.search("contract", top_k=10)
        doc_idxs = {r.doc_index for r in results.results}
        assert {0, 2}.issubset(doc_idxs), f"expected doc_index 0 and 2; got {doc_idxs}"

    def test_block_ref_round_trips(self, small_corpus: list[ContentDocument]) -> None:
        from kaos_content.indexing import SearchableCorpus

        corpus = SearchableCorpus(small_corpus)
        results = corpus.search("contract")
        # Every result's block_ref must reference a real block in the
        # named document — preserves AST grounding.
        for r in results.results:
            assert r.doc_index is not None
            assert r.block_ref
            doc = corpus.documents[r.doc_index]
            block_idx = int(r.block_ref.rsplit("/", 1)[-1])
            assert 0 <= block_idx < len(doc.body)

    def test_doc_for_row_inverse(self, small_corpus: list[ContentDocument]) -> None:
        from kaos_content.indexing import SearchableCorpus

        corpus = SearchableCorpus(small_corpus)
        # Row 0 → doc 0; row 2 → doc 1; row 4 → doc 2.
        assert corpus.doc_for_row(0)[0] == 0
        assert corpus.doc_for_row(2)[0] == 1
        assert corpus.doc_for_row(4)[0] == 2

    def test_doc_for_row_out_of_range(self, small_corpus: list[ContentDocument]) -> None:
        from kaos_content.indexing import SearchableCorpus

        corpus = SearchableCorpus(small_corpus)
        with pytest.raises(IndexError):
            corpus.doc_for_row(corpus.size)


# ─── Embedding / hybrid (fake-tokenizer based) ─────────────────────────────


def _patch_embedding_layer(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
    """Replace `_get_embedding_model` with a deterministic fake."""
    import numpy as np

    from kaos_content import indexing as indexing_mod
    from kaos_content import search as search_mod

    seen: list[str | None] = []

    class _FakeModel:
        dim = 4

        def embed(self, texts: Iterable[str]) -> object:
            text_list = list(texts)
            n = len(text_list)
            # Deterministic but text-dependent vectors so cosine
            # similarity is non-degenerate.
            mat = np.zeros((n, self.dim), dtype=np.float32)
            for i, t in enumerate(text_list):
                # Project on dim from first 4 chars' ord values.
                for j, ch in enumerate(t[:4]):
                    mat[i, j] = (ord(ch) % 16) / 16.0
            # Avoid zero vectors so L2 normalize doesn't divide by zero.
            mat[:, 0] += 0.1
            return mat

        def count_tokens(self, texts: list[str]) -> list[int]:
            return [len(t.split()) for t in texts]

    def _fake_get(model_id: str | None) -> _FakeModel:
        seen.append(model_id)
        return _FakeModel()

    monkeypatch.setattr(search_mod, "_get_embedding_model", _fake_get)
    monkeypatch.setattr(search_mod, "_ensure_transformers_available", lambda: None)
    monkeypatch.setattr(indexing_mod, "_ensure_transformers_available", lambda: None)
    return seen


@pytest.mark.skipif(not _has_nlp, reason="kaos-nlp-core not installed")
class TestEmbeddings:
    def test_lazy_matrix_build(
        self,
        monkeypatch: pytest.MonkeyPatch,
        small_corpus: list[ContentDocument],
    ) -> None:
        _patch_embedding_layer(monkeypatch)
        from kaos_content.indexing import SearchableCorpus

        corpus = SearchableCorpus(small_corpus, retrieval="embeddings")
        assert corpus._doc_embeddings is None
        results = corpus.search("contract")
        assert corpus._doc_embeddings is not None
        assert results.results

    def test_matrix_reused_across_queries(
        self,
        monkeypatch: pytest.MonkeyPatch,
        small_corpus: list[ContentDocument],
    ) -> None:
        _patch_embedding_layer(monkeypatch)
        from kaos_content.indexing import SearchableCorpus

        corpus = SearchableCorpus(small_corpus, retrieval="embeddings")
        corpus.search("contract")
        first_matrix = corpus._doc_embeddings
        corpus.search("flour")
        # Same object — second query reuses the cached matrix.
        assert corpus._doc_embeddings is first_matrix

    def test_max_embed_rows_guardrail(
        self,
        monkeypatch: pytest.MonkeyPatch,
        small_corpus: list[ContentDocument],
    ) -> None:
        _patch_embedding_layer(monkeypatch)
        from kaos_content.indexing import SearchableCorpus

        corpus = SearchableCorpus(small_corpus, retrieval="embeddings", max_embed_rows=3)
        # Construction succeeds; the dense query is what raises.
        with pytest.raises(ValueError, match="max_embed_rows"):
            corpus.search("contract")

    def test_model_id_propagated(
        self,
        monkeypatch: pytest.MonkeyPatch,
        small_corpus: list[ContentDocument],
    ) -> None:
        seen = _patch_embedding_layer(monkeypatch)
        from kaos_content.indexing import SearchableCorpus

        corpus = SearchableCorpus(
            small_corpus,
            retrieval="embeddings",
            model_id="custom/model",
        )
        corpus.search("contract")
        assert seen, "expected at least one _get_embedding_model call"
        assert all(mid == "custom/model" for mid in seen), seen


@pytest.mark.skipif(not _has_nlp, reason="kaos-nlp-core not installed")
class TestHybrid:
    def test_hybrid_uses_corpus_wide_candidates(
        self,
        monkeypatch: pytest.MonkeyPatch,
        small_corpus: list[ContentDocument],
    ) -> None:
        _patch_embedding_layer(monkeypatch)
        from kaos_content.indexing import SearchableCorpus

        corpus = SearchableCorpus(small_corpus, retrieval="hybrid")
        results = corpus.search("contract", rerank_candidate_k=10)
        # Hybrid pulls candidates from corpus-wide BM25, then reranks. We
        # don't pin a specific score order (depends on the fake matrix),
        # but every returned result must trace back to a real row.
        assert results.results
        for r in results.results:
            assert r.doc_index is not None
            assert 0 <= r.doc_index < corpus.num_documents

    def test_rerank_top_k_default(
        self,
        monkeypatch: pytest.MonkeyPatch,
        small_corpus: list[ContentDocument],
    ) -> None:
        _patch_embedding_layer(monkeypatch)
        from kaos_content.indexing import SearchableCorpus

        corpus = SearchableCorpus(small_corpus, retrieval="hybrid")
        results = corpus.search("contract", top_k=2, rerank_top_k=None)
        # When rerank_top_k is None, top_k acts as the post-rerank cap.
        assert len(results.results) <= 2

    def test_reranker_model_id_stored(
        self,
        monkeypatch: pytest.MonkeyPatch,
        small_corpus: list[ContentDocument],
    ) -> None:
        _patch_embedding_layer(monkeypatch)
        from kaos_content.indexing import SearchableCorpus

        corpus = SearchableCorpus(
            small_corpus,
            retrieval="hybrid",
            reranker_model_id="custom/reranker",
        )
        # The class itself does not invoke a reranker (per design §6);
        # it just stores the id for downstream consumers.
        assert corpus.reranker_model_id == "custom/reranker"


@pytest.mark.skipif(not _has_nlp, reason="kaos-nlp-core not installed")
class TestEmpty:
    def test_no_documents(self) -> None:
        from kaos_content.indexing import SearchableCorpus

        corpus = SearchableCorpus([])
        assert corpus.size == 0
        assert corpus.num_documents == 0
        results = corpus.search("anything")
        assert results.results == []
        assert results.total_matches == 0

    def test_all_empty_documents(self) -> None:
        from kaos_content.indexing import SearchableCorpus

        empty = ContentDocument(metadata=DocumentMetadata(), body=())
        corpus = SearchableCorpus([empty, empty])
        assert corpus.size == 0
        assert corpus.num_documents == 2
        results = corpus.search("anything")
        assert results.results == []


@pytest.mark.skipif(not _has_nlp, reason="kaos-nlp-core not installed")
class TestSentenceLevel:
    def test_sentence_offsets_threaded(self) -> None:
        from kaos_content.indexing import SearchableCorpus

        # Two-sentence paragraph in one of the docs.
        doc = ContentDocument(
            metadata=DocumentMetadata(source=SourceRef(uri="multi.pdf")),
            body=(_para("First sentence about contracts. Second sentence about damages."),),
        )
        corpus = SearchableCorpus([doc], level="sentence")
        results = corpus.search("contracts")
        assert results.results
        r = results.results[0]
        assert r.char_start is not None
        assert r.char_end is not None
        assert r.char_end > r.char_start
        assert r.doc_index == 0
        assert r.doc_uri == "multi.pdf"


# ─── Backwards compatibility ───────────────────────────────────────────────


class TestBackcompat:
    def test_search_result_doc_fields_default_to_none(self) -> None:
        """Existing single-doc callers see SearchResult.doc_index == None."""
        from kaos_content.search import SearchResult

        r = SearchResult(
            text="hello",
            score=1.0,
            block_ref="#/body/0",
            page=1,
            section_ref=None,
            section_title=None,
        )
        assert r.doc_index is None
        assert r.doc_uri is None

    @pytest.mark.skipif(not _has_nlp, reason="kaos-nlp-core not installed")
    def test_search_document_results_lack_doc_fields(self) -> None:
        """`search_document` continues to return doc_index=None."""
        from kaos_content.search import search_document

        doc = _doc("local.pdf", ["Hello, world.", "Goodbye, world."])
        results = search_document(doc, "hello")
        for r in results.results:
            assert r.doc_index is None
            assert r.doc_uri is None
