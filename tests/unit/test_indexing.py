"""Tests for SearchableDocument — pre-built indexed search over ContentDocument.

Verifies that SearchableDocument caches the index and that search results
carry char_start/char_end at sentence level and AST addresses at both levels.
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
from kaos_content.search import SearchResults

_has_nlp = importlib.util.find_spec("kaos_nlp_core") is not None

_source = SourceRef(uri="test.pdf")


def _prov(page: int) -> Provenance:
    return Provenance(source=_source, page=page)


def _para(text: str, page: int) -> Paragraph:
    return Paragraph(children=(Text(value=text),), provenance=_prov(page))


def _heading(text: str, depth: int, page: int) -> Heading:
    return Heading(children=(Text(value=text),), depth=depth, provenance=_prov(page))


@pytest.fixture()
def multi_section_doc() -> ContentDocument:
    """A document with two sections across two pages."""
    return ContentDocument(
        metadata=DocumentMetadata(title="Test Document"),
        body=(
            _heading("Introduction", 1, 1),
            _para(
                "This contract governs the sale of commercial real estate "
                "between the buyer and seller parties.",
                1,
            ),
            _para(
                "The closing date shall be no later than thirty days after "
                "the execution of this agreement.",
                1,
            ),
            _heading("Remedies", 1, 2),
            _para(
                "In the event of a material breach, the non-breaching party "
                "may pursue all available legal remedies including damages.",
                2,
            ),
            _para(
                "Specific performance may be granted by the court when "
                "monetary damages are inadequate to compensate the injured party.",
                2,
            ),
        ),
    )


@pytest.mark.skipif(not _has_nlp, reason="kaos-nlp-core not installed")
class TestSearchableDocumentParagraph:
    """Paragraph-level SearchableDocument tests."""

    def test_basic_search(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="paragraph")
        results = sdoc.search("breach remedies")
        assert isinstance(results, SearchResults)
        assert results.total_matches > 0
        assert "breach" in results.results[0].text.lower()

    def test_block_ref_present(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="paragraph")
        results = sdoc.search("contract sale")
        for r in results.results:
            assert r.block_ref.startswith("#/body/")

    def test_page_present(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="paragraph")
        results = sdoc.search("specific performance")
        perf = [r for r in results.results if "specific performance" in r.text.lower()]
        assert perf
        assert perf[0].page == 2

    def test_section_ref_and_title(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="paragraph")
        results = sdoc.search("breach damages")
        breach = [r for r in results.results if "breach" in r.text.lower()]
        assert breach
        assert breach[0].section_ref is not None
        assert breach[0].section_title == "Remedies"

    def test_path_full_breadcrumb_nested(self) -> None:
        from kaos_content.indexing import SearchableDocument

        doc = ContentDocument(
            metadata=DocumentMetadata(title="Nested"),
            body=(
                _heading("Chapter 1", 1, 1),
                _heading("Section 1.1", 2, 1),
                _para("needle clause", 1),
            ),
        )
        sdoc = SearchableDocument(doc, level="paragraph")
        hit = sdoc.search("needle").results[0]
        assert hit.heading_path == ("Chapter 1",)
        assert hit.section_title == "Section 1.1"
        assert hit.path == ("Chapter 1", "Section 1.1")

    def test_no_char_offsets_at_paragraph_level(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="paragraph")
        results = sdoc.search("contract")
        for r in results.results:
            assert r.char_start is None
            assert r.char_end is None

    def test_reuse_index(self, multi_section_doc: ContentDocument) -> None:
        """Multiple queries should use the same pre-built index."""
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="paragraph")
        r1 = sdoc.search("contract")
        r2 = sdoc.search("breach")
        r3 = sdoc.search("court")
        assert r1.total_matches > 0
        assert r2.total_matches > 0
        assert r3.total_matches > 0

    def test_empty_query_raises(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="paragraph")
        with pytest.raises(ValueError, match="empty"):
            sdoc.search("")

    def test_top_k_limits(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="paragraph")
        results = sdoc.search("the", top_k=2)
        assert len(results.results) <= 2

    def test_preview_length(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="paragraph")
        results = sdoc.search("contract", preview_length=30)
        for r in results.results:
            assert len(r.text) <= 34  # 30 + len("...")

    def test_properties(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="paragraph")
        assert sdoc.document is multi_section_doc
        assert sdoc.level == "paragraph"
        assert sdoc.view is not None
        assert len(sdoc.units) > 0


@pytest.mark.skipif(not _has_nlp, reason="kaos-nlp-core not installed")
class TestSearchableDocumentSentence:
    """Sentence-level SearchableDocument tests — verifies char offsets."""

    def test_basic_sentence_search(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="sentence")
        results = sdoc.search("closing date")
        assert results.total_matches > 0
        assert "closing" in results.results[0].text.lower()

    def test_char_offsets_populated(self, multi_section_doc: ContentDocument) -> None:
        """Sentence-level results must carry char_start and char_end."""
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="sentence")
        results = sdoc.search("breach")
        breach = [r for r in results.results if "breach" in r.text.lower()]
        assert breach
        r = breach[0]
        assert r.char_start is not None
        assert r.char_end is not None
        assert r.char_start >= 0
        assert r.char_end > r.char_start

    def test_char_offsets_correct(self, multi_section_doc: ContentDocument) -> None:
        """char_start/char_end should slice the paragraph text to the sentence."""
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="sentence")
        results = sdoc.search("breach")

        # Find the paragraph that contains the breach sentence
        for r in results.results:
            if r.char_start is None:
                continue
            # Look up the paragraph via the view
            para = next(
                (p for p in sdoc.view.paragraphs if p.block_ref == r.block_ref),
                None,
            )
            if para is None:
                continue
            # The char offsets should slice the paragraph text to produce
            # text that matches (or is contained in) the result text
            sliced = para.text[r.char_start : r.char_end]
            # The result text may be truncated by preview_length, so check
            # that the slice starts the same way
            assert sliced.startswith(r.text[: min(len(r.text), 20)])

    def test_sentence_block_ref(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="sentence")
        results = sdoc.search("damages")
        for r in results.results:
            assert r.block_ref.startswith("#/body/")

    def test_sentence_page(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="sentence")
        results = sdoc.search("court monetary")
        court = [r for r in results.results if "court" in r.text.lower()]
        assert court
        assert court[0].page == 2

    def test_sentence_section_title(self, multi_section_doc: ContentDocument) -> None:
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(multi_section_doc, level="sentence")
        results = sdoc.search("breach")
        breach = [r for r in results.results if "breach" in r.text.lower()]
        assert breach
        assert breach[0].section_title == "Remedies"

    async def test_search_corpus_threads_sentence_passage_uri(
        self, multi_section_doc: ContentDocument
    ) -> None:
        from kaos_content.indexing import SearchableDocument
        from kaos_content.model.metadata import DocumentMetadata
        from kaos_content.search import search_corpus

        doc = multi_section_doc.model_copy(
            update={"metadata": DocumentMetadata(title="Test Document", source=_source)}
        )
        sdoc = SearchableDocument(doc, level="sentence")

        results = await search_corpus([sdoc], "breach", top_k=2)

        assert results
        assert results[0].metadata["passage_uri"].startswith("test.pdf#c")


@pytest.mark.skipif(not _has_nlp, reason="kaos-nlp-core not installed")
class TestSearchDocumentCharOffsets:
    """Test that search_document also threads char offsets at sentence level."""

    def test_sentence_char_offsets_via_search_document(
        self, multi_section_doc: ContentDocument
    ) -> None:
        from kaos_content.search import search_document

        results = search_document(multi_section_doc, "breach", level="sentence")
        breach = [r for r in results.results if "breach" in r.text.lower()]
        assert breach
        r = breach[0]
        assert r.char_start is not None
        assert r.char_end is not None
        assert r.char_end > r.char_start

    def test_paragraph_no_char_offsets_via_search_document(
        self, multi_section_doc: ContentDocument
    ) -> None:
        from kaos_content.search import search_document

        results = search_document(multi_section_doc, "contract", level="paragraph")
        for r in results.results:
            assert r.char_start is None
            assert r.char_end is None


@pytest.mark.skipif(not _has_nlp, reason="kaos-nlp-core not installed")
class TestSearchableDocumentEmpty:
    """Edge cases: empty documents."""

    def test_empty_document(self) -> None:
        from kaos_content.indexing import SearchableDocument

        doc = ContentDocument(metadata=DocumentMetadata(), body=())
        sdoc = SearchableDocument(doc, level="paragraph")
        results = sdoc.search("anything")
        assert results.total_matches == 0
        assert results.results == []

    def test_empty_document_sentence(self) -> None:
        from kaos_content.indexing import SearchableDocument

        doc = ContentDocument(metadata=DocumentMetadata(), body=())
        sdoc = SearchableDocument(doc, level="sentence")
        results = sdoc.search("anything")
        assert results.total_matches == 0
        assert results.results == []


# ─── KNT-601 audit M-3: passage_uri provenance for synthetic vs real docs ───
#
# `search_corpus(dict[uri, text])` constructs synthetic SearchableDocuments
# from plain strings via `DocumentBuilder().paragraph(text)`. Those docs
# always have block_ref="#/body/0" — the same value a *real* one-paragraph
# document also produces. The `_kaos_synthetic_corpus` sentinel on
# `metadata.extra` lets `_searchable_passage_uri` tell them apart so a
# real doc keeps its block-ref-derived URI while a synthetic doc falls
# back to the char_start / hash form.


@pytest.mark.skipif(not _has_nlp, reason="kaos-nlp-core not installed")
class TestPassageUriSyntheticFlag:
    async def test_dict_mode_corpus_uses_hash_fallback(self) -> None:
        """search_corpus({uri: text}) hits should NOT carry a #/body/0
        passage_uri — they're synthetic single-paragraph wrappers, not
        AST addresses."""
        from kaos_content.search import search_corpus

        results = await search_corpus(
            {"contract.txt": "The seller hereby grants an exclusive license to the buyer."},
            "exclusive license",
            top_k=2,
        )
        assert results
        passage_uri = results[0].metadata["passage_uri"]
        # Synthetic docs fall through to the hash fallback (paragraph
        # level has no char_start).
        assert "#/body/0" not in passage_uri, (
            f"synthetic dict-mode passage_uri leaked block_ref: {passage_uri!r}"
        )
        assert passage_uri.startswith("contract.txt#h"), passage_uri

    async def test_real_one_paragraph_doc_keeps_block_ref(
        self,
        multi_section_doc: ContentDocument,
    ) -> None:
        """A LEGITIMATE one-paragraph SearchableDocument keeps its
        ``#/body/0`` passage_uri because the block_ref is meaningful."""
        from kaos_content.indexing import SearchableDocument
        from kaos_content.model.attr import SourceRef
        from kaos_content.model.metadata import DocumentMetadata
        from kaos_content.search import search_corpus

        # Build a single-paragraph document with a real source URI and
        # NO synthetic-corpus flag. The audit's M-3 hazard was that the
        # old block_ref != "#/body/0" heuristic conflated this case
        # with the synthetic-dict path.
        single_para = ContentDocument(
            metadata=DocumentMetadata(source=SourceRef(uri="real.pdf")),
            body=(_para("The buyer shall indemnify the seller for losses.", page=1),),
        )
        sdoc = SearchableDocument(single_para, level="paragraph")
        results = await search_corpus([sdoc], "indemnify", top_k=2)
        assert results
        passage_uri = results[0].metadata["passage_uri"]
        # Real doc's block_ref must round-trip into the URI.
        assert passage_uri == "real.pdf#/body/0", (
            f"real one-paragraph doc lost block_ref provenance: {passage_uri!r}"
        )


# ─── KNT-601 P6.2: model_id propagation ─────────────────────────────────────
#
# The embedding-backed retrieval modes ("embeddings", "hybrid") accept a
# `model_id` argument that flows from `search_document` / `SearchableDocument`
# /` search_corpus` down to `kaos_nlp_transformers.EmbeddingModel.load`. We
# verify the plumbing without requiring the real Rust extension by
# monkeypatching the module-level cache helper — the real load is exercised
# in tests/integration when kaos-nlp-transformers is installed.


class TestModelIdPropagation:
    """`model_id` flows from public surface to `EmbeddingModel.load`.

    Patches `kaos_content.search._get_embedding_model` so the test runs
    whether or not kaos-nlp-transformers is installed. The fake records
    the model_id seen on each call and returns a numpy-backed stub
    sufficient for the embedding code path.
    """

    @staticmethod
    def _patch_embedding_layer(monkeypatch: pytest.MonkeyPatch) -> list[str | None]:
        """Replace `_get_embedding_model` with a fake; return the call log.

        Also bypasses the optional-dep check so embedding paths don't
        short-circuit on missing kaos-nlp-transformers.
        """
        import numpy as np

        from kaos_content import search as search_mod

        seen: list[str | None] = []

        class _FakeModel:
            dim = 4

            def embed(self, texts: Iterable[str]) -> object:
                # Accept any sequence; return a deterministic (N, dim) matrix.
                n = len(list(texts))
                return np.ones((n, self.dim), dtype=np.float32)

        def _fake_get(model_id: str | None) -> _FakeModel:
            seen.append(model_id)
            return _FakeModel()

        monkeypatch.setattr(search_mod, "_get_embedding_model", _fake_get)
        monkeypatch.setattr(search_mod, "_ensure_transformers_available", lambda: None)
        # `kaos_content.indexing` re-imports `_ensure_transformers_available`
        # at module load, so its rebound reference also needs patching.
        from kaos_content import indexing as indexing_mod

        monkeypatch.setattr(indexing_mod, "_ensure_transformers_available", lambda: None)
        return seen

    def test_default_model_id_is_none(
        self,
        monkeypatch: pytest.MonkeyPatch,
        multi_section_doc: ContentDocument,
    ) -> None:
        seen = self._patch_embedding_layer(monkeypatch)
        from kaos_content.search import search_document

        results = search_document(
            multi_section_doc,
            "breach",
            retrieval="embeddings",
        )
        # Returns *something* (results may be empty or non-empty depending on
        # the fake's similarity outputs; the contract here is propagation).
        assert results is not None
        assert seen, "expected at least one _get_embedding_model call"
        assert all(mid is None for mid in seen), seen

    def test_explicit_model_id_propagates_through_search_document(
        self,
        monkeypatch: pytest.MonkeyPatch,
        multi_section_doc: ContentDocument,
    ) -> None:
        seen = self._patch_embedding_layer(monkeypatch)
        from kaos_content.search import search_document

        search_document(
            multi_section_doc,
            "breach",
            retrieval="embeddings",
            model_id="intfloat/e5-large-v2",
        )
        assert seen and all(mid == "intfloat/e5-large-v2" for mid in seen), seen

    @pytest.mark.skipif(not _has_nlp, reason="kaos-nlp-core not installed")
    def test_searchable_document_records_and_propagates_model_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
        multi_section_doc: ContentDocument,
    ) -> None:
        seen = self._patch_embedding_layer(monkeypatch)
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(
            multi_section_doc,
            retrieval="embeddings",
            model_id="custom/model",
        )
        assert sdoc.model_id == "custom/model"

        sdoc.search("breach")
        # Both _embed_texts (corpus) and _embed_query (query) must see the
        # same custom model id — H-1/H-2 fix.
        assert seen, "expected at least one _get_embedding_model call"
        assert all(mid == "custom/model" for mid in seen), seen

    @pytest.mark.skipif(not _has_nlp, reason="kaos-nlp-core not installed")
    def test_hybrid_propagates_model_id(
        self,
        monkeypatch: pytest.MonkeyPatch,
        multi_section_doc: ContentDocument,
    ) -> None:
        seen = self._patch_embedding_layer(monkeypatch)
        from kaos_content.indexing import SearchableDocument

        sdoc = SearchableDocument(
            multi_section_doc,
            retrieval="hybrid",
            model_id="custom/model",
        )
        sdoc.search("breach", rerank_top_k=3, rerank_candidate_k=10)
        assert seen and all(mid == "custom/model" for mid in seen), seen
