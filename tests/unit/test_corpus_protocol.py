"""Conformance suite for the ``kaos_content.corpus.Corpus`` Protocol.

Verifies every known implementation satisfies the formal Protocol contract
introduced in WS-3.3. New implementations (TabularDocumentCorpus,
FilesystemCorpus, SourceIterCorpus — WS-3.3 follow-ons) must be added to
the parametrization below and pass the shared invariants.

Covered today:
- ``kaos_content.corpus.ContentDocumentCorpus``
- ``kaos_ml_core.Corpus`` (via the alias methods added in WS-3.3)

Both must satisfy:
1. ``isinstance(corpus, Corpus)`` passes.
2. ``corpus.size > 0`` for a non-empty corpus.
3. ``list(corpus.iter_passages())`` has length ``corpus.size``.
4. Each yielded object satisfies the ``Passage`` Protocol.
5. ``corpus.get_passage(row)`` round-trips for every valid row.
6. Out-of-range rows raise ``IndexError`` or ``KeyError`` (impl choice).
"""

from __future__ import annotations

import importlib.util

import pytest

from kaos_content.corpus import ContentDocumentCorpus, Corpus, Passage
from kaos_content.model.attr import Provenance, SourceRef
from kaos_content.model.blocks import Heading, Paragraph
from kaos_content.model.document import ContentDocument
from kaos_content.model.inlines import Text
from kaos_content.model.metadata import DocumentMetadata

_has_ml_core = importlib.util.find_spec("kaos_ml_core") is not None


_source = SourceRef(uri="doc:test-corpus-protocol")


def _prov(page: int) -> Provenance:
    return Provenance(source=_source, page=page)


def _para(text: str, page: int = 1) -> Paragraph:
    return Paragraph(children=(Text(value=text),), provenance=_prov(page))


def _heading(text: str, depth: int = 1, page: int = 1) -> Heading:
    return Heading(children=(Text(value=text),), depth=depth, provenance=_prov(page))


@pytest.fixture()
def sample_document() -> ContentDocument:
    """A small document with two sections and four non-empty paragraphs."""
    return ContentDocument(
        metadata=DocumentMetadata(title="Sample"),
        body=(
            _heading("Alpha", 1, 1),
            _para("Alpha first paragraph — introduces the alpha topic.", 1),
            _para("Alpha second paragraph continues the discussion.", 1),
            _heading("Beta", 1, 2),
            _para("Beta first paragraph shifts to a different subject.", 2),
            _para("Beta second paragraph wraps up the example corpus.", 2),
        ),
    )


def _corpus_factories(document: ContentDocument) -> list[tuple[str, Corpus]]:
    """Return (label, corpus_instance) pairs for every known implementation."""
    factories: list[tuple[str, Corpus]] = [
        ("ContentDocumentCorpus", ContentDocumentCorpus([document])),
    ]
    if _has_ml_core:
        from kaos_ml_core.corpus import Corpus as MLCorpus  # ty: ignore[unresolved-import]

        factories.append(("kaos_ml_core.Corpus", MLCorpus.from_paragraphs(document)))
    return factories


@pytest.mark.unit
class TestCorpusProtocolConformance:
    def test_protocol_is_runtime_checkable(self) -> None:
        assert hasattr(Corpus, "__instancecheck__"), (
            "Corpus Protocol must be @runtime_checkable so RAG can isinstance() against it"
        )

    def test_passage_protocol_is_runtime_checkable(self) -> None:
        assert hasattr(Passage, "__instancecheck__")

    def test_all_known_corpora_satisfy_protocol(self, sample_document: ContentDocument) -> None:
        for label, corpus in _corpus_factories(sample_document):
            assert isinstance(corpus, Corpus), (
                f"{label} should satisfy the Corpus Protocol but isinstance failed. "
                "Either a required method is missing or the Protocol signature drifted. "
                "See docs/design/corpus-actual-state.md §8."
            )

    def test_size_is_positive_int(self, sample_document: ContentDocument) -> None:
        for label, corpus in _corpus_factories(sample_document):
            assert isinstance(corpus.size, int), f"{label}: size is not int"
            assert corpus.size > 0, f"{label}: sample_document must produce non-empty corpus"

    def test_iter_passages_length_matches_size(self, sample_document: ContentDocument) -> None:
        for label, corpus in _corpus_factories(sample_document):
            passages = list(corpus.iter_passages())
            assert len(passages) == corpus.size, (
                f"{label}: iter_passages yielded {len(passages)} items, size says {corpus.size}"
            )

    def test_yielded_items_satisfy_passage_protocol(self, sample_document: ContentDocument) -> None:
        for label, corpus in _corpus_factories(sample_document):
            for i, passage in enumerate(corpus.iter_passages()):
                assert isinstance(passage, Passage), (
                    f"{label}: item {i} is not a Passage Protocol instance: {passage!r}"
                )

    def test_get_passage_roundtrip(self, sample_document: ContentDocument) -> None:
        for label, corpus in _corpus_factories(sample_document):
            for row, expected in enumerate(corpus.iter_passages()):
                actual = corpus.get_passage(row)
                assert actual.row == expected.row == row, (
                    f"{label}: row mismatch at {row}: got {actual.row}, expected {row}"
                )
                assert actual.text == expected.text, f"{label}: text mismatch at {row}"
                assert actual.block_ref == expected.block_ref, (
                    f"{label}: block_ref mismatch at {row}"
                )

    def test_get_passage_out_of_range(self, sample_document: ContentDocument) -> None:
        for _label, corpus in _corpus_factories(sample_document):
            with pytest.raises((IndexError, KeyError)):
                corpus.get_passage(corpus.size + 100)
            with pytest.raises((IndexError, KeyError)):
                corpus.get_passage(-1)


@pytest.mark.unit
class TestContentDocumentCorpus:
    """Impl-specific sanity for the new lightweight adapter."""

    def test_dense_rows_across_documents(self) -> None:
        doc_a = ContentDocument(
            metadata=DocumentMetadata(title="A"),
            body=(_para("alpha one"), _para("alpha two")),
        )
        doc_b = ContentDocument(
            metadata=DocumentMetadata(title="B"),
            body=(_para("beta one"), _para("beta two"), _para("beta three")),
        )
        corpus = ContentDocumentCorpus([doc_a, doc_b])
        rows = [p.row for p in corpus.iter_passages()]
        assert rows == [0, 1, 2, 3, 4], (
            f"rows must be dense and contiguous across the document boundary; got {rows}"
        )
        assert corpus.size == 5

    def test_empty_documents_sequence_produces_empty_corpus(self) -> None:
        corpus = ContentDocumentCorpus([])
        assert corpus.size == 0
        assert list(corpus.iter_passages()) == []
        # Empty corpus still satisfies the Protocol — size/iter_passages/get_passage exist.
        assert isinstance(corpus, Corpus)


@pytest.mark.unit
@pytest.mark.skipif(not _has_ml_core, reason="kaos_ml_core not installed")
class TestMLCoreCorpusProtocolAliases:
    """Smoke for the Protocol alias methods added to kaos_ml_core.Corpus."""

    def test_aliases_delegate_to_canonical_accessors(
        self, sample_document: ContentDocument
    ) -> None:
        from kaos_ml_core.corpus import Corpus as MLCorpus  # ty: ignore[unresolved-import]

        corpus = MLCorpus.from_paragraphs(sample_document)
        assert corpus.size == len(corpus)
        assert list(corpus.iter_passages()) == list(corpus)
        for row in range(corpus.size):
            assert corpus.get_passage(row) is corpus.unit(row)
