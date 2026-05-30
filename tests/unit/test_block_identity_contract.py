"""Cross-module contract: block identity + sentence count.

kaos-agents builds a multi-document provenance map keyed by ``id(block)``
(``_resolve_corpus_view_with_document``'s ``block_id_to_source_uri``) and
later resolves a finding's source file by looking up
``id(view.document.body[idx])``. That only works if **``ContentDocument``
construction preserves block object identity** — if it ever copied/
re-validated blocks, every lookup would miss and findings would lose
their source attribution silently (the 2026-05-30 NDA-matrix attribution
class of bug). These tests pin that contract here, at the source, so a
change to copy-on-construct fails loudly in kaos-content rather than
silently degrading a downstream consumer.

The sentence-count test pins the other contract kaos-agents depends on:
``DocumentView.sentences`` is the cost driver behind the full-scan-vs-
narrow budget decision, so it must be stable + deterministic.
"""

from __future__ import annotations

from kaos_content import ContentDocument, DocumentMetadata, Paragraph, Text
from kaos_content.views import DocumentView


class _MockSegmenter:
    """Sentence segmenter matching the PunktTokenizer span API."""

    def tokenize_spans(self, text: str) -> list[tuple[int, int]]:
        spans, start = [], 0
        while start < len(text):
            end = text.find(". ", start)
            if end == -1:
                spans.append((start, len(text)))
                break
            spans.append((start, end + 1))
            start = end + 2
        return spans


def _para(text: str) -> Paragraph:
    return Paragraph(children=(Text(value=text),))


def test_construction_preserves_block_identity() -> None:
    """``ContentDocument(body=blocks)`` must reuse the same block objects
    by reference — kaos-agents keys its source-uri map by ``id(block)``."""
    blocks = tuple(_para(f"clause {i}") for i in range(6))
    doc = ContentDocument(body=blocks)
    for original, in_doc in zip(blocks, doc.body, strict=True):
        assert in_doc is original, "ContentDocument must not copy body blocks"


def test_merged_multidoc_provenance_map_resolves() -> None:
    """Reproduce kaos-agents' merge: gather blocks from several sources
    into one body, key ``id(block) -> source`` BEFORE construction, then
    resolve via ``id(doc.body[idx])`` AFTER construction. Every block
    must resolve to its originating source."""
    sources = {
        "EMNA Mutual NDA.docx": [
            _para("EMNA term: two years"),
            _para("EMNA governing law: Delaware"),
        ],
        "MNDA - Acme.docx": [
            _para("Acme term: no fixed end date"),
            _para("Acme governing law: Michigan"),
        ],
    }
    merged: list[Paragraph] = []
    id_to_source: dict[int, str] = {}
    for src, blocks in sources.items():
        for b in blocks:
            merged.append(b)
            id_to_source[id(b)] = src

    doc = ContentDocument(metadata=DocumentMetadata(title="merged"), body=tuple(merged))

    # The provenance lookup kaos-agents performs at finding-emit time.
    resolved = [id_to_source.get(id(b)) for b in doc.body]
    assert resolved == [
        "EMNA Mutual NDA.docx",
        "EMNA Mutual NDA.docx",
        "MNDA - Acme.docx",
        "MNDA - Acme.docx",
    ]
    assert None not in resolved, "every merged block must resolve to a source"


def test_sentence_count_is_stable_and_deterministic() -> None:
    """``DocumentView.sentences`` is the cost driver for the full-scan
    budget decision — it must be stable across calls (cached) and count
    every segmented sentence."""
    doc = ContentDocument(
        body=(_para("First sentence. Second sentence. Third sentence."), _para("Lone sentence.")),
    )
    view = DocumentView(doc, sentence_segmenter=_MockSegmenter())
    assert view.has_sentences
    first = len(view.sentences)
    assert first == 4  # 3 + 1
    assert len(view.sentences) == first  # cached / stable on repeat access
