"""Revisions inside footnotes must be visible to read + transform.

Regression: ``Revisions.from_document`` walked ``body`` but not the
``footnotes`` dict, so tracked changes in footnotes were invisible — and
because ``accept_all`` / ``reject_all`` derive their id sets from that
walk (and ``_apply`` only rewrote ``body``), footnote revisions were left
unresolved. Footnotes are common in contracts and briefs.
"""

from __future__ import annotations

from kaos_content.model.blocks import Paragraph
from kaos_content.model.document import ContentDocument, DocumentMetadata
from kaos_content.model.inlines import Text
from kaos_content.revision import (
    Revisions,
    RevisionType,
    accept_all,
    make_inline_deletion,
    make_inline_insertion,
    reject_all,
)
from kaos_content.traversal.visitor import extract_text


def _doc_with_footnote(span: object) -> ContentDocument:
    return ContentDocument(
        metadata=DocumentMetadata(title=""),
        body=(Paragraph(children=(Text(value="Body paragraph."),)),),
        footnotes={"1": (Paragraph(children=(Text(value="See "), span)),)},  # type: ignore[arg-type]
    )


def _footnote_text(doc: ContentDocument) -> str:
    return "".join(extract_text(b) for blocks in doc.footnotes.values() for b in blocks)


class TestFootnoteRevisions:
    def test_from_document_finds_footnote_insertion(self) -> None:
        ins = make_inline_insertion(Text(value="ADDED"), author="A", revision_id="0")
        doc = _doc_with_footnote(ins)
        revs = Revisions.from_document(doc)
        assert len(revs) == 1
        assert revs.items[0].change_type == RevisionType.INSERTION

    def test_accept_all_resolves_footnote_insertion(self) -> None:
        ins = make_inline_insertion(Text(value="ADDED"), author="A", revision_id="0")
        doc = _doc_with_footnote(ins)
        accepted = accept_all(doc)
        assert "ADDED" in _footnote_text(accepted)
        assert len(Revisions.from_document(accepted)) == 0

    def test_reject_all_drops_footnote_insertion(self) -> None:
        ins = make_inline_insertion(Text(value="ADDED"), author="A", revision_id="0")
        doc = _doc_with_footnote(ins)
        rejected = reject_all(doc)
        assert "ADDED" not in _footnote_text(rejected)
        assert len(Revisions.from_document(rejected)) == 0

    def test_footnote_deletion_restored_on_reject(self) -> None:
        dele = make_inline_deletion(Text(value="GONE"), author="A", revision_id="0")
        doc = _doc_with_footnote(dele)
        assert "GONE" in _footnote_text(reject_all(doc))
        assert "GONE" not in _footnote_text(accept_all(doc))
