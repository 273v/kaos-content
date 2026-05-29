"""Revisions inside table cells must be visible to read + transform.

Regression: ``Revisions.from_document`` walked ``children`` / ``content``
but not a Table's head/bodies/foot → rows → cells, so tracked changes in
table cells were invisible. Because ``accept_all`` / ``reject_all`` derive
their id sets from that walk, table-cell revisions were also left
unresolved. Tables are pervasive in contracts, so this matters for
redlines.
"""

from __future__ import annotations

from kaos_content.model.blocks import Paragraph, Table
from kaos_content.model.document import ContentDocument, DocumentMetadata
from kaos_content.model.inlines import Text
from kaos_content.model.table import Cell, Row, TableSection
from kaos_content.revision import (
    Revisions,
    RevisionType,
    accept_all,
    make_inline_deletion,
    make_inline_insertion,
    reject_all,
)
from kaos_content.traversal.visitor import extract_text


def _doc_with_cell_inline(span: object, *, lead: str = "before ") -> ContentDocument:
    cell = Cell(content=(Paragraph(children=(Text(value=lead), span)),))  # type: ignore[arg-type]
    table = Table(bodies=(TableSection(rows=(Row(cells=(cell,)),)),))
    return ContentDocument(metadata=DocumentMetadata(title=""), body=(table,))


def _body_text(doc: ContentDocument) -> str:
    return "".join(extract_text(b) for b in doc.body)


class TestTableCellRevisions:
    def test_from_document_finds_cell_insertion(self) -> None:
        ins = make_inline_insertion(Text(value="ADDED"), author="A", revision_id="0")
        doc = _doc_with_cell_inline(ins)
        revs = Revisions.from_document(doc)
        assert len(revs) == 1
        assert revs.items[0].change_type == RevisionType.INSERTION

    def test_accept_all_resolves_cell_insertion(self) -> None:
        ins = make_inline_insertion(Text(value="ADDED"), author="A", revision_id="0")
        doc = _doc_with_cell_inline(ins)
        accepted = accept_all(doc)
        assert "ADDED" in _body_text(accepted)
        assert len(Revisions.from_document(accepted)) == 0

    def test_reject_all_drops_cell_insertion(self) -> None:
        ins = make_inline_insertion(Text(value="ADDED"), author="A", revision_id="0")
        doc = _doc_with_cell_inline(ins)
        rejected = reject_all(doc)
        assert "ADDED" not in _body_text(rejected)
        assert len(Revisions.from_document(rejected)) == 0

    def test_cell_deletion_restored_on_reject_dropped_on_accept(self) -> None:
        dele = make_inline_deletion(Text(value="GONE"), author="A", revision_id="0")
        doc = _doc_with_cell_inline(dele)
        assert "GONE" in _body_text(reject_all(doc))
        assert "GONE" not in _body_text(accept_all(doc))
