"""Tests for the move authoring helpers and the ``view()`` convenience."""

from __future__ import annotations

from kaos_content.model.attr import Attr
from kaos_content.model.blocks import Div, Paragraph
from kaos_content.model.document import ContentDocument, DocumentMetadata
from kaos_content.model.inlines import Span, Text
from kaos_content.revision import (
    Revisions,
    RevisionType,
    RevisionView,
    accept_all,
    make_block_move_from,
    make_block_move_to,
    make_inline_move_from,
    make_inline_move_to,
    reject_all,
    view,
)


def _redline_doc() -> ContentDocument:
    """A document with one inline insertion expressed as a rev-ins span."""
    return ContentDocument(
        metadata=DocumentMetadata(title=""),
        body=(
            Paragraph(
                children=(
                    Text(value="Keep "),
                    Span(
                        attr=Attr(classes=("rev-ins",), kv={"rev:id": "0", "rev:author": "A"}),
                        children=(Text(value="added"),),
                    ),
                )
            ),
        ),
    )


class TestMoveConstructors:
    def test_inline_move_from(self) -> None:
        span = make_inline_move_from(Text(value="clause"), author="Alice", move_name="move0")
        assert isinstance(span, Span)
        assert "rev-move-from" in span.attr.classes
        assert span.attr.kv["rev:move-name"] == "move0"

    def test_inline_move_to(self) -> None:
        span = make_inline_move_to(Text(value="clause"), author="Alice", move_name="move0")
        assert "rev-move-to" in span.attr.classes
        assert span.attr.kv["rev:move-name"] == "move0"

    def test_block_move_pair_share_name(self) -> None:
        para = Paragraph(children=(Text(value="x"),))
        mf = make_block_move_from(para, author="A", move_name="move7", revision_id="1")
        mt = make_block_move_to(para, author="A", move_name="move7", revision_id="2")
        assert isinstance(mf, Div)
        assert isinstance(mt, Div)
        assert mf.attr.kv["rev:move-name"] == mt.attr.kv["rev:move-name"] == "move7"
        assert mf.attr.kv["rev:id"] != mt.attr.kv["rev:id"]

    def test_move_revision_types_round_trip(self) -> None:
        para = Paragraph(children=(Text(value="x"),))
        doc = ContentDocument(
            metadata=DocumentMetadata(title=""),
            body=(
                make_block_move_from(para, author="A", move_name="m", revision_id="0"),
                make_block_move_to(para, author="A", move_name="m", revision_id="1"),
            ),
        )
        types = {r.change_type for r in Revisions.from_document(doc)}
        assert types == {RevisionType.MOVE_FROM, RevisionType.MOVE_TO}


class TestView:
    def test_markup_is_identity(self) -> None:
        doc = _redline_doc()
        assert view(doc, RevisionView.MARKUP) is doc

    def test_final_equals_accept_all(self) -> None:
        doc = _redline_doc()
        assert view(doc, RevisionView.FINAL) == accept_all(doc)

    def test_original_equals_reject_all(self) -> None:
        doc = _redline_doc()
        assert view(doc, RevisionView.ORIGINAL) == reject_all(doc)

    def test_final_keeps_insertion_text(self) -> None:
        from kaos_content.traversal.visitor import extract_text

        doc = _redline_doc()
        assert "added" in extract_text(view(doc, RevisionView.FINAL).body[0])

    def test_original_drops_insertion_text(self) -> None:
        from kaos_content.traversal.visitor import extract_text

        doc = _redline_doc()
        assert "added" not in extract_text(view(doc, RevisionView.ORIGINAL).body[0])
