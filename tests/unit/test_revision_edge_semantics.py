"""Pins for edge-case revision-transform semantics.

These are not bugs — they document deliberate behavior so a future change
is intentional: duplicate ``rev:id`` values, accepting only one half of a
move pair, and (pathological) nested revision wrappers, which the
``_collect`` walk explicitly does not recurse into ("revisions never nest
inside one another in OOXML").
"""

from __future__ import annotations

from kaos_content.model.attr import Attr
from kaos_content.model.blocks import Paragraph
from kaos_content.model.document import ContentDocument, DocumentMetadata
from kaos_content.model.inlines import Span, Text
from kaos_content.revision import (
    Revisions,
    accept,
    accept_all,
    make_block_insertion,
    make_block_move_from,
    make_block_move_to,
    reject_all,
)
from kaos_content.traversal.visitor import extract_text


def _body_text(doc: ContentDocument) -> str:
    return "\n".join(extract_text(b) for b in doc.body)


def test_duplicate_revision_id_resolves_all_matching_nodes() -> None:
    doc = ContentDocument(
        metadata=DocumentMetadata(title=""),
        body=(
            make_block_insertion(
                Paragraph(children=(Text(value="first"),)), author="A", revision_id="0"
            ),
            make_block_insertion(
                Paragraph(children=(Text(value="second"),)), author="A", revision_id="0"
            ),
        ),
    )
    assert len(Revisions.from_document(doc)) == 2
    accepted = accept(doc, "0")
    # accept by id applies to every node carrying that id.
    assert "first" in _body_text(accepted)
    assert "second" in _body_text(accepted)
    assert len(Revisions.from_document(accepted)) == 0


def test_accepting_one_half_of_move_leaves_the_other() -> None:
    doc = ContentDocument(
        metadata=DocumentMetadata(title=""),
        body=(
            make_block_move_from(
                Paragraph(children=(Text(value="clause"),)),
                author="A",
                move_name="m",
                revision_id="0",
            ),
            make_block_move_to(
                Paragraph(children=(Text(value="clause"),)),
                author="A",
                move_name="m",
                revision_id="1",
            ),
        ),
    )
    # Accepting only the move-to keeps its content and leaves the move-from
    # pending — a move pair should be resolved together.
    accepted = accept(doc, "1")
    assert len(Revisions.from_document(accepted)) == 1


def test_nested_wrappers_resolve_by_outer_only() -> None:
    inner = Span(
        attr=Attr(classes=("rev-ins",), kv={"rev:id": "1", "rev:author": "A"}),
        children=(Text(value="inner"),),
    )
    outer = Span(
        attr=Attr(classes=("rev-del",), kv={"rev:id": "0", "rev:author": "A"}),
        children=(Text(value="x "), inner),
    )
    doc = ContentDocument(metadata=DocumentMetadata(title=""), body=(Paragraph(children=(outer,)),))
    # Only the outer revision is collected (no nesting in OOXML).
    revs = Revisions.from_document(doc)
    assert len(revs) == 1
    # Accept drops the outer deletion (and its subtree); reject unwraps it.
    assert _body_text(accept_all(doc)) == ""
    assert "x inner" in _body_text(reject_all(doc))
