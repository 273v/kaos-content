"""Tests for revision authoring helpers (UC4).

Verifies the ``make_*`` constructors and document-level helpers that
create tracked-change wrappers on clean documents so agents can propose
edits that Word displays as redlines.
"""

from __future__ import annotations

from datetime import UTC, datetime

from kaos_content.model.attr import Attr
from kaos_content.model.blocks import Div, Paragraph
from kaos_content.model.document import ContentDocument, DocumentMetadata
from kaos_content.model.inlines import Span, Text
from kaos_content.revision import (
    Revisions,
    append_block_insertion,
    delete_block_at,
    insert_block_after,
    make_block_deletion,
    make_block_insertion,
    make_inline_deletion,
    make_inline_insertion,
)


def _plain_doc() -> ContentDocument:
    return ContentDocument(
        metadata=DocumentMetadata(title=""),
        body=(
            Paragraph(children=(Text(value="First."),)),
            Paragraph(children=(Text(value="Second."),)),
            Paragraph(children=(Text(value="Third."),)),
        ),
    )


# ---------------------------------------------------------------------------
# Inline constructors
# ---------------------------------------------------------------------------


class TestInlineConstructors:
    def test_make_inline_insertion_returns_span(self) -> None:
        span = make_inline_insertion(
            Text(value="new"), author="Alice", date=datetime(2026, 4, 18, tzinfo=UTC)
        )
        assert isinstance(span, Span)
        assert "rev-ins" in span.attr.classes
        assert span.attr.kv["rev:author"] == "Alice"
        assert span.attr.kv["rev:date"] == "2026-04-18T00:00:00Z"

    def test_make_inline_deletion_returns_span(self) -> None:
        span = make_inline_deletion(Text(value="old"), author="Bob")
        assert isinstance(span, Span)
        assert "rev-del" in span.attr.classes
        assert span.attr.kv["rev:author"] == "Bob"
        # date omitted → not in kv
        assert "rev:date" not in span.attr.kv

    def test_make_inline_insertion_accepts_tuple(self) -> None:
        span = make_inline_insertion((Text(value="a"), Text(value="b")), author="Alice")
        assert len(span.children) == 2

    def test_make_inline_explicit_id(self) -> None:
        span = make_inline_insertion(Text(value="x"), author="A", revision_id="42")
        assert span.attr.kv["rev:id"] == "42"


# ---------------------------------------------------------------------------
# Block constructors
# ---------------------------------------------------------------------------


class TestBlockConstructors:
    def test_make_block_insertion_returns_div(self) -> None:
        div = make_block_insertion(Paragraph(children=(Text(value="para"),)), author="Alice")
        assert isinstance(div, Div)
        assert "rev-ins" in div.attr.classes

    def test_make_block_deletion_returns_div(self) -> None:
        div = make_block_deletion(Paragraph(children=(Text(value="para"),)), author="Bob")
        assert isinstance(div, Div)
        assert "rev-del" in div.attr.classes


# ---------------------------------------------------------------------------
# Document-level helpers: append, insert, delete
# ---------------------------------------------------------------------------


class TestDocumentLevelHelpers:
    def test_append_block_insertion(self) -> None:
        doc = _plain_doc()
        result = append_block_insertion(
            doc,
            Paragraph(children=(Text(value="Added at end."),)),
            author="AI",
            date=datetime(2026, 4, 18, tzinfo=UTC),
        )
        assert len(result.body) == len(doc.body) + 1
        last = result.body[-1]
        assert isinstance(last, Div)
        assert "rev-ins" in last.attr.classes

    def test_insert_block_after_middle(self) -> None:
        doc = _plain_doc()
        result = insert_block_after(
            doc,
            block_index=0,  # after First
            content=Paragraph(children=(Text(value="Inserted."),)),
            author="AI",
        )
        assert len(result.body) == 4
        # Order: First, Inserted(wrapped), Second, Third
        assert isinstance(result.body[1], Div)
        assert "rev-ins" in result.body[1].attr.classes

    def test_insert_block_clamps_high_index(self) -> None:
        """block_index >= len(body) appends at the end."""
        doc = _plain_doc()
        result = insert_block_after(doc, 99, Paragraph(children=(Text(value="End."),)), author="AI")
        assert len(result.body) == 4
        assert isinstance(result.body[-1], Div)

    def test_insert_block_clamps_negative_index(self) -> None:
        """Negative block_index inserts at the start."""
        doc = _plain_doc()
        result = insert_block_after(
            doc, -5, Paragraph(children=(Text(value="Start."),)), author="AI"
        )
        # Clamped to 0 → inserted after position 0 (so between First and Second)
        # This is consistent with min(block_index + 1, len(body)), max(0, ...)
        # -5 + 1 = -4; max(0, -4) = 0 → inserted at position 0 (before First)
        assert len(result.body) == 4
        assert isinstance(result.body[0], Div)

    def test_delete_block_at(self) -> None:
        doc = _plain_doc()
        result = delete_block_at(doc, 1, author="AI")
        assert len(result.body) == 3
        # body[1] was "Second" — now wrapped in rev-del Div
        assert isinstance(result.body[1], Div)
        assert "rev-del" in result.body[1].attr.classes
        # Inner content preserved
        inner_block = result.body[1].children[0]
        assert isinstance(inner_block, Paragraph)
        inner_text = inner_block.children[0]
        assert isinstance(inner_text, Text)
        assert inner_text.value == "Second."

    def test_delete_block_out_of_range_raises(self) -> None:
        doc = _plain_doc()
        try:
            delete_block_at(doc, 99, author="AI")
        except IndexError:
            return
        raise AssertionError("Expected IndexError for out-of-range block_index")


# ---------------------------------------------------------------------------
# Auto ID assignment
# ---------------------------------------------------------------------------


class TestAutoIdAssignment:
    def test_fresh_id_when_none(self) -> None:
        doc = _plain_doc()
        result = append_block_insertion(doc, Paragraph(children=(Text(value="x"),)), author="A")
        div = result.body[-1]
        assert isinstance(div, Div)
        assert div.attr.kv["rev:id"] == "0"

    def test_id_skips_existing(self) -> None:
        """New revisions get IDs that don't collide with existing ones."""
        # Start with a doc containing rev-ins id="0"
        doc = ContentDocument(
            metadata=DocumentMetadata(title=""),
            body=(
                Paragraph(
                    children=(
                        Span(
                            attr=Attr(
                                classes=("rev-ins",),
                                kv={"rev:id": "0", "rev:author": "Prev"},
                            ),
                            children=(Text(value="existing"),),
                        ),
                    )
                ),
            ),
        )
        # Add a new one — should get id "1"
        result = append_block_insertion(doc, Paragraph(children=(Text(value="new"),)), author="A")
        revs = Revisions.from_document(result)
        ids = {r.id for r in revs}
        assert "0" in ids
        assert "1" in ids


# ---------------------------------------------------------------------------
# End-to-end: author then accept
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_author_then_accept_produces_clean_doc(self) -> None:
        """UC4 happy path: build a redline, then accept it to get the final state."""
        from kaos_content.model.blocks import BaseBlock  # noqa: F401
        from kaos_content.revision import accept_all
        from kaos_content.traversal.visitor import extract_text

        doc = _plain_doc()
        with_edits = append_block_insertion(
            doc,
            Paragraph(children=(Text(value="My proposed edit."),)),
            author="AI",
            date=datetime(2026, 4, 18, tzinfo=UTC),
        )
        # One revision present
        assert len(Revisions.from_document(with_edits)) == 1

        accepted = accept_all(with_edits)
        # Revisions gone, edit content is now canonical
        assert len(Revisions.from_document(accepted)) == 0
        texts = [extract_text(b) for b in accepted.body]
        assert "My proposed edit." in texts

    def test_delete_then_reject_restores(self) -> None:
        """Reject a delete → content unwrapped, revision gone."""
        from kaos_content.revision import reject_all

        doc = _plain_doc()
        modified = delete_block_at(doc, 1, author="AI")
        # Revision present
        assert len(Revisions.from_document(modified)) == 1
        # Reject it
        restored = reject_all(modified)
        assert len(Revisions.from_document(restored)) == 0
        # Original content restored
        texts = []
        for block in restored.body:
            if isinstance(block, Paragraph) and block.children:
                first_child = block.children[0]
                if isinstance(first_child, Text):
                    texts.append(first_child.value)
                    continue
            texts.append("")
        assert "Second." in texts
