"""Tests for the typed revision wrapper API (kaos_content.revision)."""

from __future__ import annotations

from datetime import UTC, datetime

from kaos_content.model.attr import Attr
from kaos_content.model.blocks import Div, Paragraph
from kaos_content.model.document import ContentDocument, DocumentMetadata
from kaos_content.model.inlines import Span, Text
from kaos_content.revision import Revision, Revisions, RevisionType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _single_inline_pair() -> ContentDocument:
    """ "The deadline is {-Monday-}{+Friday+}."""
    return ContentDocument(
        metadata=DocumentMetadata(title=""),
        body=(
            Paragraph(
                children=(
                    Text(value="The deadline is "),
                    Span(
                        attr=Attr(
                            classes=("rev-del",),
                            kv={
                                "rev:id": "0",
                                "rev:author": "Alice",
                                "rev:date": "2026-04-15T10:30:00Z",
                            },
                        ),
                        children=(Text(value="Monday"),),
                    ),
                    Span(
                        attr=Attr(
                            classes=("rev-ins",),
                            kv={
                                "rev:id": "1",
                                "rev:author": "Alice",
                                "rev:date": "2026-04-15T10:30:00Z",
                            },
                        ),
                        children=(Text(value="Friday"),),
                    ),
                    Text(value="."),
                ),
            ),
        ),
    )


def _multi_author() -> ContentDocument:
    return ContentDocument(
        metadata=DocumentMetadata(title=""),
        body=(
            Paragraph(
                children=(
                    Span(
                        attr=Attr(
                            classes=("rev-ins",),
                            kv={
                                "rev:id": "1",
                                "rev:author": "Alice",
                                "rev:date": "2026-04-15T10:00:00Z",
                            },
                        ),
                        children=(Text(value="a1"),),
                    ),
                ),
            ),
            Paragraph(
                children=(
                    Span(
                        attr=Attr(
                            classes=("rev-del",),
                            kv={
                                "rev:id": "2",
                                "rev:author": "Bob",
                                "rev:date": "2026-04-16T11:00:00Z",
                            },
                        ),
                        children=(Text(value="b1"),),
                    ),
                ),
            ),
            Paragraph(
                children=(
                    Span(
                        attr=Attr(
                            classes=("rev-ins",),
                            kv={
                                "rev:id": "3",
                                "rev:author": "Alice",
                                "rev:date": "2026-04-17T09:00:00Z",
                            },
                        ),
                        children=(Text(value="a2"),),
                    ),
                ),
            ),
        ),
    )


def _block_insertion() -> ContentDocument:
    return ContentDocument(
        metadata=DocumentMetadata(title=""),
        body=(
            Paragraph(children=(Text(value="First."),)),
            Div(
                attr=Attr(
                    classes=("rev-ins",),
                    kv={"rev:id": "5", "rev:author": "Bob", "rev:date": "2026-04-20T12:00:00Z"},
                ),
                children=(Paragraph(children=(Text(value="Inserted paragraph."),)),),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Revision dataclass
# ---------------------------------------------------------------------------


class TestRevisionDataclass:
    def test_fields_populated(self) -> None:
        revs = Revisions.from_document(_single_inline_pair())
        assert len(revs) == 2
        first = revs.items[0]
        assert first.id == "0"
        assert first.author == "Alice"
        assert first.date == datetime(2026, 4, 15, 10, 30, 0, tzinfo=UTC)
        assert first.change_type == RevisionType.DELETION

    def test_is_block_false_for_span(self) -> None:
        revs = Revisions.from_document(_single_inline_pair())
        assert all(not r.is_block for r in revs.items)

    def test_is_block_true_for_div(self) -> None:
        revs = Revisions.from_document(_block_insertion())
        assert any(r.is_block for r in revs.items)

    def test_text_extraction(self) -> None:
        revs = Revisions.from_document(_single_inline_pair())
        texts = {r.id: r.text for r in revs.items}
        assert texts == {"0": "Monday", "1": "Friday"}

    def test_preview_short(self) -> None:
        revs = Revisions.from_document(_single_inline_pair())
        first = revs.items[0]
        assert first.preview == "Monday"

    def test_frozen(self) -> None:
        """Revision should be immutable."""
        import dataclasses

        rev = Revision(
            node=None,
            node_ref="#/body/0",
            id="x",
            author="a",
            date=None,
            change_type=RevisionType.INSERTION,
        )
        try:
            rev.__setattr__("id", "y")
        except dataclasses.FrozenInstanceError:
            return
        raise AssertionError("Revision should be frozen")


# ---------------------------------------------------------------------------
# Revisions collection
# ---------------------------------------------------------------------------


class TestRevisionsCollection:
    def test_from_document_empty(self) -> None:
        doc = ContentDocument(metadata=DocumentMetadata(title=""), body=())
        revs = Revisions.from_document(doc)
        assert len(revs) == 0
        assert not revs

    def test_from_document_no_revisions(self) -> None:
        doc = ContentDocument(
            metadata=DocumentMetadata(title=""),
            body=(Paragraph(children=(Text(value="plain"),)),),
        )
        revs = Revisions.from_document(doc)
        assert len(revs) == 0

    def test_from_document_counts(self) -> None:
        revs = Revisions.from_document(_single_inline_pair())
        assert len(revs) == 2

    def test_iteration(self) -> None:
        revs = Revisions.from_document(_single_inline_pair())
        ids = [r.id for r in revs]
        assert ids == ["0", "1"]

    def test_by_id(self) -> None:
        revs = Revisions.from_document(_single_inline_pair())
        r0 = revs.by_id("0")
        assert r0 is not None
        assert r0.change_type == RevisionType.DELETION
        assert revs.by_id("does-not-exist") is None

    def test_by_author(self) -> None:
        revs = Revisions.from_document(_multi_author())
        alice = revs.by_author("Alice")
        bob = revs.by_author("Bob")
        assert len(alice) == 2
        assert len(bob) == 1
        assert revs.by_author("Nonexistent") == []

    def test_by_type(self) -> None:
        revs = Revisions.from_document(_multi_author())
        insertions = revs.by_type(RevisionType.INSERTION)
        deletions = revs.by_type(RevisionType.DELETION)
        assert len(insertions) == 2
        assert len(deletions) == 1

    def test_authors_distinct_in_order(self) -> None:
        revs = Revisions.from_document(_multi_author())
        assert revs.authors() == ["Alice", "Bob"]

    def test_between_filters_by_date(self) -> None:
        revs = Revisions.from_document(_multi_author())
        start = datetime(2026, 4, 16, tzinfo=UTC)
        end = datetime(2026, 4, 16, 23, 59, 59, tzinfo=UTC)
        result = revs.between(start=start, end=end)
        assert len(result) == 1
        assert result[0].author == "Bob"

    def test_between_open_ended(self) -> None:
        revs = Revisions.from_document(_multi_author())
        all_rev = revs.between()
        assert len(all_rev) == 3

    def test_sorted_by_date(self) -> None:
        revs = Revisions.from_document(_multi_author())
        sorted_revs = revs.sorted_by_date()
        ids = [r.id for r in sorted_revs]
        assert ids == ["1", "2", "3"]

    def test_summary(self) -> None:
        revs = Revisions.from_document(_multi_author())
        summary = revs.summary()
        assert summary == {
            "Alice": {"insertion": 2},
            "Bob": {"deletion": 1},
        }


# ---------------------------------------------------------------------------
# DOCX round-trip (integration via the reader)
# ---------------------------------------------------------------------------


class TestRealDocumentIntegration:
    """Live-ish: wire the typed API to a document built from raw pydantic models
    in the same shape the DOCX reader produces.
    """

    def test_block_revision_detected(self) -> None:
        revs = Revisions.from_document(_block_insertion())
        assert len(revs) == 1
        r = revs.items[0]
        assert r.is_block
        assert r.change_type == RevisionType.INSERTION
        assert r.author == "Bob"

    def test_node_ref_format(self) -> None:
        """node_ref should be a JSON-pointer-style string anchored at the doc root."""
        revs = Revisions.from_document(_single_inline_pair())
        for r in revs.items:
            assert r.node_ref.startswith("#/body/")

    def test_missing_date_tolerated(self) -> None:
        """Revision without date should parse with date=None."""
        doc = ContentDocument(
            metadata=DocumentMetadata(title=""),
            body=(
                Paragraph(
                    children=(
                        Span(
                            attr=Attr(
                                classes=("rev-ins",),
                                kv={"rev:id": "x", "rev:author": "Someone"},
                            ),
                            children=(Text(value="content"),),
                        ),
                    ),
                ),
            ),
        )
        revs = Revisions.from_document(doc)
        assert len(revs) == 1
        assert revs.items[0].date is None
