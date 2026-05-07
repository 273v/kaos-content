"""Tests for revision transforms (accept/reject/at_time).

Covers the accept/reject semantics matrix:

=============== ============ ============
rev_class       accept       reject
=============== ============ ============
rev-ins         unwrap       drop
rev-del         drop         unwrap
rev-move-to     unwrap       drop
rev-move-from   drop         unwrap
=============== ============ ============

And the higher-level helpers: accept_all, reject_all, accept_by_author,
reject_by_author, at_time.
"""

from __future__ import annotations

from datetime import UTC, datetime

from kaos_content.model.annotation import Annotation, AnnotationTarget, AnnotationType
from kaos_content.model.attr import Attr
from kaos_content.model.blocks import Div, Paragraph
from kaos_content.model.document import ContentDocument, DocumentMetadata
from kaos_content.model.inlines import Span, Text
from kaos_content.revision import (
    Revisions,
    accept,
    accept_all,
    accept_by_author,
    at_time,
    reject,
    reject_all,
    reject_by_author,
)
from kaos_content.serializers.text import serialize_text

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _inline_redline() -> ContentDocument:
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
        annotations=(
            Annotation(
                id="ann-0",
                type=AnnotationType.TRACKED_CHANGE,
                targets=(AnnotationTarget(node_ref="#/body/0/children/1"),),
                body={"revision_id": "0", "change_type": "deletion", "author": "Alice"},
            ),
            Annotation(
                id="ann-1",
                type=AnnotationType.TRACKED_CHANGE,
                targets=(AnnotationTarget(node_ref="#/body/0/children/2"),),
                body={"revision_id": "1", "change_type": "insertion", "author": "Alice"},
            ),
        ),
    )


def _multi_author() -> ContentDocument:
    """Alice at T=15, Bob at T=16, Alice at T=17."""
    return ContentDocument(
        metadata=DocumentMetadata(title=""),
        body=(
            Paragraph(
                children=(
                    Text(value="a:"),
                    Span(
                        attr=Attr(
                            classes=("rev-ins",),
                            kv={
                                "rev:id": "1",
                                "rev:author": "Alice",
                                "rev:date": "2026-04-15T10:00:00Z",
                            },
                        ),
                        children=(Text(value="alpha"),),
                    ),
                ),
            ),
            Paragraph(
                children=(
                    Text(value="b:"),
                    Span(
                        attr=Attr(
                            classes=("rev-del",),
                            kv={
                                "rev:id": "2",
                                "rev:author": "Bob",
                                "rev:date": "2026-04-16T11:00:00Z",
                            },
                        ),
                        children=(Text(value="beta"),),
                    ),
                ),
            ),
            Paragraph(
                children=(
                    Text(value="c:"),
                    Span(
                        attr=Attr(
                            classes=("rev-ins",),
                            kv={
                                "rev:id": "3",
                                "rev:author": "Alice",
                                "rev:date": "2026-04-17T09:00:00Z",
                            },
                        ),
                        children=(Text(value="gamma"),),
                    ),
                ),
            ),
        ),
    )


def _block_insertion() -> ContentDocument:
    """An entire paragraph inserted as rev-ins Div."""
    return ContentDocument(
        metadata=DocumentMetadata(title=""),
        body=(
            Paragraph(children=(Text(value="Original."),)),
            Div(
                attr=Attr(
                    classes=("rev-ins",),
                    kv={"rev:id": "5", "rev:author": "Bob", "rev:date": "2026-04-20T12:00:00Z"},
                ),
                children=(Paragraph(children=(Text(value="Inserted."),)),),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# Basic accept / reject semantics
# ---------------------------------------------------------------------------


class TestAcceptReject:
    def test_accept_insertion_unwraps(self) -> None:
        """accept rev-ins unwraps content. rev-del untouched but hidden in view=final."""
        doc = _inline_redline()
        result = accept(doc, "1")
        text = serialize_text(result, view="markup")
        assert "Friday" in text  # unwrapped from rev-ins
        assert "{-Monday-}" in text  # rev-del still there, untouched
        assert len(Revisions.from_document(result)) == 1

    def test_accept_deletion_drops(self) -> None:
        """accept rev-del drops content entirely."""
        doc = _inline_redline()
        result = accept(doc, "0")
        text = serialize_text(result, view="markup")
        assert "Monday" not in text  # dropped
        assert "{+Friday+}" in text  # rev-ins still there

    def test_reject_insertion_drops(self) -> None:
        """reject rev-ins drops content entirely."""
        doc = _inline_redline()
        result = reject(doc, "1")
        text = serialize_text(result, view="markup")
        assert "Friday" not in text  # dropped
        assert "{-Monday-}" in text  # rev-del still there

    def test_reject_deletion_unwraps(self) -> None:
        """reject rev-del unwraps — content stays, no longer wrapped."""
        doc = _inline_redline()
        result = reject(doc, "0")
        text = serialize_text(result, view="markup")
        assert "Monday" in text
        # And no longer wrapped in {-...-} markers
        assert "{-Monday-}" not in text

    def test_unmatched_id_noop(self) -> None:
        doc = _inline_redline()
        result = accept(doc, "999")
        # Nothing changed
        assert len(Revisions.from_document(result)) == 2

    def test_annotation_removed_on_accept(self) -> None:
        doc = _inline_redline()
        assert len(doc.annotations) == 2
        result = accept(doc, "1")
        # ann-1 removed, ann-0 still there
        assert len(result.annotations) == 1
        assert result.annotations[0].body["revision_id"] == "0"


# ---------------------------------------------------------------------------
# Bulk: accept_all / reject_all
# ---------------------------------------------------------------------------


class TestBulkTransforms:
    def test_accept_all_final(self) -> None:
        doc = _inline_redline()
        result = accept_all(doc)
        text = serialize_text(result).strip()
        assert text == "The deadline is Friday."
        assert len(Revisions.from_document(result)) == 0
        # All TRACKED_CHANGE annotations consumed
        tc = [a for a in result.annotations if a.type == AnnotationType.TRACKED_CHANGE]
        assert tc == []

    def test_reject_all_original(self) -> None:
        doc = _inline_redline()
        result = reject_all(doc)
        text = serialize_text(result).strip()
        assert text == "The deadline is Monday."
        assert len(Revisions.from_document(result)) == 0

    def test_accept_all_matches_view_final(self) -> None:
        """accept_all(doc) text should equal serialize_text(doc, view='final')."""
        doc = _inline_redline()
        accepted_text = serialize_text(accept_all(doc))
        final_view = serialize_text(doc, view="final")
        assert accepted_text.strip() == final_view.strip()

    def test_reject_all_matches_view_original(self) -> None:
        doc = _inline_redline()
        rejected_text = serialize_text(reject_all(doc))
        original_view = serialize_text(doc, view="original")
        assert rejected_text.strip() == original_view.strip()


# ---------------------------------------------------------------------------
# By-author
# ---------------------------------------------------------------------------


class TestByAuthor:
    def test_accept_by_author_affects_only_target(self) -> None:
        doc = _multi_author()
        result = accept_by_author(doc, "Alice")
        # Alice's two insertions are accepted (unwrapped → content stays)
        # Bob's deletion is untouched
        remaining = Revisions.from_document(result)
        assert len(remaining) == 1
        assert remaining.items[0].author == "Bob"

    def test_reject_by_author_affects_only_target(self) -> None:
        doc = _multi_author()
        result = reject_by_author(doc, "Alice")
        # Alice's two insertions rejected → dropped
        # Bob's deletion untouched — use view=markup to see it through
        text = serialize_text(result, view="markup")
        assert "alpha" not in text
        assert "gamma" not in text
        assert "{-beta-}" in text  # Bob's rev-del still present, rendered as markup
        remaining = Revisions.from_document(result)
        assert len(remaining) == 1
        assert remaining.items[0].author == "Bob"


# ---------------------------------------------------------------------------
# at_time: the time machine
# ---------------------------------------------------------------------------


class TestAtTime:
    def test_before_all_revisions(self) -> None:
        """t before any revisions → every revision rejected → original document."""
        doc = _multi_author()
        t = datetime(2020, 1, 1, tzinfo=UTC)
        result = at_time(doc, t)
        text = serialize_text(result)
        # All three insertions/deletions rejected:
        # - rev-ins rejected → dropped (alpha, gamma gone)
        # - rev-del rejected → unwrapped (beta stays)
        assert "alpha" not in text
        assert "beta" in text
        assert "gamma" not in text

    def test_after_all_revisions(self) -> None:
        """t after all revisions → every revision accepted → final document."""
        doc = _multi_author()
        t = datetime(2030, 1, 1, tzinfo=UTC)
        result = at_time(doc, t)
        text = serialize_text(result)
        # All accepted:
        # - rev-ins accepted → unwrapped (alpha, gamma stay)
        # - rev-del accepted → dropped (beta gone)
        assert "alpha" in text
        assert "beta" not in text
        assert "gamma" in text

    def test_between_revisions(self) -> None:
        """At T=16.5, only Alice's first change (T=15) is applied."""
        doc = _multi_author()
        t = datetime(2026, 4, 16, 12, 0, 0, tzinfo=UTC)  # between Bob (T=16) and Alice #2 (T=17)
        result = at_time(doc, t)
        text = serialize_text(result)
        # Alice #1 (T=15): accepted → alpha stays
        # Bob (T=16): accepted → beta gone (deletion)
        # Alice #2 (T=17): rejected → gamma gone (insertion)
        assert "alpha" in text
        assert "beta" not in text
        assert "gamma" not in text

    def test_exactly_at_revision_date_inclusive(self) -> None:
        doc = _multi_author()
        # Bob's date is 2026-04-16T11:00:00Z. At exactly that time, Bob is accepted.
        t = datetime(2026, 4, 16, 11, 0, 0, tzinfo=UTC)
        result = at_time(doc, t)
        text = serialize_text(result)
        # Alice #1 (T=15): accepted → alpha stays
        # Bob (T=16): accepted → beta gone
        # Alice #2 (T=17): rejected → gamma gone
        assert "alpha" in text
        assert "beta" not in text
        assert "gamma" not in text


# ---------------------------------------------------------------------------
# Block-level transforms
# ---------------------------------------------------------------------------


class TestBlockLevelTransforms:
    def test_accept_block_insertion_unwraps(self) -> None:
        doc = _block_insertion()
        result = accept_all(doc)
        # The rev-ins Div is unwrapped — inner Paragraph survives
        assert len(result.body) == 2  # Original paragraph + inserted paragraph
        text = serialize_text(result)
        assert "Original" in text
        assert "Inserted" in text

    def test_reject_block_insertion_drops(self) -> None:
        doc = _block_insertion()
        result = reject_all(doc)
        # The rev-ins Div is dropped entirely
        assert len(result.body) == 1
        text = serialize_text(result)
        assert "Original" in text
        assert "Inserted" not in text


# ---------------------------------------------------------------------------
# Immutability: transforms never mutate the input
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_input_not_mutated(self) -> None:
        doc = _inline_redline()
        orig_text = serialize_text(doc)
        orig_annotations = doc.annotations
        _ = accept_all(doc)
        # Original doc unchanged
        assert serialize_text(doc) == orig_text
        assert doc.annotations == orig_annotations

    def test_returns_new_document_instance(self) -> None:
        doc = _inline_redline()
        result = accept(doc, "0")
        assert result is not doc


# ---------------------------------------------------------------------------
# Empty / no-revision documents
# ---------------------------------------------------------------------------


class TestNoRevisions:
    def _plain_doc(self) -> ContentDocument:
        return ContentDocument(
            metadata=DocumentMetadata(title=""),
            body=(Paragraph(children=(Text(value="plain"),)),),
        )

    def test_accept_all_noop(self) -> None:
        doc = self._plain_doc()
        result = accept_all(doc)
        assert serialize_text(result) == serialize_text(doc)

    def test_reject_all_noop(self) -> None:
        doc = self._plain_doc()
        result = reject_all(doc)
        assert serialize_text(result) == serialize_text(doc)

    def test_at_time_noop(self) -> None:
        doc = self._plain_doc()
        result = at_time(doc, datetime.now(UTC))
        assert serialize_text(result) == serialize_text(doc)
