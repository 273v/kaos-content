"""Date handling in the revision time-machine must tolerate mixed tz-awareness.

Regression: a date-only ``w:date`` parses to a *naive* datetime; comparing
it against a timezone-*aware* ``at_time`` / ``between`` argument (and vice
versa) raised ``TypeError: can't compare offset-naive and offset-aware
datetimes`` and crashed the time-machine. Dates are now normalized to
aware-UTC at the boundaries.
"""

from __future__ import annotations

from datetime import UTC, datetime

from kaos_content.model.attr import Attr
from kaos_content.model.blocks import Paragraph
from kaos_content.model.document import ContentDocument, DocumentMetadata
from kaos_content.model.inlines import Span, Text
from kaos_content.revision import Revisions, at_time, make_block_insertion
from kaos_content.traversal.visitor import extract_text


def _body_text(doc: ContentDocument) -> str:
    return "\n".join(extract_text(b) for b in doc.body)


def _date_only_ins_doc() -> ContentDocument:
    """A revision whose w:date is date-only → parses naive (pre-fix)."""
    return ContentDocument(
        metadata=DocumentMetadata(title=""),
        body=(
            Paragraph(
                children=(
                    Span(
                        attr=Attr(
                            classes=("rev-ins",),
                            kv={"rev:id": "0", "rev:author": "A", "rev:date": "2026-04-16"},
                        ),
                        children=(Text(value="added"),),
                    ),
                )
            ),
        ),
    )


def test_parsed_date_is_timezone_aware() -> None:
    rev = Revisions.from_document(_date_only_ins_doc()).items[0]
    assert rev.date is not None
    assert rev.date.tzinfo is not None


def test_at_time_aware_arg_against_naive_source_revision() -> None:
    doc = _date_only_ins_doc()
    after = at_time(doc, datetime(2026, 5, 1, tzinfo=UTC))
    before = at_time(doc, datetime(2026, 1, 1, tzinfo=UTC))
    assert "added" in _body_text(after)
    assert "added" not in _body_text(before)


def test_at_time_naive_arg_against_aware_source_revision() -> None:
    doc = ContentDocument(
        metadata=DocumentMetadata(title=""),
        body=(
            make_block_insertion(
                Paragraph(children=(Text(value="y"),)),
                author="A",
                date=datetime(2026, 4, 16, tzinfo=UTC),
                revision_id="0",
            ),
        ),
    )
    snap = at_time(doc, datetime(2026, 5, 1))  # naive arg, no crash
    assert "y" in _body_text(snap)


def test_between_aware_bounds_against_naive_source() -> None:
    revs = Revisions.from_document(_date_only_ins_doc())
    assert len(revs.between(start=datetime(2026, 1, 1, tzinfo=UTC))) == 1
    assert len(revs.between(end=datetime(2026, 1, 1, tzinfo=UTC))) == 0


def test_sorted_by_date_mixed_awareness_does_not_crash() -> None:
    doc = ContentDocument(
        metadata=DocumentMetadata(title=""),
        body=(
            Paragraph(
                children=(
                    Span(
                        attr=Attr(
                            classes=("rev-ins",),
                            kv={"rev:id": "0", "rev:author": "A", "rev:date": "2026-04-16"},
                        ),
                        children=(Text(value="a"),),
                    ),
                )
            ),
            make_block_insertion(
                Paragraph(children=(Text(value="b"),)),
                author="A",
                date=datetime(2026, 4, 17, tzinfo=UTC),
                revision_id="1",
            ),
        ),
    )
    ordered = Revisions.from_document(doc).sorted_by_date()
    assert len(ordered) == 2
    assert [r.id for r in ordered] == ["0", "1"]
