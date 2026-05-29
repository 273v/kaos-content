"""Tests for the redline engine (``kaos_content.diff.compare_documents``).

The central invariant: comparing ``original`` and ``revised`` and then
``accept_all`` must reproduce ``revised``'s text, while ``reject_all``
must reproduce ``original``'s text. Specific change shapes (insert,
delete, word-level edit, move) are asserted on top of that invariant.
"""

from __future__ import annotations

from kaos_content import compare_documents
from kaos_content.model.blocks import Paragraph
from kaos_content.model.document import ContentDocument, DocumentMetadata
from kaos_content.model.inlines import Text
from kaos_content.revision import (
    Revisions,
    RevisionType,
    accept_all,
    reject_all,
)
from kaos_content.traversal.visitor import extract_text


def _doc(*paras: str) -> ContentDocument:
    return ContentDocument(
        metadata=DocumentMetadata(title=""),
        body=tuple(Paragraph(children=(Text(value=p),)) for p in paras),
    )


def _body_text(doc: ContentDocument) -> str:
    return "\n".join(extract_text(b) for b in doc.body)


def _assert_roundtrip(original: ContentDocument, revised: ContentDocument) -> ContentDocument:
    """compare → accept_all == revised text; reject_all == original text."""
    redline = compare_documents(original, revised, author="Tester")
    assert _body_text(accept_all(redline)) == _body_text(revised)
    assert _body_text(reject_all(redline)) == _body_text(original)
    return redline


class TestIdentical:
    def test_no_revisions_when_unchanged(self) -> None:
        doc = _doc("Alpha.", "Beta.", "Gamma.")
        redline = compare_documents(doc, doc, author="Tester")
        assert len(Revisions.from_document(redline)) == 0
        assert _body_text(redline) == _body_text(doc)


class TestInsert:
    def test_inserted_paragraph_is_block_insertion(self) -> None:
        original = _doc("Alpha.", "Gamma.")
        revised = _doc("Alpha.", "Beta.", "Gamma.")
        redline = _assert_roundtrip(original, revised)
        revs = Revisions.from_document(redline)
        assert [r.change_type for r in revs] == [RevisionType.INSERTION]
        assert revs.items[0].text == "Beta."


class TestDelete:
    def test_deleted_paragraph_is_block_deletion(self) -> None:
        original = _doc("Alpha.", "Beta.", "Gamma.")
        revised = _doc("Alpha.", "Gamma.")
        redline = _assert_roundtrip(original, revised)
        revs = Revisions.from_document(redline)
        assert [r.change_type for r in revs] == [RevisionType.DELETION]
        assert revs.items[0].text == "Beta."


class TestWordLevelEdit:
    def test_changed_paragraph_produces_inline_ins_and_del(self) -> None:
        original = _doc("The quick brown fox jumps.")
        revised = _doc("The quick red fox leaps.")
        redline = _assert_roundtrip(original, revised)
        revs = Revisions.from_document(redline)
        types = {r.change_type for r in revs}
        assert RevisionType.INSERTION in types
        assert RevisionType.DELETION in types
        # "brown"→"red" and "jumps"→"leaps" both edited; "The quick fox"
        # is shared and must survive in both views.
        assert "quick" in _body_text(reject_all(redline))
        assert "red" in _body_text(accept_all(redline))
        assert "brown" not in _body_text(accept_all(redline))

    def test_dissimilar_paragraph_is_full_replace(self) -> None:
        original = _doc("Completely different original sentence here.")
        revised = _doc("Nothing alike whatsoever in this one.")
        redline = _assert_roundtrip(original, revised)
        revs = Revisions.from_document(redline)
        # Below the similarity threshold → a block delete + a block insert.
        assert {r.change_type for r in revs} == {
            RevisionType.DELETION,
            RevisionType.INSERTION,
        }


class TestMoveDetection:
    def test_moved_paragraph_is_move_pair(self) -> None:
        moved = "This entire clause relocates to a new position in the document."
        original = _doc(moved, "Stable tail paragraph.")
        revised = _doc("Stable tail paragraph.", moved)
        redline = compare_documents(original, revised, author="Tester", detect_moves=True)
        revs = Revisions.from_document(redline)
        by_type = {r.change_type for r in revs}
        assert RevisionType.MOVE_FROM in by_type
        assert RevisionType.MOVE_TO in by_type
        # The two halves share a move name.
        names = {r.move_name for r in revs if r.move_name}
        assert len(names) == 1

    def test_moves_disabled_yields_insert_delete(self) -> None:
        moved = "This entire clause relocates to a new position in the document."
        original = _doc(moved, "Stable tail paragraph.")
        revised = _doc("Stable tail paragraph.", moved)
        redline = compare_documents(original, revised, author="Tester", detect_moves=False)
        by_type = {r.change_type for r in Revisions.from_document(redline)}
        assert RevisionType.MOVE_FROM not in by_type
        assert RevisionType.MOVE_TO not in by_type


class TestAuthorStamp:
    def test_author_recorded_on_every_revision(self) -> None:
        original = _doc("Alpha.")
        revised = _doc("Alpha.", "Beta.")
        redline = compare_documents(original, revised, author="Jane Reviewer")
        revs = Revisions.from_document(redline)
        assert revs
        assert all(r.author == "Jane Reviewer" for r in revs)
