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


class TestMultipleMoves:
    def test_three_distinct_blocks_relocated(self) -> None:
        a = "First clause about indemnification obligations and limits."
        b = "Second clause about governing law and venue selection."
        c = "Third clause about confidentiality and permitted disclosures."
        original = _doc(a, b, c, "Anchor paragraph that does not move.")
        revised = _doc("Anchor paragraph that does not move.", c, a, b)
        redline = compare_documents(original, revised, author="T", detect_moves=True)
        revs = Revisions.from_document(redline)
        froms = [r for r in revs if r.change_type == RevisionType.MOVE_FROM]
        tos = [r for r in revs if r.change_type == RevisionType.MOVE_TO]
        # Each relocated clause is a from/to pair with a shared, unique name.
        assert len(froms) == len(tos) >= 2
        names = {r.move_name for r in froms}
        assert len(names) == len(froms)  # every move pair has its own name
        _assert_roundtrip(original, revised)


class TestMoveBudget:
    def test_move_detection_skipped_above_budget(self) -> None:
        """Huge insert/delete counts skip move detection but still round-trip.

        The cap keeps the O(deleted x inserted) pairing from stalling; the
        diff stays correct (relocations become delete + insert).
        """
        original = _doc(*[f"Old unique paragraph number {i} here." for i in range(240)])
        revised = _doc(*[f"New unique paragraph number {i} here." for i in range(240)])
        # 240 * 240 = 57600 > 50000 budget → move detection skipped.
        redline = compare_documents(original, revised, author="T", detect_moves=True)
        by_type = {r.change_type for r in Revisions.from_document(redline)}
        assert RevisionType.MOVE_FROM not in by_type
        assert by_type <= {RevisionType.INSERTION, RevisionType.DELETION}
        _assert_roundtrip(original, revised)


class TestTables:
    def _table_doc(self, cell_text: str) -> ContentDocument:
        from kaos_content.model.blocks import Table
        from kaos_content.model.table import Cell, Row, TableSection

        table = Table(
            head=TableSection(
                rows=(Row(cells=(Cell(content=(Paragraph(children=(Text(value="Header"),)),)),)),)
            ),
            bodies=(
                TableSection(
                    rows=(
                        Row(cells=(Cell(content=(Paragraph(children=(Text(value=cell_text),)),)),)),
                    )
                ),
            ),
        )
        return ContentDocument(metadata=DocumentMetadata(title=""), body=(table,))

    def test_changed_table_round_trips(self) -> None:
        original = self._table_doc("Original cell value.")
        revised = self._table_doc("Revised cell value.")
        redline = compare_documents(original, revised, author="T")
        assert Revisions.from_document(redline)
        _assert_roundtrip(original, revised)


class TestLargerDocument:
    def test_many_paragraphs_few_edits_round_trip(self) -> None:
        base = [f"Section {i}: standard contractual language and provisions." for i in range(200)]
        original = _doc(*base)
        edited = list(base)
        edited[50] = "Section 50: AMENDED contractual language and provisions."
        del edited[120]
        edited.insert(10, "Section 9b: a newly inserted provision.")
        revised = _doc(*edited)
        redline = _assert_roundtrip(original, revised)
        # A handful of edits should not explode into hundreds of revisions.
        assert len(Revisions.from_document(redline)) < 20
