"""Tests for the redline engine (``kaos_content.diff.compare_documents``).

The central invariant: comparing ``original`` and ``revised`` and then
``accept_all`` must reproduce ``revised``'s text, while ``reject_all``
must reproduce ``original``'s text. Specific change shapes (insert,
delete, word-level edit, move) are asserted on top of that invariant.
"""

from __future__ import annotations

from kaos_content import compare_documents
from kaos_content.model.blocks import Heading, Paragraph
from kaos_content.model.document import ContentDocument, DocumentMetadata
from kaos_content.model.inlines import Strong, Text
from kaos_content.revision import (
    Revisions,
    RevisionType,
    accept_all,
    reject_all,
)
from kaos_content.traversal import walk
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


class TestEmptyAndDegenerate:
    def test_empty_to_empty_has_no_revisions(self) -> None:
        empty = ContentDocument(metadata=DocumentMetadata(title=""), body=())
        redline = compare_documents(empty, empty)
        assert len(Revisions.from_document(redline)) == 0

    def test_empty_to_one_block_is_insertion(self) -> None:
        empty = ContentDocument(metadata=DocumentMetadata(title=""), body=())
        one = _doc("Brand new content.")
        redline = _assert_roundtrip(empty, one)
        assert [r.change_type for r in Revisions.from_document(redline)] == [RevisionType.INSERTION]

    def test_one_block_to_empty_is_deletion(self) -> None:
        empty = ContentDocument(metadata=DocumentMetadata(title=""), body=())
        one = _doc("Will be removed.")
        redline = _assert_roundtrip(one, empty)
        assert [r.change_type for r in Revisions.from_document(redline)] == [RevisionType.DELETION]

    def test_empty_paragraph_gains_text(self) -> None:
        a = ContentDocument(metadata=DocumentMetadata(title=""), body=(Paragraph(children=()),))
        b = _doc("Now it has words.")
        # An empty paragraph and a filled one are dissimilar → full replace.
        _assert_roundtrip(a, b)

    def test_whitespace_only_change_round_trips(self) -> None:
        a = _doc("alpha beta")
        b = _doc("alpha  beta")
        # Normalization collapses whitespace for *alignment*, but the exact
        # revised text must still be reproducible by accept_all.
        _assert_roundtrip(a, b)


class TestDeterminism:
    def test_same_inputs_produce_same_content(self) -> None:
        a = _doc("the cat sat on the mat", "second clause unchanged")
        b = _doc("the dog sat on the mat", "second clause unchanged")
        r1 = compare_documents(a, b, author="X")
        r2 = compare_documents(a, b, author="X")
        # Node UUIDs differ by design; content (text + change types + rev ids)
        # must be identical run to run.
        revs1 = Revisions.from_document(r1)
        revs2 = Revisions.from_document(r2)
        assert [(x.change_type, x.id, x.text) for x in revs1] == [
            (x.change_type, x.id, x.text) for x in revs2
        ]
        assert _body_text(r1) == _body_text(r2)


class TestBlockTypeChange:
    def test_heading_to_paragraph_same_text_is_currently_no_op(self) -> None:
        """Documents a known limitation: alignment keys on text only.

        Changing a heading to a paragraph with identical text aligns as
        ``equal`` and produces no revision (the text didn't change, only the
        block style). A future style-change feature (pPrChange) would model
        it; this test pins today's behavior so a change is intentional.
        """
        a = ContentDocument(
            metadata=DocumentMetadata(title=""),
            body=(Heading(depth=1, children=(Text(value="Governing Law"),)),),
        )
        b = _doc("Governing Law")
        redline = compare_documents(a, b)
        assert len(Revisions.from_document(redline)) == 0


class TestWordDiffFidelity:
    def test_unicode_smart_quotes_round_trip(self) -> None:
        a = _doc("He said “hello world” today.")
        b = _doc("He said “goodbye world” today.")
        redline = _assert_roundtrip(a, b)
        # The unchanged smart-quoted span survives in the final text.
        assert "“goodbye world”" in _body_text(accept_all(redline))

    def test_cjk_round_trip(self) -> None:
        a = _doc("合同条款一 unchanged tail")
        b = _doc("合同条款二 unchanged tail")
        _assert_roundtrip(a, b)

    def test_punctuation_only_change(self) -> None:
        a = _doc("Payment is due, on receipt.")
        b = _doc("Payment is due on receipt.")
        _assert_roundtrip(a, b)

    def test_single_word_edit_keeps_surrounding_text_unmarked(self) -> None:
        a = _doc("The party shall deliver the goods within ten days of order.")
        b = _doc("The party shall deliver the goods within thirty days of order.")
        redline = _assert_roundtrip(a, b)
        revs = Revisions.from_document(redline)
        # Only the changed word produces revisions; the long shared remainder
        # is plain text, so the revision text is small.
        joined = " ".join(r.text for r in revs)
        assert "ten" in joined
        assert "thirty" in joined
        assert "deliver" not in joined  # untouched word not marked

    def test_inline_formatting_lost_in_changed_paragraph_is_pinned(self) -> None:
        """Known limitation: a *changed* paragraph is rebuilt from plain text,
        so inline run formatting (Strong/Emphasis/Link) inside it is dropped.
        Unchanged paragraphs keep full fidelity. Pinned so a future
        improvement is a deliberate change, not an accident.
        """
        a = ContentDocument(
            metadata=DocumentMetadata(title=""),
            body=(
                Paragraph(
                    children=(
                        Text(value="The "),
                        Strong(children=(Text(value="material"),)),
                        Text(value=" term is alpha."),
                    )
                ),
            ),
        )
        b = ContentDocument(
            metadata=DocumentMetadata(title=""),
            body=(
                Paragraph(
                    children=(
                        Text(value="The "),
                        Strong(children=(Text(value="material"),)),
                        Text(value=" term is beta."),
                    )
                ),
            ),
        )
        redline = _assert_roundtrip(a, b)
        # Text round-trips, but Strong is not preserved in the changed para.
        assert not any(type(n).__name__ == "Strong" for block in redline.body for n in walk(block))


class TestMixedChanges:
    def test_insert_delete_and_edit_in_one_diff(self) -> None:
        original = _doc(
            "Intro paragraph stays the same.",
            "This clause will be deleted entirely.",
            "The fee shall be ten dollars per unit.",
        )
        revised = _doc(
            "Intro paragraph stays the same.",
            "The fee shall be twelve dollars per unit.",
            "A newly added closing paragraph.",
        )
        redline = _assert_roundtrip(original, revised)
        kinds = {r.change_type for r in Revisions.from_document(redline)}
        assert RevisionType.INSERTION in kinds
        assert RevisionType.DELETION in kinds


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
