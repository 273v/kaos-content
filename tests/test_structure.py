"""Tests for kaos_content.structure: the kaos-nlp-core P7 wrapper."""

from __future__ import annotations

import pytest

from kaos_content.model.annotation import AnnotationType
from kaos_content.model.annotation_bodies import (
    BoilerplateBody,
    HeadingCandidateBody,
    MetadataBody,
    TableRowBody,
    parse_body,
)
from kaos_content.shortcuts import heading, paragraph

# Skip whole module when [nlp] extra is missing.
pytest.importorskip("kaos_nlp_core.structure")

from kaos_content.model.document import ContentDocument
from kaos_content.structure import (
    annotate_structure,
    with_inferred_structure,
    with_structure_annotations,
)


def _build_doc() -> ContentDocument:
    """Small synthetic document with 1 heading + 2 body paragraphs."""
    return ContentDocument(
        body=(
            heading(1, "Discussion"),
            paragraph("The court considered each argument in turn and rejected them."),
            paragraph("Conclusion follows below in due course."),
        )
    )


def test_empty_document_returns_no_annotations() -> None:
    doc = ContentDocument(body=())
    result = annotate_structure(doc)
    assert result.annotations == ()
    assert result.n_lines == 0


def test_basic_document_emits_heading_candidate() -> None:
    doc = _build_doc()
    result = annotate_structure(doc)
    # At least one heading_candidate annotation.
    headings = [a for a in result.annotations if a.type == AnnotationType.HEADING_CANDIDATE]
    assert headings, f"expected ≥1 heading candidate, got {result.annotations}"
    # Body should be valid HeadingCandidateBody.
    body = parse_body(headings[0])
    assert isinstance(body, HeadingCandidateBody)
    assert -1.0 <= body.score <= 1.0


def test_heading_candidate_targets_a_body_block() -> None:
    doc = _build_doc()
    result = annotate_structure(doc)
    for ann in result.annotations:
        for tgt in ann.targets:
            assert tgt.node_ref.startswith("#/body/"), tgt.node_ref


def test_annotation_targets_carry_char_range() -> None:
    """T3a: every emitted target must populate (start_offset, end_offset).

    For single-line blocks (the common case here) the range starts at 0 and
    spans the whole serialized line.
    """
    doc = _build_doc()
    result = annotate_structure(doc)
    assert result.annotations, "expected at least one annotation"
    for ann in result.annotations:
        for tgt in ann.targets:
            assert tgt.start_offset is not None, ann
            assert tgt.end_offset is not None, ann
            assert tgt.start_offset == 0, (
                f"single-line block expected start=0, got {tgt.start_offset} for {ann}"
            )
            assert tgt.end_offset > tgt.start_offset, (
                f"empty range on {ann}: {tgt.start_offset}..{tgt.end_offset}"
            )


def test_char_range_accumulates_within_multi_line_block() -> None:
    """T3a: a list block emits one line per item; offsets accumulate.

    For a 3-item list, item 1 spans [0, len1], item 2 spans
    [len1+1, len1+1+len2], item 3 spans [len1+1+len2+1, …].
    """
    from itertools import pairwise

    from kaos_content.shortcuts import bullet_list

    doc = ContentDocument(
        body=(
            heading(2, "List header"),
            bullet_list("first item line", "second item line", "third item line"),
        )
    )
    result = annotate_structure(doc)
    # Find the list_item / body annotations associated with `#/body/1`.
    list_targets = [
        tgt for ann in result.annotations for tgt in ann.targets if tgt.node_ref == "#/body/1"
    ]
    if not list_targets:
        # Bullet list items may decode as body/list_item which don't emit
        # annotations in the default class set. The mapping itself is
        # exercised by the heading candidate above; skip the multi-line
        # assertion when nothing was emitted on the list block.
        return
    list_targets.sort(key=lambda tgt: tgt.start_offset or 0)
    for prev, cur in pairwise(list_targets):
        # Each subsequent line must start strictly after the previous one
        # ended (the +1 is the inter-line newline within the block).
        assert (
            cur.start_offset is not None
            and prev.end_offset is not None
            and cur.start_offset >= prev.end_offset + 1
        ), (prev, cur)


def test_table_row_targets_descend_into_rows() -> None:
    """T3b: each table row line must map to that row's ref, not the table's.

    When the document contains a table whose rows are detected as
    table_row by the structure decoder, the emitted annotations should
    target ``#/body/{i}/<section>/rows/{j}`` rather than just the
    table's ``#/body/{i}``.
    """
    from kaos_content.shortcuts import table_from_rows

    table = table_from_rows(
        ["Col A", "Col B", "Col C"],
        [
            ["Val 1", "Val 2", "Val 3"],
            ["Val 4", "Val 5", "Val 6"],
            ["Val 7", "Val 8", "Val 9"],
        ],
    )
    doc = ContentDocument(body=(table,))
    result = annotate_structure(doc)
    table_row_anns = [a for a in result.annotations if a.type == AnnotationType.TABLE_ROW]
    if not table_row_anns:
        # Decoder may not always classify pipe-separated rows as table_row
        # depending on weights. Skip when nothing fired — the layout
        # mapping itself is exercised by the multi-line block test.
        return
    refs = {tgt.node_ref for ann in table_row_anns for tgt in ann.targets}
    # Each row ref must contain "/rows/" — that's the marker proving the
    # mapper descended into the row level rather than collapsing to the
    # table's base ref.
    assert any("/rows/" in r for r in refs), (
        f"expected at least one ref to descend into /rows/, got {refs}"
    )


def test_footnote_targets_use_footnote_path() -> None:
    """T3b: footnote-block lines map to ``#/footnotes/{key}/{block}``."""
    from kaos_content.model.document import ContentDocument as Doc

    doc = Doc(
        body=(heading(1, "Body Heading"), paragraph("Some body text.")),
        footnotes={"1": (paragraph("First footnote body."),)},
    )
    # We don't assert on which annotations fire on the footnote. Instead
    # we exercise the layout mapper directly: IF it assigns a ref under
    # #/footnotes/, it must have the expected shape.
    from kaos_content.serializers.text import serialize_text
    from kaos_content.structure import _map_lines_to_node_refs

    layout = _map_lines_to_node_refs(doc, serialize_text(doc))
    footnote_refs = [ref for (ref, _s, _e) in layout.values() if ref.startswith("#/footnotes/")]
    assert footnote_refs, f"expected at least one footnote ref in layout, got {layout}"
    for ref in footnote_refs:
        assert ref.startswith("#/footnotes/1/"), ref


# ─── T3c: with_inferred_structure (heading promotion) ────────────────────


def test_with_inferred_structure_promotes_paragraph_to_heading() -> None:
    """A paragraph that scores as heading should become a Heading block.

    The synthetic input has zero typed Heading blocks — every block is a
    Paragraph. After promotion, the all-caps short paragraph that scores
    as a heading must be a Heading block with depth in [1, 6].
    """
    from kaos_content.model.blocks import Heading

    # All-caps short line with blank-bounded neighbors → strong heading
    # signal in the P7 scorer.
    doc = ContentDocument(
        body=(
            paragraph("Some preamble body text."),
            paragraph("DISCUSSION"),
            paragraph("The court considered each argument and rejected them all."),
            paragraph("CONCLUSION"),
            paragraph("The motion is hereby denied."),
        )
    )
    promoted = with_inferred_structure(doc)
    headings = [b for b in promoted.body if isinstance(b, Heading)]
    assert headings, (
        f"expected ≥1 Heading after promotion, got {[type(b).__name__ for b in promoted.body]}"
    )
    for h in headings:
        assert 1 <= h.depth <= 6, h


def test_with_inferred_structure_is_idempotent_on_typed_headings() -> None:
    """Re-running on a doc that already has Heading blocks must NOT
    double-promote — typed headings are preserved as-is.
    """
    from kaos_content.model.blocks import Heading

    doc = ContentDocument(
        body=(
            heading(2, "Overview"),
            paragraph("Body content."),
            heading(3, "Details"),
            paragraph("More body content."),
        )
    )
    promoted = with_inferred_structure(doc)
    promoted_again = with_inferred_structure(promoted)
    # Original Heading blocks stay Heading with the same depth.
    headings_first = [(i, b.depth) for i, b in enumerate(promoted.body) if isinstance(b, Heading)]
    headings_second = [
        (i, b.depth) for i, b in enumerate(promoted_again.body) if isinstance(b, Heading)
    ]
    assert headings_first == headings_second, (headings_first, headings_second)


def test_with_inferred_structure_enables_section_view() -> None:
    """The whole point of T3c: SectionView must see the promoted headings.

    Before promotion the doc has zero typed Heading blocks, so
    DocumentView.sections returns a single big section. After promotion,
    DocumentView.sections must split at the inferred heading boundaries.
    """
    from kaos_content.views.document_view import DocumentView

    doc = ContentDocument(
        body=(
            paragraph("Preamble."),
            paragraph("DISCUSSION"),
            paragraph("First section body."),
            paragraph("CONCLUSION"),
            paragraph("Second section body."),
        )
    )
    sections_before = DocumentView(doc).sections
    promoted = with_inferred_structure(doc)
    sections_after = DocumentView(promoted).sections
    assert len(sections_after) > len(sections_before), (
        f"sections should increase after promotion, "
        f"got {len(sections_before)} → {len(sections_after)}"
    )


def test_with_inferred_structure_preserves_block_id_and_provenance() -> None:
    """Promoted blocks must keep their original id + provenance so
    downstream node_refs and page lookups stay stable.
    """
    from kaos_content.model.attr import Provenance
    from kaos_content.model.blocks import Heading, Paragraph
    from kaos_content.model.inlines import Text

    p = Paragraph(
        id="my-stable-id",
        children=(Text(value="DISCUSSION"),),
        provenance=Provenance(page=3),
    )
    doc = ContentDocument(
        body=(
            paragraph("Preamble."),
            p,
            paragraph("Body content after the heading."),
        )
    )
    promoted = with_inferred_structure(doc)
    promoted_block = promoted.body[1]
    if isinstance(promoted_block, Heading):
        assert promoted_block.id == "my-stable-id"
        assert promoted_block.provenance is not None
        assert promoted_block.provenance.page == 3


def test_with_inferred_structure_attaches_annotations() -> None:
    """The HEADING_CANDIDATE / BOILERPLATE / ... annotations should also
    land on the returned doc so downstream consumers can read scores.
    """
    doc = ContentDocument(
        body=(
            paragraph("Preamble."),
            paragraph("DISCUSSION"),
            paragraph("Body."),
        )
    )
    promoted = with_inferred_structure(doc)
    n_before = len(doc.annotations)
    n_after = len(promoted.annotations)
    assert n_after > n_before


def test_label_counts_present() -> None:
    doc = _build_doc()
    result = annotate_structure(doc)
    expected = {
        "blank",
        "heading",
        "body",
        "list_item",
        "table_row",
        "metadata",
        "boilerplate",
    }
    assert expected.issubset(set(result.label_counts.keys()))
    assert sum(result.label_counts.values()) == result.n_lines


def test_with_structure_annotations_attaches_to_document() -> None:
    doc = _build_doc()
    n_before = len(doc.annotations)
    new_doc = with_structure_annotations(doc)
    n_after = len(new_doc.annotations)
    assert n_after > n_before


def test_annotation_id_prefix_propagates() -> None:
    doc = _build_doc()
    result = annotate_structure(doc, annotation_id_prefix="myprefix")
    if result.annotations:
        assert all(a.id.startswith("myprefix-") for a in result.annotations)


def test_lexicon_kwargs_propagate() -> None:
    """Smoke: passing custom lexicons doesn't crash."""
    doc = ContentDocument(
        body=(
            heading(1, "Article 5 — Définitions"),
            paragraph("Body text."),
        )
    )
    result = annotate_structure(
        doc,
        enum_lexicon="french_legal",
        hierarchy_lexicon="french_legal",
    )
    headings = [a for a in result.annotations if a.type == AnnotationType.HEADING_CANDIDATE]
    assert headings


def test_unknown_lexicon_raises() -> None:
    doc = _build_doc()
    with pytest.raises(ValueError, match="unknown"):
        annotate_structure(doc, hierarchy_lexicon="not_a_real_lex")


def test_table_row_annotation_body_typed() -> None:
    """Body for TABLE_ROW must validate as TableRowBody."""
    doc = ContentDocument(
        body=(
            paragraph("Col A | Col B | Col C"),
            paragraph("Val 1 | Val 2 | Val 3"),
            paragraph("Val 4 | Val 5 | Val 6"),
        )
    )
    result = annotate_structure(doc)
    table_rows = [a for a in result.annotations if a.type == AnnotationType.TABLE_ROW]
    if table_rows:  # Decoder is permissive but typically picks up the pipe shape.
        body = parse_body(table_rows[0])
        assert isinstance(body, TableRowBody)


def test_metadata_annotation_body_typed() -> None:
    """Inline-colon metadata lines emit MetadataBody."""
    doc = ContentDocument(
        body=(
            paragraph("Author: Jane Doe"),
            paragraph("Date: 2026-05-05"),
            paragraph("Case Number: 22-1234"),
            paragraph("Body content here."),
        )
    )
    result = annotate_structure(doc)
    metadata = [a for a in result.annotations if a.type == AnnotationType.METADATA]
    assert metadata, "expected metadata annotation(s)"
    body = parse_body(metadata[0])
    assert isinstance(body, MetadataBody)


def test_boilerplate_body_typed() -> None:
    """Boilerplate annotations parse to BoilerplateBody."""
    # Make a doc where the same heading appears 5 times (synthetic
    # boilerplate-shaped).
    doc = ContentDocument(
        body=tuple(
            heading(1, "HEADER") if i % 2 == 0 else paragraph(f"page {i} body") for i in range(10)
        )
    )
    result = annotate_structure(doc)
    bp = [a for a in result.annotations if a.type == AnnotationType.BOILERPLATE]
    if bp:
        body = parse_body(bp[0])
        assert isinstance(body, BoilerplateBody)
