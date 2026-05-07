"""Tests for ``kaos_content.indexing.AnnotationIndex`` (P4.5).

The wrapper builds a ``kaos_nlp_core.structures.SpanIndex`` from a
``ContentDocument.annotations`` tuple and exposes node-ref-aware queries.
Falls back to a pure-Python path when the optional ``[nlp]`` extra is
not installed.
"""

from __future__ import annotations

from kaos_content.indexing import AnnotationIndex
from kaos_content.model.annotation import (
    Annotation,
    AnnotationTarget,
    AnnotationType,
)
from kaos_content.model.document import ContentDocument


def _doc_with(annotations: list[Annotation]) -> ContentDocument:
    return ContentDocument(annotations=tuple(annotations))


def test_empty_document_returns_empty() -> None:
    idx = AnnotationIndex(ContentDocument())
    assert idx.annotations_for("/body/0") == []
    assert idx.annotations_containing_offset("/body/0", 5) == []


def test_annotations_for_finds_node_match() -> None:
    a = Annotation(
        id="ann-1",
        type=AnnotationType.HIGHLIGHT,
        targets=(AnnotationTarget(node_ref="/body/0"),),
    )
    b = Annotation(
        id="ann-2",
        type=AnnotationType.COMMENT,
        targets=(AnnotationTarget(node_ref="/body/1"),),
    )
    doc = _doc_with([a, b])
    idx = AnnotationIndex(doc)
    hits = idx.annotations_for("/body/0")
    assert {h.id for h in hits} == {"ann-1"}


def test_annotations_containing_offset_excludes_non_matching_range() -> None:
    a = Annotation(
        id="ann-1",
        type=AnnotationType.HIGHLIGHT,
        targets=(AnnotationTarget(node_ref="/body/0", start_offset=10, end_offset=20),),
    )
    doc = _doc_with([a])
    idx = AnnotationIndex(doc)
    assert {h.id for h in idx.annotations_containing_offset("/body/0", 15)} == {"ann-1"}
    assert idx.annotations_containing_offset("/body/0", 25) == []
    assert idx.annotations_containing_offset("/body/0", 20) == []  # half-open
    # Wrong node ref → no match.
    assert idx.annotations_containing_offset("/body/1", 15) == []


def test_whole_node_annotation_returns_unconditionally() -> None:
    """An annotation with no offsets covers the whole node — every offset
    in that node should match."""
    a = Annotation(
        id="ann-1",
        type=AnnotationType.REDACTION,
        targets=(AnnotationTarget(node_ref="/body/0"),),
    )
    doc = _doc_with([a])
    idx = AnnotationIndex(doc)
    for offset in (0, 5, 100, 999):
        hits = idx.annotations_containing_offset("/body/0", offset)
        assert {h.id for h in hits} == {"ann-1"}, f"offset={offset}"


def test_multi_target_annotation() -> None:
    """Annotations can span multiple nodes — both targets register."""
    a = Annotation(
        id="ann-1",
        type=AnnotationType.CROSS_REFERENCE,
        targets=(
            AnnotationTarget(node_ref="/body/0", start_offset=0, end_offset=10),
            AnnotationTarget(node_ref="/body/1", start_offset=5, end_offset=15),
        ),
    )
    doc = _doc_with([a])
    idx = AnnotationIndex(doc)
    assert {h.id for h in idx.annotations_for("/body/0")} == {"ann-1"}
    assert {h.id for h in idx.annotations_for("/body/1")} == {"ann-1"}
    # Node-ref-keyed offset queries respect each target's range.
    assert {h.id for h in idx.annotations_containing_offset("/body/0", 5)} == {"ann-1"}
    assert idx.annotations_containing_offset("/body/0", 12) == []
    assert {h.id for h in idx.annotations_containing_offset("/body/1", 10)} == {"ann-1"}


def test_overlapping_annotations_all_returned() -> None:
    """Multiple annotations on the same node, with overlapping ranges,
    should all be returned for an offset that lies in their intersection."""
    a = Annotation(
        id="ann-A",
        type=AnnotationType.HIGHLIGHT,
        targets=(AnnotationTarget(node_ref="/body/0", start_offset=0, end_offset=20),),
    )
    b = Annotation(
        id="ann-B",
        type=AnnotationType.COMMENT,
        targets=(AnnotationTarget(node_ref="/body/0", start_offset=5, end_offset=15),),
    )
    c = Annotation(
        id="ann-C",
        type=AnnotationType.ENTITY,
        targets=(AnnotationTarget(node_ref="/body/0", start_offset=10, end_offset=12),),
    )
    doc = _doc_with([a, b, c])
    idx = AnnotationIndex(doc)
    hits = idx.annotations_containing_offset("/body/0", 11)
    assert {h.id for h in hits} == {"ann-A", "ann-B", "ann-C"}


def test_has_nlp_backend_smoke() -> None:
    """Either has the backend (test env has [nlp] installed) or doesn't —
    both are acceptable; the API just must not raise."""
    idx = AnnotationIndex(ContentDocument())
    # Calling has_nlp_backend triggers the lazy import path.
    flag = idx.has_nlp_backend()
    assert isinstance(flag, bool)


def test_pure_python_path_matches_backed_path() -> None:
    """Smoke test — when both paths agree on the synthetic corpus we know
    the backed path's prefiltering is consistent with the naive walk."""
    a = Annotation(
        id="ann-1",
        type=AnnotationType.HIGHLIGHT,
        targets=(AnnotationTarget(node_ref="/body/0", start_offset=0, end_offset=10),),
    )
    b = Annotation(
        id="ann-2",
        type=AnnotationType.COMMENT,
        targets=(AnnotationTarget(node_ref="/body/0", start_offset=5, end_offset=15),),
    )
    doc = _doc_with([a, b])
    idx = AnnotationIndex(doc)
    # Force-build then query.
    idx._ensure_built()  # type: ignore[attr-defined]
    backed_hits = idx.annotations_containing_offset("/body/0", 7)
    # Reference path: walk annotations.
    naive = [
        ann
        for ann in doc.annotations
        if any(
            tgt.node_ref == "/body/0"
            and (
                tgt.start_offset is None
                or tgt.end_offset is None
                or (tgt.start_offset <= 7 < tgt.end_offset)
            )
            for tgt in ann.targets
        )
    ]
    assert {h.id for h in backed_hits} == {n.id for n in naive}
