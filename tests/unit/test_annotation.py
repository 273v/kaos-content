"""Tests for annotation types."""

from kaos_content import (
    Annotation,
    AnnotationTarget,
    AnnotationType,
    Provenance,
)


class TestAnnotationType:
    def test_values(self) -> None:
        assert AnnotationType.REDACTION == "redaction"
        assert AnnotationType.EXTERNAL_CITATION == "external_citation"
        assert AnnotationType.ENTITY == "entity"

    def test_all_members(self) -> None:
        expected = {
            "HIGHLIGHT",
            "COMMENT",
            "BOOKMARK",
            "REDACTION",
            "DEFINED_TERM",
            "TERM_DEFINITION",
            "CROSS_REFERENCE",
            "EXTERNAL_CITATION",
            "AMENDMENT",
            "PROVISION",
            "ENTITY",
            "SENTIMENT",
            "CLASSIFICATION",
            "EXTRACTED_CELL",
            "TRACKED_CHANGE",
            "HEADING_CANDIDATE",
            "BOILERPLATE",
            "TABLE_ROW",
            "METADATA",
        }
        assert {m.name for m in AnnotationType} == expected

    def test_tracked_change_value(self) -> None:
        assert AnnotationType.TRACKED_CHANGE == "tracked_change"


class TestAnnotationTarget:
    def test_node_only(self) -> None:
        t = AnnotationTarget(node_ref="#/body/0")
        assert t.start_offset is None
        assert t.end_offset is None

    def test_with_offsets(self) -> None:
        t = AnnotationTarget(node_ref="#/body/0", start_offset=5, end_offset=15)
        assert t.start_offset == 5
        assert t.end_offset == 15

    def test_json_roundtrip(self) -> None:
        t = AnnotationTarget(node_ref="#/body/2", start_offset=0, end_offset=10)
        assert AnnotationTarget.model_validate_json(t.model_dump_json()) == t


class TestAnnotation:
    def test_basic(self) -> None:
        a = Annotation(
            id="ann-1",
            type=AnnotationType.REDACTION,
            targets=(AnnotationTarget(node_ref="#/body/0"),),
        )
        assert a.id == "ann-1"
        assert a.type == AnnotationType.REDACTION
        assert a.body == {}

    def test_with_body(self) -> None:
        a = Annotation(
            id="ann-2",
            type=AnnotationType.DEFINED_TERM,
            targets=(AnnotationTarget(node_ref="#/body/3", start_offset=0, end_offset=14),),
            body={"definition_id": "dt-001", "definition_text": "Force Majeure"},
        )
        assert a.body["definition_id"] == "dt-001"

    def test_multi_target(self) -> None:
        a = Annotation(
            id="ann-3",
            type=AnnotationType.HIGHLIGHT,
            targets=(
                AnnotationTarget(node_ref="#/body/0", start_offset=10, end_offset=20),
                AnnotationTarget(node_ref="#/body/1", start_offset=0, end_offset=5),
            ),
        )
        assert len(a.targets) == 2

    def test_with_provenance(self) -> None:
        a = Annotation(
            id="ann-4",
            type=AnnotationType.ENTITY,
            targets=(AnnotationTarget(node_ref="#/body/0"),),
            body={"entity_type": "organization", "text": "ACME Corp"},
            provenance=Provenance(extractor="spacy"),
        )
        assert a.provenance is not None
        assert a.provenance.extractor == "spacy"

    def test_json_roundtrip(self) -> None:
        a = Annotation(
            id="ann-5",
            type=AnnotationType.EXTERNAL_CITATION,
            targets=(AnnotationTarget(node_ref="#/body/7", start_offset=0, end_offset=30),),
            body={"reporter": "F.3d", "volume": "123", "page": "456"},
        )
        restored = Annotation.model_validate_json(a.model_dump_json())
        assert restored == a

    def test_tracked_change_annotation(self) -> None:
        a = Annotation(
            id="rev-1",
            type=AnnotationType.TRACKED_CHANGE,
            targets=(AnnotationTarget(node_ref="#/body/3/children/1"),),
            body={
                "change_type": "insertion",
                "author": "Jane Smith",
                "date": "2026-04-15T10:30:00Z",
                "revision_id": "1",
            },
        )
        assert a.type == AnnotationType.TRACKED_CHANGE
        assert a.body["change_type"] == "insertion"
        # Round-trip
        restored = Annotation.model_validate_json(a.model_dump_json())
        assert restored == a
