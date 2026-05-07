"""Tests for kaos_content.model.annotation_bodies — typed Annotation.body schemas."""

from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError

from kaos_content.model.annotation import (
    Annotation,
    AnnotationTarget,
    AnnotationType,
)
from kaos_content.model.annotation_bodies import (
    BODY_REGISTRY,
    ClassificationBody,
    CommentBody,
    CrossReferenceBody,
    DefinedTermBody,
    EntityBody,
    ExternalCitationBody,
    ExtractedCellBody,
    RedactionBody,
    SentimentBody,
    TrackedChangeBody,
    parse_body,
)


class TestBodySchemas:
    def test_comment_body_requires_author_and_text(self) -> None:
        body = CommentBody(author="Alice", text="hello")
        assert body.author == "Alice"
        assert body.text == "hello"
        assert body.date is None
        assert body.initials == ""

    def test_comment_body_missing_required(self) -> None:
        with pytest.raises(ValidationError):
            CommentBody.model_validate({})

    def test_tracked_change_body_valid(self) -> None:
        body = TrackedChangeBody(
            change_type="insertion",
            author="Bob",
            revision_id="1",
        )
        assert body.change_type == "insertion"
        assert body.move_name is None

    def test_tracked_change_invalid_change_type(self) -> None:
        with pytest.raises(ValidationError):
            TrackedChangeBody(
                change_type=cast(Any, "bogus"),
                author="x",
                revision_id="1",
            )

    def test_sentiment_body_literals(self) -> None:
        body = SentimentBody(sentiment="positive", score=0.9)
        assert body.sentiment == "positive"
        with pytest.raises(ValidationError):
            SentimentBody(sentiment=cast(Any, "happy"))

    def test_all_bodies_instantiable(self) -> None:
        """Smoke test: every body type can be constructed with minimum fields."""
        CommentBody(author="a", text="t")
        TrackedChangeBody(change_type="insertion", author="a", revision_id="1")
        RedactionBody()
        DefinedTermBody(definition_id="d1")
        CrossReferenceBody(target_ref="#/body/0")
        ExtractedCellBody(column_id="c", schema_version=1, cell_ref="r", doc_id="d")
        ExternalCitationBody()
        EntityBody(entity_type="ORG", text="ACME")
        SentimentBody(sentiment="neutral")
        ClassificationBody(label="x")


class TestConstructionFromTypedBody:
    def test_annotation_with_comment_body(self) -> None:
        body = CommentBody(author="Alice", text="please review", initials="AM")
        ann = Annotation(
            id="c0",
            type=AnnotationType.COMMENT,
            targets=(AnnotationTarget(node_ref="#/body/0"),),
            body=body.model_dump(),
        )
        assert ann.body["author"] == "Alice"
        assert ann.body["text"] == "please review"

    def test_annotation_json_roundtrip(self) -> None:
        body = TrackedChangeBody(
            change_type="deletion",
            author="Bob",
            date="2026-04-18T12:00:00Z",
            revision_id="5",
        )
        ann = Annotation(
            id="r0",
            type=AnnotationType.TRACKED_CHANGE,
            targets=(),
            body=body.model_dump(),
        )
        restored = Annotation.model_validate_json(ann.model_dump_json())
        assert restored == ann


class TestParseBody:
    def test_comment_parse(self) -> None:
        ann = Annotation(
            id="c0",
            type=AnnotationType.COMMENT,
            targets=(),
            body={"author": "Alice", "text": "hi"},
        )
        parsed = parse_body(ann)
        assert isinstance(parsed, CommentBody)
        assert parsed.author == "Alice"

    def test_tracked_change_parse(self) -> None:
        ann = Annotation(
            id="r0",
            type=AnnotationType.TRACKED_CHANGE,
            targets=(),
            body={
                "change_type": "insertion",
                "author": "A",
                "revision_id": "1",
            },
        )
        parsed = parse_body(ann)
        assert isinstance(parsed, TrackedChangeBody)
        assert parsed.change_type == "insertion"

    def test_unregistered_type_returns_dict(self) -> None:
        ann = Annotation(
            id="a0",
            type=AnnotationType.HIGHLIGHT,  # not in registry
            targets=(),
            body={"color": "yellow"},
        )
        parsed = parse_body(ann)
        assert parsed == {"color": "yellow"}

    def test_invalid_body_raises(self) -> None:
        """parse_body raises on invalid body shapes — caller can catch."""
        ann = Annotation(
            id="bad",
            type=AnnotationType.TRACKED_CHANGE,
            targets=(),
            body={"author": "no change_type or revision_id"},
        )
        with pytest.raises(ValidationError):
            parse_body(ann)


class TestRegistry:
    def test_registry_covers_major_types(self) -> None:
        assert AnnotationType.COMMENT in BODY_REGISTRY
        assert AnnotationType.TRACKED_CHANGE in BODY_REGISTRY
        assert AnnotationType.REDACTION in BODY_REGISTRY
        assert AnnotationType.EXTRACTED_CELL in BODY_REGISTRY

    def test_every_registered_body_is_pydantic_model(self) -> None:
        from pydantic import BaseModel

        for body_cls in BODY_REGISTRY.values():
            assert issubclass(body_cls, BaseModel)
