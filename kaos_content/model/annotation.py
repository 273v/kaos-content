"""Annotation types for standoff markup."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kaos_content.model.attr import Provenance


class AnnotationType(StrEnum):
    """Annotation type discriminator."""

    # Generic
    HIGHLIGHT = "highlight"
    COMMENT = "comment"
    BOOKMARK = "bookmark"

    # Legal domain
    REDACTION = "redaction"
    DEFINED_TERM = "defined_term"
    TERM_DEFINITION = "term_definition"
    CROSS_REFERENCE = "cross_reference"
    EXTERNAL_CITATION = "external_citation"
    AMENDMENT = "amendment"
    PROVISION = "provision"

    # NLP
    ENTITY = "entity"
    SENTIMENT = "sentiment"
    CLASSIFICATION = "classification"

    # Structured extraction (KAOS WS-TR)
    EXTRACTED_CELL = "extracted_cell"
    """Bidirectional click-through anchor: a source span that contributed
    to an extracted cell. Body shape: ``{"column_id": str,
    "schema_version": int, "cell_ref": str, "doc_id": str}``."""

    # Tracked changes / revisions (OOXML w:ins, w:del, w:moveFrom, w:moveTo, w:*Change)
    TRACKED_CHANGE = "tracked_change"
    """Tracked change metadata for a revision in the document. Points at the
    Span/Div node(s) carrying the ``rev-*`` classes with revision content.
    Body shape: ``{"change_type": str, "author": str, "date": str | None,
    "revision_id": str, "move_name": str | None}``. Valid ``change_type`` values:
    ``"insertion"``, ``"deletion"``, ``"move_from"``, ``"move_to"``,
    ``"format_change"``, ``"paragraph_property_change"``."""

    # Document-structure analysis (kaos-nlp-core P7)
    HEADING_CANDIDATE = "heading_candidate"
    """Heading-detector output. Body shape: ``{"score": float,
    "hierarchy_level": int | None, "numeric_depth": int | None,
    "atx_depth": int | None, "enumerator_kind": str | None,
    "lexicon_used": str | None}``. Per Q8 of the
    ``SECTION_HEADING_PRIMITIVES_RESEARCH.md`` design reference."""

    BOILERPLATE = "boilerplate"
    """Recurring header/footer/page-number/caption run detected by the
    boilerplate detector. Body shape: ``{"kind": str, "occurrences": int,
    "fingerprint": int, "language_hint": str | None}``. Valid ``kind``
    values: ``"page_number"``, ``"caption"``, ``"header"``, ``"footer"``,
    ``"unknown"``, ``"exact_dup"``, ``"near_dup"``."""

    TABLE_ROW = "table_row"
    """Line classified as a table-row by the structure decoder (P7). Body
    shape: ``{"row_index": int, "column_count": int | None}``. Useful as
    a hint to downstream table-extraction passes."""

    METADATA = "metadata"
    """Inline-colon metadata line (`Author: Jane Doe`). Body shape:
    ``{"kind": str, "canonical_value": str | None}``. Valid ``kind``
    values: ``"author"``, ``"date"``, ``"case_number"``, ``"footer"``,
    ``"front_matter"``, ``"unknown"``."""


class AnnotationTarget(BaseModel):
    """Identifies the span an annotation covers."""

    model_config = ConfigDict(frozen=True)
    node_ref: str
    start_offset: int | None = None
    end_offset: int | None = None

    @model_validator(mode="after")
    def _validate_offsets(self) -> Self:
        if self.start_offset is not None and self.start_offset < 0:
            msg = "start_offset must be non-negative"
            raise ValueError(msg)
        if self.end_offset is not None and self.end_offset < 0:
            msg = "end_offset must be non-negative"
            raise ValueError(msg)
        if (
            self.start_offset is not None
            and self.end_offset is not None
            and self.start_offset > self.end_offset
        ):
            msg = "start_offset must be <= end_offset"
            raise ValueError(msg)
        return self


class Annotation(BaseModel):
    """Standoff annotation that can span across tree boundaries."""

    model_config = ConfigDict(frozen=True)
    id: str
    type: AnnotationType
    targets: tuple[AnnotationTarget, ...]
    # Per-instance dict via default_factory — see Attr.kv for the
    # full reasoning on shared-mutable-default.
    body: dict[str, Any] = Field(default_factory=dict)
    provenance: Provenance | None = None
