"""Typed schemas for ``Annotation.body`` contents.

The ``Annotation.body`` field is ``dict[str, Any]`` for maximum
flexibility — readers from different backends record different shapes.
These Pydantic models document and validate the conventional body shape
for each ``AnnotationType`` used by KAOS readers and writers.

Two usage patterns:

1. **Construction** — build a typed body, serialize to dict::

       ann = Annotation(
           id="c0",
           type=AnnotationType.COMMENT,
           targets=(...),
           body=CommentBody(author="Alice", text="hi").model_dump(),
       )

2. **Typed access** — validate an existing annotation::

       from kaos_content.model.annotation import parse_body
       body = parse_body(ann)  # returns CommentBody | TrackedChangeBody | ...

Backward compatible: ``Annotation.body`` is still a plain dict on disk.
These schemas are opt-in validation and IDE support.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from kaos_content.model.annotation import AnnotationType


class CommentBody(BaseModel):
    """Conventional body shape for ``AnnotationType.COMMENT``."""

    model_config = ConfigDict(frozen=True)

    author: str
    date: str | None = None  # ISO-8601
    text: str
    initials: str = ""
    comment_id: str | None = None  # optional source ID (DOCX w:id, etc.)


class TrackedChangeBody(BaseModel):
    """Conventional body shape for ``AnnotationType.TRACKED_CHANGE``."""

    model_config = ConfigDict(frozen=True)

    change_type: Literal[
        "insertion",
        "deletion",
        "move_from",
        "move_to",
        "format_change",
        "paragraph_property_change",
    ]
    author: str
    date: str | None = None  # ISO-8601
    revision_id: str
    move_name: str | None = None  # pairs moveFrom/moveTo


class RedactionBody(BaseModel):
    """Conventional body shape for ``AnnotationType.REDACTION``."""

    model_config = ConfigDict(frozen=True)

    reason: str | None = None
    redactor: str | None = None


class DefinedTermBody(BaseModel):
    """Body for ``AnnotationType.DEFINED_TERM`` — capitalized term marker."""

    model_config = ConfigDict(frozen=True)

    definition_id: str
    definition_text: str | None = None
    definition_ref: str | None = None  # node_ref of the defining text


class CrossReferenceBody(BaseModel):
    """Body for ``AnnotationType.CROSS_REFERENCE``."""

    model_config = ConfigDict(frozen=True)

    target_ref: str
    label: str | None = None


class ExtractedCellBody(BaseModel):
    """Body for ``AnnotationType.EXTRACTED_CELL`` (WS-TR click-through)."""

    model_config = ConfigDict(frozen=True)

    column_id: str
    schema_version: int
    cell_ref: str
    doc_id: str


class ExternalCitationBody(BaseModel):
    """Body for ``AnnotationType.EXTERNAL_CITATION``."""

    model_config = ConfigDict(frozen=True)

    reporter: str | None = None
    volume: str | None = None
    page: str | None = None
    url: str | None = None
    citation_string: str | None = None


class EntityBody(BaseModel):
    """Body for ``AnnotationType.ENTITY`` (NER output)."""

    model_config = ConfigDict(frozen=True)

    entity_type: str
    text: str
    confidence: float | None = None


class SentimentBody(BaseModel):
    """Body for ``AnnotationType.SENTIMENT``."""

    model_config = ConfigDict(frozen=True)

    sentiment: Literal["positive", "negative", "neutral", "mixed"]
    score: float | None = None


class ClassificationBody(BaseModel):
    """Body for ``AnnotationType.CLASSIFICATION``."""

    model_config = ConfigDict(frozen=True)

    label: str
    confidence: float | None = None
    classifier: str | None = None


class HeadingCandidateBody(BaseModel):
    """Body for ``AnnotationType.HEADING_CANDIDATE``.

    Emitted by the kaos-nlp-core structure layer (P7) for every heading
    line the decoder labels as ``heading``. Carries both the
    hierarchy-keyword depth and the numeric / ATX depth so consumers can
    pick which one to honor.
    """

    model_config = ConfigDict(frozen=True)

    score: float
    hierarchy_level: int | None = None
    numeric_depth: int | None = None
    atx_depth: int | None = None
    enumerator_kind: str | None = None
    lexicon_used: str | None = None


class BoilerplateBody(BaseModel):
    """Body for ``AnnotationType.BOILERPLATE``.

    Emitted by the kaos-nlp-core boilerplate detector (P5) for each
    detected run of repeated header / footer / page-number / caption
    lines. The ``language_hint`` is set only for caption runs whose
    prefix matched a Western-language caption lexicon
    (``"english"``, ``"german"``, ``"french"``, ``"spanish"``,
    ``"italian"``, ``"portuguese"``).
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal[
        "page_number",
        "caption",
        "header",
        "footer",
        "unknown",
        "exact_dup",
        "near_dup",
    ]
    occurrences: int
    fingerprint: int | None = None
    language_hint: str | None = None


class TableRowBody(BaseModel):
    """Body for ``AnnotationType.TABLE_ROW``.

    Hint to downstream table-extraction passes that the line shape
    (`HAS_PIPE` or `HAS_COLUMN_GAPS`) suggests a tabular row.
    """

    model_config = ConfigDict(frozen=True)

    row_index: int | None = None
    column_count: int | None = None


class MetadataBody(BaseModel):
    """Body for ``AnnotationType.METADATA``.

    Inline-colon metadata line (`Author: Jane Doe`,
    `Date: 2026-05-05`). The ``kind`` is a coarse classification — the
    structure layer emits ``"unknown"`` by default; downstream passes
    can refine it.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal[
        "author",
        "date",
        "case_number",
        "footer",
        "front_matter",
        "unknown",
    ] = "unknown"
    canonical_value: str | None = None


# ---------------------------------------------------------------------------
# Registry and parser
# ---------------------------------------------------------------------------


BODY_REGISTRY: dict[AnnotationType, type[BaseModel]] = {
    AnnotationType.COMMENT: CommentBody,
    AnnotationType.TRACKED_CHANGE: TrackedChangeBody,
    AnnotationType.REDACTION: RedactionBody,
    AnnotationType.DEFINED_TERM: DefinedTermBody,
    AnnotationType.CROSS_REFERENCE: CrossReferenceBody,
    AnnotationType.EXTRACTED_CELL: ExtractedCellBody,
    AnnotationType.EXTERNAL_CITATION: ExternalCitationBody,
    AnnotationType.ENTITY: EntityBody,
    AnnotationType.SENTIMENT: SentimentBody,
    AnnotationType.CLASSIFICATION: ClassificationBody,
    AnnotationType.HEADING_CANDIDATE: HeadingCandidateBody,
    AnnotationType.BOILERPLATE: BoilerplateBody,
    AnnotationType.TABLE_ROW: TableRowBody,
    AnnotationType.METADATA: MetadataBody,
}


def parse_body(annotation: Any) -> BaseModel | dict[str, Any]:
    """Validate and return a typed body for the annotation.

    If the annotation's ``type`` has a registered body schema, returns an
    instance of that schema validated against ``annotation.body``. If
    the type is unregistered or validation fails softly, returns the raw
    dict unchanged.

    Raises ``pydantic.ValidationError`` if validation fails strictly;
    callers that want best-effort behavior should catch and fall back to
    the raw dict.
    """
    body_cls = BODY_REGISTRY.get(annotation.type)
    if body_cls is None:
        return annotation.body
    return body_cls.model_validate(annotation.body)
