"""Extraction primitives — ExtractionCell, ExtractionCitation, ExtractionError.

These types form the per-cell audit shape for KAOS structured extraction.
They are read by tabular sinks (TabularDocument.from_cells) and written
by extraction programs (kaos-llm-core's Extract). The type-level dependency
flows one direction only: kaos-content owns the data shape; LLM-runtime
fields are populated as opaque payload by the writer.

Shape converges Relativity aiR's per-cell audit tuple
{prediction, score, rationale, considerations, citations, errors}
with Reducto's citation schema {bbox, page, original_page, confidence,
parent_block} and Harvey's multi-color flags. block_ref anchors back to
the kaos-content AST — a KAOS-native advantage no surveyed vendor offers.

See ``docs/design/structured-extraction-roadmap.md`` §2.5 and
``docs/design/structured-extraction-integration.md`` §1 for design rationale.
"""

from __future__ import annotations

import datetime
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kaos_content.model.attr import BoundingBox

CellStatus = Literal["extracted", "not_in_document", "unclear", "error"]
"""Three-state refusal semantics. Distinguishes:

- ``extracted`` — value found and the model is confident.
- ``not_in_document`` — model searched and confirmed absence.
- ``unclear`` — found possible evidence but low confidence.
- ``error`` — extraction failure (retry candidate).

Only Relativity aiR and Reducto preserve all four cleanly across the
vendor landscape; KAOS adopts the same discipline.
"""


ExtractionErrorCode = Literal[
    "doc_text_empty",
    "doc_text_too_long",
    "model_api_error",
    "schema_validation_failed",
    "span_verification_failed",
    "refusal",
    "timeout",
    "other",
]
"""Mirrors Relativity aiR's documented error strings plus a few KAOS-specific
codes for span-verification and schema-validation failures.
"""


class ExtractionCitation(BaseModel):
    """Per-span provenance for an extracted value.

    Strictly an extension of :class:`kaos_content.model.attr.Provenance` with
    two additions required for tabular review:

    - ``block_ref`` — JSON-pointer ref into the kaos-content AST (not in
      Provenance). This is the KAOS-native anchor that lets the same Cell
      serialize cleanly into Reducto-style (bbox+page), Harvey-style
      (snippet+page), or Docugami-style (chunk+tag) citation views.
    - ``parent_block_ref`` — the surrounding block for context. Reducto
      surface their Excel-extraction equivalent as ``parentBlock``; useful
      for UIs that show context without re-reading the source.

    Coordinates: ``bbox`` is in source-document pixel space (kaos-content
    convention via :class:`BoundingBox`). To export 0-1 normalized
    coordinates (Reducto convention), use :func:`normalize_bbox`.

    The ``snippet_sha256`` field carries WS-1's normalized-snippet hash,
    enabling tamper-evident verification across pipelines.
    """

    model_config = ConfigDict(frozen=True)

    block_ref: str
    """JSON-pointer ref into the source ContentDocument AST.

    Format: ``#/body/{i}`` or ``#/body/{i}/children/{j}/...`` —
    resolvable via :class:`kaos_content.traversal.NodeIndex`.
    """

    page: int = Field(ge=1)
    """1-indexed page number in the source document."""

    bbox: BoundingBox | None = None
    """Spatial bounds in source-document pixel space. ``None`` for
    text-only formats without a layout (HTML, plain text)."""

    char_span: tuple[int, int]
    """``(start, end)`` character offsets into the resolved block's text.
    ``end`` is exclusive (Python slice convention)."""

    snippet: str
    """Verbatim text extracted from the source span.

    Must match ``source[char_span[0]:char_span[1]]`` modulo the
    normalization used by the verification strategy."""

    snippet_sha256: str = Field(min_length=64, max_length=64)
    """SHA-256 hex of the normalized snippet (WS-1 convention).

    Computed by :func:`kaos_llm_core.signatures.span_tagging.normalize`
    + ``hashlib.sha256``. Tamper-evident across pipelines."""

    confidence: float = Field(ge=0.0, le=1.0)
    """Per-citation reliability score in ``[0, 1]``."""

    parent_block_ref: str | None = None
    """JSON-pointer ref of the containing block, when meaningful (e.g.,
    a sentence inside a paragraph). Optional context for UIs."""


class ExtractionError(BaseModel):
    """Per-cell error record. Mirrors Relativity aiR's error column.

    Carries enough information for an agent to decide whether to retry
    automatically (``retry_recommended``) or escalate to a reviewer.
    """

    model_config = ConfigDict(frozen=True)

    code: ExtractionErrorCode
    message: str
    retry_recommended: bool = False
    attempt: int = Field(default=1, ge=1)


class ExtractionCell(BaseModel):
    """Per-cell extraction record — the canonical KAOS audit tuple.

    One ``ExtractionCell`` corresponds to one ``(doc_id, column_id)`` pair
    in the tabular review model. Cells are independent units of work
    (matching Harvey's per-cell-is-a-model-call discipline), enabling:

    - **Partial-failure retry**: re-run only failed cells, not whole rows.
    - **Mid-run schema edits**: editing a column rebuilds only that column.
    - **Concurrent execution**: futures over ``(doc_id, column_id)`` pairs.
    - **Reviewer overlay**: ``ai_value`` is immutable after extraction;
      reviewer edits land in ``reviewed_value`` (Relativity aiR pattern).

    Pydantic so :func:`kaos_llm_core.programs.batch.batch_run` round-trips
    cells through JSONL natively via ``model_dump()``.
    """

    model_config = ConfigDict(frozen=True)

    # -- Identity --------------------------------------------------------
    doc_id: str
    """Stable identifier for the source document."""

    column_id: str
    """Stable identifier for the schema column. Matches
    ``ColumnSpec.id`` in :mod:`kaos_llm_core.signatures.extraction`."""

    schema_version: int = Field(ge=1)
    """Monotonic schema version. Cells with stale versions are
    re-extraction candidates when the schema evolves."""

    # -- Refusal status -------------------------------------------------
    status: CellStatus
    """Three-state semantic enum. See :data:`CellStatus`."""

    # -- Model output ---------------------------------------------------
    ai_value: Any | None = None
    """The extracted value, typed per the schema's ``ColumnSpec``.
    ``None`` when ``status != "extracted"``."""

    rationale: str | None = None
    """Natural-language justification for the extracted value."""

    considerations: str | None = None
    """Adversarial self-critique — assumptions, alternative readings,
    missing facts that could reverse the prediction. Borrowed from
    Relativity aiR's ``Considerations`` field. Opt-in via
    ``provenance="grounded"`` to avoid the doubled token cost."""

    citations: tuple[ExtractionCitation, ...] = ()
    """Source spans verifying the extracted value. Empty for refusals."""

    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    """Overall confidence in ``[0, 1]``. Required for ``extracted`` and
    ``unclear``; ``None`` for refusals and errors."""

    error: ExtractionError | None = None
    """Error record. Present iff ``status == "error"``."""

    # -- Reviewer overlay (Relativity aiR-vs-user-field separation) ----
    reviewed_value: Any | None = None
    """Reviewer's final coding. Never overwrites ``ai_value`` —
    preserves a complete audit trail of model-vs-human."""

    reviewer_id: str | None = None
    reviewed_at: datetime.datetime | None = None
    review_note: str | None = None

    # -- Multi-color flags (Harvey) -------------------------------------
    flags: tuple[str, ...] = ()
    """Free-form flag tokens (e.g., ``"red"``, ``"high-risk"``,
    ``"needs-second-review"``). Harvey's UI uses these for risk
    triage; KAOS treats them as opaque tokens."""

    # -- Cost / observability -------------------------------------------
    model: str = ""
    """Provider/model identifier that produced this cell
    (e.g., ``"anthropic:claude-haiku-4-5"``)."""

    cost_usd: float = Field(default=0.0, ge=0.0)
    """USD cost of this cell's LLM call(s)."""

    tokens_total: int = Field(default=0, ge=0)
    """Total tokens (prompt + completion) for this cell."""

    latency_ms: float = Field(default=0.0, ge=0.0)
    """Wall-clock latency for this cell's extraction in milliseconds."""

    @model_validator(mode="after")
    def _check_status_invariants(self) -> Self:
        """Enforce the status/value/error/citations invariants."""
        if self.status == "extracted":
            if self.ai_value is None:
                msg = "status='extracted' requires ai_value to be set"
                raise ValueError(msg)
            if self.confidence is None:
                msg = "status='extracted' requires confidence to be set"
                raise ValueError(msg)
        elif self.status == "unclear":
            if self.confidence is None:
                msg = "status='unclear' requires confidence to be set"
                raise ValueError(msg)
        elif self.status == "not_in_document":
            if self.ai_value is not None:
                msg = "status='not_in_document' requires ai_value to be None"
                raise ValueError(msg)
            if self.error is not None:
                msg = "status='not_in_document' requires error to be None"
                raise ValueError(msg)
            if self.citations:
                msg = "status='not_in_document' requires citations to be empty"
                raise ValueError(msg)
        elif self.status == "error":
            if self.error is None:
                msg = "status='error' requires error record to be set"
                raise ValueError(msg)
            if self.ai_value is not None:
                msg = "status='error' requires ai_value to be None"
                raise ValueError(msg)

        # Reviewer overlay integrity
        has_overlay = self.reviewed_value is not None or self.review_note is not None
        if has_overlay and (self.reviewer_id is None or self.reviewed_at is None):
            msg = (
                "reviewer overlay requires both reviewer_id and reviewed_at "
                "when reviewed_value or review_note is set"
            )
            raise ValueError(msg)

        return self


def normalize_bbox(bbox: BoundingBox, page_width: float, page_height: float) -> dict[str, float]:
    """Convert a pixel-space :class:`BoundingBox` to 0-1 normalized coords.

    Returns a dict matching Reducto's published citation schema:
    ``{"left", "top", "width", "height"}`` all in ``[0, 1]``.

    This is the export boundary — keep absolute pixels internally
    (kaos-content convention) and normalize only when emitting to
    vendors that expect normalized coords.

    Args:
        bbox: The pixel-space bounding box.
        page_width: Source page width in the same units as bbox coordinates.
        page_height: Source page height in the same units as bbox coordinates.

    Returns:
        ``{"left": float, "top": float, "width": float, "height": float}``
        with each value in ``[0, 1]``.

    Raises:
        ValueError: If ``page_width`` or ``page_height`` is non-positive.
    """
    if page_width <= 0 or page_height <= 0:
        msg = f"page dimensions must be positive, got width={page_width}, height={page_height}"
        raise ValueError(msg)
    width = bbox.right - bbox.left
    height = bbox.bottom - bbox.top
    return {
        "left": bbox.left / page_width,
        "top": bbox.top / page_height,
        "width": width / page_width,
        "height": height / page_height,
    }
