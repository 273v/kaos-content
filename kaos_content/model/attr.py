"""Shared types: Attr, Provenance, BoundingBox, SourceRef, Caption, ColSpec."""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, model_validator

if TYPE_CHECKING:
    from kaos_content.model.blocks import Block
    from kaos_content.model.inlines import Inline


class CoordOrigin(StrEnum):
    """Coordinate origin for bounding boxes."""

    TOP_LEFT = "top_left"
    BOTTOM_LEFT = "bottom_left"


class Alignment(StrEnum):
    """Column or cell alignment."""

    LEFT = "left"
    CENTER = "center"
    RIGHT = "right"


class BoundingBox(BaseModel):
    """Spatial position on a page.

    Coordinates may be negative on translated pages; we do not enforce
    sign. We DO enforce that ``right >= left`` and ``bottom >= top``
    when ``coord_origin`` is the conventional ``top_left``, since a
    zero/negative-extent box is not a valid region.
    """

    model_config = ConfigDict(frozen=True)
    left: float
    top: float
    right: float
    bottom: float
    coord_origin: CoordOrigin = CoordOrigin.TOP_LEFT

    @model_validator(mode="after")
    def _check_extents(self) -> BoundingBox:
        # In TOP_LEFT origin, a valid box has right >= left, bottom >= top.
        # In BOTTOM_LEFT origin (PDF-style), bottom < top is the convention,
        # so we only enforce horizontal extent.
        if self.right < self.left:
            msg = f"BoundingBox: right ({self.right}) must be >= left ({self.left})"
            raise ValueError(msg)
        if self.coord_origin is CoordOrigin.TOP_LEFT and self.bottom < self.top:
            msg = f"BoundingBox(top_left): bottom ({self.bottom}) must be >= top ({self.top})"
            raise ValueError(msg)
        return self


class SourceRef(BaseModel):
    """Reference to a source artifact."""

    model_config = ConfigDict(frozen=True)
    uri: str
    mime_type: str | None = None
    artifact_id: str | None = None


class Provenance(BaseModel):
    """Source location metadata.

    Field constraints:
    - ``page`` is 1-indexed (``ge=1``); 0 is the historical "unknown
      page" sentinel and is rejected — callers should use ``None``.
    - ``confidence`` is bounded to ``[0.0, 1.0]``. Values outside that
      range were silently accepted before 0.1.0a1 and made it into
      downstream ranking and reviewer-overlay logic.
    - ``char_span`` is a half-open ``(start, end)`` pair where ``start``
      and ``end`` are non-negative and ``end >= start``.
    """

    model_config = ConfigDict(frozen=True)
    source: SourceRef | None = None
    page: int | None = Field(default=None, ge=1)
    bbox: BoundingBox | None = None
    char_span: tuple[int, int] | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    extractor: str | None = None

    @model_validator(mode="after")
    def _check_char_span(self) -> Provenance:
        if self.char_span is None:
            return self
        start, end = self.char_span
        if start < 0 or end < 0:
            msg = f"Provenance.char_span: must be non-negative; got {self.char_span}"
            raise ValueError(msg)
        if end < start:
            msg = f"Provenance.char_span: end ({end}) must be >= start ({start})"
            raise ValueError(msg)
        return self


class Attr(BaseModel):
    """Universal node attributes (Pandoc's Attr triple)."""

    model_config = ConfigDict(frozen=True)
    id: str | None = None
    classes: tuple[str, ...] = ()
    # ``default_factory=dict`` — each Attr instance gets its own dict.
    # Sharing a single ``{}`` literal across instances would be a
    # subtle aliasing bug if any caller ever mutated the dict
    # post-construction. The frozen contract blocks reassignment but
    # not nested mutation, so ``default_factory`` is the right
    # belt-and-braces fix.
    kv: dict[str, str] = Field(default_factory=dict)


class ColSpec(BaseModel):
    """Column specification for tables."""

    model_config = ConfigDict(frozen=True)
    alignment: Alignment | None = None
    width: float | None = None


class Caption(BaseModel):
    """Table or figure caption."""

    model_config = ConfigDict(frozen=True)
    short: tuple[Inline, ...] | None = None
    body: tuple[Block, ...] = ()
