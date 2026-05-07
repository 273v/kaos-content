"""Document-level metadata."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from kaos_content.model.attr import SourceRef


class PageSetup(BaseModel):
    """Page geometry for formats that distinguish it (DOCX, PPTX, PDF).

    All dimensions are in typographic **points** (1 pt = 1/72 inch).
    Readers convert format-native units (OOXML twips, EMUs, pixels) to
    points at the boundary so downstream code has a single unit system.

    All fields are optional — ``None`` means "not specified; use the
    writer's default". A fresh ``PageSetup()`` is equivalent to absent
    metadata.
    """

    model_config = ConfigDict(frozen=True)
    page_width_pt: float | None = None
    page_height_pt: float | None = None
    margin_top_pt: float | None = None
    margin_bottom_pt: float | None = None
    margin_left_pt: float | None = None
    margin_right_pt: float | None = None
    header_distance_pt: float | None = None
    footer_distance_pt: float | None = None


SectionBreakType = Literal["continuous", "nextPage", "nextColumn", "evenPage", "oddPage"]


class Section(BaseModel):
    """A contiguous range of body blocks sharing page-layout properties.

    Real-world documents often contain multiple regions with different
    page size, margins, or columns: a portrait cover page followed by a
    landscape tables section, region-specific headers, mid-document
    orientation changes. OOXML models this with one ``<w:sectPr>`` per
    region; ``Section`` is the kaos-content equivalent.

    Sections index into ``ContentDocument.body``: section ``i`` covers
    ``body[prev_end:section.end_block_index]`` where ``prev_end`` is the
    previous section's ``end_block_index`` (0 for the first). The final
    section's ``end_block_index`` must equal ``len(body)``. When
    ``ContentDocument.sections`` is empty the whole body is a single
    implicit section described by ``metadata.page_setup``.

    ``break_type`` is the OOXML ``w:sectPr/w:type/@w:val`` value — the
    default ``nextPage`` is what Word writes when the type attribute is
    omitted, and covers the common case of "new section starts on a new
    page".
    """

    model_config = ConfigDict(frozen=True)
    end_block_index: int
    page_setup: PageSetup | None = None
    break_type: SectionBreakType = "nextPage"


class DocumentMetadata(BaseModel):
    """Document-level metadata."""

    model_config = ConfigDict(frozen=True)
    title: str | None = None
    authors: tuple[str, ...] = ()
    date: str | None = None
    language: str | None = None
    source: SourceRef | None = None
    document_type: str | None = None
    page_setup: PageSetup | None = None
    # Per-instance dict via default_factory — see Attr.kv for the
    # full reasoning on shared-mutable-default.
    extra: dict[str, Any] = Field(default_factory=dict)
