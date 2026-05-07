"""View models for hierarchical document navigation.

All models are frozen Pydantic v2 (matching kaos-content conventions).
These are computed views — never stored in the AST.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from kaos_content.model.blocks import Block


class PageView(BaseModel):
    """Blocks grouped by provenance page number."""

    model_config = ConfigDict(frozen=True)

    page_number: int
    """1-indexed page number (matching provenance convention)."""

    blocks: tuple[Block, ...] = ()
    """All blocks on this page, in reading order."""

    block_refs: tuple[str, ...] = ()
    """JSON pointer refs for each block (parallel to blocks)."""

    section_refs: tuple[str, ...] = ()
    """Heading refs of sections that start on or span this page."""


class SectionView(BaseModel):
    """Content grouped by heading hierarchy (recursive)."""

    model_config = ConfigDict(frozen=True)

    heading_ref: str | None = None
    """JSON pointer ref of the heading block. None for preamble."""

    heading_text: str = ""
    """Extracted text of the heading. Empty for preamble."""

    depth: int = 0
    """Heading depth (1-6). 0 for preamble (content before first heading)."""

    blocks: tuple[Block, ...] = ()
    """Direct content blocks in this section (NOT including subsection blocks)."""

    block_refs: tuple[str, ...] = ()
    """JSON pointer refs for each block (parallel to blocks)."""

    subsections: tuple[SectionView, ...] = ()
    """Child sections (headings of deeper depth)."""

    page_range: tuple[int, int] | None = None
    """(start_page, end_page) inclusive. None if no provenance pages."""


class ParagraphView(BaseModel):
    """A paragraph with its location context."""

    model_config = ConfigDict(frozen=True)

    block_ref: str
    """JSON pointer ref of the paragraph block."""

    text: str
    """Extracted text content."""

    page: int | None = None
    """Page number from provenance (1-indexed)."""

    section_ref: str | None = None
    """Heading ref of the containing section."""

    confidence: float | None = None
    """Source-level confidence from the extraction layer (e.g. Tesseract
    OCR confidence on a scanned PDF). When the underlying block's
    Provenance carries a ``confidence`` value, it propagates here so
    downstream retrieval / verification can refuse to cite low-quality
    passages. ``None`` for born-digital text or when the extractor
    didn't report confidence. Range ``[0.0, 1.0]``. N6."""


class SentenceView(BaseModel):
    """A sentence within a paragraph, with full context."""

    model_config = ConfigDict(frozen=True)

    text: str
    """Sentence text."""

    start: int
    """Character offset within the paragraph text."""

    end: int
    """Character offset end within the paragraph text."""

    confidence: float = 1.0
    """Segmentation confidence from the tokenizer."""

    paragraph_ref: str
    """Block ref of the containing paragraph."""

    page: int | None = None
    """Page from provenance."""

    section_ref: str | None = None
    """Heading ref of the containing section."""


# Rebuild models for forward references (SectionView self-references via subsections)
SectionView.model_rebuild()
