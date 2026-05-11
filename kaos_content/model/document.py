"""ContentDocument root container."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from kaos_content.model.annotation import Annotation
from kaos_content.model.blocks import Block
from kaos_content.model.metadata import DocumentMetadata, Section
from kaos_content.model.summary import DocumentSummary


class ContentDocument(BaseModel):
    """Root document container."""

    model_config = ConfigDict(frozen=True)

    # ``default_factory`` everywhere a default is mutable — each
    # ContentDocument instance gets its own dicts and its own default
    # DocumentMetadata. Sharing the literal across instances would be
    # a latent aliasing bug if any caller mutated the defaults, even
    # though ``frozen=True`` blocks reassignment.
    metadata: DocumentMetadata = Field(default_factory=DocumentMetadata)
    body: tuple[Block, ...] = ()
    footnotes: dict[str, tuple[Block, ...]] = Field(default_factory=dict)
    definitions: dict[str, str] = Field(default_factory=dict)
    annotations: tuple[Annotation, ...] = ()
    headers: dict[str, tuple[Block, ...]] = Field(default_factory=dict)
    """Page header content keyed by type. Standard keys: ``"default"``,
    ``"first"`` (title page), ``"even"`` (even-numbered pages when
    different from odd). Values are block sequences."""
    footers: dict[str, tuple[Block, ...]] = Field(default_factory=dict)
    """Page footer content keyed by type — same keys as ``headers``."""
    sections: tuple[Section, ...] = ()
    """Page-layout sections. Empty means the whole body is a single
    implicit section described by ``metadata.page_setup`` (backward
    compat). A multi-section document produces one ``Section`` per
    ``<w:sectPr>`` in order. Each section's ``end_block_index`` is
    exclusive; the last section's value must equal ``len(body)``."""

    summary: DocumentSummary | None = None
    """Optional cheap, deterministic summary for corpus-scale triage
    (head tokens + top/bottom n-grams + entity counts). Populated on
    demand by :func:`kaos_content.summarize.build_document_summary`;
    ``None`` is the canonical "not yet computed" state and is
    backward-compatible with serialised documents that predate this
    field. See ``docs/design/findings-entities-summary.md``."""
