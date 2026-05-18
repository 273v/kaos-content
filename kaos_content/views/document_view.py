"""DocumentView: dynamic hierarchical views over a flat ContentDocument AST.

Lazily computes page, section, paragraph, and sentence views from the
canonical flat block sequence. Adapts to what the document actually has:
pages (from provenance), sections (from headings), sentences (from NLP).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from kaos_core.logging import get_logger

from kaos_content.model.blocks import Heading
from kaos_content.model.document import ContentDocument
from kaos_content.model.node import BaseBlock
from kaos_content.serializers.markdown import serialize_markdown
from kaos_content.traversal.index import NodeIndex
from kaos_content.traversal.visitor import extract_text
from kaos_content.views.models import PageView, ParagraphView, SectionView, SentenceView

if TYPE_CHECKING:
    from kaos_content.model.blocks import Block

logger = get_logger(__name__)


class DocumentView:
    """Computed hierarchical views over a flat ContentDocument AST.

    All views are lazily computed on first access and cached. The document
    itself is never modified.

    Args:
        document: The ContentDocument to create views for.
        sentence_segmenter: Optional NLP segmenter (duck-typed: must have
            ``tokenize_spans(text) -> list[tuple[int, int]]``). If not
            provided, sentence views are unavailable.
    """

    __slots__ = (
        "_block_to_section",
        "_document",
        "_flat_sections",
        "_index",
        "_pages",
        "_paragraphs",
        "_section_ancestors",
        "_section_map",
        "_sections",
        "_segmenter",
        "_sentences",
    )

    def __init__(
        self,
        document: ContentDocument,
        *,
        sentence_segmenter: Any | None = None,
    ) -> None:
        self._document = document
        self._index = NodeIndex(document)
        self._segmenter = sentence_segmenter

        # Lazy caches (None = not yet computed)
        self._pages: tuple[PageView, ...] | None = None
        self._sections: tuple[SectionView, ...] | None = None
        self._flat_sections: tuple[SectionView, ...] | None = None
        self._paragraphs: tuple[ParagraphView, ...] | None = None
        self._sentences: tuple[SentenceView, ...] | None = None
        self._section_map: dict[str, SectionView] | None = None
        self._block_to_section: dict[str, str | None] | None = None
        # Cache for ``block_path``: section_heading_ref → ordered heading-text
        # chain (root-first, INCLUSIVE of the section itself). Built lazily on
        # first call to :meth:`block_path`.
        self._section_ancestors: dict[str, tuple[str, ...]] | None = None

    # ── Properties ──

    @property
    def document(self) -> ContentDocument:
        """Underlying ContentDocument the view wraps."""
        return self._document

    @property
    def index(self) -> NodeIndex:
        """NodeIndex over the document — JSON-pointer ref → block lookup."""
        return self._index

    # ── Pages ──

    @property
    def has_pages(self) -> bool:
        """True if any block has provenance.page set."""
        return any(
            b.provenance is not None and b.provenance.page is not None for b in self._document.body
        )

    @property
    def pages(self) -> tuple[PageView, ...]:
        """All pages, derived from per-block ``provenance.page``. Lazy and cached."""
        if self._pages is None:
            self._pages = self._compute_pages()
        return self._pages

    @property
    def page_count(self) -> int:
        """Number of distinct pages with at least one block. Zero for non-paginated docs."""
        return len(self.pages)

    def page(self, page_number: int) -> PageView:
        """Get a specific page by 1-indexed page number."""
        for p in self.pages:
            if p.page_number == page_number:
                return p
        msg = f"Page {page_number} not found"
        raise KeyError(msg)

    def page_as_markdown(self, page_number: int) -> str:
        """Serialize a single page's content as markdown."""
        pv = self.page(page_number)
        temp_doc = ContentDocument(
            metadata=self._document.metadata,
            body=pv.blocks,
        )
        return serialize_markdown(temp_doc)

    # ── Sections ──

    @property
    def has_sections(self) -> bool:
        """True if the document has any headings."""
        return len(self._index.headings) > 0

    @property
    def sections(self) -> tuple[SectionView, ...]:
        """Top-level sections (recursive structure via subsections)."""
        if self._sections is None:
            self._sections = self._compute_sections()
        return self._sections

    @property
    def flat_sections(self) -> tuple[SectionView, ...]:
        """All sections flattened depth-first."""
        if self._flat_sections is None:
            result: list[SectionView] = []
            self._flatten_sections(self.sections, result)
            self._flat_sections = tuple(result)
        return self._flat_sections

    def section_by_ref(self, heading_ref: str) -> SectionView | None:
        """Find a section by its heading's JSON pointer ref."""
        if self._section_map is None:
            self._section_map = {}
            for s in self.flat_sections:
                if s.heading_ref is not None:
                    self._section_map[s.heading_ref] = s
        return self._section_map.get(heading_ref)

    def section_as_markdown(self, heading_ref: str) -> str:
        """Serialize a section's content as markdown."""
        sv = self.section_by_ref(heading_ref)
        if sv is None:
            msg = f"Section {heading_ref} not found"
            raise KeyError(msg)
        # Collect all blocks: this section + subsections
        all_blocks = self.collect_section_blocks(sv)
        temp_doc = ContentDocument(
            metadata=self._document.metadata,
            body=tuple(all_blocks),
        )
        return serialize_markdown(temp_doc)

    def block_path(self, block_ref: str) -> tuple[str, ...]:
        """Structural breadcrumb for a block — the chain of enclosing
        heading texts from the document root down to (and INCLUDING) the
        nearest containing section.

        Returns an empty tuple when the document has no headings, when the
        block lives in the pre-heading preamble, or when ``block_ref`` is
        not known to the view (no `KeyError` — empty tuple is the explicit
        "no structural position available" contract).

        Examples for an NDA with one top-level heading ``"11. GOVERNING LAW"``::

            view.block_path("#/body/22") == ("11. GOVERNING LAW",)

        For a nested structure (``"Chapter 1" → "1.1 Background"`` containing
        the block)::

            view.block_path("#/body/9") == ("Chapter 1", "1.1 Background")

        The chain is empty (``()``) for blocks in the preamble — callers
        should treat empty path as a contract that section identifiers
        cannot be cited for this block.
        """
        # Trigger section computation (populates _block_to_section).
        _ = self.sections
        if self._block_to_section is None:
            return ()
        section_heading_ref = self._block_to_section.get(block_ref)
        if section_heading_ref is None:
            return ()
        if self._section_ancestors is None:
            self._section_ancestors = self._compute_section_ancestors()
        return self._section_ancestors.get(section_heading_ref, ())

    # ── Paragraphs ──

    @property
    def paragraphs(self) -> tuple[ParagraphView, ...]:
        if self._paragraphs is None:
            self._paragraphs = self._compute_paragraphs()
        return self._paragraphs

    # ── Sentences ──

    @property
    def has_sentences(self) -> bool:
        return self._segmenter is not None

    @property
    def sentences(self) -> tuple[SentenceView, ...]:
        if self._sentences is None:
            self._sentences = self._compute_sentences()
        return self._sentences

    def sentences_for_paragraph(self, paragraph_ref: str) -> tuple[SentenceView, ...]:
        """Get sentences for a specific paragraph."""
        return tuple(s for s in self.sentences if s.paragraph_ref == paragraph_ref)

    # ── Typed-entity filters (K2) ──
    #
    # Thin wrappers around free functions in
    # ``kaos_content.views.entity_filters``. The functions hold the
    # actual logic + value types; these methods exist for ergonomic
    # discoverability on the view object. See entity_filters.py for
    # design rationale.

    def sentences_with_entity(self, entity_type: str) -> tuple[Any, ...]:
        """Filter ``self.sentences`` to those containing >=1 match.

        See :func:`kaos_content.views.entity_filters.iter_sentences_with_entity`.
        ``entity_type`` must be one of ``ENTITY_TYPES``.
        """
        from kaos_content.views.entity_filters import iter_sentences_with_entity

        return tuple(iter_sentences_with_entity(self, entity_type))

    def paragraphs_with_entity(self, entity_type: str) -> tuple[Any, ...]:
        """Filter ``self.paragraphs`` to those containing >=1 match.

        See :func:`kaos_content.views.entity_filters.iter_paragraphs_with_entity`.
        """
        from kaos_content.views.entity_filters import iter_paragraphs_with_entity

        return tuple(iter_paragraphs_with_entity(self, entity_type))

    # ── Internal: page computation ──

    def _compute_pages(self) -> tuple[PageView, ...]:
        """Group blocks by provenance.page."""
        page_blocks: dict[int, list[tuple[Block, str]]] = {}
        current_page = 1

        for i, block in enumerate(self._document.body):
            ref = f"#/body/{i}"
            if block.provenance is not None and block.provenance.page is not None:
                current_page = block.provenance.page
            page_blocks.setdefault(current_page, []).append((block, ref))

        # Build section cross-references
        section_refs_by_page: dict[int, set[str]] = {}
        for sv in self.flat_sections:
            if sv.page_range is not None and sv.heading_ref is not None:
                for p in range(sv.page_range[0], sv.page_range[1] + 1):
                    section_refs_by_page.setdefault(p, set()).add(sv.heading_ref)

        pages: list[PageView] = []
        for page_num in sorted(page_blocks.keys()):
            items = page_blocks[page_num]
            blocks = tuple(b for b, _ in items)
            refs = tuple(r for _, r in items)
            sec_refs = tuple(sorted(section_refs_by_page.get(page_num, set())))
            pages.append(
                PageView(
                    page_number=page_num,
                    blocks=blocks,
                    block_refs=refs,
                    section_refs=sec_refs,
                )
            )

        return tuple(pages)

    # ── Internal: section computation ──

    def _compute_sections(self) -> tuple[SectionView, ...]:
        """Build recursive section tree from heading hierarchy."""
        body = self._document.body
        if not body:
            return ()

        # Build flat list of (heading_index, depth) pairs
        heading_positions: list[tuple[int, int]] = []
        for i, block in enumerate(body):
            if isinstance(block, Heading):
                heading_positions.append((i, block.depth))

        if not heading_positions:
            # No headings — entire document is one section
            blocks = body
            refs = tuple(f"#/body/{i}" for i in range(len(blocks)))
            page_range = self._compute_page_range(blocks)
            return (
                SectionView(
                    blocks=blocks,
                    block_refs=refs,
                    page_range=page_range,
                ),
            )

        # Build sections recursively
        sections: list[SectionView] = []

        # Preamble: blocks before first heading
        first_heading_idx = heading_positions[0][0]
        if first_heading_idx > 0:
            pre_blocks = body[:first_heading_idx]
            pre_refs = tuple(f"#/body/{i}" for i in range(first_heading_idx))
            sections.append(
                SectionView(
                    blocks=pre_blocks,
                    block_refs=pre_refs,
                    page_range=self._compute_page_range(pre_blocks),
                )
            )

        # Process headings
        top_level = self._build_section_tree(body, heading_positions, 0, len(heading_positions))
        sections.extend(top_level)

        # Build block→section mapping for paragraph views
        self._block_to_section = {}
        self._map_blocks_to_sections(sections)

        return tuple(sections)

    def _build_section_tree(
        self,
        body: tuple[Block, ...],
        positions: list[tuple[int, int]],
        start: int,
        end: int,
    ) -> list[SectionView]:
        """Build section tree for a range of heading positions."""
        if start >= end:
            return []

        sections: list[SectionView] = []
        i = start

        while i < end:
            heading_idx, depth = positions[i]
            heading_ref = f"#/body/{heading_idx}"
            heading = body[heading_idx]
            heading_text = extract_text(heading) if isinstance(heading, Heading) else ""

            # Find where this section ends (next heading of same or shallower depth)
            child_start = i + 1
            next_same_or_higher = end

            for j in range(i + 1, end):
                _, d = positions[j]
                if d <= depth:
                    next_same_or_higher = j
                    break

            # Direct content: blocks between this heading and the first child heading
            if child_start < next_same_or_higher:
                first_child_idx = positions[child_start][0]
                direct_blocks = body[heading_idx:first_child_idx]
            else:
                # No child headings — content goes to next same/higher heading or end
                if next_same_or_higher < end:
                    content_end = positions[next_same_or_higher][0]
                else:
                    content_end = len(body)
                direct_blocks = body[heading_idx:content_end]

            direct_refs = tuple(f"#/body/{heading_idx + k}" for k in range(len(direct_blocks)))

            # Subsections: child headings (deeper depth) within this section
            subsections = self._build_section_tree(
                body, positions, child_start, next_same_or_higher
            )

            # Page range: union of this section + subsections
            page_range = self._compute_page_range(direct_blocks)
            for sub in subsections:
                if sub.page_range is not None:
                    if page_range is None:
                        page_range = sub.page_range
                    else:
                        page_range = (
                            min(page_range[0], sub.page_range[0]),
                            max(page_range[1], sub.page_range[1]),
                        )

            sections.append(
                SectionView(
                    heading_ref=heading_ref,
                    heading_text=heading_text,
                    depth=depth,
                    blocks=tuple(direct_blocks),
                    block_refs=direct_refs,
                    subsections=tuple(subsections),
                    page_range=page_range,
                )
            )

            i = next_same_or_higher

        return sections

    def _compute_page_range(
        self, blocks: tuple[Block, ...] | list[Block]
    ) -> tuple[int, int] | None:
        """Compute (min_page, max_page) from block provenance."""
        pages: list[int] = []
        for b in blocks:
            if b.provenance is not None and b.provenance.page is not None:
                pages.append(b.provenance.page)
        if not pages:
            return None
        return (min(pages), max(pages))

    def _map_blocks_to_sections(
        self, sections: tuple[SectionView, ...] | list[SectionView]
    ) -> None:
        """Populate _block_to_section mapping."""
        for sv in sections:
            for ref in sv.block_refs:
                if self._block_to_section is not None:
                    self._block_to_section[ref] = sv.heading_ref
            self._map_blocks_to_sections(sv.subsections)

    def _flatten_sections(
        self,
        sections: tuple[SectionView, ...],
        result: list[SectionView],
    ) -> None:
        for s in sections:
            result.append(s)
            self._flatten_sections(s.subsections, result)

    def _compute_section_ancestors(self) -> dict[str, tuple[str, ...]]:
        """Build ``heading_ref → (root_text, ..., own_text)`` mapping.

        Each entry's chain is root-first and INCLUDES the section's own
        heading text as the final element. Used by :meth:`block_path`.
        Preamble sections (``heading_ref is None``) are skipped — their
        contained blocks return an empty path.
        """
        out: dict[str, tuple[str, ...]] = {}

        def _walk(
            sections: tuple[SectionView, ...],
            ancestors: tuple[str, ...],
        ) -> None:
            for sec in sections:
                if sec.heading_ref is None:
                    # Preamble — recurse without recording; subsections
                    # of a preamble section keep the same ancestor chain.
                    _walk(sec.subsections, ancestors)
                    continue
                own_chain = (*ancestors, sec.heading_text or "")
                out[sec.heading_ref] = own_chain
                _walk(sec.subsections, own_chain)

        _walk(self.sections, ())
        return out

    def collect_section_blocks(self, sv: SectionView) -> list[Block]:
        """Collect all blocks from a section and its subsections."""
        blocks: list[Block] = list(sv.blocks)
        for sub in sv.subsections:
            blocks.extend(self.collect_section_blocks(sub))
        return blocks

    # ── Internal: paragraph computation ──

    def _compute_paragraphs(self) -> tuple[ParagraphView, ...]:
        """Extract paragraph views with context.

        Walks the block tree recursively so that paragraphs nested inside
        container blocks (BlockQuote, ListItem, Figure, Div, Admonition)
        are indexed for search.
        """
        # Ensure sections are computed for block→section mapping
        _ = self.sections

        result: list[ParagraphView] = []
        self._collect_paragraphs(self._document.body, "#/body", result)
        return tuple(result)

    def _collect_paragraphs(
        self,
        blocks: tuple[Block, ...],
        prefix: str,
        result: list[ParagraphView],
    ) -> None:
        """Recursively collect ParagraphViews from a block sequence."""
        for i, block in enumerate(blocks):
            ref = f"{prefix}/{i}"
            if block.node_type == "paragraph":
                text = extract_text(block)
                page = block.provenance.page if block.provenance else None
                # N6: thread Provenance.confidence (typically Tesseract
                # OCR confidence on scanned PDFs) into the paragraph view
                # so retrieval can refuse to cite low-quality passages.
                confidence = block.provenance.confidence if block.provenance else None
                section_ref = (
                    self._block_to_section.get(ref) if self._block_to_section is not None else None
                )
                result.append(
                    ParagraphView(
                        block_ref=ref,
                        text=text,
                        page=page,
                        section_ref=section_ref,
                        confidence=confidence,
                    )
                )
            # Recurse into container blocks that hold nested Block children
            # (BlockQuote, ListItem, Div, Admonition, BulletList, ...). Skip
            # Inline-bearing containers (Paragraph, Heading, Emphasis, ...);
            # their children are inlines, not paragraphs, and descending into
            # them produces nonsensical block_refs and wastes traversal time.
            children = getattr(block, "children", None)
            if isinstance(children, tuple) and children and isinstance(children[0], BaseBlock):
                self._collect_paragraphs(children, f"{ref}/children", result)

    # ── Internal: sentence computation ──

    def _compute_sentences(self) -> tuple[SentenceView, ...]:
        """Segment paragraphs into sentences using the provided segmenter."""
        if self._segmenter is None:
            return ()

        result: list[SentenceView] = []
        for pv in self.paragraphs:
            if not pv.text.strip():
                continue
            try:
                spans = self._segmenter.tokenize_spans(pv.text)
            except Exception as exc:
                logger.debug(
                    "document_view: sentence segmentation failed for paragraph %s: %s",
                    pv.block_ref,
                    exc,
                )
                continue
            for start, end in spans:
                sent_text = pv.text[start:end]
                if sent_text.strip():
                    result.append(
                        SentenceView(
                            text=sent_text,
                            start=start,
                            end=end,
                            paragraph_ref=pv.block_ref,
                            page=pv.page,
                            section_ref=pv.section_ref,
                        )
                    )
        return tuple(result)

    # ── Repr ──

    def __repr__(self) -> str:
        parts = [f"DocumentView(blocks={len(self._document.body)}"]
        if self.has_pages:
            parts.append(f"pages={self.page_count}")
        if self.has_sections:
            parts.append(f"sections={len(self.flat_sections)}")
        parts.append(f"paragraphs={len(self.paragraphs)}")
        if self.has_sentences:
            parts.append(f"sentences={len(self.sentences)}")
        return ", ".join(parts) + ")"
