"""DocumentBuilder: fluent API for constructing ContentDocument instances."""

from __future__ import annotations

from typing import Any, Self, cast

from kaos_content.model.annotation import Annotation, AnnotationTarget, AnnotationType
from kaos_content.model.attr import (
    Alignment,
    Attr,
    BoundingBox,
    Caption,
    ColSpec,
    Provenance,
    SourceRef,
)
from kaos_content.model.blocks import (
    Admonition,
    Block,
    BlockQuote,
    BulletList,
    CodeBlock,
    DefinitionItem,
    DefinitionList,
    Div,
    Figure,
    Heading,
    ListItem,
    MathBlock,
    OrderedList,
    PageBreak,
    Paragraph,
    RawBlock,
    Table,
    ThematicBreak,
)
from kaos_content.model.document import ContentDocument
from kaos_content.model.inlines import (
    Code,
    Emphasis,
    FootnoteRef,
    Image,
    Inline,
    Link,
    Math,
    Strong,
    Text,
)
from kaos_content.model.metadata import DocumentMetadata
from kaos_content.model.table import Cell, Row, TableSection


class DocumentBuilder:
    """Fluent API for constructing ContentDocument instances.

    The builder accumulates blocks into a mutable list, then freezes
    everything into an immutable ContentDocument on ``build()``.

    Nesting is managed via a stack: ``begin_blockquote()`` pushes a new
    accumulator, ``end()`` pops it and wraps the accumulated blocks into
    the appropriate container node.

    Example::

        doc = (
            DocumentBuilder(title="Contract")
            .heading(1, "Article I")
            .paragraph("This is the first paragraph.")
            .begin_list(ordered=True)
            .begin_list_item()
            .paragraph("First item")
            .end()
            .begin_list_item()
            .paragraph("Second item")
            .end()
            .end()
            .build()
        )
    """

    def __init__(self, title: str | None = None) -> None:
        self._metadata_kwargs: dict[str, Any] = {}
        if title is not None:
            self._metadata_kwargs["title"] = title
        self._source: SourceRef | None = None
        self._blocks: list[Block] = []
        self._stack: list[_StackFrame] = []
        self._last_node: Block | Inline | None = None
        self._footnotes: dict[str, list[Block]] = {}
        self._definitions: dict[str, str] = {}
        self._annotations: list[Annotation] = []
        self._headers: dict[str, tuple[Block, ...]] = {}
        self._footers: dict[str, tuple[Block, ...]] = {}
        self._sections: list[Any] = []  # list[Section] — avoid import cycle

    # ── Metadata ──

    def set_metadata(self, **kwargs: Any) -> Self:
        """Set document metadata fields (title, authors, etc.)."""
        self._metadata_kwargs.update(kwargs)
        return self

    def set_source(self, uri: str, mime_type: str | None = None) -> Self:
        """Set the source reference for provenance."""
        self._source = SourceRef(uri=uri, mime_type=mime_type)
        return self

    def set_header(self, kind: str, *blocks: Block) -> Self:
        """Attach a page-header block sequence. ``kind`` is conventionally
        ``"default"``, ``"first"`` (title page), or ``"even"``.
        """
        self._headers[kind] = tuple(blocks)
        return self

    def set_footer(self, kind: str, *blocks: Block) -> Self:
        """Attach a page-footer block sequence. Same keys as ``set_header``."""
        self._footers[kind] = tuple(blocks)
        return self

    def set_sections(self, sections: tuple[Any, ...] | list[Any]) -> Self:
        """Attach page-layout sections for multi-section documents.

        See :class:`kaos_content.model.metadata.Section` for the invariants
        the tuple must satisfy (contiguous, ``end_block_index`` exclusive,
        last section ends at ``len(body)``). Called by readers that
        enumerate per-region page properties (e.g. DOCX ``<w:sectPr>``).
        """
        self._sections = list(sections)
        return self

    # ── Block construction (flat) ──

    def heading(self, depth: int, text: str, **attr_kv: str) -> Self:
        """Add a heading block."""
        attr = Attr(kv=attr_kv) if attr_kv else Attr()
        node = Heading(depth=depth, children=(Text(value=text),), attr=attr)
        self._append_block(node)
        return self

    def paragraph(self, *inlines: Inline | str) -> Self:
        """Add a paragraph. String arguments are auto-wrapped as Text nodes."""
        children = tuple(_coerce_inline(i) for i in inlines)
        node = Paragraph(children=children)
        self._append_block(node)
        return self

    def blockquote(self, *blocks: Block) -> Self:
        """Add a blockquote with pre-built blocks."""
        node = BlockQuote(children=blocks)
        self._append_block(node)
        return self

    def code_block(self, code: str, language: str | None = None) -> Self:
        """Add a fenced code block."""
        node = CodeBlock(value=code, language=language)
        self._append_block(node)
        return self

    def math_block(self, value: str) -> Self:
        """Add a display math block."""
        node = MathBlock(value=value)
        self._append_block(node)
        return self

    def table(
        self,
        headers: list[str],
        rows: list[list[str]],
        *,
        alignments: list[Alignment | None] | None = None,
    ) -> Self:
        """Add a simple table from string data.

        For complex tables with block content, cell spans, or provenance,
        construct Table nodes directly and use ``add_block()``.
        """
        col_specs: tuple[ColSpec, ...] = ()
        if alignments:
            col_specs = tuple(ColSpec(alignment=a) for a in alignments)

        header_cells = tuple(Cell(content=(Paragraph(children=(Text(value=h),)),)) for h in headers)
        head = TableSection(rows=(Row(cells=header_cells),))

        body_rows = tuple(
            Row(cells=tuple(Cell(content=(Paragraph(children=(Text(value=c),)),)) for c in row))
            for row in rows
        )
        body = TableSection(rows=body_rows)

        node = Table(head=head, bodies=(body,), col_specs=col_specs)
        self._append_block(node)
        return self

    def thematic_break(self) -> Self:
        """Add a horizontal rule."""
        self._append_block(ThematicBreak())
        return self

    def page_break(self) -> Self:
        """Add a page break marker."""
        self._append_block(PageBreak())
        return self

    def image(
        self,
        src: str,
        alt: str | None = None,
        title: str | None = None,
        width: float | None = None,
        height: float | None = None,
    ) -> Self:
        """Add an image as a figure with optional dimensions.

        ``width`` and ``height`` follow the kaos-content convention: points
        (1/72 inch) for office/print content. See ``Image.width`` docstring
        for details.
        """
        img = Image(src=src, alt=alt or "", title=title, width=width, height=height)
        node = Figure(children=(Paragraph(children=(img,)),))
        self._append_block(node)
        return self

    def raw_block(self, value: str, fmt: str = "html") -> Self:
        """Add a raw block (HTML, LaTeX, etc.)."""
        self._append_block(RawBlock(format=fmt, value=value))
        return self

    def admonition(self, kind: str, *blocks: Block, title: str | None = None) -> Self:
        """Add an admonition/callout with pre-built blocks."""
        node = Admonition(kind=kind, title=title, children=blocks)
        self._append_block(node)
        return self

    def definition_list(self, *items: tuple[str, str]) -> Self:
        """Add a definition list from (term, definition) pairs."""
        di_nodes = tuple(
            DefinitionItem(
                term=(Text(value=term),),
                definitions=((Paragraph(children=(Text(value=defn),)),),),
            )
            for term, defn in items
        )
        self._append_block(DefinitionList(children=di_nodes))
        return self

    def add_block(self, block: Block) -> Self:
        """Add a pre-built block node directly."""
        self._append_block(block)
        return self

    # ── Nested block construction ──

    def begin_blockquote(self) -> Self:
        """Start a blockquote. Call ``end()`` to close it."""
        self._stack.append(_StackFrame("blockquote"))
        return self

    def begin_list(self, ordered: bool = False, start: int = 1) -> Self:
        """Start a list. Call ``end()`` to close it."""
        kind = "ordered_list" if ordered else "bullet_list"
        self._stack.append(_StackFrame(kind, start=start))
        return self

    def begin_list_item(self, checked: bool | None = None) -> Self:
        """Start a list item. Call ``end()`` to close it."""
        self._stack.append(_StackFrame("list_item", checked=checked))
        return self

    def begin_div(
        self,
        *,
        classes: tuple[str, ...] | str = (),
        kv: dict[str, str] | None = None,
        **attr_kv: str,
    ) -> Self:
        """Start a generic div container. Call ``end()`` to close it.

        Args:
            classes: Attr.classes to set on the emitted Div. Accepts a
                tuple of class names or a single class name as a string
                (convenience shorthand).
            kv: Explicit kv dict (merged with ``attr_kv`` kwargs). Use
                this when kv keys contain characters that cannot appear
                in Python identifiers (e.g. ``"rev:id"``).
            **attr_kv: Additional kv entries as keyword arguments.
        """
        if isinstance(classes, str):
            normalized_classes: tuple[str, ...] = (classes,) if classes else ()
        else:
            normalized_classes = tuple(classes)
        merged_kv: dict[str, str] = dict(kv) if kv else {}
        merged_kv.update(attr_kv)
        self._stack.append(_StackFrame("div", attr_kv=merged_kv, attr_classes=normalized_classes))
        return self

    def begin_figure(self, caption_text: str | None = None) -> Self:
        """Start a figure. Call ``end()`` to close it."""
        self._stack.append(_StackFrame("figure", caption_text=caption_text))
        return self

    def end(self) -> Self:
        """Close the current nesting level, wrapping accumulated blocks."""
        if not self._stack:
            msg = "end() called without matching begin_*()"
            raise ValueError(msg)

        frame = self._stack.pop()
        children = tuple(frame.blocks)
        node: Block

        if frame.kind == "blockquote":
            node = BlockQuote(children=children)
        elif frame.kind == "ordered_list":
            node = OrderedList(children=cast("tuple[ListItem, ...]", children), start=frame.start)
        elif frame.kind == "bullet_list":
            node = BulletList(children=cast("tuple[ListItem, ...]", children))
        elif frame.kind == "list_item":
            node = ListItem(children=children, checked=frame.checked)
        elif frame.kind == "div":
            if frame.attr_kv or frame.attr_classes:
                attr = Attr(classes=frame.attr_classes, kv=frame.attr_kv)
            else:
                attr = Attr()
            node = Div(children=children, attr=attr)
        elif frame.kind == "figure":
            caption = None
            if frame.caption_text is not None:
                caption = Caption(body=(Paragraph(children=(Text(value=frame.caption_text),)),))
            node = Figure(children=children, caption=caption)
        else:
            msg = f"Unknown stack frame kind: {frame.kind}"
            raise ValueError(msg)

        self._append_block(node)
        return self

    # ── Inline helpers (return nodes, not Self) ──

    @staticmethod
    def text(value: str) -> Text:
        """Create a Text inline node."""
        return Text(value=value)

    @staticmethod
    def bold(text: str) -> Strong:
        """Create a Strong (bold) inline node."""
        return Strong(children=(Text(value=text),))

    @staticmethod
    def italic(text: str) -> Emphasis:
        """Create an Emphasis (italic) inline node."""
        return Emphasis(children=(Text(value=text),))

    @staticmethod
    def link(text: str, url: str, title: str | None = None) -> Link:
        """Create a Link inline node."""
        return Link(url=url, title=title, children=(Text(value=text),))

    @staticmethod
    def code(value: str) -> Code:
        """Create a Code inline node."""
        return Code(value=value)

    @staticmethod
    def math(value: str) -> Math:
        """Create a Math inline node."""
        return Math(value=value)

    @staticmethod
    def footnote_ref(identifier: str) -> FootnoteRef:
        """Create a footnote reference inline node."""
        return FootnoteRef(identifier=identifier)

    # ── Provenance ──

    def with_provenance(
        self,
        *,
        page: int | None = None,
        bbox: BoundingBox | None = None,
        char_span: tuple[int, int] | None = None,
        confidence: float | None = None,
        extractor: str | None = None,
    ) -> Self:
        """Attach provenance to the most recently added block.

        Must be called immediately after a block-adding method.
        Creates a new node with provenance (since nodes are frozen).
        """
        if self._last_node is None:
            msg = "with_provenance() called before adding any block"
            raise ValueError(msg)

        prov = Provenance(
            source=self._source,
            page=page,
            bbox=bbox,
            char_span=char_span,
            confidence=confidence,
            extractor=extractor,
        )

        # Replace the last block with a copy that has provenance
        target = self._current_blocks()
        if target and target[-1] is self._last_node:
            # Reconstruct with provenance using model_copy
            updated = self._last_node.model_copy(update={"provenance": prov})
            target[-1] = updated
            self._last_node = updated

        return self

    # ── Footnotes ──

    def add_footnote(self, identifier: str, *blocks: Block | str) -> Self:
        """Add a footnote definition."""
        fn_blocks = [
            Paragraph(children=(Text(value=b),)) if isinstance(b, str) else b for b in blocks
        ]
        self._footnotes[identifier] = fn_blocks
        return self

    # ── Definitions ──

    def add_definition(self, key: str, url: str) -> Self:
        """Add a link/reference definition."""
        self._definitions[key] = url
        return self

    # ── Annotations ──

    def annotate(
        self,
        annotation_type: AnnotationType,
        targets: list[AnnotationTarget],
        *,
        annotation_id: str | None = None,
        body: dict[str, Any] | None = None,
    ) -> Self:
        """Add an annotation to the document."""
        from kaos_content.model.node import _generate_node_id

        ann = Annotation(
            id=annotation_id or _generate_node_id(),
            type=annotation_type,
            targets=tuple(targets),
            body=body or {},
        )
        self._annotations.append(ann)
        return self

    # ── Build ──

    def build(self) -> ContentDocument:
        """Finalize and return the immutable ContentDocument.

        Raises ValueError if there are unclosed nesting levels.
        """
        if self._stack:
            kinds = [f.kind for f in self._stack]
            msg = f"Unclosed nesting levels: {kinds}"
            raise ValueError(msg)

        metadata = DocumentMetadata(**self._metadata_kwargs)

        # Convert mutable lists to tuples for the frozen model
        footnotes = {k: tuple(v) for k, v in self._footnotes.items()}

        return ContentDocument(
            metadata=metadata,
            body=tuple(self._blocks),
            footnotes=footnotes,
            definitions=self._definitions,
            annotations=tuple(self._annotations),
            headers=dict(self._headers),
            footers=dict(self._footers),
            sections=tuple(self._sections),
        )

    # ── Internal helpers ──

    def _current_blocks(self) -> list[Block]:
        """Return the block list for the current nesting level."""
        if self._stack:
            return self._stack[-1].blocks
        return self._blocks

    def _append_block(self, block: Block) -> None:
        """Append a block to the current nesting level."""
        self._current_blocks().append(block)
        self._last_node = block


class _StackFrame:
    """Internal state for a nesting level in the builder."""

    __slots__ = (
        "attr_classes",
        "attr_kv",
        "blocks",
        "caption_text",
        "checked",
        "kind",
        "start",
    )

    def __init__(
        self,
        kind: str,
        *,
        start: int = 1,
        checked: bool | None = None,
        attr_kv: dict[str, str] | None = None,
        attr_classes: tuple[str, ...] = (),
        caption_text: str | None = None,
    ) -> None:
        self.kind = kind
        self.blocks: list[Block] = []
        self.start = start
        self.checked = checked
        self.attr_kv = attr_kv or {}
        self.attr_classes = attr_classes
        self.caption_text = caption_text


def _coerce_inline(value: Inline | str) -> Inline:
    """Convert a string to a Text node, pass Inline nodes through."""
    if isinstance(value, str):
        return Text(value=value)
    return value
