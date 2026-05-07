"""NodeIndex: O(1) ref lookup, typed collections, annotation queries."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeVar, cast

from kaos_core.logging import get_logger

from kaos_content.model.node import BaseNode

if TYPE_CHECKING:
    from kaos_content.model.annotation import Annotation
    from kaos_content.model.blocks import (
        CodeBlock,
        Heading,
        Table,
    )
    from kaos_content.model.document import ContentDocument
    from kaos_content.model.inlines import Image, Link

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseNode)


class NodeIndex:
    """O(1) node lookup by JSON pointer ref, typed collection access.

    Built from a single depth-first traversal over a ContentDocument.
    The index is a companion object — the document itself is immutable
    and carries no index state.
    """

    def __init__(self, document: ContentDocument) -> None:
        self._document = document
        self._ref_map: dict[str, BaseNode] = {}
        self._id_map: dict[str, BaseNode] = {}
        self._by_type: dict[type, list[BaseNode]] = {}
        self._by_page: dict[int, list[BaseNode]] = {}
        self._annotation_map: dict[str, list[Annotation]] = {}

        self._build()

    def _build(self) -> None:
        """Single depth-first pass to populate all lookup structures."""
        # Index body blocks
        for i, block in enumerate(self._document.body):
            self._index_node(block, f"#/body/{i}")

        # Index footnote blocks
        for key, blocks in self._document.footnotes.items():
            for i, block in enumerate(blocks):
                self._index_node(block, f"#/footnotes/{key}/{i}")

        # Build annotation map
        for ann in self._document.annotations:
            for target in ann.targets:
                self._annotation_map.setdefault(target.node_ref, []).append(ann)

        # Validate annotations
        invalid = self.validate_annotations()
        if invalid:
            logger.warning(
                "Document has %d annotation target(s) referencing non-existent nodes: %s",
                len(invalid),
                invalid[:5],
            )

    def _index_node(self, node: BaseNode, ref: str) -> None:
        """Register a node and recursively index its children."""
        self._ref_map[ref] = node
        self._id_map[node.id] = node
        self._by_type.setdefault(type(node), []).append(node)

        if node.provenance is not None and node.provenance.page is not None:
            self._by_page.setdefault(node.provenance.page, []).append(node)

        # Dispatch to child indexing based on available fields
        self._index_children(node, ref)

    def _index_children(self, node: BaseNode, ref: str) -> None:
        """Index child nodes, building correct JSON pointer paths."""
        # "children" field — used by most block/inline container types
        children = getattr(node, "children", None)
        if children is not None:
            for i, child in enumerate(children):
                if isinstance(child, BaseNode):
                    self._index_node(child, f"{ref}/children/{i}")

        # DefinitionItem: term + definitions
        term = getattr(node, "term", None)
        if term is not None:
            for i, t in enumerate(term):
                if isinstance(t, BaseNode):
                    self._index_node(t, f"{ref}/term/{i}")

        definitions = getattr(node, "definitions", None)
        if definitions is not None:
            for i, def_blocks in enumerate(definitions):
                for j, block in enumerate(def_blocks):
                    if isinstance(block, BaseNode):
                        self._index_node(block, f"{ref}/definitions/{i}/{j}")

        # Table: caption, head, bodies, foot
        caption = getattr(node, "caption", None)
        if caption is not None:
            if caption.short is not None:
                for i, inline in enumerate(caption.short):
                    if isinstance(inline, BaseNode):
                        self._index_node(inline, f"{ref}/caption/short/{i}")
            for i, block in enumerate(caption.body):
                if isinstance(block, BaseNode):
                    self._index_node(block, f"{ref}/caption/body/{i}")

        head = getattr(node, "head", None)
        if head is not None:
            self._index_table_section(head, f"{ref}/head")

        bodies = getattr(node, "bodies", None)
        if bodies is not None:
            for i, section in enumerate(bodies):
                self._index_table_section(section, f"{ref}/bodies/{i}")

        foot = getattr(node, "foot", None)
        if foot is not None:
            self._index_table_section(foot, f"{ref}/foot")

    def _index_table_section(self, section: object, ref: str) -> None:
        """Index a TableSection and all its children (rows, cells, cell content)."""
        # Register the TableSection itself if it's a BaseNode
        if isinstance(section, BaseNode):
            self._index_node_only(section, ref)

        rows = getattr(section, "rows", ())
        for ri, row in enumerate(rows):
            row_ref = f"{ref}/rows/{ri}"
            # Register the Row itself if it's a BaseNode
            if isinstance(row, BaseNode):
                self._index_node_only(row, row_ref)

            cells = getattr(row, "cells", ())
            for ci, cell in enumerate(cells):
                cell_ref = f"{row_ref}/cells/{ci}"
                # Register the Cell itself if it's a BaseNode
                if isinstance(cell, BaseNode):
                    self._index_node_only(cell, cell_ref)

                content = getattr(cell, "content", ())
                for bi, block in enumerate(content):
                    if isinstance(block, BaseNode):
                        self._index_node(block, f"{cell_ref}/content/{bi}")

    def _index_node_only(self, node: BaseNode, ref: str) -> None:
        """Register a node without recursing into its children (used for structural nodes)."""
        self._ref_map[ref] = node
        self._id_map[node.id] = node
        self._by_type.setdefault(type(node), []).append(node)

        if node.provenance is not None and node.provenance.page is not None:
            self._by_page.setdefault(node.provenance.page, []).append(node)

    # ── Public API ──

    def get(self, node_ref: str) -> BaseNode | None:
        """Look up a node by its JSON pointer ref. Returns None if not found."""
        return self._ref_map.get(node_ref)

    def get_by_id(self, node_id: str) -> BaseNode | None:
        """Look up a node by its UUID. Returns None if not found."""
        return self._id_map.get(node_id)

    def __getitem__(self, node_ref: str) -> BaseNode:
        """Look up a node by ref. Raises KeyError if not found."""
        try:
            return self._ref_map[node_ref]
        except KeyError:
            msg = f"No node at ref {node_ref!r}"
            raise KeyError(msg) from None

    def __len__(self) -> int:
        """Total number of indexed nodes."""
        return len(self._ref_map)

    def __contains__(self, node_ref: str) -> bool:
        return node_ref in self._ref_map

    @property
    def refs(self) -> list[str]:
        """All node refs in traversal order."""
        return list(self._ref_map.keys())

    @property
    def headings(self) -> list[Heading]:
        """All Heading nodes."""
        from kaos_content.model.blocks import Heading

        return cast("list[Heading]", self._by_type.get(Heading, []))

    @property
    def tables(self) -> list[Table]:
        """All Table nodes."""
        from kaos_content.model.blocks import Table

        return cast("list[Table]", self._by_type.get(Table, []))

    @property
    def images(self) -> list[Image]:
        """All Image nodes."""
        from kaos_content.model.inlines import Image

        return cast("list[Image]", self._by_type.get(Image, []))

    @property
    def code_blocks(self) -> list[CodeBlock]:
        """All CodeBlock nodes."""
        from kaos_content.model.blocks import CodeBlock

        return cast("list[CodeBlock]", self._by_type.get(CodeBlock, []))

    @property
    def links(self) -> list[Link]:
        """All Link nodes."""
        from kaos_content.model.inlines import Link

        return cast("list[Link]", self._by_type.get(Link, []))

    def by_type(self, node_type: type[T]) -> list[T]:
        """All nodes of a specific type."""
        return cast("list[T]", self._by_type.get(node_type, []))

    def by_provenance_page(self, page: int) -> list[BaseNode]:
        """All nodes with provenance pointing to the given page number."""
        return self._by_page.get(page, [])

    def annotations_for(self, node_ref: str) -> list[Annotation]:
        """All annotations targeting the given node ref."""
        return self._annotation_map.get(node_ref, [])

    def validate_annotations(self) -> list[str]:
        """Check that all annotation targets reference existing nodes.

        Returns a list of invalid node_ref strings (empty means all valid).
        """
        invalid: list[str] = []
        for ann in self._document.annotations:
            for target in ann.targets:
                if target.node_ref not in self._ref_map:
                    invalid.append(target.node_ref)
        return invalid
