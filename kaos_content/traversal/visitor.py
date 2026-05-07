"""Depth-first tree traversal, text extraction, and content hashing."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

from kaos_content.model.node import BaseBlock, BaseInline, BaseNode

if TYPE_CHECKING:
    from kaos_content.model.document import ContentDocument


def _iter_children(node: BaseNode) -> Iterator[BaseNode]:
    """Yield direct child nodes (BaseNode instances only) from any AST node."""
    # Most block/inline types use "children"
    children = getattr(node, "children", None)
    if children is not None:
        yield from children
        # For Figure/Table with caption, also descend into caption after children
        # (handled below)

    # DefinitionItem: term (list[Inline]) + definitions (list[list[Block]])
    term = getattr(node, "term", None)
    if term is not None:
        yield from term
    definitions = getattr(node, "definitions", None)
    if definitions is not None:
        for def_blocks in definitions:
            yield from def_blocks

    # Table structure: caption, head, bodies, foot
    caption = getattr(node, "caption", None)
    if caption is not None:
        if caption.short is not None:
            yield from caption.short
        yield from caption.body

    head = getattr(node, "head", None)
    if head is not None and isinstance(head, BaseNode):
        yield head

    bodies = getattr(node, "bodies", None)
    if bodies is not None:
        for section in bodies:
            if isinstance(section, BaseNode):
                yield section

    foot = getattr(node, "foot", None)
    if foot is not None and isinstance(foot, BaseNode):
        yield foot

    # TableSection.rows → Row nodes
    rows = getattr(node, "rows", None)
    if rows is not None and not hasattr(node, "children"):
        for row in rows:
            if isinstance(row, BaseNode):
                yield row

    # Row.cells → Cell nodes
    cells = getattr(node, "cells", None)
    if cells is not None:
        for cell in cells:
            if isinstance(cell, BaseNode):
                yield cell

    # Cell.content → Block nodes
    content = getattr(node, "content", None)
    if content is not None:
        for block in content:
            if isinstance(block, BaseNode):
                yield block


def walk(node: BaseNode) -> Iterator[BaseNode]:
    """Depth-first iteration over all nodes in the subtree (including the root)."""
    yield node
    for child in _iter_children(node):
        yield from walk(child)


def walk_blocks(document: ContentDocument) -> Iterator[BaseBlock]:
    """Iterate over all block nodes in the document body and footnotes."""
    for block in document.body:
        for node in walk(block):
            if isinstance(node, BaseBlock):
                yield node
    for fn_blocks in document.footnotes.values():
        for block in fn_blocks:
            for node in walk(block):
                if isinstance(node, BaseBlock):
                    yield node


def walk_inlines(document: ContentDocument) -> Iterator[BaseInline]:
    """Iterate over all inline nodes in the document body and footnotes."""
    for block in document.body:
        for node in walk(block):
            if isinstance(node, BaseInline):
                yield node
    for fn_blocks in document.footnotes.values():
        for block in fn_blocks:
            for node in walk(block):
                if isinstance(node, BaseInline):
                    yield node


def find(document: ContentDocument, predicate: Callable[[BaseNode], bool]) -> list[BaseNode]:
    """Find all nodes in the document matching a predicate."""
    results: list[BaseNode] = []
    for block in document.body:
        for node in walk(block):
            if predicate(node):
                results.append(node)
    for fn_blocks in document.footnotes.values():
        for block in fn_blocks:
            for node in walk(block):
                if predicate(node):
                    results.append(node)
    return results


def find_first(document: ContentDocument, predicate: Callable[[BaseNode], bool]) -> BaseNode | None:
    """Find the first node in the document matching a predicate, or None."""
    for block in document.body:
        for node in walk(block):
            if predicate(node):
                return node
    for fn_blocks in document.footnotes.values():
        for block in fn_blocks:
            for node in walk(block):
                if predicate(node):
                    return node
    return None


def extract_text(node: BaseNode) -> str:
    """Recursively extract all text content from a node subtree.

    Returns concatenated text from all Text and leaf-value nodes.
    SoftBreak → space, LineBreak → newline.
    """
    from kaos_content.model.inlines import LineBreak, SoftBreak

    parts: list[str] = []
    for descendant in walk(node):
        value = getattr(descendant, "value", None)
        if isinstance(value, str):
            parts.append(value)
        elif isinstance(descendant, SoftBreak):
            parts.append(" ")
        elif isinstance(descendant, LineBreak):
            parts.append("\n")
    return "".join(parts)


def content_hash(node: BaseNode) -> str:
    """Compute a deterministic SHA-256 hash of a node's content.

    Walks the subtree depth-first, hashing node_type and text values.
    The hash is content-addressable: same content produces the same hash,
    regardless of node id or provenance. Useful for dedup and change detection.

    Does NOT include: node id, attr, provenance (these are identity/metadata,
    not content). DOES include: node_type, value, depth, language, format,
    src, alt, url, title, identifier, kind, checked, row_span, col_span,
    start, alignment — all fields that affect the semantic content.
    """
    h = hashlib.sha256()
    for descendant in walk(node):
        node_type = getattr(descendant, "node_type", "")
        h.update(node_type.encode("utf-8"))
        # Hash leaf values
        for field in (
            "value",
            "depth",
            "language",
            "format",
            "src",
            "alt",
            "url",
            "title",
            "identifier",
            "kind",
            "start",
        ):
            val = getattr(descendant, field, None)
            if val is not None:
                h.update(f"{field}={val}".encode())
        # Hash structural attributes that affect content
        checked = getattr(descendant, "checked", None)
        if checked is not None:
            h.update(f"checked={checked}".encode())
        row_span = getattr(descendant, "row_span", None)
        if row_span is not None and row_span != 1:
            h.update(f"row_span={row_span}".encode())
        col_span = getattr(descendant, "col_span", None)
        if col_span is not None and col_span != 1:
            h.update(f"col_span={col_span}".encode())
        alignment = getattr(descendant, "alignment", None)
        if alignment is not None:
            h.update(f"alignment={alignment}".encode())
    return h.hexdigest()
