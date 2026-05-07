"""Typed tree queries over a ContentDocument.

Complements ``walk`` / ``find`` / ``find_first`` in ``traversal.visitor``
with type-safe, documented helpers for the queries that readers,
writers, and transforms actually need.

Every query walks the document body AND footnotes, matching the
behavior of ``find`` / ``find_first``.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING

from kaos_content.model.annotation import Annotation, AnnotationType
from kaos_content.model.node import BaseNode
from kaos_content.traversal.visitor import walk

if TYPE_CHECKING:
    from kaos_content.model.blocks import Heading, Table
    from kaos_content.model.document import ContentDocument
    from kaos_content.model.inlines import Image, Link


# ---------------------------------------------------------------------------
# Core queries
# ---------------------------------------------------------------------------


def find_by_type[T: BaseNode](document: ContentDocument, node_type: type[T]) -> Iterator[T]:
    """Yield every AST node of a given type in ``document.body`` and footnotes.

    Examples::

        from kaos_content.model.blocks import Heading
        for h in find_by_type(doc, Heading):
            print(h.depth, extract_text(h))
    """
    for block in document.body:
        for node in walk(block):
            if isinstance(node, node_type):
                yield node
    for fn_blocks in document.footnotes.values():
        for block in fn_blocks:
            for node in walk(block):
                if isinstance(node, node_type):
                    yield node


def find_by_class(document: ContentDocument, class_name: str) -> Iterator[BaseNode]:
    """Yield every node whose ``Attr.classes`` contains ``class_name``.

    Used to find Div/Span carriers of domain semantics (``rev-ins``,
    ``speaker-notes``, custom classes). Matches any node type; callers
    should ``isinstance``-filter if they only want Divs or Spans.
    """

    def _has_class(node: BaseNode) -> bool:
        attr = getattr(node, "attr", None)
        if attr is None:
            return False
        classes = getattr(attr, "classes", ()) or ()
        return class_name in classes

    for block in document.body:
        for node in walk(block):
            if _has_class(node):
                yield node
    for fn_blocks in document.footnotes.values():
        for block in fn_blocks:
            for node in walk(block):
                if _has_class(node):
                    yield node


def find_by_kv(document: ContentDocument, key: str, value: str | None = None) -> Iterator[BaseNode]:
    """Yield nodes whose ``Attr.kv[key]`` matches ``value``.

    If ``value`` is None, yields any node that has the key at all
    regardless of value — useful for "find every node with rev:id".
    """

    def _matches(node: BaseNode) -> bool:
        attr = getattr(node, "attr", None)
        if attr is None:
            return False
        kv = getattr(attr, "kv", None) or {}
        if key not in kv:
            return False
        return value is None or kv[key] == value

    for block in document.body:
        for node in walk(block):
            if _matches(node):
                yield node
    for fn_blocks in document.footnotes.values():
        for block in fn_blocks:
            for node in walk(block):
                if _matches(node):
                    yield node


def find_annotations_of_type(
    document: ContentDocument, annotation_type: AnnotationType
) -> Iterator[Annotation]:
    """Yield every annotation of a given type from ``document.annotations``.

    Does NOT walk the tree; annotations are document-level standoff
    markup.
    """
    for ann in document.annotations:
        if ann.type == annotation_type:
            yield ann


# ---------------------------------------------------------------------------
# Domain-specific convenience helpers
# ---------------------------------------------------------------------------


def find_headings(document: ContentDocument, *, depth: int | None = None) -> Iterator[Heading]:
    """Yield every Heading, optionally filtered by depth (1-6)."""
    from kaos_content.model.blocks import Heading

    for h in find_by_type(document, Heading):
        if depth is None or h.depth == depth:
            yield h


def find_links(document: ContentDocument) -> Iterator[Link]:
    """Yield every Link inline."""
    from kaos_content.model.inlines import Link

    return find_by_type(document, Link)


def find_tables(document: ContentDocument) -> Iterator[Table]:
    """Yield every Table block."""
    from kaos_content.model.blocks import Table

    return find_by_type(document, Table)


def find_images(document: ContentDocument) -> Iterator[Image]:
    """Yield every Image inline."""
    from kaos_content.model.inlines import Image

    return find_by_type(document, Image)


def find_footnote_refs(document: ContentDocument):
    """Yield every FootnoteRef inline."""
    from kaos_content.model.inlines import FootnoteRef

    return find_by_type(document, FootnoteRef)
