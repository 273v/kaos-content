"""Tree traversal, typed queries, and node indexing."""

from kaos_content.traversal.index import NodeIndex
from kaos_content.traversal.queries import (
    find_annotations_of_type,
    find_by_class,
    find_by_kv,
    find_by_type,
    find_footnote_refs,
    find_headings,
    find_images,
    find_links,
    find_tables,
)
from kaos_content.traversal.visitor import (
    content_hash,
    extract_text,
    find,
    find_first,
    walk,
    walk_blocks,
    walk_inlines,
)

__all__ = [
    "NodeIndex",
    "content_hash",
    "extract_text",
    "find",
    "find_annotations_of_type",
    "find_by_class",
    "find_by_kv",
    "find_by_type",
    "find_first",
    "find_footnote_refs",
    "find_headings",
    "find_images",
    "find_links",
    "find_tables",
    "walk",
    "walk_blocks",
    "walk_inlines",
]
