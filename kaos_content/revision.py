"""Typed API and transforms over tracked-change Span/Div nodes.

The DOCX reader's ``track_changes=True`` mode wraps revised content in
``Span`` / ``Div`` nodes carrying one of the ``rev-*`` classes
(``rev-ins``, ``rev-del``, ``rev-move-from``, ``rev-move-to``) and
metadata in ``Attr.kv`` (``rev:id``, ``rev:author``, ``rev:date``,
``rev:move-name``). This module provides:

1. Ergonomic, type-safe views (``Revision`` / ``Revisions``) so callers
   don't grovel through Attr strings.
2. Immutable tree transforms (``accept``, ``reject``, ``accept_all``,
   ``reject_all``, ``accept_by_author``, ``reject_by_author``,
   ``at_time``) that return new ContentDocuments with revisions
   resolved.

Transform semantics (per rev-* class):

=============== ============ ============
rev_class       accept       reject
=============== ============ ============
rev-ins         unwrap       drop
rev-del         drop         unwrap
rev-move-to     unwrap       drop
rev-move-from   drop         unwrap
=============== ============ ============

``unwrap`` keeps the wrapped content (splices children into parent).
``drop`` removes the whole node including children.

Typical usage::

    from datetime import datetime, UTC
    from kaos_content.revision import Revisions, accept_all, at_time

    doc = parse_docx("contract.docx", track_changes=True)

    # UC1: read/review
    revs = Revisions.from_document(doc)
    for rev in revs.sorted_by_date():
        print(f"{rev.date} {rev.author}: {rev.change_type} -> {rev.preview}")

    # UC2: time machine — snapshot at a point in time
    snapshot = at_time(doc, datetime(2026, 4, 16, tzinfo=UTC))

    # UC3: accept all of one author's changes
    cleaned = accept_by_author(doc, "Jane Smith")

See ``kaos-office/docs/TRACKED_CHANGES_DESIGN.md`` for the full design.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from kaos_content.model.document import ContentDocument


class RevisionType(StrEnum):
    """Discriminator for revision change types."""

    INSERTION = "insertion"
    DELETION = "deletion"
    MOVE_FROM = "move_from"
    MOVE_TO = "move_to"


# Map rev-* class names → RevisionType.
_CLASS_TO_TYPE: dict[str, RevisionType] = {
    "rev-ins": RevisionType.INSERTION,
    "rev-del": RevisionType.DELETION,
    "rev-move-from": RevisionType.MOVE_FROM,
    "rev-move-to": RevisionType.MOVE_TO,
}


@dataclass(frozen=True, slots=True)
class Revision:
    """Typed view over a single rev-* Span or Div node.

    Fields are populated from ``Attr.kv`` on the underlying node. Access
    the original AST via ``node``.
    """

    node: Any  # Span | Div (avoiding circular import at dataclass level)
    node_ref: str
    id: str
    author: str
    date: datetime | None
    change_type: RevisionType
    move_name: str | None = None

    @property
    def is_block(self) -> bool:
        """Whether this revision wraps a block (Div) vs an inline (Span)."""
        return type(self.node).__name__ == "Div"

    @property
    def text(self) -> str:
        """Extract the plain-text content of this revision."""
        from kaos_content.traversal.visitor import extract_text

        return extract_text(self.node)

    @property
    def preview(self) -> str:
        """Short preview of the text (up to 60 chars)."""
        text = self.text
        if len(text) <= 60:
            return text
        return text[:57] + "..."


@dataclass(frozen=True, slots=True)
class Revisions:
    """Typed, queryable collection of Revision objects.

    Construct with :meth:`from_document` to walk a ContentDocument and
    extract every rev-* Span and Div.
    """

    items: tuple[Revision, ...] = ()
    by_id_index: dict[str, Revision] = field(default_factory=dict)

    # ── Constructors ────────────────────────────────────────────────

    @classmethod
    def from_document(cls, document: ContentDocument) -> Revisions:
        """Walk the document and collect every rev-* wrapper into a Revisions."""
        items: list[Revision] = []
        for i, block in enumerate(document.body):
            _collect(block, f"#/body/{i}", items)
        by_id = {r.id: r for r in items if r.id}
        return cls(items=tuple(items), by_id_index=by_id)

    # ── Queries ─────────────────────────────────────────────────────

    def by_author(self, author: str) -> list[Revision]:
        """All revisions by a given author."""
        return [r for r in self.items if r.author == author]

    def by_type(self, change_type: RevisionType) -> list[Revision]:
        """All revisions with the given change_type."""
        return [r for r in self.items if r.change_type == change_type]

    def by_id(self, revision_id: str) -> Revision | None:
        """Look up a revision by its (DOCX-assigned) id."""
        return self.by_id_index.get(revision_id)

    def authors(self) -> list[str]:
        """Distinct author names in insertion order."""
        seen: set[str] = set()
        out: list[str] = []
        for r in self.items:
            if r.author and r.author not in seen:
                seen.add(r.author)
                out.append(r.author)
        return out

    def between(self, start: datetime | None = None, end: datetime | None = None) -> list[Revision]:
        """All revisions whose date falls in [start, end] (inclusive)."""
        out: list[Revision] = []
        for r in self.items:
            if r.date is None:
                continue
            if start is not None and r.date < start:
                continue
            if end is not None and r.date > end:
                continue
            out.append(r)
        return out

    def sorted_by_date(self) -> list[Revision]:
        """All dated revisions, ascending by date. Undated revisions are excluded."""
        return sorted(
            (r for r in self.items if r.date is not None),
            key=lambda r: r.date,
        )

    def summary(self) -> dict[str, dict[str, int]]:
        """Counts by (author, change_type). Useful for dashboards."""
        result: dict[str, dict[str, int]] = {}
        for r in self.items:
            bucket = result.setdefault(r.author or "<unknown>", {})
            bucket[r.change_type.value] = bucket.get(r.change_type.value, 0) + 1
        return result

    # ── Python protocols ────────────────────────────────────────────

    def __iter__(self) -> Iterator[Revision]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __bool__(self) -> bool:
        return bool(self.items)


# ── Internal walk + construction ────────────────────────────────────


def _node_revision_class(node: Any) -> str | None:
    """Return the first ``rev-*`` class on a node, or None."""
    attr = getattr(node, "attr", None)
    if attr is None:
        return None
    for cls in getattr(attr, "classes", ()) or ():
        if cls in _CLASS_TO_TYPE:
            return cls
    return None


def _parse_date(raw: str | None) -> datetime | None:
    """Parse an ISO-8601 date-time string (possibly with trailing 'Z')."""
    if not raw:
        return None
    # Normalize trailing Z → +00:00 for fromisoformat
    normalized = raw.rstrip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _make_revision(node: Any, node_ref: str, rev_class: str) -> Revision:
    """Build a Revision dataclass from a rev-* node."""
    kv = dict(getattr(node.attr, "kv", {}) or {})
    return Revision(
        node=node,
        node_ref=node_ref,
        id=kv.get("rev:id", ""),
        author=kv.get("rev:author", ""),
        date=_parse_date(kv.get("rev:date")),
        change_type=_CLASS_TO_TYPE[rev_class],
        move_name=kv.get("rev:move-name"),
    )


def _collect(node: Any, ref: str, out: list[Revision]) -> None:
    """Walk the AST and append a Revision for each rev-* wrapper.

    A rev-* node's children are NOT recursively scanned for further
    revisions — revisions never nest inside one another in OOXML.
    """
    rev_class = _node_revision_class(node)
    if rev_class is not None:
        out.append(_make_revision(node, ref, rev_class))
        return

    children = getattr(node, "children", None)
    content = getattr(node, "content", None)
    if children:
        for i, c in enumerate(children):
            _collect(c, f"{ref}/children/{i}", out)
    if content:
        for i, c in enumerate(content):
            _collect(c, f"{ref}/content/{i}", out)


# ----------------------------------------------------------------------------
# Transforms: accept / reject / at_time
# ----------------------------------------------------------------------------


# action_for_class table: (rev_class, accepting) -> "unwrap" | "drop"
_ACTION: dict[tuple[str, bool], str] = {
    ("rev-ins", True): "unwrap",
    ("rev-del", True): "drop",
    ("rev-move-to", True): "unwrap",
    ("rev-move-from", True): "drop",
    ("rev-ins", False): "drop",
    ("rev-del", False): "unwrap",
    ("rev-move-to", False): "drop",
    ("rev-move-from", False): "unwrap",
}


def _rev_id(node: Any) -> str | None:
    """Extract rev:id from a rev-* node's Attr.kv, or None."""
    attr = getattr(node, "attr", None)
    if attr is None:
        return None
    kv = getattr(attr, "kv", None) or {}
    return kv.get("rev:id")


def _apply(
    doc: ContentDocument,
    *,
    accept_ids: set[str],
    reject_ids: set[str],
) -> ContentDocument:
    """Core transform: apply accept/reject to every matching rev-* node.

    Nodes not in either set are left unchanged. Returns a new
    ``ContentDocument`` with matching TRACKED_CHANGE annotations also
    removed.
    """
    from kaos_content.model.annotation import AnnotationType

    new_body = _transform_block_tuple(doc.body, accept_ids=accept_ids, reject_ids=reject_ids)

    # Filter annotations: drop TRACKED_CHANGE entries whose revision_id we processed
    processed = accept_ids | reject_ids
    new_annotations = tuple(
        a
        for a in doc.annotations
        if a.type != AnnotationType.TRACKED_CHANGE or a.body.get("revision_id") not in processed
    )

    return doc.model_copy(update={"body": new_body, "annotations": new_annotations})


def _transform_block_tuple(
    blocks: tuple[Any, ...], *, accept_ids: set[str], reject_ids: set[str]
) -> tuple[Any, ...]:
    """Apply block-level transforms, returning a new tuple."""
    out: list[Any] = []
    for block in blocks:
        out.extend(_transform_block(block, accept_ids=accept_ids, reject_ids=reject_ids))
    return tuple(out)


def _transform_block(block: Any, *, accept_ids: set[str], reject_ids: set[str]) -> list[Any]:
    """Transform a single block. Returns 0, 1, or many blocks.

    If the block is a rev-* Div matching accept_ids or reject_ids, apply
    the resolved action (unwrap → recurse and return children; drop → []).
    Otherwise recurse into its children if any.
    """
    rev_class = _node_revision_class(block)
    rev_id = _rev_id(block)
    if rev_class and rev_id:
        if rev_id in accept_ids:
            action = _ACTION[(rev_class, True)]
        elif rev_id in reject_ids:
            action = _ACTION[(rev_class, False)]
        else:
            action = "keep"
        if action == "drop":
            return []
        if action == "unwrap":
            # Splice the Div's children into the parent, recursively transformed
            return list(
                _transform_block_tuple(
                    tuple(getattr(block, "children", ()) or ()),
                    accept_ids=accept_ids,
                    reject_ids=reject_ids,
                )
            )

    # Not a matched revision (or action=keep): recurse into children
    return [_recurse_block(block, accept_ids=accept_ids, reject_ids=reject_ids)]


def _recurse_block(block: Any, *, accept_ids: set[str], reject_ids: set[str]) -> Any:
    """Apply transforms to any revision wrappers nested inside ``block``."""
    updates: dict[str, Any] = {}

    # Block with block children (BlockQuote, Div, ListItem, Figure, Admonition, ...)
    children = getattr(block, "children", None)
    if children and _children_are_blocks(block):
        new_children = _transform_block_tuple(
            tuple(children), accept_ids=accept_ids, reject_ids=reject_ids
        )
        if new_children != tuple(children):
            updates["children"] = new_children
    # Block with inline children (Paragraph, Heading)
    elif children and _children_are_inlines(block):
        new_children = _transform_inline_tuple(
            tuple(children), accept_ids=accept_ids, reject_ids=reject_ids
        )
        if new_children != tuple(children):
            updates["children"] = new_children

    # List variants (BulletList, OrderedList) — children are ListItem
    # which already holds blocks; handled by the "block children" branch above.

    # Table: head / bodies / foot — sections with rows, rows have cells, cells have content (blocks)
    if type(block).__name__ == "Table":
        updates.update(_transform_table(block, accept_ids=accept_ids, reject_ids=reject_ids))

    if updates:
        return block.model_copy(update=updates)
    return block


def _children_are_blocks(block: Any) -> bool:
    """Heuristic: does this block's ``children`` contain Block nodes?"""
    nt = getattr(block, "node_type", None)
    return nt in {
        "blockquote",
        "div",
        "list_item",
        "bullet_list",
        "ordered_list",
        "figure",
        "admonition",
        "definition_list",
    }


def _children_are_inlines(block: Any) -> bool:
    """Heuristic: does this block's ``children`` contain Inline nodes?"""
    nt = getattr(block, "node_type", None)
    return nt in {"paragraph", "heading"}


def _transform_table(table: Any, *, accept_ids: set[str], reject_ids: set[str]) -> dict[str, Any]:
    """Recurse into a Table's head/bodies/foot to transform cell contents."""
    updates: dict[str, Any] = {}

    def _transform_section(section: Any) -> Any:
        if section is None:
            return None
        new_rows = tuple(
            _transform_row(row, accept_ids=accept_ids, reject_ids=reject_ids)
            for row in section.rows
        )
        if new_rows != section.rows:
            return section.model_copy(update={"rows": new_rows})
        return section

    head = getattr(table, "head", None)
    if head is not None:
        new_head = _transform_section(head)
        if new_head is not head:
            updates["head"] = new_head

    bodies = getattr(table, "bodies", ())
    if bodies:
        new_bodies = tuple(_transform_section(s) for s in bodies)
        if any(a is not b for a, b in zip(new_bodies, bodies, strict=True)):
            updates["bodies"] = new_bodies

    foot = getattr(table, "foot", None)
    if foot is not None:
        new_foot = _transform_section(foot)
        if new_foot is not foot:
            updates["foot"] = new_foot

    return updates


def _transform_row(row: Any, *, accept_ids: set[str], reject_ids: set[str]) -> Any:
    """Transform cells within a Row (Cell has .content: tuple[Block, ...])."""
    new_cells = tuple(
        _transform_cell(cell, accept_ids=accept_ids, reject_ids=reject_ids) for cell in row.cells
    )
    if new_cells != row.cells:
        return row.model_copy(update={"cells": new_cells})
    return row


def _transform_cell(cell: Any, *, accept_ids: set[str], reject_ids: set[str]) -> Any:
    """Transform the ``content`` tuple on a Cell."""
    new_content = _transform_block_tuple(
        tuple(cell.content or ()), accept_ids=accept_ids, reject_ids=reject_ids
    )
    if new_content != cell.content:
        return cell.model_copy(update={"content": new_content})
    return cell


def _transform_inline_tuple(
    inlines: tuple[Any, ...], *, accept_ids: set[str], reject_ids: set[str]
) -> tuple[Any, ...]:
    """Apply inline-level transforms, returning a new tuple."""
    out: list[Any] = []
    for inline in inlines:
        out.extend(_transform_inline(inline, accept_ids=accept_ids, reject_ids=reject_ids))
    return tuple(out)


def _transform_inline(inline: Any, *, accept_ids: set[str], reject_ids: set[str]) -> list[Any]:
    """Transform a single inline. Returns 0, 1, or many inlines."""
    rev_class = _node_revision_class(inline)
    rev_id = _rev_id(inline)
    if rev_class and rev_id:
        if rev_id in accept_ids:
            action = _ACTION[(rev_class, True)]
        elif rev_id in reject_ids:
            action = _ACTION[(rev_class, False)]
        else:
            action = "keep"
        if action == "drop":
            return []
        if action == "unwrap":
            return list(
                _transform_inline_tuple(
                    tuple(getattr(inline, "children", ()) or ()),
                    accept_ids=accept_ids,
                    reject_ids=reject_ids,
                )
            )

    # Not a matched revision: recurse into children if any
    children = getattr(inline, "children", None)
    if children:
        new_children = _transform_inline_tuple(
            tuple(children), accept_ids=accept_ids, reject_ids=reject_ids
        )
        if new_children != tuple(children):
            return [inline.model_copy(update={"children": new_children})]
    return [inline]


# ── Public transform API ────────────────────────────────────────────


def accept(doc: ContentDocument, rev_id: str) -> ContentDocument:
    """Accept a single revision by ID. Returns a new ContentDocument."""
    return _apply(doc, accept_ids={rev_id}, reject_ids=set())


def reject(doc: ContentDocument, rev_id: str) -> ContentDocument:
    """Reject a single revision by ID. Returns a new ContentDocument."""
    return _apply(doc, accept_ids=set(), reject_ids={rev_id})


def accept_all(doc: ContentDocument) -> ContentDocument:
    """Accept every tracked change. Equivalent to the ``final`` view."""
    revs = Revisions.from_document(doc)
    return _apply(doc, accept_ids={r.id for r in revs if r.id}, reject_ids=set())


def reject_all(doc: ContentDocument) -> ContentDocument:
    """Reject every tracked change. Equivalent to the ``original`` view."""
    revs = Revisions.from_document(doc)
    return _apply(doc, accept_ids=set(), reject_ids={r.id for r in revs if r.id})


def accept_by_author(doc: ContentDocument, author: str) -> ContentDocument:
    """Accept every revision by a given author; leave others in place."""
    revs = Revisions.from_document(doc)
    ids = {r.id for r in revs.by_author(author) if r.id}
    return _apply(doc, accept_ids=ids, reject_ids=set())


def reject_by_author(doc: ContentDocument, author: str) -> ContentDocument:
    """Reject every revision by a given author; leave others in place."""
    revs = Revisions.from_document(doc)
    ids = {r.id for r in revs.by_author(author) if r.id}
    return _apply(doc, accept_ids=set(), reject_ids=ids)


def at_time(doc: ContentDocument, t: datetime) -> ContentDocument:
    """Reconstruct the document as it was at time ``t``.

    For every revision:
    - If ``revision.date <= t``: the change had been made, accept it.
    - If ``revision.date > t`` or ``date is None``: the change had not
      yet happened, reject it.
    """
    revs = Revisions.from_document(doc)
    accept_ids: set[str] = set()
    reject_ids: set[str] = set()
    for rev in revs:
        if not rev.id:
            continue
        if rev.date is not None and rev.date <= t:
            accept_ids.add(rev.id)
        else:
            reject_ids.add(rev.id)
    return _apply(doc, accept_ids=accept_ids, reject_ids=reject_ids)


# ----------------------------------------------------------------------------
# Authoring (UC4): construct revisions on a clean document
# ----------------------------------------------------------------------------


def _next_revision_id(doc: ContentDocument) -> str:
    """Pick the next free numeric rev:id in the document.

    OOXML requires ``w:id`` be unique within the document. We use string
    numeric IDs like ``"0"``, ``"1"``, ``"2"`` — matching Word's convention.
    """
    existing_ids: set[str] = set()
    for rev in Revisions.from_document(doc):
        existing_ids.add(rev.id)
    i = 0
    while str(i) in existing_ids:
        i += 1
    return str(i)


def _fmt_date(date: datetime | None) -> str:
    """Render a datetime as the OOXML ISO-8601 string used by Word."""
    if date is None:
        return ""
    # Normalize to an ISO-8601 form. Z suffix for UTC is idiomatic for OOXML.
    iso = date.isoformat()
    if iso.endswith("+00:00"):
        iso = iso[:-6] + "Z"
    return iso


def _revision_attr(
    change_type: str,
    *,
    author: str,
    date: datetime | None,
    revision_id: str,
    move_name: str | None = None,
):
    """Build the ``Attr`` for a rev-* wrapper node."""
    from kaos_content.model.attr import Attr

    rev_class = {
        "insertion": "rev-ins",
        "deletion": "rev-del",
        "move_from": "rev-move-from",
        "move_to": "rev-move-to",
    }[change_type]
    kv: dict[str, str] = {
        "rev:id": revision_id,
        "rev:author": author,
    }
    if date is not None:
        kv["rev:date"] = _fmt_date(date)
    if move_name is not None:
        kv["rev:move-name"] = move_name
    return Attr(classes=(rev_class,), kv=kv)


def make_inline_insertion(
    content: Any | tuple[Any, ...],
    *,
    author: str,
    date: datetime | None = None,
    revision_id: str | None = None,
) -> Any:
    """Wrap one or more inline nodes in a ``Span`` with rev-ins metadata.

    Args:
        content: An Inline node or tuple of Inline nodes to mark as inserted.
        author: Author name (goes into ``w:author``).
        date: Timestamp for ``w:date``. Omit for an undated revision.
        revision_id: Optional fixed ID. If None, callers should use
            ``mark_inserted`` for auto-ID assignment against a full document.
    """
    from kaos_content.model.inlines import Span

    children = _as_tuple(content)
    rid = revision_id if revision_id is not None else "0"
    attr = _revision_attr("insertion", author=author, date=date, revision_id=rid)
    return Span(attr=attr, children=children)


def make_inline_deletion(
    content: Any | tuple[Any, ...],
    *,
    author: str,
    date: datetime | None = None,
    revision_id: str | None = None,
) -> Any:
    """Wrap inline content in a ``Span`` with rev-del metadata."""
    from kaos_content.model.inlines import Span

    children = _as_tuple(content)
    rid = revision_id if revision_id is not None else "0"
    attr = _revision_attr("deletion", author=author, date=date, revision_id=rid)
    return Span(attr=attr, children=children)


def make_block_insertion(
    content: Any | tuple[Any, ...],
    *,
    author: str,
    date: datetime | None = None,
    revision_id: str | None = None,
) -> Any:
    """Wrap one or more block nodes in a ``Div`` with rev-ins metadata."""
    from kaos_content.model.blocks import Div

    children = _as_tuple(content)
    rid = revision_id if revision_id is not None else "0"
    attr = _revision_attr("insertion", author=author, date=date, revision_id=rid)
    return Div(attr=attr, children=children)


def make_block_deletion(
    content: Any | tuple[Any, ...],
    *,
    author: str,
    date: datetime | None = None,
    revision_id: str | None = None,
) -> Any:
    """Wrap block content in a ``Div`` with rev-del metadata."""
    from kaos_content.model.blocks import Div

    children = _as_tuple(content)
    rid = revision_id if revision_id is not None else "0"
    attr = _revision_attr("deletion", author=author, date=date, revision_id=rid)
    return Div(attr=attr, children=children)


def _as_tuple(content: Any) -> tuple[Any, ...]:
    """Normalize ``content`` to a tuple of AST nodes."""
    if isinstance(content, tuple):
        return content
    if isinstance(content, list):
        return tuple(content)
    return (content,)


# ── High-level helpers that operate on a whole document ──


def append_block_insertion(
    doc: ContentDocument,
    content: Any | tuple[Any, ...],
    *,
    author: str,
    date: datetime | None = None,
    revision_id: str | None = None,
) -> ContentDocument:
    """Append a block-level insertion to the end of the document.

    Returns a new ``ContentDocument`` with a ``Div`` wrapper (rev-ins class)
    containing the new block(s). Generates a fresh revision_id if none given.
    """
    rid = revision_id or _next_revision_id(doc)
    wrapper = make_block_insertion(content, author=author, date=date, revision_id=rid)
    return doc.model_copy(update={"body": (*doc.body, wrapper)})


def insert_block_after(
    doc: ContentDocument,
    block_index: int,
    content: Any | tuple[Any, ...],
    *,
    author: str,
    date: datetime | None = None,
    revision_id: str | None = None,
) -> ContentDocument:
    """Insert a block-level revision after ``doc.body[block_index]``.

    ``block_index`` is 0-based. A negative index inserts at the start.
    ``block_index >= len(body)`` appends at the end.
    """
    rid = revision_id or _next_revision_id(doc)
    wrapper = make_block_insertion(content, author=author, date=date, revision_id=rid)
    body = list(doc.body)
    insert_at = max(0, min(block_index + 1, len(body)))
    body.insert(insert_at, wrapper)
    return doc.model_copy(update={"body": tuple(body)})


def delete_block_at(
    doc: ContentDocument,
    block_index: int,
    *,
    author: str,
    date: datetime | None = None,
    revision_id: str | None = None,
) -> ContentDocument:
    """Mark the block at ``doc.body[block_index]`` as deleted.

    Wraps the existing block in a ``Div(classes=("rev-del",))``. The target
    block is preserved inside the wrapper so the deletion can still be
    rejected or viewed in ``markup`` / ``original`` modes.
    """
    body = list(doc.body)
    if not 0 <= block_index < len(body):
        msg = f"block_index {block_index} out of range [0, {len(body)})"
        raise IndexError(msg)
    rid = revision_id or _next_revision_id(doc)
    target = body[block_index]
    wrapper = make_block_deletion(target, author=author, date=date, revision_id=rid)
    body[block_index] = wrapper
    return doc.model_copy(update={"body": tuple(body)})
