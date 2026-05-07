"""Terse constructors for inline and block AST nodes.

Removes the Text(value="x") boilerplate that otherwise bloats every
test fixture and reader/writer call site. Composable: ``bold(italic("x"))``
nests correctly.

No new AST types — these are pure construction helpers that return the
same frozen Pydantic models as direct instantiation.

Example::

    from kaos_content.shortcuts import bold, italic, link, paragraph, heading

    # Before
    Paragraph(children=(
        Text(value="Visit "),
        Strong(children=(Text(value="our site"),)),
        Text(value="."),
    ))

    # After
    paragraph("Visit ", bold("our site"), ".")
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from kaos_content.model.blocks import (
    BulletList,
    CodeBlock,
    Heading,
    ListItem,
    OrderedList,
    Paragraph,
    Table,
)
from kaos_content.model.inlines import (
    Code,
    Emphasis,
    Image,
    Inline,
    LineBreak,
    Link,
    SoftBreak,
    Strikethrough,
    Strong,
    Subscript,
    Superscript,
    Text,
    Underline,
)
from kaos_content.model.node import BaseInline
from kaos_content.model.table import Cell, Row, TableSection

# ---------------------------------------------------------------------------
# Inline coercion
# ---------------------------------------------------------------------------


def _to_inline(x: Inline | str) -> Inline:
    """Coerce a string or existing Inline into an Inline node."""
    if isinstance(x, str):
        return Text(value=x)
    return x


def _to_inline_tuple(args: Iterable[Inline | str]) -> tuple[Inline, ...]:
    return tuple(_to_inline(a) for a in args)


# ---------------------------------------------------------------------------
# Inline constructors
# ---------------------------------------------------------------------------


def text(value: str) -> Text:
    """Create a ``Text`` node."""
    return Text(value=value)


def bold(*args: Inline | str) -> Strong:
    """Wrap content in ``Strong``.

    Accepts any mix of strings and Inline nodes::

        bold("word")                         # Strong(Text("word"))
        bold("hello ", italic("world"))      # Strong(Text, Emphasis(Text))
    """
    return Strong(children=_to_inline_tuple(args))


def italic(*args: Inline | str) -> Emphasis:
    """Wrap content in ``Emphasis``."""
    return Emphasis(children=_to_inline_tuple(args))


def strike(*args: Inline | str) -> Strikethrough:
    """Wrap content in ``Strikethrough``."""
    return Strikethrough(children=_to_inline_tuple(args))


def underline(*args: Inline | str) -> Underline:
    """Wrap content in ``Underline``."""
    return Underline(children=_to_inline_tuple(args))


def code(value: str) -> Code:
    """Create an inline ``Code`` node."""
    return Code(value=value)


def link(url: str, *args: Inline | str, title: str | None = None) -> Link:
    """Create a ``Link`` to ``url``. Children are the visible link text.

    Example: ``link("https://a.com", "click ", bold("here"))``
    """
    return Link(url=url, title=title, children=_to_inline_tuple(args))


def sup(*args: Inline | str) -> Superscript:
    """Wrap content in ``Superscript``."""
    return Superscript(children=_to_inline_tuple(args))


def sub(*args: Inline | str) -> Subscript:
    """Wrap content in ``Subscript``."""
    return Subscript(children=_to_inline_tuple(args))


def linebreak() -> LineBreak:
    """Hard line break."""
    return LineBreak()


def softbreak() -> SoftBreak:
    """Soft line break (collapses to whitespace in most renderers)."""
    return SoftBreak()


def image(
    src: str,
    alt: str | None = None,
    *,
    title: str | None = None,
    width: float | None = None,
    height: float | None = None,
) -> Image:
    """Create an ``Image`` inline."""
    return Image(src=src, alt=alt, title=title, width=width, height=height)


# ---------------------------------------------------------------------------
# Block constructors
# ---------------------------------------------------------------------------


def paragraph(*args: Inline | str) -> Paragraph:
    """Create a ``Paragraph`` from any mix of strings and inlines.

    Example: ``paragraph("Visit ", bold("our site"), ".")``
    """
    return Paragraph(children=_to_inline_tuple(args))


def heading(depth: int, *args: Inline | str) -> Heading:
    """Create a ``Heading`` at a given depth (1-6).

    Example: ``heading(1, "Main Title")``
    """
    return Heading(depth=depth, children=_to_inline_tuple(args))


def code_block(value: str, language: str | None = None) -> CodeBlock:
    """Create a ``CodeBlock``."""
    return CodeBlock(value=value, language=language)


def _to_list_item(x: Any) -> ListItem:
    """Accept strings, Inline, or ListItem and produce a ListItem."""
    if isinstance(x, ListItem):
        return x
    if isinstance(x, str):
        return ListItem(children=(paragraph(x),))
    # Inline nodes all inherit from BaseInline
    if isinstance(x, BaseInline):
        return ListItem(children=(Paragraph(children=(x,)),))
    # Assume Block
    return ListItem(children=(x,))


def bullet_list(*items: Any) -> BulletList:
    """Create a ``BulletList``. Items may be strings, inlines, or ListItems."""
    return BulletList(children=tuple(_to_list_item(i) for i in items))


def ordered_list(*items: Any, start: int = 1) -> OrderedList:
    """Create an ``OrderedList``. Items may be strings, inlines, or ListItems."""
    return OrderedList(children=tuple(_to_list_item(i) for i in items), start=start)


def table_from_rows(
    headers: list[str | Inline] | None,
    rows: list[list[str | Inline]],
) -> Table:
    """Build a simple Table from string/inline cell values.

    ``headers`` may be None for a table with no head section.
    Every cell becomes a single-Paragraph ``Cell``.
    """
    head = None
    if headers:
        head_cells = tuple(Cell(content=(Paragraph(children=(_to_inline(h),)),)) for h in headers)
        head = TableSection(rows=(Row(cells=head_cells),))

    body_rows = tuple(
        Row(cells=tuple(Cell(content=(Paragraph(children=(_to_inline(c),)),)) for c in row))
        for row in rows
    )
    body = TableSection(rows=body_rows)
    return Table(head=head, bodies=(body,))
