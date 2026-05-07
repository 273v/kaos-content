"""Block AST node types."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, model_validator

from kaos_content.model.attr import Caption, ColSpec
from kaos_content.model.inlines import Inline
from kaos_content.model.node import BaseBlock
from kaos_content.model.table import TableSection


class Paragraph(BaseBlock):
    """Block containing inline content."""

    node_type: Literal["paragraph"] = "paragraph"
    children: tuple[Inline, ...]


class Heading(BaseBlock):
    """Section heading, depth 1-6."""

    node_type: Literal["heading"] = "heading"
    depth: int
    children: tuple[Inline, ...]

    @model_validator(mode="after")
    def _check_depth(self) -> Heading:
        if not 1 <= self.depth <= 6:
            msg = f"Heading depth must be 1-6, got {self.depth}"
            raise ValueError(msg)
        return self


class BlockQuote(BaseBlock):
    """Quoted block content."""

    node_type: Literal["blockquote"] = "blockquote"
    children: tuple[Block, ...]


class OrderedList(BaseBlock):
    """Ordered (numbered) list."""

    node_type: Literal["ordered_list"] = "ordered_list"
    start: int = 1
    children: tuple[ListItem, ...]


class BulletList(BaseBlock):
    """Unordered (bulleted) list."""

    node_type: Literal["bullet_list"] = "bullet_list"
    children: tuple[ListItem, ...]


class ListItem(BaseBlock):
    """Single item in a list. May contain nested blocks."""

    node_type: Literal["list_item"] = "list_item"
    checked: bool | None = None
    children: tuple[Block, ...]


class DefinitionList(BaseBlock):
    """Definition list (term + definitions)."""

    node_type: Literal["definition_list"] = "definition_list"
    children: tuple[DefinitionItem, ...]


class DefinitionItem(BaseBlock):
    """Single term + its definitions."""

    node_type: Literal["definition_item"] = "definition_item"
    term: tuple[Inline, ...]
    definitions: tuple[tuple[Block, ...], ...]


class Table(BaseBlock):
    """Full table with optional caption, column specs, head/body/foot sections."""

    node_type: Literal["table"] = "table"
    caption: Caption | None = None
    col_specs: tuple[ColSpec, ...] = ()
    head: TableSection | None = None
    bodies: tuple[TableSection, ...] = ()
    foot: TableSection | None = None


class CodeBlock(BaseBlock):
    """Fenced or indented code block."""

    node_type: Literal["codeblock"] = "codeblock"
    language: str | None = None
    value: str


class ThematicBreak(BaseBlock):
    """Horizontal rule."""

    node_type: Literal["thematic_break"] = "thematic_break"


class Figure(BaseBlock):
    """Figure with optional caption. Contains an image or other block content."""

    node_type: Literal["figure"] = "figure"
    caption: Caption | None = None
    children: tuple[Block, ...]


class PageBreak(BaseBlock):
    """Explicit page break (from source document layout)."""

    node_type: Literal["page_break"] = "page_break"


class Div(BaseBlock):
    """Generic block container. Carries Attr for domain-specific semantics."""

    node_type: Literal["div"] = "div"
    children: tuple[Block, ...]


class RawBlock(BaseBlock):
    """Raw content in a specific format (HTML, LaTeX, etc.)."""

    node_type: Literal["raw_block"] = "raw_block"
    format: str
    value: str


class MathBlock(BaseBlock):
    """Display math (LaTeX)."""

    node_type: Literal["math_block"] = "math_block"
    value: str


class Admonition(BaseBlock):
    """Callout/alert block (note, warning, tip, etc.)."""

    node_type: Literal["admonition"] = "admonition"
    kind: str
    title: str | None = None
    children: tuple[Block, ...]


Block = Annotated[
    Paragraph
    | Heading
    | BlockQuote
    | OrderedList
    | BulletList
    | ListItem
    | DefinitionList
    | DefinitionItem
    | Table
    | CodeBlock
    | ThematicBreak
    | Figure
    | PageBreak
    | Div
    | RawBlock
    | MathBlock
    | Admonition,
    Field(discriminator="node_type"),
]
