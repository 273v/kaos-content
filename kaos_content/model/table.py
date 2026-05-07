"""Table structural components: TableSection, Row, Cell."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import Field

from kaos_content.model.attr import Alignment
from kaos_content.model.node import BaseNode

if TYPE_CHECKING:
    from kaos_content.model.blocks import Block


class Cell(BaseNode):
    """A table cell with optional span.

    ``row_span`` and ``col_span`` must each be ``>= 1``. A span of 0
    or negative is structurally invalid (a cell that doesn't occupy
    its own slot) and was silently accepted before 0.1.0a1.
    """

    node_type: Literal["cell"] = "cell"
    alignment: Alignment | None = None
    row_span: int = Field(default=1, ge=1)
    col_span: int = Field(default=1, ge=1)
    content: tuple[Block, ...] = ()


class Row(BaseNode):
    """A table row."""

    node_type: Literal["row"] = "row"
    cells: tuple[Cell, ...] = ()


class TableSection(BaseNode):
    """A group of rows (head, body, or foot)."""

    node_type: Literal["table_section"] = "table_section"
    rows: tuple[Row, ...] = ()
