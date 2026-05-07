"""Serialize a ContentDocument AST to plain text.

Flattens document structure to plain text with configurable separators.
Useful for search indexing, simple text extraction, and LLM prompts.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from kaos_content.serializers._revision import (
    ViewMode,
    revision_class,
    should_skip_revision,
    wrap_text_markup,
)

if TYPE_CHECKING:
    from kaos_content.model.document import ContentDocument


def serialize_text(
    document: ContentDocument,
    *,
    block_separator: str = "\n\n",
    heading_separator: str = "\n",
    list_indent: str = "  ",
    table_format: Literal["csv", "plain"] = "plain",
    view: ViewMode = "final",
) -> str:
    """Serialize a ContentDocument to plain text.

    Parameters
    ----------
    document:
        The document to serialize.
    block_separator:
        Separator between top-level blocks.
    heading_separator:
        Separator after headings (before content).
    list_indent:
        Indentation per nesting level for list items.
    table_format:
        ``"plain"`` renders cells separated by ``|``,
        ``"csv"`` renders cells comma-separated.
    view:
        Tracked-changes view mode. ``"final"`` (default) renders the
        accepted version and is backward compatible. ``"original"``
        renders the pre-change version. ``"markup"`` shows both with
        ``{+...+}`` for insertions and ``{-...-}`` for deletions.
    """
    ctx = _TextContext(
        block_separator=block_separator,
        heading_separator=heading_separator,
        list_indent=list_indent,
        table_format=table_format,
        view=view,
    )
    parts: list[str] = []
    for block in document.body:
        rendered = ctx.render_block(block)
        if rendered:
            parts.append(rendered)

    result = block_separator.join(parts)

    # Footnotes at end
    if document.footnotes:
        fn_parts: list[str] = []
        for key, blocks in document.footnotes.items():
            fn_body = block_separator.join(
                ctx.render_block(b) for b in blocks if ctx.render_block(b)
            )
            fn_parts.append(f"[{key}]: {fn_body}")
        result += block_separator + block_separator.join(fn_parts)

    return result.strip() + "\n"


class _TextContext:
    """Stateless context for plain text rendering."""

    def __init__(
        self,
        *,
        block_separator: str,
        heading_separator: str,
        list_indent: str,
        table_format: Literal["csv", "plain"],
        view: ViewMode = "final",
    ) -> None:
        self._block_sep = block_separator
        self._heading_sep = heading_separator
        self._list_indent = list_indent
        self._table_format = table_format
        self._view = view

    def render_block(self, block: Any, indent: str = "") -> str:
        nt = block.node_type

        if nt == "paragraph":
            return indent + self._render_inlines(block.children)

        if nt == "heading":
            text = self._render_inlines(block.children)
            return indent + text + self._heading_sep

        if nt == "blockquote":
            inner = self._render_block_children(block.children, indent)
            return inner

        if nt == "bullet_list":
            return self._render_list(block.children, indent, ordered=False)

        if nt == "ordered_list":
            return self._render_list(block.children, indent, ordered=True, start=block.start)

        if nt == "list_item":
            return self._render_block_children(block.children, indent)

        if nt == "definition_list":
            return self._render_definition_list(block, indent)

        if nt == "table":
            return self._render_table(block, indent)

        if nt == "codeblock":
            return indent + block.value

        if nt == "thematic_break":
            return indent + "---"

        if nt == "figure":
            return self._render_block_children(block.children, indent)

        if nt == "page_break":
            return ""

        if nt == "div":
            rev_cls = revision_class(block)
            if rev_cls and should_skip_revision(rev_cls, self._view):
                return ""
            rendered = self._render_block_children(block.children, indent)
            if rev_cls and self._view == "markup":
                return wrap_text_markup(rendered, rev_cls)
            return rendered

        if nt == "raw_block":
            return indent + block.value

        if nt == "math_block":
            return indent + block.value

        if nt == "admonition":
            kind = getattr(block, "kind", "note").upper()
            inner = self._render_block_children(block.children, indent)
            return f"{indent}[{kind}] {inner}"

        return ""

    def _render_block_children(self, blocks: Sequence[Any], indent: str) -> str:
        parts = [self.render_block(b, indent) for b in blocks]
        return self._block_sep.join(p for p in parts if p)

    def _render_list(
        self,
        items: Sequence[Any],
        indent: str,
        *,
        ordered: bool,
        start: int = 1,
    ) -> str:
        parts: list[str] = []
        for i, item in enumerate(items):
            marker = f"{start + i}. " if ordered else "- "
            checked = getattr(item, "checked", None)
            prefix = ""
            if checked is True:
                prefix = "[x] "
            elif checked is False:
                prefix = "[ ] "

            inner = self._render_block_children(item.children, indent + self._list_indent)
            parts.append(f"{indent}{marker}{prefix}{inner.lstrip()}")
        return "\n".join(parts)

    def _render_definition_list(self, dl: Any, indent: str) -> str:
        parts: list[str] = []
        for item in dl.children:
            term = self._render_inlines(item.term)
            parts.append(f"{indent}{term}")
            for def_blocks in item.definitions:
                def_text = self._render_block_children(def_blocks, indent + self._list_indent)
                parts.append(f"{indent}{self._list_indent}{def_text.lstrip()}")
        return "\n".join(parts)

    def _render_table(self, table: Any, indent: str) -> str:
        all_rows: list[Any] = []
        head = getattr(table, "head", None)
        if head is not None:
            all_rows.extend(head.rows)
        for body_section in getattr(table, "bodies", ()):
            all_rows.extend(body_section.rows)
        foot = getattr(table, "foot", None)
        if foot is not None:
            all_rows.extend(foot.rows)

        if not all_rows:
            return ""

        sep = ", " if self._table_format == "csv" else " | "
        lines: list[str] = []
        for row in all_rows:
            cells = [self._render_cell(cell) for cell in row.cells]
            lines.append(indent + sep.join(cells))
        return "\n".join(lines)

    def _render_cell(self, cell: Any) -> str:
        if not cell.content:
            return ""
        from kaos_content.traversal.visitor import extract_text

        return extract_text(cell).strip()

    # ── Inline rendering ──

    def _render_inlines(self, inlines: Sequence[Any]) -> str:
        return "".join(self._render_inline(i) for i in inlines)

    def _render_inline(self, inline: Any) -> str:
        nt = inline.node_type

        if nt == "text":
            return inline.value
        if nt == "span":
            rev_cls = revision_class(inline)
            if rev_cls and should_skip_revision(rev_cls, self._view):
                return ""
            rendered = self._render_inlines(inline.children)
            if rev_cls and self._view == "markup":
                return wrap_text_markup(rendered, rev_cls)
            return rendered
        if nt in (
            "emphasis",
            "strong",
            "strikethrough",
            "superscript",
            "subscript",
            "underline",
        ):
            return self._render_inlines(inline.children)
        if nt == "code":
            return inline.value
        if nt == "link":
            return self._render_inlines(inline.children)
        if nt == "image":
            return inline.alt or ""
        if nt == "footnote_ref":
            return f"[{inline.identifier}]"
        if nt == "citation":
            return self._render_inlines(inline.children)
        if nt == "math":
            return inline.value
        if nt == "raw_inline":
            return inline.value
        if nt == "line_break":
            return "\n"
        if nt == "soft_break":
            return " "
        return ""
