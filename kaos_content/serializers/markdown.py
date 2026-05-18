"""Serialize a ContentDocument AST to CommonMark + GFM markdown."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from kaos_content.serializers._revision import (
    ViewMode,
    revision_class,
    should_skip_revision,
    wrap_markdown_markup,
)

if TYPE_CHECKING:
    from kaos_content.model.document import ContentDocument


def serialize_markdown(
    document: ContentDocument,
    *,
    view: ViewMode = "final",
    allow_raw_html: bool = False,
) -> str:
    """Serialize a ContentDocument to markdown string.

    Args:
        document: The ContentDocument to render.
        view: Tracked-changes view mode. ``"final"`` (default) renders
            the accepted version (backward compatible). ``"original"``
            renders the pre-change version. ``"markup"`` shows both with
            HTML5 ``<ins>`` and ``<del>`` elements.
        allow_raw_html: If False (default), raw HTML/raw markdown blocks
            are stripped and link/image URLs whose canonical scheme is
            in :data:`kaos_content._security.UNSAFE_SCHEMES`
            (``javascript``, ``data``, ``vbscript``, ``file``) are
            replaced with ``#``. If True, the caller asserts that the
            document AST is trusted; raw blocks pass through verbatim
            and unsafe URLs are emitted as-is. **Set this to True only
            when serializing AST that you control end-to-end.**
            Markdown rendered to HTML by a downstream consumer can
            execute the same XSS payloads as direct HTML.
    """
    ctx = _SerializerContext(document, view=view, allow_raw_html=allow_raw_html)
    md = ctx.serialize()
    return _merge_adjacent_spans(md)


# Merge adjacent bold/italic spans: **a** **b** → **a b**, *a* *b* → *a b*
# Bold must be checked first (** before *) to avoid partial matches.
_ADJACENT_BOLD = re.compile(r"\*\*([^*]+)\*\* \*\*([^*]+)\*\*")
_ADJACENT_ITALIC = re.compile(r"(?<!\*)\*([^*]+)\* \*([^*]+)\*(?!\*)")


def _merge_adjacent_spans(md: str) -> str:
    """Merge adjacent same-formatted inline spans.

    DOCX often splits a single bold/italic phrase into separate runs
    (one per word). This produces ``**word1** **word2**`` instead of
    ``**word1 word2**``. This post-pass merges them.
    """
    # Iterate until stable — a single pass may not catch chains of 3+
    for _ in range(5):
        merged = _ADJACENT_BOLD.sub(r"**\1 \2**", md)
        merged = _ADJACENT_ITALIC.sub(r"*\1 \2*", merged)
        if merged == md:
            break
        md = merged
    return md


# ── Escaping ──
#
# CommonMark escaping is context-dependent. Characters that are dangerous at
# the start of a line (would be parsed as block syntax) differ from characters
# that are dangerous in inline context.

# Inline-context characters that always need escaping in text nodes.
# NOT escaped: ( ) { } # + - . ! | > &  — these are only dangerous in
# specific contexts (line-start, table cells) handled separately.
_INLINE_ESCAPE = re.compile(r"([\\`*_\[\]<~])")

# Characters/patterns dangerous only at line start (would trigger block syntax).
# We check these per-line in _escape_text.
_LINE_START_HEADING = re.compile(r"^(#{1,6})(\s)")
_LINE_START_ULIST = re.compile(r"^([*+-])(\s)")
_LINE_START_OLIST = re.compile(r"^(\d{1,9}[.)]) ")
_LINE_START_BLOCKQUOTE = re.compile(r"^>")
_LINE_START_THEMATIC = re.compile(r"^([-*_])\s*\1\s*\1")


def _escape_text(text: str, *, in_table_cell: bool = False) -> str:
    """Escape markdown special characters in plain text.

    Context-aware: only escapes characters that would change meaning.
    - Inline: \\, `, *, _, [, ], <, &, ~
    - Line-start: #, -, +, *, >, digit+. (only when they'd trigger block syntax)
    - Table cell: additionally escape |
    """
    lines = text.split("\n")
    result_lines: list[str] = []

    for line in lines:
        # Inline escaping (always)
        escaped = _INLINE_ESCAPE.sub(r"\\\1", line)

        # Line-start escaping (only patterns that would trigger block syntax)
        escaped = _LINE_START_HEADING.sub(r"\\\1\2", escaped)
        escaped = _LINE_START_ULIST.sub(r"\\\1\2", escaped)
        escaped = _LINE_START_OLIST.sub(lambda m: "\\" + m.group(0), escaped)
        escaped = _LINE_START_BLOCKQUOTE.sub(r"\>", escaped)
        # Don't escape thematic breaks — they require 3+ chars and are rare in text

        # Table cell: escape pipe
        if in_table_cell:
            escaped = escaped.replace("|", "\\|")

        result_lines.append(escaped)

    return "\n".join(result_lines)


class _SerializerContext:
    """Maintains state during markdown serialization."""

    def __init__(
        self,
        document: ContentDocument,
        *,
        view: ViewMode = "final",
        allow_raw_html: bool = False,
    ) -> None:
        self._document = document
        self._view = view
        self._allow_raw_html = allow_raw_html
        self._redacted_refs: set[str] = set()
        self._build_redaction_set()

    def _safe_url(self, url: str) -> str:
        """Return ``url`` if safe; otherwise ``"#"``.

        Defers to :func:`kaos_core.security.is_safe_url`. The returned
        string is suitable for the URL slot of a markdown link or image:
        ``[text](URL)`` / ``![alt](URL)``. When ``allow_raw_html=True``
        is set on the serializer, the URL is returned unchanged.
        """
        from kaos_core.security import is_safe_url

        if self._allow_raw_html or is_safe_url(url):
            return url
        return "#"

    def _build_redaction_set(self) -> None:
        from kaos_content.model.annotation import AnnotationType

        for ann in self._document.annotations:
            if ann.type == AnnotationType.REDACTION:
                for target in ann.targets:
                    self._redacted_refs.add(target.node_ref)

    def _is_redacted(self, ref: str) -> bool:
        """Check if a node ref is targeted by a REDACTION annotation."""
        return ref in self._redacted_refs

    def serialize(self) -> str:
        parts: list[str] = []

        for i, block in enumerate(self._document.body):
            rendered = self._render_block(block, indent="", ref=f"#/body/{i}")
            parts.append(rendered)

        # Footnotes at end
        if self._document.footnotes:
            parts.append("")  # blank line before footnotes
            for key, blocks in self._document.footnotes.items():
                # Footnote blocks use refs like #/footnotes/{key}/{i} (no /children/ segment)
                fn_parts: list[str] = []
                for fi, block in enumerate(blocks):
                    fn_ref = f"#/footnotes/{key}/{fi}"
                    fn_parts.append(self._render_block(block, indent="    ", ref=fn_ref))
                fn_content = "\n\n".join(fn_parts)
                # First line gets the footnote marker
                if fn_content:
                    fn_lines = fn_content.split("\n")
                    fn_lines[0] = f"[^{key}]: {fn_lines[0].lstrip()}"
                    # Subsequent lines indented
                    for i in range(1, len(fn_lines)):
                        if fn_lines[i].strip():
                            fn_lines[i] = f"    {fn_lines[i].lstrip()}"
                    parts.append("\n".join(fn_lines))

        result = "\n\n".join(parts)
        # Clean up excessive blank lines
        while "\n\n\n" in result:
            result = result.replace("\n\n\n", "\n\n")
        return result.strip() + "\n"

    # ── Block rendering ──

    def _render_block(self, block: Any, indent: str, ref: str = "") -> str:
        """Dispatch block rendering by node_type."""
        # Redaction check: if this block is targeted, emit placeholder
        if ref and self._is_redacted(ref):
            return f"{indent}[REDACTED]"

        node_type = block.node_type
        if node_type == "paragraph":
            return self._render_paragraph(block, indent, ref)
        if node_type == "heading":
            return self._render_heading(block, indent, ref)
        if node_type == "blockquote":
            return self._render_blockquote(block, indent, ref)
        if node_type == "bullet_list":
            return self._render_bullet_list(block, indent, ref)
        if node_type == "ordered_list":
            return self._render_ordered_list(block, indent, ref)
        if node_type == "list_item":
            return self._render_list_item(block, indent, ref=ref)
        if node_type == "definition_list":
            return self._render_definition_list(block, indent, ref)
        if node_type == "definition_item":
            return self._render_definition_item(block, indent, ref)
        if node_type == "table":
            return self._render_table(block, indent, ref)
        if node_type == "codeblock":
            return self._render_code_block(block, indent)
        if node_type == "thematic_break":
            return f"{indent}---"
        if node_type == "figure":
            return self._render_figure(block, indent, ref)
        if node_type == "page_break":
            return f"{indent}---"
        if node_type == "div":
            rev_cls = revision_class(block)
            if rev_cls and should_skip_revision(rev_cls, self._view):
                return ""
            rendered = self._render_block_children(block.children, indent, ref_prefix=ref)
            if rev_cls and self._view == "markup":
                return wrap_markdown_markup(rendered, rev_cls)
            return rendered
        if node_type == "raw_block":
            if block.format in ("markdown", "html"):
                if self._allow_raw_html:
                    return f"{indent}{block.value}"
                # Safe default — raw HTML/markdown blocks are dropped.
                # The marker is rendered as a markdown HTML comment so
                # downstream markdown -> HTML conversion preserves the
                # forensic trail.
                marker = f"<!-- raw {block.format} stripped (set allow_raw_html=True to emit) -->"
                return f"{indent}{marker}"
            return ""
        if node_type == "math_block":
            value_lines = block.value.split("\n")
            indented_value = "\n".join(f"{indent}{line}" for line in value_lines)
            return f"{indent}$$\n{indented_value}\n{indent}$$"
        if node_type == "admonition":
            return self._render_admonition(block, indent, ref)
        return ""

    def _render_paragraph(self, para: Any, indent: str, ref: str = "") -> str:
        inlines = self._render_inlines(para.children, ref_prefix=ref)
        label = getattr(para, "numbering_label", None)
        if label:
            return f"{indent}{label} {inlines}"
        return f"{indent}{inlines}"

    def _render_heading(self, heading: Any, indent: str, ref: str = "") -> str:
        prefix = "#" * heading.depth
        inlines = self._render_inlines(heading.children, ref_prefix=ref)
        label = getattr(heading, "numbering_label", None)
        if label:
            return f"{indent}{prefix} {label} {inlines}"
        return f"{indent}{prefix} {inlines}"

    def _render_blockquote(self, bq: Any, indent: str, ref: str = "") -> str:
        inner = self._render_block_children(bq.children, indent="", ref_prefix=ref)
        lines = inner.split("\n")
        return "\n".join(f"{indent}> {line}" if line.strip() else f"{indent}>" for line in lines)

    def _render_bullet_list(self, bl: Any, indent: str, ref: str = "") -> str:
        items: list[str] = []
        for i, item in enumerate(bl.children):
            child_ref = f"{ref}/children/{i}" if ref else ""
            label = getattr(item, "numbering_label", None)
            # A bullet list whose items each carry a rendered label
            # came from a numbered Word list misidentified as a bullet
            # list (e.g. an unrecognized numFmt). Emit the label
            # verbatim so attorney citations survive the round trip.
            marker = f"{label} " if label else "- "
            rendered = self._render_list_item(item, indent, marker=marker, ref=child_ref)
            items.append(rendered)
        return "\n".join(items)

    def _render_ordered_list(self, ol: Any, indent: str, ref: str = "") -> str:
        items: list[str] = []
        start = getattr(ol, "start", 1)
        # Compute the widest marker so all items align consistently
        last_num = start + len(ol.children) - 1
        marker_width = len(f"{last_num}. ")
        for i, item in enumerate(ol.children):
            child_ref = f"{ref}/children/{i}" if ref else ""
            label = getattr(item, "numbering_label", None)
            if label:
                # Rendered Word numeral wins outright — preserves
                # "Section 11(a)(i)" instead of resequencing to
                # "1.". Width-aligned to the longest label so wrapped
                # continuation indents still align.
                marker = f"{label} "
            else:
                num = start + i
                marker = f"{num}. ".ljust(marker_width)
            rendered = self._render_list_item(item, indent, marker=marker, ref=child_ref)
            items.append(rendered)
        return "\n".join(items)

    def _render_list_item(self, item: Any, indent: str, marker: str = "- ", ref: str = "") -> str:
        # Redaction check on the list item itself
        if ref and self._is_redacted(ref):
            return f"{indent}{marker}[REDACTED]"

        if not item.children:
            checked = getattr(item, "checked", None)
            if checked is True:
                return f"{indent}{marker}[x]"
            if checked is False:
                return f"{indent}{marker}[ ]"
            return f"{indent}{marker}"

        parts: list[str] = []
        continuation_indent = indent + " " * len(marker)

        for idx, child in enumerate(item.children):
            child_ref = f"{ref}/children/{idx}" if ref else ""
            rendered = self._render_block(child, indent="", ref=child_ref)
            child_lines = rendered.split("\n")
            if idx == 0:
                # First child: attach to marker
                checked = getattr(item, "checked", None)
                prefix = ""
                if checked is True:
                    prefix = "[x] "
                elif checked is False:
                    prefix = "[ ] "
                child_lines[0] = f"{indent}{marker}{prefix}{child_lines[0]}"
                for li in range(1, len(child_lines)):
                    child_lines[li] = f"{continuation_indent}{child_lines[li]}"
            else:
                # Subsequent children: blank line + continuation indent
                for li in range(len(child_lines)):
                    child_lines[li] = f"{continuation_indent}{child_lines[li]}"
            parts.append("\n".join(child_lines))

        return "\n".join(parts)

    def _render_definition_list(self, dl: Any, indent: str, ref: str = "") -> str:
        items: list[str] = []
        for i, item in enumerate(dl.children):
            child_ref = f"{ref}/children/{i}" if ref else ""
            items.append(self._render_definition_item(item, indent, child_ref))
        return "\n\n".join(items)

    def _render_definition_item(self, di: Any, indent: str, ref: str = "") -> str:
        term_text = self._render_inlines(di.term, ref_prefix=f"{ref}/term" if ref else "")
        parts = [f"{indent}{term_text}"]
        for i, def_blocks in enumerate(di.definitions):
            for j, block in enumerate(def_blocks):
                child_ref = f"{ref}/definitions/{i}/{j}" if ref else ""
                rendered = self._render_block(block, indent="", ref=child_ref)
                # First line gets the `:   ` marker, continuation lines get 4-space indent
                lines = rendered.split("\n")
                lines[0] = f"{indent}:   {lines[0]}"
                for li in range(1, len(lines)):
                    if lines[li].strip():
                        lines[li] = f"{indent}    {lines[li]}"
                parts.append("\n".join(lines))
        return "\n".join(parts)

    def _render_table(self, table: Any, indent: str, ref: str = "") -> str:
        parts: list[str] = []
        col_specs = getattr(table, "col_specs", [])

        # Collect all rows
        header_rows: list[Any] = []
        body_rows: list[Any] = []

        head = getattr(table, "head", None)
        if head is not None:
            header_rows.extend(head.rows)

        bodies = getattr(table, "bodies", [])
        for section in bodies:
            body_rows.extend(section.rows)

        foot = getattr(table, "foot", None)
        if foot is not None:
            body_rows.extend(foot.rows)

        all_rows = header_rows + body_rows
        if not all_rows:
            return ""

        # Determine column count
        n_cols = max(len(row.cells) for row in all_rows) if all_rows else 0

        # Render header
        if header_rows:
            for row in header_rows:
                parts.append(self._render_table_row(row, n_cols, indent))
        else:
            # Synthesize empty header
            parts.append(f"{indent}|" + " |" * n_cols)

        # Separator row with alignment
        sep_cells: list[str] = []
        for ci in range(n_cols):
            alignment = _get_col_alignment(col_specs, ci)
            sep_cells.append(_alignment_marker(alignment))
        parts.append(f"{indent}|" + "|".join(sep_cells) + "|")

        # Body rows
        for row in body_rows:
            parts.append(self._render_table_row(row, n_cols, indent))

        result = "\n".join(parts)

        # Caption
        caption = getattr(table, "caption", None)
        if caption is not None and caption.body:
            cap_text = self._render_block_children(caption.body, indent="")
            result += f"\n\n{indent}*{cap_text.strip()}*"

        return result

    def _render_table_row(self, row: Any, n_cols: int, indent: str) -> str:
        cells: list[str] = []
        for ci in range(n_cols):
            if ci < len(row.cells):
                cell = row.cells[ci]
                if cell.content:
                    text = self._render_cell_content(cell.content)
                    cells.append(f" {text} ")
                else:
                    cells.append("  ")
            else:
                cells.append("  ")
        return f"{indent}|" + "|".join(cells) + "|"

    def _render_cell_content(self, blocks: Sequence[Any]) -> str:
        """Render block content for a GFM table cell (inline-only output).

        GFM cells only support inline content on a single line. Block-level
        structures are flattened: headings → bold, code blocks → inline code,
        lists → comma-separated, others → plain text.
        """
        from kaos_content.traversal.visitor import extract_text

        parts: list[str] = []
        for block in blocks:
            nt = block.node_type
            if nt == "paragraph":
                parts.append(self._render_inlines(block.children))
            elif nt == "heading":
                inner = self._render_inlines(block.children)
                parts.append(f"**{inner}**")
            elif nt == "codeblock":
                # Inline code, not fenced block
                value = block.value.replace("\n", " ")
                parts.append(f"`{value}`")
            elif nt in ("bullet_list", "ordered_list"):
                # Flatten list items to comma-separated
                item_texts: list[str] = []
                for item in block.children:
                    item_text = extract_text(item).strip()
                    if item_text:
                        item_texts.append(item_text)
                parts.append(", ".join(item_texts))
            elif nt == "blockquote":
                text = extract_text(block).strip()
                parts.append(text)
            else:
                # Fallback: extract plain text
                text = extract_text(block).strip()
                if text:
                    parts.append(text)

        result = " <br> ".join(parts) if len(parts) > 1 else (parts[0] if parts else "")
        # Escape pipes (they'd break the table row)
        result = result.replace("|", "\\|")
        # No newlines allowed in a GFM cell
        result = result.replace("\n", " ")
        return result

    def _render_code_block(self, cb: Any, indent: str) -> str:
        lang = getattr(cb, "language", None) or ""
        fence = _choose_code_fence(cb.value)
        value_lines = cb.value.split("\n")
        indented_value = "\n".join(f"{indent}{line}" for line in value_lines)
        return f"{indent}{fence}{lang}\n{indented_value}\n{indent}{fence}"

    def _render_figure(self, fig: Any, indent: str, ref: str = "") -> str:
        parts: list[str] = []
        parts.append(self._render_block_children(fig.children, indent, ref_prefix=ref))
        caption = getattr(fig, "caption", None)
        if caption is not None and caption.body:
            cap_text = self._render_block_children(
                caption.body, indent="", ref_prefix=f"{ref}/caption/body" if ref else ""
            )
            parts.append(f"{indent}*{cap_text.strip()}*")
        return "\n\n".join(p for p in parts if p)

    def _render_admonition(self, adm: Any, indent: str, ref: str = "") -> str:
        kind = getattr(adm, "kind", "note").upper()
        inner = self._render_block_children(adm.children, indent="", ref_prefix=ref)
        lines = inner.split("\n")
        result_lines = [f"{indent}> [!{kind}]"]
        for line in lines:
            if line.strip():
                result_lines.append(f"{indent}> {line}")
            else:
                result_lines.append(f"{indent}>")
        return "\n".join(result_lines)

    def _render_block_children(
        self, blocks: Sequence[Any], indent: str, ref_prefix: str = ""
    ) -> str:
        parts: list[str] = []
        for i, block in enumerate(blocks):
            child_ref = f"{ref_prefix}/children/{i}" if ref_prefix else ""
            parts.append(self._render_block(block, indent, ref=child_ref))
        return "\n\n".join(parts)

    # ── Inline rendering ──

    def _render_inlines(
        self, inlines: Sequence[Any], *, in_table_cell: bool = False, ref_prefix: str = ""
    ) -> str:
        return "".join(
            self._render_inline(
                i,
                in_table_cell=in_table_cell,
                ref=f"{ref_prefix}/children/{idx}" if ref_prefix else "",
            )
            for idx, i in enumerate(inlines)
        )

    def _render_inline(
        self,
        inline: Any,
        *,
        in_table_cell: bool = False,
        parent_emph: str = "",
        ref: str = "",
        prev_delim: str = "",
    ) -> str:
        # Redaction check on inline nodes
        if ref and self._is_redacted(ref):
            return "[REDACTED]"

        node_type = inline.node_type
        if node_type == "text":
            return _escape_text(inline.value, in_table_cell=in_table_cell)
        if node_type == "emphasis":
            return self._render_emphasis(
                inline,
                in_table_cell=in_table_cell,
                parent_emph=parent_emph,
                ref=ref,
                prev_delim=prev_delim,
            )
        if node_type == "strong":
            return self._render_strong(
                inline,
                in_table_cell=in_table_cell,
                parent_emph=parent_emph,
                ref=ref,
                prev_delim=prev_delim,
            )
        if node_type == "strikethrough":
            inner = self._render_inlines_with_context(
                inline.children,
                in_table_cell=in_table_cell,
                parent_emph=parent_emph,
                ref_prefix=ref,
            )
            return f"~~{inner}~~"
        if node_type == "code":
            return _render_inline_code(inline.value)
        if node_type == "link":
            inner = self._render_inlines_with_context(
                inline.children,
                in_table_cell=in_table_cell,
                parent_emph=parent_emph,
                ref_prefix=ref,
            )
            url = _escape_link_url(self._safe_url(inline.url))
            title_part = f' "{_escape_link_title(inline.title)}"' if inline.title else ""
            return f"[{inner}]({url}{title_part})"
        if node_type == "image":
            alt = _escape_link_alt(inline.alt or "")
            url = _escape_link_url(self._safe_url(inline.src))
            title_part = f' "{_escape_link_title(inline.title)}"' if inline.title else ""
            return f"![{alt}]({url}{title_part})"
        if node_type == "footnote_ref":
            return f"[^{inline.identifier}]"
        if node_type == "citation":
            return self._render_inlines_with_context(
                inline.children,
                in_table_cell=in_table_cell,
                parent_emph=parent_emph,
                ref_prefix=ref,
            )
        if node_type == "math":
            return f"${inline.value}$"
        if node_type == "raw_inline":
            # Gate raw inline content on the same allow_raw_html flag
            # that governs RawBlock — otherwise an attacker who can
            # construct a RawInline(format="html", value="<script>...")
            # bypasses the safe-default contract that the surrounding
            # serializer claims to enforce. ``markdown`` is treated the
            # same as ``html`` because it can carry inline HTML by spec.
            if inline.format in ("markdown", "html"):
                if self._allow_raw_html:
                    return inline.value
                return f"<!-- raw {inline.format} stripped -->"
            return ""
        if node_type == "line_break":
            return "\\\n"
        if node_type == "soft_break":
            return "\n"
        if node_type == "span":
            rev_cls = revision_class(inline)
            if rev_cls and should_skip_revision(rev_cls, self._view):
                return ""
            rendered = self._render_inlines_with_context(
                inline.children,
                in_table_cell=in_table_cell,
                parent_emph=parent_emph,
                ref_prefix=ref,
            )
            if rev_cls and self._view == "markup":
                return wrap_markdown_markup(rendered, rev_cls)
            return rendered
        if node_type == "superscript":
            inner = self._render_inlines_with_context(
                inline.children,
                in_table_cell=in_table_cell,
                parent_emph=parent_emph,
                ref_prefix=ref,
            )
            return f"<sup>{inner}</sup>"
        if node_type == "subscript":
            inner = self._render_inlines_with_context(
                inline.children,
                in_table_cell=in_table_cell,
                parent_emph=parent_emph,
                ref_prefix=ref,
            )
            return f"<sub>{inner}</sub>"
        if node_type == "underline":
            inner = self._render_inlines_with_context(
                inline.children,
                in_table_cell=in_table_cell,
                parent_emph=parent_emph,
                ref_prefix=ref,
            )
            return f"<u>{inner}</u>"
        return ""

    def _render_inlines_with_context(
        self,
        inlines: Sequence[Any],
        *,
        in_table_cell: bool = False,
        parent_emph: str = "",
        ref_prefix: str = "",
    ) -> str:
        # Track the *outermost* delimiter of the previously-emitted inline
        # so an adjacent emphasis/strong sibling using the same delimiter
        # can flip to its alternate. Without this, three adjacent
        # ``Emphasis(Text("0"))`` siblings serialise to ``*0**0**0*``,
        # which markdown-it re-tokenises as ``Emphasis(Text("0"),
        # Strong(Text("0")), Text("0"))`` — a structural diff on
        # round-trip. Switching the second to ``_`` (and back to ``*``
        # for the third) gives ``*0*_0_*0*`` which round-trips cleanly.
        parts: list[str] = []
        prev_delim = ""
        for idx, i in enumerate(inlines):
            rendered = self._render_inline(
                i,
                in_table_cell=in_table_cell,
                parent_emph=parent_emph,
                ref=f"{ref_prefix}/children/{idx}" if ref_prefix else "",
                prev_delim=prev_delim,
            )
            parts.append(rendered)
            prev_delim = _outer_emph_delim(i, rendered, parent_emph)
        return "".join(parts)

    def _render_emphasis(
        self,
        node: Any,
        *,
        in_table_cell: bool = False,
        parent_emph: str = "",
        ref: str = "",
        prev_delim: str = "",
    ) -> str:
        """Render emphasis, alternating delimiters when nested OR when an
        adjacent sibling used the same delimiter, and expelling whitespace."""
        # Default to *. Alternate to _ when:
        # - the parent emphasis already uses *, OR
        # - the immediately-previous inline sibling emitted * (which would
        #   merge with our opening delim into ** — see CommonMark flanker
        #   rules and audit M-roundtrip).
        delim = "_" if parent_emph == "*" or prev_delim == "*" else "*"

        inner = self._render_inlines_with_context(
            node.children, in_table_cell=in_table_cell, parent_emph=delim, ref_prefix=ref
        )

        # Expel leading/trailing whitespace outside the delimiters
        leading, inner, trailing = _expel_whitespace(inner)
        if not inner:
            return leading + trailing
        return f"{leading}{delim}{inner}{delim}{trailing}"

    def _render_strong(
        self,
        node: Any,
        *,
        in_table_cell: bool = False,
        parent_emph: str = "",
        ref: str = "",
        prev_delim: str = "",
    ) -> str:
        """Render strong, alternating delimiters when nested OR when an
        adjacent sibling used the same delimiter, and expelling whitespace."""
        # Default to **. Alternate to __ when:
        # - the parent emphasis used *, OR
        # - the previous sibling emitted ** (would merge into ****).
        delim = "__" if parent_emph == "*" or prev_delim == "**" else "**"

        inner = self._render_inlines_with_context(
            node.children, in_table_cell=in_table_cell, parent_emph=delim[0], ref_prefix=ref
        )

        leading, inner, trailing = _expel_whitespace(inner)
        if not inner:
            return leading + trailing
        return f"{leading}{delim}{inner}{delim}{trailing}"


# ── Helper functions ──


def _outer_emph_delim(node: Any, rendered: str, parent_emph: str) -> str:
    """Return the outermost markdown delimiter the rendered inline
    emitted, or ``""`` if it isn't an emphasis-like node.

    Used by ``_render_inlines_with_context`` to track adjacency and
    let the next sibling pick a non-clashing delimiter. Mirrors the
    delimiter-selection logic in ``_render_emphasis`` /
    ``_render_strong`` (must stay in sync).
    """
    node_type = getattr(node, "node_type", "")
    if node_type == "emphasis":
        return "_" if parent_emph == "*" else "*"
    if node_type == "strong":
        return "__" if parent_emph == "*" else "**"
    # For text and other inlines, the trailing character can also act
    # as a flanker. If the rendered text ends with ``*`` the next
    # emphasis sibling should still alternate to ``_``.
    if rendered.endswith("**"):
        return "**"
    if rendered.endswith("*"):
        return "*"
    return ""


def _expel_whitespace(text: str) -> tuple[str, str, str]:
    """Split leading and trailing whitespace from text.

    Returns (leading_ws, inner_text, trailing_ws).
    Emphasis delimiters must not be flanked by whitespace per CommonMark,
    so we move whitespace outside the delimiters.
    """
    stripped = text.strip()
    if not stripped:
        return (text, "", "")
    leading = text[: text.index(stripped[0])]
    trailing = text[text.rindex(stripped[-1]) + 1 :]
    return (leading, stripped, trailing)


def _render_inline_code(value: str) -> str:
    """Render inline code with proper backtick delimiter selection."""
    if "`" not in value:
        return f"`{value}`"
    max_run = _longest_backtick_run(value)
    delim = "`" * (max_run + 1)
    # If value starts or ends with backtick, add space padding
    if value.startswith("`") or value.endswith("`"):
        return f"{delim} {value} {delim}"
    return f"{delim}{value}{delim}"


def _escape_link_title(title: str) -> str:
    """Escape a link/image title for use inside double quotes."""
    return title.replace("\\", "\\\\").replace('"', '\\"')


def _escape_link_url(url: str) -> str:
    """Escape a URL for use in CommonMark inline link/image destinations.

    Per the CommonMark spec, an inline destination forbids unescaped
    ``<``, ``>``, control chars, line breaks, and unbalanced parens.
    Pre-Sec-2 (finding #1) this function only handled unbalanced parens
    by wrapping in ``<>``; that left a parens-balancing breakout open
    where a URL like ``"https://example.com) <script>...alert(1)..."``
    serialized to ``[text](https://example.com) <script>...``, the
    parser closed the link at the first ``)``, and ``<script>`` reached
    the rendered HTML.

    Fix: always backslash-escape ``\\``, ``(``, ``)``, ``<``, ``>`` and
    drop control chars + newlines. The result is unambiguously inside
    the destination — no breakout vector. The ``<>`` angle-bracket
    wrapper is unnecessary because every char that would require it
    is now escaped.

    NOTE: order matters — backslash MUST be escaped first or it would
    double-escape every subsequent escape token.
    """
    # Drop ASCII control chars (0x00..0x1f), DEL (0x7f), and ALL ASCII
    # whitespace. Per RFC 3986 a URL cannot contain literal whitespace;
    # it must be percent-encoded by the producer. CommonMark also ends
    # the inline destination at the first whitespace char, so leaving
    # spaces in is the same parens-balancing breakout vector in
    # disguise: ``"https://x.com/foo) <script>...(`` would become
    # ``[text](https://x.com/foo\)`` followed by `` <script>...`` as
    # plain text — escaping the parens isn't enough on its own.
    safe = "".join(c for c in url if c > " " and c != "\x7f")
    return (
        safe.replace("\\", "\\\\")
        .replace("(", "\\(")
        .replace(")", "\\)")
        .replace("<", "\\<")
        .replace(">", "\\>")
    )


def _escape_link_alt(alt: str) -> str:
    """Escape image alt-text for use inside ``![alt](url)`` syntax.

    Per the CommonMark spec, alt text is parsed as link text and
    forbids unescaped ``[`` and ``]`` (which would close the alt
    bracket prematurely) and ``\\`` (which is the escape introducer).
    Pre-Sec-2 (finding #1) ``Image.alt`` was emitted unescaped at the
    image render site, allowing a payload like ``"](x) <script>..."``
    to break out of the alt syntax: the result rendered as raw HTML.

    Fix: backslash-escape ``\\``, ``[``, ``]``. Order matters — the
    backslash escape must come first.
    """
    return alt.replace("\\", "\\\\").replace("[", "\\[").replace("]", "\\]")


def _get_col_alignment(col_specs: Sequence[Any], index: int) -> Any:
    if index < len(col_specs):
        return getattr(col_specs[index], "alignment", None)
    return None


def _longest_backtick_run(text: str) -> int:
    """Return the length of the longest consecutive run of backticks in text."""
    max_run = 0
    current_run = 0
    for ch in text:
        if ch == "`":
            current_run += 1
            if current_run > max_run:
                max_run = current_run
        else:
            current_run = 0
    return max_run


def _choose_code_fence(content: str) -> str:
    """Choose a backtick fence that won't be confused with content."""
    max_run = _longest_backtick_run(content)
    fence_len = max(3, max_run + 1)
    return "`" * fence_len


def _alignment_marker(alignment: Any) -> str:
    if alignment is None:
        return " --- "
    from kaos_content.model.attr import Alignment

    if alignment == Alignment.LEFT:
        return " :--- "
    if alignment == Alignment.RIGHT:
        return " ---: "
    if alignment == Alignment.CENTER:
        return " :---: "
    return " --- "
