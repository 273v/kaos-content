"""Serialize a ContentDocument AST to semantic HTML5.

Provenance is encoded as ``data-*`` attributes. Annotations are rendered
as ``<mark>``, ``<span class="...">``, or ``[REDACTED]`` depending on type.

Security
--------

By default the serializer is XSS-safe:

- Raw HTML blocks (``raw_block`` / ``raw_inline`` with format ``"html"``)
  are dropped and replaced with an HTML comment marker. Set
  ``allow_raw_html=True`` to opt back in (caller asserts the AST is
  trusted).
- ``<a href>`` and ``<img src>`` URLs whose canonical scheme is in
  :data:`kaos_content._security.UNSAFE_SCHEMES`
  (``javascript``, ``data``, ``vbscript``, ``file``) are replaced with
  ``#`` and the original is captured in a ``data-unsafe-url``
  attribute for forensics. Setting ``allow_raw_html=True`` *also*
  disables the URL filter (caller takes full responsibility).
"""

from __future__ import annotations

import html as html_lib
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from kaos_core.security import is_safe_url

from kaos_content.serializers._revision import (
    ViewMode,
    revision_class,
    should_skip_revision,
    wrap_html_markup,
)

if TYPE_CHECKING:
    from kaos_content.model.document import ContentDocument


def serialize_html(
    document: ContentDocument,
    *,
    include_provenance: bool = True,
    view: ViewMode = "final",
    allow_raw_html: bool = False,
) -> str:
    """Serialize a ContentDocument to an HTML5 string.

    Parameters
    ----------
    document:
        The document to serialize.
    include_provenance:
        If True (default), emit provenance as ``data-*`` attributes.
    view:
        Tracked-changes view mode. ``"final"`` (default) renders the
        accepted version (backward compatible). ``"original"`` renders
        the pre-change version. ``"markup"`` shows both with semantic
        ``<ins>`` / ``<del>`` elements.
    allow_raw_html:
        If False (default), raw HTML blocks/inlines are stripped and
        ``<a href>``/``<img src>`` URLs with unsafe schemes are
        replaced with ``#``. If True, the caller asserts that the
        document AST is trusted; raw HTML passes through verbatim and
        unsafe URLs are emitted as-is. **Set this to True only when
        serializing AST that you control end-to-end.** This flag
        defaults to False because the typical caller is rendering
        user-supplied content into an HTML page.
    """
    ctx = _HtmlContext(
        document,
        include_provenance=include_provenance,
        view=view,
        allow_raw_html=allow_raw_html,
    )
    return ctx.serialize()


class _HtmlContext:
    """Maintains state during HTML serialization."""

    def __init__(
        self,
        document: ContentDocument,
        *,
        include_provenance: bool,
        view: ViewMode = "final",
        allow_raw_html: bool = False,
    ) -> None:
        self._document = document
        self._include_provenance = include_provenance
        self._view = view
        self._allow_raw_html = allow_raw_html
        self._redacted_refs: set[str] = set()
        self._build_redaction_set()

    def _build_redaction_set(self) -> None:
        from kaos_content.model.annotation import AnnotationType

        for ann in self._document.annotations:
            if ann.type == AnnotationType.REDACTION:
                for target in ann.targets:
                    self._redacted_refs.add(target.node_ref)

    def _is_redacted(self, ref: str) -> bool:
        return ref in self._redacted_refs

    def _prov_attrs(self, node: Any) -> str:
        """Build data-* attribute string from provenance."""
        if not self._include_provenance:
            return ""
        prov = getattr(node, "provenance", None)
        if prov is None:
            return ""
        attrs: list[str] = []
        if prov.page is not None:
            attrs.append(f'data-page="{prov.page}"')
        if prov.bbox is not None:
            b = prov.bbox
            attrs.append(f'data-bbox="{b.left},{b.top},{b.right},{b.bottom}"')
        if prov.confidence is not None:
            attrs.append(f'data-confidence="{prov.confidence}"')
        if prov.extractor is not None:
            attrs.append(f'data-extractor="{html_lib.escape(prov.extractor)}"')
        if prov.char_span is not None:
            attrs.append(f'data-char-span="{prov.char_span[0]},{prov.char_span[1]}"')
        return " " + " ".join(attrs) if attrs else ""

    def serialize(self) -> str:
        parts: list[str] = []
        for i, block in enumerate(self._document.body):
            ref = f"#/body/{i}"
            parts.append(self._render_block(block, ref=ref))

        # Footnotes as an ordered list at the end
        if self._document.footnotes:
            fn_parts: list[str] = []
            for key, blocks in self._document.footnotes.items():
                fn_body = "\n".join(
                    self._render_block(b, ref=f"#/footnotes/{key}/{fi}")
                    for fi, b in enumerate(blocks)
                )
                fn_parts.append(f'<li id="fn-{html_lib.escape(key)}">{fn_body}</li>')
            parts.append(
                '<section class="footnotes"><ol>' + "\n".join(fn_parts) + "</ol></section>"
            )

        return "\n".join(parts)

    # ── Block rendering ──

    def _render_block(self, block: Any, ref: str = "") -> str:
        if ref and self._is_redacted(ref):
            return '<p class="redacted">[REDACTED]</p>'

        nt = block.node_type
        prov = self._prov_attrs(block)

        if nt == "paragraph":
            inner = self._render_inlines(block.children, ref_prefix=ref)
            return f"<p{prov}>{inner}</p>"

        if nt == "heading":
            d = block.depth
            inner = self._render_inlines(block.children, ref_prefix=ref)
            return f"<h{d}{prov}>{inner}</h{d}>"

        if nt == "blockquote":
            inner = self._render_block_children(block.children, ref)
            return f"<blockquote{prov}>\n{inner}\n</blockquote>"

        if nt == "bullet_list":
            items = self._render_list_items(block.children, ref)
            return f"<ul{prov}>\n{items}\n</ul>"

        if nt == "ordered_list":
            start = getattr(block, "start", 1)
            start_attr = f' start="{start}"' if start != 1 else ""
            items = self._render_list_items(block.children, ref)
            return f"<ol{start_attr}{prov}>\n{items}\n</ol>"

        if nt == "list_item":
            return self._render_list_item(block, ref)

        if nt == "definition_list":
            return self._render_definition_list(block, ref, prov)

        if nt == "table":
            return self._render_table(block, ref, prov)

        if nt == "codeblock":
            lang = getattr(block, "language", None)
            code = html_lib.escape(block.value)
            lang_class = f' class="language-{html_lib.escape(lang)}"' if lang else ""
            return f"<pre{prov}><code{lang_class}>{code}</code></pre>"

        if nt == "thematic_break":
            return f"<hr{prov} />"

        if nt == "figure":
            return self._render_figure(block, ref, prov)

        if nt == "page_break":
            return f'<hr class="page-break"{prov} />'

        if nt == "div":
            rev_cls = revision_class(block)
            if rev_cls and should_skip_revision(rev_cls, self._view):
                return ""
            inner = self._render_block_children(block.children, ref)
            rendered = f"<div{prov}>\n{inner}\n</div>"
            if rev_cls and self._view == "markup":
                return wrap_html_markup(rendered, rev_cls)
            return rendered

        if nt == "raw_block":
            if block.format == "html":
                if self._allow_raw_html:
                    return block.value
                # XSS-safe default: refuse to emit untrusted raw HTML.
                # The HTML comment is harmless even if the surrounding
                # context is `<script>` or an HTML attribute (the
                # comment would just be visible).
                return "<!-- raw HTML stripped (set allow_raw_html=True to emit) -->"
            return f"<!-- raw:{block.format} -->"

        if nt == "math_block":
            return f'<div class="math-display"{prov}>$$\n{html_lib.escape(block.value)}\n$$</div>'

        if nt == "admonition":
            kind = getattr(block, "kind", "note")
            inner = self._render_block_children(block.children, ref)
            return (
                f'<div class="admonition admonition-{html_lib.escape(kind)}"{prov}>\n'
                f"{inner}\n"
                "</div>"
            )

        return ""

    def _render_block_children(self, blocks: Sequence[Any], ref_prefix: str) -> str:
        parts: list[str] = []
        for i, block in enumerate(blocks):
            child_ref = f"{ref_prefix}/children/{i}" if ref_prefix else ""
            parts.append(self._render_block(block, ref=child_ref))
        return "\n".join(parts)

    def _render_list_items(self, items: Sequence[Any], ref_prefix: str) -> str:
        parts: list[str] = []
        for i, item in enumerate(items):
            child_ref = f"{ref_prefix}/children/{i}" if ref_prefix else ""
            parts.append(self._render_list_item(item, child_ref))
        return "\n".join(parts)

    def _render_list_item(self, item: Any, ref: str) -> str:
        if ref and self._is_redacted(ref):
            return '<li class="redacted">[REDACTED]</li>'

        prov = self._prov_attrs(item)
        checked = getattr(item, "checked", None)

        inner = self._render_block_children(item.children, ref)

        if checked is True:
            return (
                f'<li class="task-list-item"{prov}>'
                f'<input type="checkbox" checked disabled /> {inner}</li>'
            )
        if checked is False:
            return (
                f'<li class="task-list-item"{prov}><input type="checkbox" disabled /> {inner}</li>'
            )
        return f"<li{prov}>{inner}</li>"

    def _render_definition_list(self, dl: Any, ref: str, prov: str) -> str:
        parts: list[str] = [f"<dl{prov}>"]
        for i, item in enumerate(dl.children):
            child_ref = f"{ref}/children/{i}" if ref else ""
            term_inner = self._render_inlines(
                item.term, ref_prefix=f"{child_ref}/term" if child_ref else ""
            )
            parts.append(f"<dt>{term_inner}</dt>")
            for di, def_blocks in enumerate(item.definitions):
                dd_inner = "\n".join(
                    self._render_block(
                        b, ref=f"{child_ref}/definitions/{di}/{bi}" if child_ref else ""
                    )
                    for bi, b in enumerate(def_blocks)
                )
                parts.append(f"<dd>{dd_inner}</dd>")
        parts.append("</dl>")
        return "\n".join(parts)

    def _render_table(self, table: Any, ref: str, prov: str) -> str:
        parts: list[str] = [f"<table{prov}>"]

        # Caption
        caption = getattr(table, "caption", None)
        if caption is not None and caption.body:
            cap_inner = self._render_block_children(
                caption.body, f"{ref}/caption/body" if ref else ""
            )
            parts.append(f"<caption>{cap_inner}</caption>")

        # Head
        head = getattr(table, "head", None)
        if head is not None and head.rows:
            parts.append("<thead>")
            for row in head.rows:
                parts.append(self._render_table_row(row, is_header=True))
            parts.append("</thead>")

        # Body
        bodies = getattr(table, "bodies", ())
        for body_section in bodies:
            parts.append("<tbody>")
            for row in body_section.rows:
                parts.append(self._render_table_row(row, is_header=False))
            parts.append("</tbody>")

        # Foot
        foot = getattr(table, "foot", None)
        if foot is not None and foot.rows:
            parts.append("<tfoot>")
            for row in foot.rows:
                parts.append(self._render_table_row(row, is_header=False))
            parts.append("</tfoot>")

        parts.append("</table>")
        return "\n".join(parts)

    def _render_table_row(self, row: Any, *, is_header: bool) -> str:
        tag = "th" if is_header else "td"
        cells: list[str] = []
        for cell in row.cells:
            attrs: list[str] = []
            if cell.row_span != 1:
                attrs.append(f'rowspan="{cell.row_span}"')
            if cell.col_span != 1:
                attrs.append(f'colspan="{cell.col_span}"')
            if cell.alignment is not None:
                from kaos_content.model.attr import Alignment

                align_map = {
                    Alignment.LEFT: "left",
                    Alignment.CENTER: "center",
                    Alignment.RIGHT: "right",
                }
                if cell.alignment in align_map:
                    attrs.append(f'style="text-align:{align_map[cell.alignment]}"')

            attr_str = " " + " ".join(attrs) if attrs else ""
            if cell.content:
                inner = "\n".join(self._render_block(b) for b in cell.content)
                cells.append(f"<{tag}{attr_str}>{inner}</{tag}>")
            else:
                cells.append(f"<{tag}{attr_str}></{tag}>")
        return "<tr>" + "".join(cells) + "</tr>"

    def _render_figure(self, fig: Any, ref: str, prov: str) -> str:
        parts: list[str] = [f"<figure{prov}>"]
        parts.append(self._render_block_children(fig.children, ref))
        caption = getattr(fig, "caption", None)
        if caption is not None and caption.body:
            cap_inner = self._render_block_children(
                caption.body, f"{ref}/caption/body" if ref else ""
            )
            parts.append(f"<figcaption>{cap_inner}</figcaption>")
        parts.append("</figure>")
        return "\n".join(parts)

    # ── Inline rendering ──

    def _render_inlines(self, inlines: Sequence[Any], ref_prefix: str = "") -> str:
        return "".join(
            self._render_inline(i, ref=f"{ref_prefix}/children/{idx}" if ref_prefix else "")
            for idx, i in enumerate(inlines)
        )

    def _safe_url_attr(self, url: str) -> tuple[str, str]:
        """Return ``(href_value, extra_attr)`` for a URL.

        When the canonicalised scheme is in ``UNSAFE_SCHEMES`` and
        ``allow_raw_html`` is False, the href is replaced with ``"#"``
        and the original (HTML-escaped) URL is captured in a
        ``data-unsafe-url`` attribute for forensics. Otherwise the URL
        is HTML-attribute-escaped and returned unchanged with an
        empty extra-attr string.

        Two-return-value shape lets callers compose the attribute into
        an existing string template without conditional logic at every
        call site.
        """
        if self._allow_raw_html or is_safe_url(url):
            return html_lib.escape(url, quote=True), ""
        original = html_lib.escape(url, quote=True)
        return "#", f' data-unsafe-url="{original}"'

    def _render_inline(self, inline: Any, ref: str = "") -> str:
        if ref and self._is_redacted(ref):
            return '<span class="redacted">[REDACTED]</span>'

        nt = inline.node_type

        if nt == "text":
            return html_lib.escape(inline.value)

        if nt == "emphasis":
            inner = self._render_inlines(inline.children, ref_prefix=ref)
            return f"<em>{inner}</em>"

        if nt == "strong":
            inner = self._render_inlines(inline.children, ref_prefix=ref)
            return f"<strong>{inner}</strong>"

        if nt == "strikethrough":
            inner = self._render_inlines(inline.children, ref_prefix=ref)
            return f"<del>{inner}</del>"

        if nt == "code":
            return f"<code>{html_lib.escape(inline.value)}</code>"

        if nt == "link":
            inner = self._render_inlines(inline.children, ref_prefix=ref)
            href, unsafe = self._safe_url_attr(inline.url)
            title = f' title="{html_lib.escape(inline.title, quote=True)}"' if inline.title else ""
            return f'<a href="{href}"{unsafe}{title}>{inner}</a>'

        if nt == "image":
            alt = html_lib.escape(inline.alt or "", quote=True)
            src, unsafe = self._safe_url_attr(inline.src)
            title = f' title="{html_lib.escape(inline.title, quote=True)}"' if inline.title else ""
            return f'<img src="{src}"{unsafe} alt="{alt}"{title} />'

        if nt == "footnote_ref":
            ident = html_lib.escape(inline.identifier)
            return f'<sup><a href="#fn-{ident}">[{ident}]</a></sup>'

        if nt == "citation":
            inner = self._render_inlines(inline.children, ref_prefix=ref)
            return f"<cite>{inner}</cite>"

        if nt == "math":
            return f'<span class="math-inline">${html_lib.escape(inline.value)}$</span>'

        if nt == "raw_inline":
            if inline.format == "html":
                if self._allow_raw_html:
                    return inline.value
                return "<!-- raw HTML stripped -->"
            return ""

        if nt == "line_break":
            return "<br />"

        if nt == "soft_break":
            return "\n"

        if nt == "span":
            rev_cls = revision_class(inline)
            if rev_cls and should_skip_revision(rev_cls, self._view):
                return ""
            inner = self._render_inlines(inline.children, ref_prefix=ref)
            if rev_cls and self._view == "markup":
                return wrap_html_markup(inner, rev_cls)
            return f"<span>{inner}</span>"

        if nt == "superscript":
            inner = self._render_inlines(inline.children, ref_prefix=ref)
            return f"<sup>{inner}</sup>"

        if nt == "subscript":
            inner = self._render_inlines(inline.children, ref_prefix=ref)
            return f"<sub>{inner}</sub>"

        if nt == "underline":
            inner = self._render_inlines(inline.children, ref_prefix=ref)
            return f"<u>{inner}</u>"

        return ""
