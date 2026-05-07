"""Parse CommonMark + GFM markdown into a ContentDocument AST.

Uses markdown-it-py as the tokenizer. Supports:
- CommonMark block/inline elements
- GFM tables (with column alignment)
- GFM strikethrough
- Footnotes (via mdit-py-plugins)
- Definition lists (via mdit-py-plugins)
- Dollar-sign math (inline and display, via mdit-py-plugins)
- YAML front matter (via mdit-py-plugins)
- GitHub-style admonitions (> [!NOTE], > [!WARNING], etc.)
- Task lists ([x] / [ ] in list items)
- HTML inline tags: <sup>, <sub>, <u>
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

from markdown_it import MarkdownIt
from markdown_it.token import Token
from mdit_py_plugins.deflist import deflist_plugin
from mdit_py_plugins.dollarmath import dollarmath_plugin
from mdit_py_plugins.footnote import footnote_plugin
from mdit_py_plugins.front_matter import front_matter_plugin

from kaos_content.model.annotation import Annotation
from kaos_content.model.attr import Alignment, ColSpec, Provenance, SourceRef
from kaos_content.model.blocks import (
    Admonition,
    BlockQuote,
    BulletList,
    CodeBlock,
    DefinitionItem,
    DefinitionList,
    Heading,
    ListItem,
    MathBlock,
    OrderedList,
    Paragraph,
    RawBlock,
    ThematicBreak,
)
from kaos_content.model.document import ContentDocument
from kaos_content.model.inlines import (
    Code,
    Emphasis,
    FootnoteRef,
    Image,
    LineBreak,
    Link,
    Math,
    RawInline,
    SoftBreak,
    Strikethrough,
    Strong,
    Subscript,
    Superscript,
    Text,
    Underline,
)
from kaos_content.model.metadata import DocumentMetadata
from kaos_content.model.table import Cell, Row, TableSection

if TYPE_CHECKING:
    from kaos_content.model.blocks import Block, Table
    from kaos_content.model.inlines import Inline


# GitHub admonition pattern: [!KIND] at the start of a blockquote paragraph
_ADMONITION_RE = re.compile(r"^\[!([A-Z_]+)\]$")

# HTML inline tag patterns for sup/sub/u
_HTML_TAG_RE = re.compile(r"^<(/?)(sup|sub|u)>$", re.IGNORECASE)


def parse_markdown(
    text: str,
    *,
    source: SourceRef | None = None,
) -> ContentDocument:
    """Parse markdown text into a ContentDocument.

    If *source* is provided, all nodes receive Provenance with ``char_span``
    offsets derived from token source-map positions.

    Requires the ``markdown`` optional dependency (``markdown-it-py[plugins]``).
    """
    md = _make_parser()
    tokens = md.parse(text)

    ctx = _ParseContext(text=text, source=source)
    body = ctx.parse_blocks(tokens)

    metadata = _extract_frontmatter(tokens)
    # When a SourceRef is provided, propagate it to ``metadata.source`` so
    # multi-document corpora (``ContentDocumentCorpus``,
    # ``Corpus.from_documents``) thread ``doc_uri`` without an explicit
    # ``doc_uris`` kwarg. Mirrors ``parse_plain_text`` and the
    # ``extract_pdf`` / ``parse_docx`` dual set_source+set_metadata pattern.
    if source is not None and metadata.source is None:
        metadata = metadata.model_copy(update={"source": source})

    return ContentDocument(
        metadata=metadata,
        body=tuple(body),
        footnotes=ctx.footnotes,
        annotations=tuple(ctx.annotations),
    )


# ── Parser factory ──


def _make_parser() -> MarkdownIt:
    """Create a markdown-it-py instance with GFM + extensions."""
    md = MarkdownIt("commonmark", {"typographer": False})
    md.enable(["table", "strikethrough"])
    footnote_plugin(md)
    front_matter_plugin(md)
    deflist_plugin(md)
    dollarmath_plugin(md)
    return md


# ── Front matter ──


def _extract_frontmatter(tokens: list[Token]) -> DocumentMetadata:
    """Extract YAML front matter into DocumentMetadata."""
    for token in tokens:
        if token.type == "front_matter":
            return _parse_yaml_frontmatter(token.content)
    return DocumentMetadata()


def _parse_yaml_frontmatter(content: str) -> DocumentMetadata:
    """Best-effort parse of YAML front matter into DocumentMetadata."""
    fields: dict[str, Any] = {}
    for line in content.strip().splitlines():
        if ":" in line:
            key, _, value = line.partition(":")
            key = key.strip().lower()
            value = value.strip()
            if not value:
                continue
            if key == "title":
                fields["title"] = value
            elif key == "author":
                fields["authors"] = (value,)
            elif key == "authors":
                # Simple comma-separated
                fields["authors"] = tuple(a.strip() for a in value.split(",") if a.strip())
            elif key == "date":
                fields["date"] = value
            elif key == "language" or key == "lang":
                fields["language"] = value
            elif key == "type" or key == "document_type":
                fields["document_type"] = value
    return DocumentMetadata(**fields)


# ── Parse context ──


class _ParseContext:
    """Stateful context for parsing a token stream into AST nodes."""

    def __init__(self, text: str, source: SourceRef | None) -> None:
        self._text = text
        self._source = source
        self._lines: list[str] | None = None
        self.footnotes: dict[str, tuple[Block, ...]] = {}
        self.annotations: list[Annotation] = []

    def _line_offsets(self) -> list[int]:
        """Compute cumulative character offset for each line (lazily)."""
        if self._lines is None:
            self._lines = self._text.split("\n")
        offsets = [0]
        for line in self._lines:
            offsets.append(offsets[-1] + len(line) + 1)  # +1 for '\n'
        return offsets

    def _make_provenance(self, token: Token) -> Provenance | None:
        """Build provenance from a token's source map, if available."""
        if self._source is None or token.map is None:
            return None
        line_offsets = self._line_offsets()
        start_line, end_line = token.map
        start_char = line_offsets[start_line] if start_line < len(line_offsets) else 0
        end_char = line_offsets[end_line] if end_line < len(line_offsets) else len(self._text)
        return Provenance(
            source=self._source,
            char_span=(start_char, end_char),
        )

    # ── Block-level parsing ──

    def parse_blocks(self, tokens: list[Token]) -> list[Block]:
        """Parse a flat token list into a list of Block AST nodes."""
        blocks: list[Block] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            tt = token.type

            # Skip front_matter (handled separately)
            if tt == "front_matter":
                i += 1
                continue

            # Heading
            if tt == "heading_open":
                block, i = self._parse_heading(tokens, i)
                blocks.append(block)
                continue

            # Paragraph
            if tt == "paragraph_open":
                block, i = self._parse_paragraph(tokens, i)
                blocks.append(block)
                continue

            # Blockquote (may become Admonition)
            if tt == "blockquote_open":
                block, i = self._parse_blockquote(tokens, i)
                blocks.append(block)
                continue

            # Bullet list
            if tt == "bullet_list_open":
                block, i = self._parse_bullet_list(tokens, i)
                blocks.append(block)
                continue

            # Ordered list
            if tt == "ordered_list_open":
                block, i = self._parse_ordered_list(tokens, i)
                blocks.append(block)
                continue

            # Fenced code block
            if tt == "fence":
                prov = self._make_provenance(token)
                lang = token.info.strip() or None
                blocks.append(
                    CodeBlock(language=lang, value=token.content.rstrip("\n"), provenance=prov)
                )
                i += 1
                continue

            # Code block (indented)
            if tt == "code_block":
                prov = self._make_provenance(token)
                blocks.append(CodeBlock(value=token.content.rstrip("\n"), provenance=prov))
                i += 1
                continue

            # Thematic break
            if tt == "hr":
                prov = self._make_provenance(token)
                blocks.append(ThematicBreak(provenance=prov))
                i += 1
                continue

            # Table
            if tt == "table_open":
                block, i = self._parse_table(tokens, i)
                blocks.append(block)
                continue

            # Definition list
            if tt == "dl_open":
                block, i = self._parse_definition_list(tokens, i)
                blocks.append(block)
                continue

            # Math block (display)
            if tt == "math_block":
                prov = self._make_provenance(token)
                value = token.content.strip()
                blocks.append(MathBlock(value=value, provenance=prov))
                i += 1
                continue

            # HTML block
            if tt == "html_block":
                prov = self._make_provenance(token)
                blocks.append(
                    RawBlock(format="html", value=token.content.rstrip("\n"), provenance=prov)
                )
                i += 1
                continue

            # Footnote block (contains footnote definitions)
            if tt == "footnote_block_open":
                i = self._parse_footnote_block(tokens, i)
                continue

            # Skip tokens we don't handle
            i += 1

        return blocks

    def _parse_heading(self, tokens: list[Token], start: int) -> tuple[Heading, int]:
        """Parse heading_open .. inline .. heading_close."""
        open_tok = tokens[start]
        depth = int(open_tok.tag[1])  # h1 → 1, h2 → 2, etc.
        prov = self._make_provenance(open_tok)

        inlines: list[Inline] = []
        i = start + 1
        while i < len(tokens) and tokens[i].type != "heading_close":
            if tokens[i].type == "inline":
                inlines = self._parse_inlines(tokens[i].children or [])
            i += 1
        i += 1  # skip heading_close

        return Heading(depth=depth, children=tuple(inlines), provenance=prov), i

    def _parse_paragraph(self, tokens: list[Token], start: int) -> tuple[Paragraph, int]:
        """Parse paragraph_open .. inline .. paragraph_close."""
        open_tok = tokens[start]
        prov = self._make_provenance(open_tok)

        inlines: list[Inline] = []
        i = start + 1
        while i < len(tokens) and tokens[i].type != "paragraph_close":
            if tokens[i].type == "inline":
                inlines = self._parse_inlines(tokens[i].children or [])
            i += 1
        i += 1  # skip paragraph_close

        return Paragraph(children=tuple(inlines), provenance=prov), i

    def _parse_blockquote(
        self, tokens: list[Token], start: int
    ) -> tuple[BlockQuote | Admonition, int]:
        """Parse blockquote_open .. children .. blockquote_close.

        Detects GitHub-style admonitions (> [!NOTE]) and returns an Admonition node.
        """
        open_tok = tokens[start]
        prov = self._make_provenance(open_tok)

        # Collect child tokens
        i = start + 1
        depth = 1
        child_tokens: list[Token] = []
        while i < len(tokens) and depth > 0:
            if tokens[i].type == "blockquote_open":
                depth += 1
            elif tokens[i].type == "blockquote_close":
                depth -= 1
                if depth == 0:
                    break
            child_tokens.append(tokens[i])
            i += 1
        i += 1  # skip blockquote_close

        children = self.parse_blocks(child_tokens)

        # Check for admonition pattern: first child is a paragraph starting with [!KIND]
        adm = self._try_extract_admonition(children, prov)
        if adm is not None:
            return adm, i

        return BlockQuote(children=tuple(children), provenance=prov), i

    def _try_extract_admonition(
        self, children: list[Block], prov: Provenance | None
    ) -> Admonition | None:
        """Check if blockquote children match the GitHub admonition pattern."""
        if not children:
            return None
        first = children[0]
        if not isinstance(first, Paragraph):
            return None

        first_inlines = first.children
        if not first_inlines:
            return None

        # The first inline must be a Text node matching [!KIND]
        first_inline = first_inlines[0]
        if not isinstance(first_inline, Text):
            return None

        m = _ADMONITION_RE.match(first_inline.value)
        if m is None:
            return None

        kind = m.group(1).lower()

        # Remaining inlines from the first paragraph (after the [!KIND] marker)
        # become the admonition's first paragraph content (skip leading soft breaks)
        rest_inlines = list(first_inlines[1:])
        while rest_inlines and rest_inlines[0].node_type == "soft_break":
            rest_inlines = rest_inlines[1:]

        adm_children: list[Block] = []
        if rest_inlines:
            adm_children.append(Paragraph(children=tuple(rest_inlines)))
        adm_children.extend(children[1:])

        return Admonition(kind=kind, children=tuple(adm_children), provenance=prov)

    def _parse_bullet_list(self, tokens: list[Token], start: int) -> tuple[BulletList, int]:
        """Parse bullet_list_open .. list_item* .. bullet_list_close."""
        open_tok = tokens[start]
        prov = self._make_provenance(open_tok)

        items: list[ListItem] = []
        i = start + 1
        while i < len(tokens) and tokens[i].type != "bullet_list_close":
            if tokens[i].type == "list_item_open":
                item, i = self._parse_list_item(tokens, i)
                items.append(item)
            else:
                i += 1

        i += 1  # skip bullet_list_close
        return BulletList(children=tuple(items), provenance=prov), i

    def _parse_ordered_list(self, tokens: list[Token], start: int) -> tuple[OrderedList, int]:
        """Parse ordered_list_open .. list_item* .. ordered_list_close."""
        open_tok = tokens[start]
        prov = self._make_provenance(open_tok)
        list_start = 1
        if open_tok.attrs and "start" in open_tok.attrs:
            list_start = int(open_tok.attrs["start"])

        items: list[ListItem] = []
        i = start + 1
        while i < len(tokens) and tokens[i].type != "ordered_list_close":
            if tokens[i].type == "list_item_open":
                item, i = self._parse_list_item(tokens, i)
                items.append(item)
            else:
                i += 1

        i += 1  # skip ordered_list_close
        return OrderedList(start=list_start, children=tuple(items), provenance=prov), i

    def _parse_list_item(self, tokens: list[Token], start: int) -> tuple[ListItem, int]:
        """Parse list_item_open .. children .. list_item_close.

        Detects task list markers ([x] / [ ]) in the first paragraph.
        """
        open_tok = tokens[start]
        prov = self._make_provenance(open_tok)

        # Collect child tokens
        i = start + 1
        depth = 1
        child_tokens: list[Token] = []
        while i < len(tokens) and depth > 0:
            if tokens[i].type == "list_item_open":
                depth += 1
            elif tokens[i].type == "list_item_close":
                depth -= 1
                if depth == 0:
                    break
            child_tokens.append(tokens[i])
            i += 1
        i += 1  # skip list_item_close

        children = self.parse_blocks(child_tokens)

        # Detect task list markers
        checked = self._extract_task_marker(children)

        if checked is not None:
            # Mutate the first paragraph to strip the [x]/[ ] prefix
            children = self._strip_task_marker(children)

        return ListItem(children=tuple(children), checked=checked, provenance=prov), i

    def _extract_task_marker(self, children: list[Block]) -> bool | None:
        """Detect [x] or [ ] at the start of a list item's first paragraph."""
        if not children:
            return None
        first_block = children[0]
        if not isinstance(first_block, Paragraph) or not first_block.children:
            return None

        first = first_block.children[0]
        if not isinstance(first, Text):
            return None

        if first.value.startswith("[x] ") or first.value == "[x]":
            return True
        if first.value.startswith("[ ] ") or first.value == "[ ]":
            return False
        return None

    def _strip_task_marker(self, children: list[Block]) -> list[Block]:
        """Strip the [x]/[ ] prefix from the first paragraph's first text node."""
        if not children:
            return children
        first_block = children[0]
        if not isinstance(first_block, Paragraph) or not first_block.children:
            return children

        first = first_block.children[0]
        if not isinstance(first, Text):
            return children

        if first.value.startswith("[x] ") or first.value.startswith("[ ] "):
            new_text = first.value[4:]
        elif first.value in ("[x]", "[ ]"):
            new_text = ""
        else:
            return children

        if new_text:
            new_first = Text(value=new_text)
            new_para = Paragraph(
                children=(new_first, *first_block.children[1:]),
                provenance=first_block.provenance,
            )
        elif len(first_block.children) > 1:
            new_para = Paragraph(
                children=first_block.children[1:], provenance=first_block.provenance
            )
        else:
            new_para = Paragraph(children=(), provenance=first_block.provenance)

        return [new_para, *children[1:]]

    def _parse_table(self, tokens: list[Token], start: int) -> tuple[Table, int]:
        """Parse table_open .. (thead, tbody) .. table_close."""
        from kaos_content.model.blocks import Table as TableNode

        open_tok = tokens[start]
        prov = self._make_provenance(open_tok)

        head: TableSection | None = None
        bodies: list[TableSection] = []
        col_specs: list[ColSpec] = []
        alignments: list[Alignment | None] = []

        i = start + 1
        while i < len(tokens) and tokens[i].type != "table_close":
            tt = tokens[i].type

            if tt == "thead_open":
                section, i, aligns = self._parse_table_section(
                    tokens, i, "thead_close", is_head=True
                )
                head = section
                alignments = aligns
                continue

            if tt == "tbody_open":
                section, i, _ = self._parse_table_section(tokens, i, "tbody_close")
                bodies.append(section)
                continue

            i += 1

        i += 1  # skip table_close

        # Build col_specs from detected alignments
        col_specs = [ColSpec(alignment=a) for a in alignments]

        return TableNode(
            head=head,
            bodies=tuple(bodies),
            col_specs=tuple(col_specs),
            provenance=prov,
        ), i

    def _parse_table_section(
        self, tokens: list[Token], start: int, close_type: str, *, is_head: bool = False
    ) -> tuple[TableSection, int, list[Alignment | None]]:
        """Parse thead/tbody section into TableSection."""
        rows: list[Row] = []
        alignments: list[Alignment | None] = []
        i = start + 1  # skip open token

        while i < len(tokens) and tokens[i].type != close_type:
            if tokens[i].type == "tr_open":
                row, i, row_aligns = self._parse_table_row(tokens, i, is_head=is_head)
                rows.append(row)
                if is_head and not alignments:
                    alignments = row_aligns
            else:
                i += 1

        i += 1  # skip close token
        return TableSection(rows=tuple(rows)), i, alignments

    def _parse_table_row(
        self, tokens: list[Token], start: int, *, is_head: bool = False
    ) -> tuple[Row, int, list[Alignment | None]]:
        """Parse tr_open .. (th|td)* .. tr_close."""
        cells: list[Cell] = []
        alignments: list[Alignment | None] = []
        i = start + 1  # skip tr_open

        while i < len(tokens) and tokens[i].type != "tr_close":
            if tokens[i].type in ("th_open", "td_open"):
                cell, i, align = self._parse_table_cell(tokens, i)
                cells.append(cell)
                alignments.append(align)
            else:
                i += 1

        i += 1  # skip tr_close
        return Row(cells=tuple(cells)), i, alignments

    def _parse_table_cell(
        self, tokens: list[Token], start: int
    ) -> tuple[Cell, int, Alignment | None]:
        """Parse th_open/td_open .. inline .. th_close/td_close."""
        open_tok = tokens[start]
        close_type = open_tok.type.replace("_open", "_close")

        # Extract alignment from style attribute
        alignment = None
        if open_tok.attrs:
            style = open_tok.attrs.get("style", "")
            if isinstance(style, str):
                if "text-align:center" in style:
                    alignment = Alignment.CENTER
                elif "text-align:right" in style:
                    alignment = Alignment.RIGHT
                elif "text-align:left" in style:
                    alignment = Alignment.LEFT

        inlines: list[Inline] = []
        i = start + 1
        while i < len(tokens) and tokens[i].type != close_type:
            if tokens[i].type == "inline":
                inlines = self._parse_inlines(tokens[i].children or [])
            i += 1
        i += 1  # skip close token

        # Table cells contain blocks; wrap inlines in a paragraph
        content: tuple[Block, ...] = ()
        if inlines:
            content = (Paragraph(children=tuple(inlines)),)

        return Cell(content=content, alignment=alignment), i, alignment

    def _parse_definition_list(self, tokens: list[Token], start: int) -> tuple[DefinitionList, int]:
        """Parse dl_open .. (dt/dd pairs) .. dl_close."""
        items: list[DefinitionItem] = []
        i = start + 1  # skip dl_open

        current_term: list[Inline] | None = None
        current_defs: list[tuple[Block, ...]] = []

        while i < len(tokens) and tokens[i].type != "dl_close":
            tt = tokens[i].type

            if tt == "dt_open":
                # If we have a pending term, save the previous item
                if current_term is not None:
                    items.append(
                        DefinitionItem(
                            term=tuple(current_term),
                            definitions=tuple(current_defs),
                        )
                    )
                    current_defs = []

                # Parse term inlines
                i += 1
                current_term = []
                while i < len(tokens) and tokens[i].type != "dt_close":
                    if tokens[i].type == "inline":
                        current_term = self._parse_inlines(tokens[i].children or [])
                    i += 1
                i += 1  # skip dt_close
                continue

            if tt == "dd_open":
                # Collect child tokens for definition body
                i += 1
                dd_tokens: list[Token] = []
                depth = 1
                while i < len(tokens) and depth > 0:
                    if tokens[i].type == "dd_open":
                        depth += 1
                    elif tokens[i].type == "dd_close":
                        depth -= 1
                        if depth == 0:
                            break
                    dd_tokens.append(tokens[i])
                    i += 1
                i += 1  # skip dd_close

                def_blocks = self.parse_blocks(dd_tokens)
                current_defs.append(tuple(def_blocks))
                continue

            i += 1

        # Save last item
        if current_term is not None:
            items.append(
                DefinitionItem(
                    term=tuple(current_term),
                    definitions=tuple(current_defs),
                )
            )

        i += 1  # skip dl_close
        return DefinitionList(children=tuple(items)), i

    def _parse_footnote_block(self, tokens: list[Token], start: int) -> int:
        """Parse footnote_block_open .. footnote* .. footnote_block_close."""
        i = start + 1  # skip footnote_block_open

        while i < len(tokens) and tokens[i].type != "footnote_block_close":
            if tokens[i].type == "footnote_open":
                i = self._parse_footnote(tokens, i)
            else:
                i += 1

        i += 1  # skip footnote_block_close
        return i

    def _parse_footnote(self, tokens: list[Token], start: int) -> int:
        """Parse footnote_open .. children .. footnote_close.

        Extracts the footnote label and body blocks.
        """
        open_tok = tokens[start]
        label = str(open_tok.meta.get("label", str(open_tok.meta.get("id", ""))))

        # Collect child tokens
        i = start + 1
        child_tokens: list[Token] = []
        while i < len(tokens) and tokens[i].type != "footnote_close":
            # Skip footnote_anchor tokens (they're rendering artifacts)
            if tokens[i].type != "footnote_anchor":
                child_tokens.append(tokens[i])
            i += 1
        i += 1  # skip footnote_close

        body = self.parse_blocks(child_tokens)
        self.footnotes[label] = tuple(body)
        return i

    # ── Inline-level parsing ──

    def _parse_inlines(self, tokens: list[Token]) -> list[Inline]:
        """Parse a list of inline tokens into Inline AST nodes."""
        result: list[Inline] = []
        i = 0
        while i < len(tokens):
            token = tokens[i]
            tt = token.type

            if tt == "text":
                if token.content:
                    result.append(Text(value=token.content))
                i += 1
                continue

            if tt == "code_inline":
                result.append(Code(value=token.content))
                i += 1
                continue

            if tt == "softbreak":
                result.append(SoftBreak())
                i += 1
                continue

            if tt == "hardbreak":
                result.append(LineBreak())
                i += 1
                continue

            if tt == "em_open":
                children, i = self._collect_inline_children(tokens, i, "em_close")
                result.append(Emphasis(children=tuple(children)))
                continue

            if tt == "strong_open":
                children, i = self._collect_inline_children(tokens, i, "strong_close")
                result.append(Strong(children=tuple(children)))
                continue

            if tt == "s_open":
                children, i = self._collect_inline_children(tokens, i, "s_close")
                result.append(Strikethrough(children=tuple(children)))
                continue

            if tt == "link_open":
                href = ""
                title = None
                if token.attrs:
                    href = str(token.attrs.get("href", ""))
                    raw_title = token.attrs.get("title")
                    if raw_title:
                        title = str(raw_title)
                children, i = self._collect_inline_children(tokens, i, "link_close")
                result.append(Link(url=href, title=title, children=tuple(children)))
                continue

            if tt == "image":
                src = ""
                alt = token.content or None
                title = None
                if token.attrs:
                    src = str(token.attrs.get("src", ""))
                    raw_title = token.attrs.get("title")
                    if raw_title:
                        title = str(raw_title)
                    raw_alt = token.attrs.get("alt")
                    if raw_alt:
                        alt = str(raw_alt) or alt
                # Prefer content for alt (it has the parsed inline text)
                if token.content:
                    alt = token.content
                result.append(Image(src=src, alt=alt, title=title))
                i += 1
                continue

            if tt == "footnote_ref":
                label = token.meta.get("label", "") if token.meta else ""
                result.append(FootnoteRef(identifier=str(label)))
                i += 1
                continue

            if tt == "math_inline":
                result.append(Math(value=token.content))
                i += 1
                continue

            if tt == "html_inline":
                inline = self._try_parse_html_inline(tokens, i, result)
                if inline is not None:
                    i = inline
                    continue
                # Fallback: raw HTML inline
                result.append(RawInline(format="html", value=token.content))
                i += 1
                continue

            # Unknown inline token: skip
            i += 1

        return result

    def _collect_inline_children(
        self, tokens: list[Token], start: int, close_type: str
    ) -> tuple[list[Inline], int]:
        """Collect inline children between open and close tokens."""
        i = start + 1  # skip open token
        child_tokens: list[Token] = []
        depth = 1
        open_type = close_type.replace("_close", "_open")

        while i < len(tokens) and depth > 0:
            if tokens[i].type == open_type:
                depth += 1
            elif tokens[i].type == close_type:
                depth -= 1
                if depth == 0:
                    break
            child_tokens.append(tokens[i])
            i += 1
        i += 1  # skip close token

        children = self._parse_inlines(child_tokens)
        return children, i

    def _try_parse_html_inline(
        self, tokens: list[Token], start: int, result: list[Inline]
    ) -> int | None:
        """Try to parse HTML inline tags (<sup>, <sub>, <u>) into AST nodes.

        Returns the new index if successful, or None if not a recognized tag.
        """
        tag_match = _HTML_TAG_RE.match(tokens[start].content)
        if tag_match is None:
            return None

        is_closing = tag_match.group(1) == "/"
        tag_name = tag_match.group(2).lower()

        if is_closing:
            # Closing tag without matching open — emit as raw
            return None

        # Find the matching close tag
        close_tag = f"</{tag_name}>"
        i = start + 1
        child_tokens: list[Token] = []
        while i < len(tokens):
            if tokens[i].type == "html_inline" and tokens[i].content.lower() == close_tag:
                # Found matching close
                children = self._parse_inlines(child_tokens)
                if tag_name == "sup":
                    result.append(Superscript(children=tuple(children)))
                elif tag_name == "sub":
                    result.append(Subscript(children=tuple(children)))
                elif tag_name == "u":
                    result.append(Underline(children=tuple(children)))
                return i + 1  # skip close tag
            child_tokens.append(tokens[i])
            i += 1

        # No matching close found — emit as raw
        return None
