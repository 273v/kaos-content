"""Phase 1 hardening tests for NodeIndex and traversal.

Attack vectors:
1. Large documents (1000+ blocks, deeply nested 30+ levels)
2. node_ref path correctness for every container type
3. Table edge cases (head + multiple bodies + foot, caption short + body, empty sections)
4. walk() completeness — every node type visited
5. extract_text edge cases (non-text-only docs, CodeBlock, etc.)
6. Annotation validation edge cases (footnote targets, prefix-but-not-exact refs)
7. by_provenance_page with multiple nodes on same page
8. find/find_first with match-all / match-none predicates
9. walk on footnote content — proper refs
10. Concurrent/independent NodeIndex builds
"""

from __future__ import annotations

from kaos_content import (
    Admonition,
    Annotation,
    AnnotationTarget,
    AnnotationType,
    BlockQuote,
    BulletList,
    Caption,
    Cell,
    Citation,
    Code,
    CodeBlock,
    ContentDocument,
    DefinitionItem,
    DefinitionList,
    Div,
    Emphasis,
    Figure,
    FootnoteRef,
    Heading,
    Image,
    LineBreak,
    Link,
    ListItem,
    Math,
    MathBlock,
    NodeIndex,
    OrderedList,
    PageBreak,
    Paragraph,
    Provenance,
    RawBlock,
    RawInline,
    Row,
    SoftBreak,
    Span,
    Strikethrough,
    Strong,
    Subscript,
    Superscript,
    Table,
    TableSection,
    Text,
    ThematicBreak,
    Underline,
    extract_text,
    find,
    find_first,
    walk,
    walk_blocks,
    walk_inlines,
)
from kaos_content.model.node import BaseBlock, BaseInline


# ---------------------------------------------------------------------------
# 1. Large documents
# ---------------------------------------------------------------------------
class TestLargeDocuments:
    def test_1000_blocks_indexed(self) -> None:
        """NodeIndex handles a document with 1000+ top-level blocks."""
        blocks = tuple(Paragraph(children=(Text(value=f"p{i}"),)) for i in range(1000))
        doc = ContentDocument(body=blocks)
        index = NodeIndex(doc)
        # 1000 paragraphs + 1000 text nodes
        assert len(index) == 2000
        # Spot-check first, middle, last
        assert index.get("#/body/0") is not None
        assert index.get("#/body/499") is not None
        assert index.get("#/body/999") is not None
        assert index.get("#/body/999/children/0") is not None
        assert isinstance(index["#/body/999/children/0"], Text)
        assert index["#/body/999/children/0"].value == "p999"

    def test_deeply_nested_30_levels(self) -> None:
        """30 levels of nested Divs do not crash walk() or NodeIndex."""
        # Build inside-out: Paragraph at the core, wrapped in 30 Divs
        inner = Paragraph(children=(Text(value="deep"),))
        node: Div | Paragraph = inner
        for _ in range(30):
            node = Div(children=(node,))
        doc = ContentDocument(body=(node,))

        # walk() should not hit recursion limit
        nodes = list(walk(node))
        # 30 Divs + 1 Paragraph + 1 Text = 32
        assert len(nodes) == 32

        index = NodeIndex(doc)
        assert len(index) == 32

        # Verify the deepest ref path
        ref = "#/body/0" + "/children/0" * 30
        deepest_para = index.get(ref)
        assert deepest_para is not None
        assert isinstance(deepest_para, Paragraph)

        deepest_text = index.get(ref + "/children/0")
        assert deepest_text is not None
        assert isinstance(deepest_text, Text)
        assert deepest_text.value == "deep"

    def test_deep_nesting_does_not_hit_recursion_limit(self) -> None:
        """Ensure walk() survives nesting up to depth that approaches default recursion limit.

        Python default recursion limit is 1000. We test 200 levels which is
        well above realistic docs but should work with default limits.
        """
        inner = Paragraph(children=(Text(value="bottom"),))
        node: Div | Paragraph = inner
        for _ in range(200):
            node = Div(children=(node,))

        nodes = list(walk(node))
        # 200 Divs + 1 Paragraph + 1 Text = 202
        assert len(nodes) == 202

    def test_wide_document(self) -> None:
        """A paragraph with 500 inline children."""
        inlines = tuple(Text(value=f"t{i}") for i in range(500))
        p = Paragraph(children=inlines)
        doc = ContentDocument(body=(p,))
        index = NodeIndex(doc)
        # 1 paragraph + 500 text nodes
        assert len(index) == 501
        assert index.get("#/body/0/children/499") is not None


# ---------------------------------------------------------------------------
# 2. node_ref path correctness for every container type
# ---------------------------------------------------------------------------
class TestNodeRefPaths:
    def test_body_block_ref(self) -> None:
        doc = ContentDocument(body=(Paragraph(children=(Text(value="a"),)),))
        index = NodeIndex(doc)
        assert "#/body/0" in index
        assert "#/body/0/children/0" in index

    def test_children_ref_for_div(self) -> None:
        inner = Paragraph(children=(Text(value="x"),))
        doc = ContentDocument(body=(Div(children=(inner,)),))
        index = NodeIndex(doc)
        assert "#/body/0/children/0" in index
        assert isinstance(index["#/body/0/children/0"], Paragraph)

    def test_definition_item_term_ref(self) -> None:
        doc = ContentDocument(
            body=(
                DefinitionList(
                    children=(
                        DefinitionItem(
                            term=(Text(value="T1"), Text(value="T2")),
                            definitions=(
                                (Paragraph(children=(Text(value="D1a"),)),),
                                (
                                    Paragraph(children=(Text(value="D2a"),)),
                                    Paragraph(children=(Text(value="D2b"),)),
                                ),
                            ),
                        ),
                    )
                ),
            )
        )
        index = NodeIndex(doc)
        # DL at #/body/0, DI at #/body/0/children/0
        di_ref = "#/body/0/children/0"
        assert isinstance(index[di_ref], DefinitionItem)

        # term[0], term[1]
        term0 = index[f"{di_ref}/term/0"]
        assert isinstance(term0, Text)
        assert term0.value == "T1"
        term1 = index[f"{di_ref}/term/1"]
        assert isinstance(term1, Text)
        assert term1.value == "T2"

        # definitions[0][0], definitions[1][0], definitions[1][1]
        assert isinstance(index[f"{di_ref}/definitions/0/0"], Paragraph)
        assert isinstance(index[f"{di_ref}/definitions/1/0"], Paragraph)
        assert isinstance(index[f"{di_ref}/definitions/1/1"], Paragraph)

        # Children of definition paragraphs
        def_child = index[f"{di_ref}/definitions/1/1/children/0"]
        assert isinstance(def_child, Text)
        assert def_child.value == "D2b"

    def test_caption_short_ref(self) -> None:
        doc = ContentDocument(
            body=(
                Table(
                    caption=Caption(
                        short=(Text(value="S1"), Emphasis(children=(Text(value="S2"),))),
                        body=(Paragraph(children=(Text(value="Full caption"),)),),
                    )
                ),
            )
        )
        index = NodeIndex(doc)
        short0 = index["#/body/0/caption/short/0"]
        assert isinstance(short0, Text)
        assert short0.value == "S1"
        assert isinstance(index["#/body/0/caption/short/1"], Emphasis)
        short1_child = index["#/body/0/caption/short/1/children/0"]
        assert isinstance(short1_child, Text)
        assert short1_child.value == "S2"

    def test_caption_body_ref(self) -> None:
        doc = ContentDocument(
            body=(
                Table(
                    caption=Caption(
                        body=(
                            Paragraph(children=(Text(value="cap p1"),)),
                            Paragraph(children=(Text(value="cap p2"),)),
                        )
                    )
                ),
            )
        )
        index = NodeIndex(doc)
        assert isinstance(index["#/body/0/caption/body/0"], Paragraph)
        assert isinstance(index["#/body/0/caption/body/1"], Paragraph)
        cap_child = index["#/body/0/caption/body/1/children/0"]
        assert isinstance(cap_child, Text)
        assert cap_child.value == "cap p2"

    def test_table_head_ref(self) -> None:
        doc = ContentDocument(
            body=(
                Table(
                    head=TableSection(
                        rows=(
                            Row(
                                cells=(
                                    Cell(content=(Paragraph(children=(Text(value="H1"),)),)),
                                    Cell(content=(Paragraph(children=(Text(value="H2"),)),)),
                                )
                            ),
                        )
                    )
                ),
            )
        )
        index = NodeIndex(doc)
        assert isinstance(index["#/body/0/head/rows/0/cells/0/content/0"], Paragraph)
        assert isinstance(index["#/body/0/head/rows/0/cells/1/content/0"], Paragraph)
        t = index["#/body/0/head/rows/0/cells/1/content/0/children/0"]
        assert isinstance(t, Text)
        assert t.value == "H2"

    def test_table_bodies_ref(self) -> None:
        doc = ContentDocument(
            body=(
                Table(
                    bodies=(
                        TableSection(
                            rows=(
                                Row(
                                    cells=(
                                        Cell(
                                            content=(Paragraph(children=(Text(value="B0R0C0"),)),)
                                        ),
                                    )
                                ),
                            )
                        ),
                        TableSection(
                            rows=(
                                Row(
                                    cells=(
                                        Cell(
                                            content=(Paragraph(children=(Text(value="B1R0C0"),)),)
                                        ),
                                    )
                                ),
                            )
                        ),
                    )
                ),
            )
        )
        index = NodeIndex(doc)
        p0 = index["#/body/0/bodies/0/rows/0/cells/0/content/0"]
        assert isinstance(p0, Paragraph)
        p1 = index["#/body/0/bodies/1/rows/0/cells/0/content/0"]
        assert isinstance(p1, Paragraph)
        b1_text = index["#/body/0/bodies/1/rows/0/cells/0/content/0/children/0"]
        assert isinstance(b1_text, Text)
        assert b1_text.value == "B1R0C0"

    def test_table_foot_ref(self) -> None:
        doc = ContentDocument(
            body=(
                Table(
                    foot=TableSection(
                        rows=(
                            Row(cells=(Cell(content=(Paragraph(children=(Text(value="F0"),)),)),)),
                        )
                    )
                ),
            )
        )
        index = NodeIndex(doc)
        assert isinstance(index["#/body/0/foot/rows/0/cells/0/content/0"], Paragraph)
        foot_child = index["#/body/0/foot/rows/0/cells/0/content/0/children/0"]
        assert isinstance(foot_child, Text)
        assert foot_child.value == "F0"

    def test_footnote_ref_path(self) -> None:
        doc = ContentDocument(
            footnotes={
                "fn1": (Paragraph(children=(Text(value="note1"),)),),
                "fn2": (
                    Paragraph(children=(Text(value="note2a"),)),
                    Paragraph(children=(Text(value="note2b"),)),
                ),
            }
        )
        index = NodeIndex(doc)
        assert isinstance(index["#/footnotes/fn1/0"], Paragraph)
        assert isinstance(index["#/footnotes/fn2/0"], Paragraph)
        assert isinstance(index["#/footnotes/fn2/1"], Paragraph)
        fn2_child = index["#/footnotes/fn2/1/children/0"]
        assert isinstance(fn2_child, Text)
        assert fn2_child.value == "note2b"

    def test_figure_caption_ref(self) -> None:
        doc = ContentDocument(
            body=(
                Figure(
                    children=(Paragraph(children=(Image(src="img.png"),)),),
                    caption=Caption(
                        short=(Text(value="Fig short"),),
                        body=(Paragraph(children=(Text(value="Fig body"),)),),
                    ),
                ),
            )
        )
        index = NodeIndex(doc)
        # Figure's children
        assert isinstance(index["#/body/0/children/0"], Paragraph)
        assert isinstance(index["#/body/0/children/0/children/0"], Image)
        # Figure's caption
        fig_short = index["#/body/0/caption/short/0"]
        assert isinstance(fig_short, Text)
        assert fig_short.value == "Fig short"
        assert isinstance(index["#/body/0/caption/body/0"], Paragraph)


# ---------------------------------------------------------------------------
# 3. Table edge cases
# ---------------------------------------------------------------------------
class TestTableEdgeCases:
    def test_table_head_multiple_bodies_foot(self) -> None:
        """Table with head, 3 body sections, and foot — all indexed."""
        doc = ContentDocument(
            body=(
                Table(
                    head=TableSection(
                        rows=(
                            Row(
                                cells=(
                                    Cell(content=(Paragraph(children=(Text(value="Header"),)),)),
                                )
                            ),
                        )
                    ),
                    bodies=tuple(
                        TableSection(
                            rows=(
                                Row(
                                    cells=(
                                        Cell(
                                            content=(Paragraph(children=(Text(value=f"Body{i}"),)),)
                                        ),
                                    )
                                ),
                            )
                        )
                        for i in range(3)
                    ),
                    foot=TableSection(
                        rows=(
                            Row(
                                cells=(
                                    Cell(content=(Paragraph(children=(Text(value="Footer"),)),)),
                                )
                            ),
                        )
                    ),
                ),
            )
        )
        index = NodeIndex(doc)
        # Verify all sections are reachable
        assert index.get("#/body/0/head/rows/0/cells/0/content/0") is not None
        for i in range(3):
            ref = f"#/body/0/bodies/{i}/rows/0/cells/0/content/0"
            node = index.get(ref)
            assert node is not None, f"Missing: {ref}"
        assert index.get("#/body/0/foot/rows/0/cells/0/content/0") is not None

    def test_table_with_caption_short_and_body(self) -> None:
        doc = ContentDocument(
            body=(
                Table(
                    caption=Caption(
                        short=(Text(value="Short"),),
                        body=(Paragraph(children=(Text(value="Long"),)),),
                    ),
                    bodies=(
                        TableSection(
                            rows=(
                                Row(
                                    cells=(
                                        Cell(content=(Paragraph(children=(Text(value="data"),)),)),
                                    )
                                ),
                            )
                        ),
                    ),
                ),
            )
        )
        index = NodeIndex(doc)
        cap_short = index["#/body/0/caption/short/0"]
        assert isinstance(cap_short, Text)
        assert cap_short.value == "Short"
        assert extract_text(index["#/body/0/caption/body/0"]) == "Long"

    def test_empty_table_sections(self) -> None:
        """Table with empty head/bodies/foot (no rows) — nothing crashes."""
        doc = ContentDocument(
            body=(
                Table(
                    head=TableSection(rows=()),
                    bodies=(TableSection(rows=()),),
                    foot=TableSection(rows=()),
                ),
            )
        )
        index = NodeIndex(doc)
        # Table + 3 TableSections (head, body, foot) are indexed
        assert len(index) == 4
        assert index.tables == [doc.body[0]]

    def test_table_multi_row_multi_cell(self) -> None:
        """Table body with 2 rows x 3 cells each."""
        doc = ContentDocument(
            body=(
                Table(
                    bodies=(
                        TableSection(
                            rows=tuple(
                                Row(
                                    cells=tuple(
                                        Cell(
                                            content=(
                                                Paragraph(children=(Text(value=f"R{r}C{c}"),)),
                                            )
                                        )
                                        for c in range(3)
                                    )
                                )
                                for r in range(2)
                            )
                        ),
                    )
                ),
            )
        )
        index = NodeIndex(doc)
        for r in range(2):
            for c in range(3):
                ref = f"#/body/0/bodies/0/rows/{r}/cells/{c}/content/0/children/0"
                node = index.get(ref)
                assert node is not None, f"Missing ref: {ref}"
                assert isinstance(node, Text)
                assert node.value == f"R{r}C{c}"

    def test_table_cell_with_multiple_blocks(self) -> None:
        """A cell containing two paragraphs."""
        doc = ContentDocument(
            body=(
                Table(
                    bodies=(
                        TableSection(
                            rows=(
                                Row(
                                    cells=(
                                        Cell(
                                            content=(
                                                Paragraph(children=(Text(value="p1"),)),
                                                Paragraph(children=(Text(value="p2"),)),
                                            )
                                        ),
                                    )
                                ),
                            )
                        ),
                    )
                ),
            )
        )
        index = NodeIndex(doc)
        cell_p1 = index["#/body/0/bodies/0/rows/0/cells/0/content/0/children/0"]
        assert isinstance(cell_p1, Text)
        assert cell_p1.value == "p1"
        cell_p2 = index["#/body/0/bodies/0/rows/0/cells/0/content/1/children/0"]
        assert isinstance(cell_p2, Text)
        assert cell_p2.value == "p2"

    def test_table_no_optional_sections(self) -> None:
        """Table with no head, no bodies, no foot, no caption."""
        doc = ContentDocument(body=(Table(),))
        index = NodeIndex(doc)
        assert len(index) == 1  # just the Table
        nodes = list(walk(doc.body[0]))
        assert len(nodes) == 1


# ---------------------------------------------------------------------------
# 4. walk() completeness — every node type
# ---------------------------------------------------------------------------
class TestWalkCompleteness:
    def _build_every_type_doc(self) -> ContentDocument:
        """Build a document containing every single Block and Inline node type."""
        return ContentDocument(
            body=(
                # Paragraph
                Paragraph(
                    children=(
                        Text(value="plain"),
                        Emphasis(children=(Text(value="em"),)),
                        Strong(children=(Text(value="strong"),)),
                        Strikethrough(children=(Text(value="strike"),)),
                        Code(value="code"),
                        Link(url="http://x", children=(Text(value="link"),)),
                        Image(src="img.png", alt="alt"),
                        FootnoteRef(identifier="fn1"),
                        Citation(identifiers=("cite1",), children=(Text(value="[1]"),)),
                        Math(value="x^2"),
                        RawInline(format="html", value="<br>"),
                        LineBreak(),
                        SoftBreak(),
                        Span(children=(Text(value="span"),)),
                        Superscript(children=(Text(value="sup"),)),
                        Subscript(children=(Text(value="sub"),)),
                        Underline(children=(Text(value="under"),)),
                    )
                ),
                # Heading
                Heading(depth=1, children=(Text(value="H1"),)),
                # BlockQuote
                BlockQuote(children=(Paragraph(children=(Text(value="quoted"),)),)),
                # OrderedList
                OrderedList(
                    children=(ListItem(children=(Paragraph(children=(Text(value="ol1"),)),)),)
                ),
                # BulletList
                BulletList(
                    children=(ListItem(children=(Paragraph(children=(Text(value="ul1"),)),)),)
                ),
                # DefinitionList
                DefinitionList(
                    children=(
                        DefinitionItem(
                            term=(Text(value="term"),),
                            definitions=((Paragraph(children=(Text(value="def"),)),),),
                        ),
                    )
                ),
                # Table
                Table(
                    caption=Caption(
                        short=(Text(value="cap_short"),),
                        body=(Paragraph(children=(Text(value="cap_body"),)),),
                    ),
                    head=TableSection(
                        rows=(
                            Row(
                                cells=(Cell(content=(Paragraph(children=(Text(value="thead"),)),)),)
                            ),
                        )
                    ),
                    bodies=(
                        TableSection(
                            rows=(
                                Row(
                                    cells=(
                                        Cell(content=(Paragraph(children=(Text(value="tbody"),)),)),
                                    )
                                ),
                            )
                        ),
                    ),
                    foot=TableSection(
                        rows=(
                            Row(
                                cells=(Cell(content=(Paragraph(children=(Text(value="tfoot"),)),)),)
                            ),
                        )
                    ),
                ),
                # CodeBlock
                CodeBlock(language="py", value="print()"),
                # ThematicBreak
                ThematicBreak(),
                # Figure
                Figure(
                    children=(Paragraph(children=(Image(src="fig.png"),)),),
                    caption=Caption(body=(Paragraph(children=(Text(value="fig_cap"),)),)),
                ),
                # PageBreak
                PageBreak(),
                # Div
                Div(children=(Paragraph(children=(Text(value="div_inner"),)),)),
                # RawBlock
                RawBlock(format="html", value="<hr>"),
                # MathBlock
                MathBlock(value="E=mc^2"),
                # Admonition
                Admonition(
                    kind="note",
                    title="Note",
                    children=(Paragraph(children=(Text(value="adm"),)),),
                ),
            ),
            footnotes={
                "fn1": (Paragraph(children=(Text(value="footnote_text"),)),),
            },
        )

    def test_walk_visits_every_block_type(self) -> None:
        doc = self._build_every_type_doc()
        block_types_visited = set()
        for block in doc.body:
            for node in walk(block):
                if isinstance(node, BaseBlock):
                    block_types_visited.add(type(node).__name__)

        expected_block_types = {
            "Paragraph",
            "Heading",
            "BlockQuote",
            "OrderedList",
            "BulletList",
            "ListItem",
            "DefinitionList",
            "DefinitionItem",
            "Table",
            "CodeBlock",
            "ThematicBreak",
            "Figure",
            "PageBreak",
            "Div",
            "RawBlock",
            "MathBlock",
            "Admonition",
        }
        assert block_types_visited == expected_block_types

    def test_walk_visits_every_inline_type(self) -> None:
        doc = self._build_every_type_doc()
        inline_types_visited = set()
        for block in doc.body:
            for node in walk(block):
                if isinstance(node, BaseInline):
                    inline_types_visited.add(type(node).__name__)

        expected_inline_types = {
            "Text",
            "Emphasis",
            "Strong",
            "Strikethrough",
            "Code",
            "Link",
            "Image",
            "FootnoteRef",
            "Citation",
            "Math",
            "RawInline",
            "LineBreak",
            "SoftBreak",
            "Span",
            "Superscript",
            "Subscript",
            "Underline",
        }
        assert inline_types_visited == expected_inline_types

    def test_walk_count_matches_manual(self) -> None:
        """Manually count nodes in a known structure and verify walk() matches."""
        # Structure: Div > [Paragraph > [Text, Strong > [Text]], BlockQuote > [Paragraph > [Text]]]
        doc = ContentDocument(
            body=(
                Div(
                    children=(
                        Paragraph(
                            children=(
                                Text(value="a"),
                                Strong(children=(Text(value="b"),)),
                            )
                        ),
                        BlockQuote(children=(Paragraph(children=(Text(value="c"),)),)),
                    )
                ),
            )
        )
        # Manual count: Div(1) + Paragraph(2) + Text("a")(3) + Strong(4) + Text("b")(5)
        #             + BlockQuote(6) + Paragraph(7) + Text("c")(8)
        nodes = list(walk(doc.body[0]))
        assert len(nodes) == 8

    def test_node_index_count_matches_walk_on_doc(self) -> None:
        """NodeIndex total count should match walk over all body + footnotes."""
        doc = self._build_every_type_doc()
        index = NodeIndex(doc)

        walk_count = 0
        for block in doc.body:
            walk_count += sum(1 for _ in walk(block))
        for fn_blocks in doc.footnotes.values():
            for block in fn_blocks:
                walk_count += sum(1 for _ in walk(block))

        assert len(index) == walk_count


# ---------------------------------------------------------------------------
# 5. extract_text edge cases
# ---------------------------------------------------------------------------
class TestExtractTextEdgeCases:
    def test_only_non_text_nodes(self) -> None:
        """Document with only images, line breaks, thematic breaks — no text content."""
        p = Paragraph(children=(Image(src="a.png"), LineBreak(), Image(src="b.png")))
        # extract_text should return only the linebreak
        assert extract_text(p) == "\n"

    def test_thematic_break_no_text(self) -> None:
        tb = ThematicBreak()
        assert extract_text(tb) == ""

    def test_page_break_no_text(self) -> None:
        pb = PageBreak()
        assert extract_text(pb) == ""

    def test_code_block_value(self) -> None:
        cb = CodeBlock(value="def foo():\n    pass")
        assert extract_text(cb) == "def foo():\n    pass"

    def test_raw_block_value(self) -> None:
        rb = RawBlock(format="html", value="<div>hello</div>")
        assert extract_text(rb) == "<div>hello</div>"

    def test_math_inline_value(self) -> None:
        m = Math(value="\\alpha + \\beta")
        assert extract_text(m) == "\\alpha + \\beta"

    def test_raw_inline_value(self) -> None:
        ri = RawInline(format="html", value="<b>bold</b>")
        assert extract_text(ri) == "<b>bold</b>"

    def test_code_inline_value(self) -> None:
        c = Code(value="x = 1")
        assert extract_text(c) == "x = 1"

    def test_footnote_ref_no_text(self) -> None:
        """FootnoteRef has no text content."""
        fr = FootnoteRef(identifier="fn1")
        assert extract_text(fr) == ""

    def test_image_no_text(self) -> None:
        """Image alt text is not returned by extract_text (alt is not 'value')."""
        img = Image(src="x.png", alt="Alt text")
        assert extract_text(img) == ""

    def test_empty_paragraph(self) -> None:
        p = Paragraph(children=())
        assert extract_text(p) == ""

    def test_mixed_soft_and_line_breaks(self) -> None:
        p = Paragraph(
            children=(
                Text(value="a"),
                SoftBreak(),
                Text(value="b"),
                LineBreak(),
                Text(value="c"),
            )
        )
        assert extract_text(p) == "a b\nc"

    def test_nested_emphasis_strong(self) -> None:
        """Deeply nested inline formatting."""
        node = Emphasis(
            children=(
                Strong(
                    children=(Underline(children=(Strikethrough(children=(Text(value="deep"),)),)),)
                ),
            )
        )
        assert extract_text(node) == "deep"

    def test_extract_text_on_table(self) -> None:
        """extract_text walks into table cells."""
        table = Table(
            head=TableSection(
                rows=(Row(cells=(Cell(content=(Paragraph(children=(Text(value="H"),)),)),)),)
            ),
            bodies=(
                TableSection(
                    rows=(Row(cells=(Cell(content=(Paragraph(children=(Text(value="D"),)),)),)),)
                ),
            ),
            foot=TableSection(
                rows=(Row(cells=(Cell(content=(Paragraph(children=(Text(value="F"),)),)),)),)
            ),
        )
        assert extract_text(table) == "HDF"

    def test_extract_text_on_definition_item(self) -> None:
        di = DefinitionItem(
            term=(Text(value="A"), Text(value="B")),
            definitions=(
                (Paragraph(children=(Text(value="D1"),)),),
                (Paragraph(children=(Text(value="D2"),)),),
            ),
        )
        assert extract_text(di) == "ABD1D2"


# ---------------------------------------------------------------------------
# 6. Annotation validation edge cases
# ---------------------------------------------------------------------------
class TestAnnotationValidation:
    def test_annotation_targeting_footnote_node(self) -> None:
        """Annotation whose target is a footnote node ref should be valid."""
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="main"),)),),
            footnotes={"fn1": (Paragraph(children=(Text(value="note"),)),)},
            annotations=(
                Annotation(
                    id="a1",
                    type=AnnotationType.HIGHLIGHT,
                    targets=(AnnotationTarget(node_ref="#/footnotes/fn1/0"),),
                ),
            ),
        )
        index = NodeIndex(doc)
        assert index.validate_annotations() == []
        anns = index.annotations_for("#/footnotes/fn1/0")
        assert len(anns) == 1
        assert anns[0].id == "a1"

    def test_annotation_prefix_of_valid_ref_but_not_exact(self) -> None:
        """node_ref '#/body/0/children' is a prefix of '#/body/0/children/0' but not exact."""
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="x"),)),),
            annotations=(
                Annotation(
                    id="bad",
                    type=AnnotationType.COMMENT,
                    targets=(AnnotationTarget(node_ref="#/body/0/children"),),
                ),
            ),
        )
        index = NodeIndex(doc)
        invalid = index.validate_annotations()
        assert "#/body/0/children" in invalid

    def test_annotation_with_empty_ref(self) -> None:
        """Empty string node_ref should be invalid."""
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="x"),)),),
            annotations=(
                Annotation(
                    id="bad",
                    type=AnnotationType.COMMENT,
                    targets=(AnnotationTarget(node_ref=""),),
                ),
            ),
        )
        index = NodeIndex(doc)
        invalid = index.validate_annotations()
        assert "" in invalid

    def test_annotation_targeting_deep_child(self) -> None:
        """Annotation targeting a deeply nested text node."""
        doc = ContentDocument(
            body=(
                Paragraph(children=(Strong(children=(Emphasis(children=(Text(value="deep"),)),)),)),
            ),
            annotations=(
                Annotation(
                    id="a1",
                    type=AnnotationType.ENTITY,
                    targets=(
                        AnnotationTarget(node_ref="#/body/0/children/0/children/0/children/0"),
                    ),
                ),
            ),
        )
        index = NodeIndex(doc)
        assert index.validate_annotations() == []
        anns = index.annotations_for("#/body/0/children/0/children/0/children/0")
        assert len(anns) == 1

    def test_multiple_annotations_same_target(self) -> None:
        """Multiple annotations can target the same node."""
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="x"),)),),
            annotations=(
                Annotation(
                    id="a1",
                    type=AnnotationType.HIGHLIGHT,
                    targets=(AnnotationTarget(node_ref="#/body/0"),),
                ),
                Annotation(
                    id="a2",
                    type=AnnotationType.COMMENT,
                    targets=(AnnotationTarget(node_ref="#/body/0"),),
                ),
                Annotation(
                    id="a3",
                    type=AnnotationType.ENTITY,
                    targets=(AnnotationTarget(node_ref="#/body/0"),),
                ),
            ),
        )
        index = NodeIndex(doc)
        assert index.validate_annotations() == []
        anns = index.annotations_for("#/body/0")
        assert len(anns) == 3
        assert {a.id for a in anns} == {"a1", "a2", "a3"}

    def test_annotation_multi_target(self) -> None:
        """Single annotation with multiple targets, some valid, some not."""
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="x"),)),),
            annotations=(
                Annotation(
                    id="mixed",
                    type=AnnotationType.HIGHLIGHT,
                    targets=(
                        AnnotationTarget(node_ref="#/body/0"),
                        AnnotationTarget(node_ref="#/body/999"),
                    ),
                ),
            ),
        )
        index = NodeIndex(doc)
        invalid = index.validate_annotations()
        assert "#/body/999" in invalid
        # Valid target still works
        anns = index.annotations_for("#/body/0")
        assert len(anns) == 1

    def test_no_annotations(self) -> None:
        """Document with no annotations — validate returns empty."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="x"),)),))
        index = NodeIndex(doc)
        assert index.validate_annotations() == []


# ---------------------------------------------------------------------------
# 7. by_provenance_page with multiple nodes on same page
# ---------------------------------------------------------------------------
class TestByProvenancePage:
    def test_multiple_nodes_same_page(self) -> None:
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(Text(value="a"),),
                    provenance=Provenance(page=1),
                ),
                Paragraph(
                    children=(Text(value="b"),),
                    provenance=Provenance(page=1),
                ),
                Paragraph(
                    children=(Text(value="c"),),
                    provenance=Provenance(page=1),
                ),
                Paragraph(
                    children=(Text(value="d"),),
                    provenance=Provenance(page=2),
                ),
            )
        )
        index = NodeIndex(doc)
        page1 = index.by_provenance_page(1)
        assert len(page1) == 3
        page2 = index.by_provenance_page(2)
        assert len(page2) == 1

    def test_nested_provenance(self) -> None:
        """Both parent and child have provenance on the same page."""
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(Text(value="a", provenance=Provenance(page=5)),),
                    provenance=Provenance(page=5),
                ),
            )
        )
        index = NodeIndex(doc)
        page5 = index.by_provenance_page(5)
        assert len(page5) == 2  # paragraph + text node

    def test_no_provenance(self) -> None:
        """Nodes without provenance are not in any page bucket."""
        doc = ContentDocument(
            body=(
                Paragraph(children=(Text(value="a"),)),
                Paragraph(children=(Text(value="b"),)),
            )
        )
        index = NodeIndex(doc)
        assert index.by_provenance_page(1) == []
        assert index.by_provenance_page(0) == []


# ---------------------------------------------------------------------------
# 8. find / find_first with predicates that match everything or nothing
# ---------------------------------------------------------------------------
class TestFindPredicateEdgeCases:
    def test_find_match_everything(self) -> None:
        doc = ContentDocument(
            body=(
                Paragraph(children=(Text(value="a"), Text(value="b"))),
                Paragraph(children=(Text(value="c"),)),
            )
        )
        # Predicate that matches all nodes
        results = find(doc, lambda n: True)
        # Should get: 2 Paragraphs + 3 Texts = 5
        assert len(results) == 5

    def test_find_match_nothing(self) -> None:
        doc = ContentDocument(body=(Paragraph(children=(Text(value="a"),)),))
        results = find(doc, lambda n: False)
        assert results == []

    def test_find_first_match_everything(self) -> None:
        doc = ContentDocument(
            body=(
                Paragraph(children=(Text(value="a"),)),
                Heading(depth=1, children=(Text(value="b"),)),
            )
        )
        result = find_first(doc, lambda n: True)
        # Should return the first body block
        assert result is doc.body[0]

    def test_find_first_match_nothing(self) -> None:
        doc = ContentDocument(body=(Paragraph(children=(Text(value="a"),)),))
        result = find_first(doc, lambda n: False)
        assert result is None

    def test_find_in_empty_doc(self) -> None:
        doc = ContentDocument()
        results = find(doc, lambda n: True)
        assert results == []
        result = find_first(doc, lambda n: True)
        assert result is None

    def test_find_first_prefers_body_over_footnotes(self) -> None:
        """find_first searches body before footnotes."""
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="body"),)),),
            footnotes={"fn1": (Paragraph(children=(Text(value="note"),)),)},
        )
        result = find_first(doc, lambda n: isinstance(n, Paragraph))
        assert result is doc.body[0]

    def test_find_first_falls_through_to_footnotes(self) -> None:
        """If nothing matches in body, find_first checks footnotes."""
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="body"),)),),
            footnotes={"fn1": (Heading(depth=1, children=(Text(value="heading"),)),)},
        )
        result = find_first(doc, lambda n: isinstance(n, Heading))
        assert result is not None
        assert isinstance(result, Heading)


# ---------------------------------------------------------------------------
# 9. walk on footnote content — proper refs in NodeIndex
# ---------------------------------------------------------------------------
class TestFootnoteNodeIndex:
    def test_footnote_nodes_indexed(self) -> None:
        doc = ContentDocument(
            footnotes={
                "fn1": (Paragraph(children=(Text(value="note1"),)),),
                "fn2": (
                    Paragraph(
                        children=(
                            Text(value="note2"),
                            Strong(children=(Text(value="bold_note"),)),
                        )
                    ),
                ),
            }
        )
        index = NodeIndex(doc)
        # fn1
        assert "#/footnotes/fn1/0" in index
        assert "#/footnotes/fn1/0/children/0" in index
        # fn2
        assert "#/footnotes/fn2/0" in index
        assert "#/footnotes/fn2/0/children/0" in index
        assert "#/footnotes/fn2/0/children/1" in index
        assert "#/footnotes/fn2/0/children/1/children/0" in index
        bold_note = index["#/footnotes/fn2/0/children/1/children/0"]
        assert isinstance(bold_note, Text)
        assert bold_note.value == "bold_note"

    def test_walk_blocks_includes_footnote_blocks(self) -> None:
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="main"),)),),
            footnotes={
                "fn1": (
                    Paragraph(children=(Text(value="note1"),)),
                    BlockQuote(children=(Paragraph(children=(Text(value="quoted_note"),)),)),
                ),
            },
        )
        blocks = list(walk_blocks(doc))
        block_types = [type(b).__name__ for b in blocks]
        assert "BlockQuote" in block_types

    def test_walk_inlines_includes_footnote_inlines(self) -> None:
        doc = ContentDocument(
            body=(),
            footnotes={
                "fn1": (Paragraph(children=(Emphasis(children=(Text(value="emph_note"),)),)),),
            },
        )
        inlines = list(walk_inlines(doc))
        inline_types = [type(i).__name__ for i in inlines]
        assert "Emphasis" in inline_types
        assert "Text" in inline_types

    def test_footnote_ref_in_index_by_type(self) -> None:
        """FootnoteRef inline nodes show up in by_type."""
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Text(value="See"),
                        FootnoteRef(identifier="fn1"),
                    )
                ),
            ),
            footnotes={"fn1": (Paragraph(children=(Text(value="note"),)),)},
        )
        index = NodeIndex(doc)
        fn_refs = index.by_type(FootnoteRef)
        assert len(fn_refs) == 1
        assert fn_refs[0].identifier == "fn1"


# ---------------------------------------------------------------------------
# 10. Independent NodeIndex builds
# ---------------------------------------------------------------------------
class TestNodeIndexIndependence:
    def test_two_indexes_same_document(self) -> None:
        """Two NodeIndex objects from the same document are independent."""
        doc = ContentDocument(
            body=(
                Paragraph(children=(Text(value="a"),)),
                Heading(depth=1, children=(Text(value="b"),)),
            )
        )
        idx1 = NodeIndex(doc)
        idx2 = NodeIndex(doc)
        assert len(idx1) == len(idx2)
        assert idx1.refs == idx2.refs
        # They should be independent objects
        assert idx1 is not idx2
        assert idx1._ref_map is not idx2._ref_map
        assert idx1._by_type is not idx2._by_type

    def test_indexes_from_different_documents(self) -> None:
        """Indexes from different documents don't share state."""
        doc1 = ContentDocument(body=(Paragraph(children=(Text(value="a"),)),))
        doc2 = ContentDocument(
            body=(
                Heading(depth=1, children=(Text(value="b"),)),
                Paragraph(children=(Text(value="c"),)),
            )
        )
        idx1 = NodeIndex(doc1)
        idx2 = NodeIndex(doc2)
        assert len(idx1) != len(idx2)
        assert idx1.headings == []
        assert len(idx2.headings) == 1

    def test_index_does_not_mutate_document(self) -> None:
        """Building a NodeIndex does not modify the document."""
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="x"),)),),
            footnotes={"fn1": (Paragraph(children=(Text(value="n"),)),)},
        )
        body_before = doc.body
        fn_before = doc.footnotes
        ann_before = doc.annotations

        _ = NodeIndex(doc)

        assert doc.body is body_before
        assert doc.footnotes is fn_before
        assert doc.annotations is ann_before


# ---------------------------------------------------------------------------
# Bonus: Additional edge cases discovered during analysis
# ---------------------------------------------------------------------------
class TestAdditionalEdgeCases:
    def test_ordered_list_start(self) -> None:
        """OrderedList with custom start value is indexed normally."""
        doc = ContentDocument(
            body=(
                OrderedList(
                    start=5,
                    children=(
                        ListItem(children=(Paragraph(children=(Text(value="item5"),)),)),
                        ListItem(children=(Paragraph(children=(Text(value="item6"),)),)),
                    ),
                ),
            )
        )
        index = NodeIndex(doc)
        assert isinstance(index["#/body/0/children/1"], ListItem)
        item6_text = index["#/body/0/children/1/children/0/children/0"]
        assert isinstance(item6_text, Text)
        assert item6_text.value == "item6"

    def test_list_item_with_checked(self) -> None:
        """Task list items (checked=True/False) are indexed."""
        doc = ContentDocument(
            body=(
                BulletList(
                    children=(
                        ListItem(
                            checked=True,
                            children=(Paragraph(children=(Text(value="done"),)),),
                        ),
                        ListItem(
                            checked=False,
                            children=(Paragraph(children=(Text(value="todo"),)),),
                        ),
                    )
                ),
            )
        )
        index = NodeIndex(doc)
        li0 = index["#/body/0/children/0"]
        assert isinstance(li0, ListItem)
        assert li0.checked is True

    def test_admonition_indexed(self) -> None:
        doc = ContentDocument(
            body=(
                Admonition(
                    kind="warning",
                    title="Beware",
                    children=(Paragraph(children=(Text(value="danger"),)),),
                ),
            )
        )
        index = NodeIndex(doc)
        assert isinstance(index["#/body/0"], Admonition)
        danger_text = index["#/body/0/children/0/children/0"]
        assert isinstance(danger_text, Text)
        assert danger_text.value == "danger"

    def test_citation_children_indexed(self) -> None:
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Citation(
                            identifiers=("ref1", "ref2"),
                            children=(Text(value="[1,2]"),),
                        ),
                    )
                ),
            )
        )
        index = NodeIndex(doc)
        cite = index["#/body/0/children/0"]
        assert isinstance(cite, Citation)
        cite_text = index["#/body/0/children/0/children/0"]
        assert isinstance(cite_text, Text)
        assert cite_text.value == "[1,2]"

    def test_refs_property_ordering(self) -> None:
        """NodeIndex.refs returns refs in DFS traversal order."""
        doc = ContentDocument(
            body=(
                Paragraph(children=(Text(value="a"),)),
                Paragraph(children=(Text(value="b"),)),
            )
        )
        index = NodeIndex(doc)
        refs = index.refs
        assert refs.index("#/body/0") < refs.index("#/body/0/children/0")
        assert refs.index("#/body/0/children/0") < refs.index("#/body/1")
        assert refs.index("#/body/1") < refs.index("#/body/1/children/0")

    def test_empty_footnotes_dict(self) -> None:
        doc = ContentDocument(body=(Paragraph(children=(Text(value="x"),)),), footnotes={})
        index = NodeIndex(doc)
        assert len(index) == 2  # Paragraph + Text

    def test_figure_without_caption(self) -> None:
        """Figure with no caption — only children indexed."""
        doc = ContentDocument(
            body=(Figure(children=(Paragraph(children=(Image(src="img.png"),)),)),)
        )
        index = NodeIndex(doc)
        assert isinstance(index["#/body/0/children/0/children/0"], Image)

    def test_link_with_no_children(self) -> None:
        """Link with empty children list."""
        doc = ContentDocument(body=(Paragraph(children=(Link(url="http://x", children=()),)),))
        index = NodeIndex(doc)
        assert isinstance(index["#/body/0/children/0"], Link)
        assert len(index) == 2  # Paragraph + Link

    def test_span_container(self) -> None:
        """Span as generic inline container."""
        doc = ContentDocument(
            body=(Paragraph(children=(Span(children=(Text(value="inside span"),)),)),)
        )
        index = NodeIndex(doc)
        assert isinstance(index["#/body/0/children/0"], Span)
        span_text = index["#/body/0/children/0/children/0"]
        assert isinstance(span_text, Text)
        assert span_text.value == "inside span"

    def test_getitem_error_message(self) -> None:
        """KeyError from __getitem__ includes the ref in the message."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="x"),)),))
        index = NodeIndex(doc)
        import pytest

        with pytest.raises(KeyError, match="No node at ref"):
            index["#/nonexistent"]

    def test_json_roundtrip_preserves_index_count(self) -> None:
        """JSON serialization round-trip preserves the same number of indexed nodes."""
        doc = ContentDocument(
            body=(
                Heading(depth=1, children=(Text(value="H1"),)),
                Paragraph(
                    children=(
                        Text(value="hello"),
                        Strong(children=(Text(value="world"),)),
                    )
                ),
                Table(
                    caption=Caption(
                        short=(Text(value="T"),),
                        body=(Paragraph(children=(Text(value="Table"),)),),
                    ),
                    bodies=(
                        TableSection(
                            rows=(
                                Row(
                                    cells=(
                                        Cell(content=(Paragraph(children=(Text(value="data"),)),)),
                                    )
                                ),
                            )
                        ),
                    ),
                ),
            ),
            footnotes={"fn1": (Paragraph(children=(Text(value="note"),)),)},
        )
        idx1 = NodeIndex(doc)
        restored = ContentDocument.model_validate_json(doc.model_dump_json())
        idx2 = NodeIndex(restored)
        assert len(idx1) == len(idx2)
        assert set(idx1.refs) == set(idx2.refs)
