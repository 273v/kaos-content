"""Tests for tree traversal visitor functions."""

from kaos_content import (
    BlockQuote,
    BulletList,
    Caption,
    Cell,
    CodeBlock,
    ContentDocument,
    DefinitionItem,
    Div,
    Emphasis,
    Figure,
    Heading,
    Image,
    LineBreak,
    Link,
    ListItem,
    MathBlock,
    Paragraph,
    Provenance,
    Row,
    SoftBreak,
    Strong,
    Table,
    TableSection,
    Text,
    extract_text,
    find,
    find_first,
    walk,
    walk_blocks,
    walk_inlines,
)
from kaos_content.model.node import BaseBlock, BaseInline


class TestWalk:
    def test_single_text(self) -> None:
        node = Text(value="hello")
        nodes = list(walk(node))
        assert len(nodes) == 1
        assert nodes[0] is node

    def test_paragraph_with_inlines(self) -> None:
        t1 = Text(value="a")
        t2 = Text(value="b")
        p = Paragraph(children=(t1, Strong(children=(t2,))))
        nodes = list(walk(p))
        # p -> t1 -> Strong -> t2
        assert len(nodes) == 4
        assert nodes[0] is p
        assert nodes[1] is t1

    def test_nested_blocks(self) -> None:
        inner_p = Paragraph(children=(Text(value="inner"),))
        bq = BlockQuote(children=(inner_p,))
        outer_p = Paragraph(children=(Text(value="outer"),))
        div = Div(children=(outer_p, bq))
        nodes = list(walk(div))
        # div -> outer_p -> Text("outer") -> bq -> inner_p -> Text("inner")
        assert len(nodes) == 6

    def test_table_traversal(self) -> None:
        cell_text = Text(value="data")
        cell_para = Paragraph(children=(cell_text,))
        table = Table(bodies=(TableSection(rows=(Row(cells=(Cell(content=(cell_para,)),)),)),))
        nodes = list(walk(table))
        # table -> TableSection -> Row -> Cell -> cell_para -> cell_text
        assert len(nodes) == 6
        assert nodes[0] is table
        assert nodes[4] is cell_para
        assert nodes[5] is cell_text

    def test_table_with_head(self) -> None:
        head_text = Text(value="Header")
        head_para = Paragraph(children=(head_text,))
        table = Table(head=TableSection(rows=(Row(cells=(Cell(content=(head_para,)),)),)))
        nodes = list(walk(table))
        # table -> TableSection -> Row -> Cell -> head_para -> head_text
        assert len(nodes) == 6
        values = [getattr(n, "value", None) for n in nodes]
        assert "Header" in values

    def test_table_with_caption(self) -> None:
        cap_text = Text(value="Table 1")
        cap_para = Paragraph(children=(cap_text,))
        table = Table(
            caption=Caption(
                short=(Text(value="T1"),),
                body=(cap_para,),
            )
        )
        nodes = list(walk(table))
        # table -> Text("T1") -> cap_para -> cap_text
        assert len(nodes) == 4

    def test_definition_item(self) -> None:
        term_text = Text(value="Term")
        def_text = Text(value="Definition")
        def_para = Paragraph(children=(def_text,))
        item = DefinitionItem(
            term=(term_text,),
            definitions=((def_para,),),
        )
        nodes = list(walk(item))
        # item -> term_text -> def_para -> def_text
        assert len(nodes) == 4

    def test_figure_with_caption(self) -> None:
        img = Paragraph(children=(Image(src="a.png"),))
        cap_text = Text(value="Fig 1")
        fig = Figure(
            children=(img,),
            caption=Caption(body=(Paragraph(children=(cap_text,)),)),
        )
        nodes = list(walk(fig))
        # fig -> img(Paragraph) -> Image -> cap_para -> cap_text
        assert len(nodes) == 5

    def test_deeply_nested(self) -> None:
        """4-level nesting: div > blockquote > list > list_item > paragraph."""
        p = Paragraph(children=(Text(value="deep"),))
        li = ListItem(children=(p,))
        bl = BulletList(children=(li,))
        bq = BlockQuote(children=(bl,))
        div = Div(children=(bq,))
        nodes = list(walk(div))
        assert len(nodes) == 6
        types = [type(n).__name__ for n in nodes]
        assert types == ["Div", "BlockQuote", "BulletList", "ListItem", "Paragraph", "Text"]


class TestWalkBlocks:
    def test_only_blocks(self) -> None:
        doc = ContentDocument(
            body=(
                Heading(depth=1, children=(Text(value="Title"),)),
                Paragraph(children=(Text(value="Body"),)),
            )
        )
        blocks = list(walk_blocks(doc))
        assert all(isinstance(b, BaseBlock) for b in blocks)
        type_names = [type(b).__name__ for b in blocks]
        assert "Heading" in type_names
        assert "Paragraph" in type_names
        assert "Text" not in type_names

    def test_includes_footnotes(self) -> None:
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="body"),)),),
            footnotes={"fn1": (Paragraph(children=(Text(value="note"),)),)},
        )
        blocks = list(walk_blocks(doc))
        assert len(blocks) == 2  # body para + footnote para


class TestWalkInlines:
    def test_only_inlines(self) -> None:
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Text(value="a"),
                        Strong(children=(Text(value="b"),)),
                    )
                ),
            )
        )
        inlines = list(walk_inlines(doc))
        assert all(isinstance(i, BaseInline) for i in inlines)
        assert len(inlines) == 3  # Text("a"), Strong, Text("b")

    def test_includes_footnotes(self) -> None:
        doc = ContentDocument(
            body=(),
            footnotes={"fn1": (Paragraph(children=(Text(value="note"),)),)},
        )
        inlines = list(walk_inlines(doc))
        assert len(inlines) == 1


class TestFind:
    def test_find_headings(self) -> None:
        doc = ContentDocument(
            body=(
                Heading(depth=1, children=(Text(value="H1"),)),
                Paragraph(children=(Text(value="P"),)),
                Heading(depth=2, children=(Text(value="H2"),)),
            )
        )
        headings = find(doc, lambda n: isinstance(n, Heading))
        assert len(headings) == 2

    def test_find_by_value(self) -> None:
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="hello"), Text(value="world"))),)
        )
        results = find(doc, lambda n: getattr(n, "value", None) == "world")
        assert len(results) == 1

    def test_find_returns_empty(self) -> None:
        doc = ContentDocument(body=(Paragraph(children=(Text(value="x"),)),))
        results = find(doc, lambda n: isinstance(n, Heading))
        assert results == []

    def test_find_in_footnotes(self) -> None:
        doc = ContentDocument(
            body=(),
            footnotes={"fn1": (Paragraph(children=(Text(value="note"),)),)},
        )
        results = find(doc, lambda n: getattr(n, "value", None) == "note")
        assert len(results) == 1


class TestFindFirst:
    def test_finds_first(self) -> None:
        doc = ContentDocument(
            body=(
                Paragraph(children=(Text(value="first"),)),
                Paragraph(children=(Text(value="second"),)),
            )
        )
        result = find_first(doc, lambda n: isinstance(n, Paragraph))
        assert result is not None
        assert result is doc.body[0]

    def test_finds_first_in_footnotes(self) -> None:
        doc = ContentDocument(
            body=(),
            footnotes={"fn1": (Paragraph(children=(Text(value="note"),)),)},
        )
        result = find_first(doc, lambda n: isinstance(n, Paragraph))
        assert result is not None

    def test_returns_none(self) -> None:
        doc = ContentDocument(body=(Paragraph(children=(Text(value="x"),)),))
        result = find_first(doc, lambda n: isinstance(n, Heading))
        assert result is None


class TestWalkTableFoot:
    def test_walk_includes_foot(self) -> None:
        foot_text = Text(value="total")
        table = Table(
            foot=TableSection(
                rows=(Row(cells=(Cell(content=(Paragraph(children=(foot_text,)),)),)),)
            )
        )
        nodes = list(walk(table))
        values = [getattr(n, "value", None) for n in nodes]
        assert "total" in values


class TestExtractText:
    def test_text_node(self) -> None:
        assert extract_text(Text(value="hello")) == "hello"

    def test_paragraph(self) -> None:
        p = Paragraph(children=(Text(value="hello "), Text(value="world")))
        assert extract_text(p) == "hello world"

    def test_nested_inlines(self) -> None:
        """emphasis inside link inside paragraph."""
        p = Paragraph(
            children=(
                Link(
                    url="u",
                    children=(
                        Text(value="click "),
                        Emphasis(children=(Text(value="here"),)),
                    ),
                ),
            )
        )
        assert extract_text(p) == "click here"

    def test_code_block(self) -> None:
        cb = CodeBlock(value="x = 1\ny = 2")
        assert extract_text(cb) == "x = 1\ny = 2"

    def test_math_block(self) -> None:
        mb = MathBlock(value="E = mc^2")
        assert extract_text(mb) == "E = mc^2"

    def test_soft_break(self) -> None:
        p = Paragraph(children=(Text(value="a"), SoftBreak(), Text(value="b")))
        assert extract_text(p) == "a b"

    def test_line_break(self) -> None:
        p = Paragraph(children=(Text(value="a"), LineBreak(), Text(value="b")))
        assert extract_text(p) == "a\nb"

    def test_table_cell(self) -> None:
        table = Table(
            bodies=(
                TableSection(
                    rows=(
                        Row(
                            cells=(
                                Cell(content=(Paragraph(children=(Text(value="A"),)),)),
                                Cell(content=(Paragraph(children=(Text(value="B"),)),)),
                            )
                        ),
                    )
                ),
            )
        )
        assert extract_text(table) == "AB"

    def test_definition_item(self) -> None:
        di = DefinitionItem(
            term=(Text(value="Term"),),
            definitions=((Paragraph(children=(Text(value="Def"),)),),),
        )
        assert extract_text(di) == "TermDef"

    def test_empty_document_block(self) -> None:
        p = Paragraph(children=())
        assert extract_text(p) == ""

    def test_figure_with_caption_text(self) -> None:
        fig = Figure(
            children=(Paragraph(children=(Image(src="a.png"),)),),
            caption=Caption(body=(Paragraph(children=(Text(value="Caption"),)),)),
        )
        assert extract_text(fig) == "Caption"

    def test_with_provenance_ignored(self) -> None:
        p = Paragraph(
            children=(Text(value="text"),),
            provenance=Provenance(page=5),
        )
        assert extract_text(p) == "text"
