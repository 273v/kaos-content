"""Tests for block AST node types."""

import pytest
from pydantic import ValidationError

from kaos_content import (
    Admonition,
    Alignment,
    Attr,
    BlockQuote,
    BulletList,
    Caption,
    Cell,
    CodeBlock,
    ColSpec,
    DefinitionItem,
    DefinitionList,
    Div,
    Figure,
    Heading,
    ListItem,
    MathBlock,
    OrderedList,
    PageBreak,
    Paragraph,
    Provenance,
    RawBlock,
    Row,
    Table,
    TableSection,
    Text,
    ThematicBreak,
)


class TestParagraph:
    def test_basic(self) -> None:
        p = Paragraph(children=(Text(value="hello"),))
        assert p.node_type == "paragraph"
        assert len(p.children) == 1

    def test_multiple_inlines(self) -> None:
        from kaos_content import Emphasis, Strong

        p = Paragraph(
            children=(
                Text(value="Hello "),
                Strong(children=(Text(value="world"),)),
                Text(value=" and "),
                Emphasis(children=(Text(value="universe"),)),
            )
        )
        assert len(p.children) == 4

    def test_frozen(self) -> None:
        p = Paragraph(children=(Text(value="x"),))
        with pytest.raises(ValidationError):
            p.attr = Attr(id="changed")

    def test_with_provenance(self) -> None:
        p = Paragraph(
            children=(Text(value="x"),),
            provenance=Provenance(page=5, confidence=0.99),
        )
        assert p.provenance is not None
        assert p.provenance.page == 5


class TestHeading:
    def test_basic(self) -> None:
        h = Heading(depth=1, children=(Text(value="Title"),))
        assert h.node_type == "heading"
        assert h.depth == 1

    def test_all_depths(self) -> None:
        for d in range(1, 7):
            h = Heading(depth=d, children=(Text(value=f"H{d}"),))
            assert h.depth == d

    def test_invalid_depth_zero(self) -> None:
        with pytest.raises(ValidationError):
            Heading(depth=0, children=(Text(value="bad"),))

    def test_invalid_depth_seven(self) -> None:
        with pytest.raises(ValidationError):
            Heading(depth=7, children=(Text(value="bad"),))

    def test_with_attr(self) -> None:
        h = Heading(
            depth=2,
            children=(Text(value="Section"),),
            attr=Attr(id="sec-overview", kv={"provision-type": "recital"}),
        )
        assert h.attr.id == "sec-overview"


class TestBlockQuote:
    def test_basic(self) -> None:
        bq = BlockQuote(children=(Paragraph(children=(Text(value="quoted"),)),))
        assert bq.node_type == "blockquote"
        assert len(bq.children) == 1

    def test_nested(self) -> None:
        bq = BlockQuote(
            children=(BlockQuote(children=(Paragraph(children=(Text(value="deep"),)),)),)
        )
        assert len(bq.children) == 1


class TestLists:
    def test_bullet_list(self) -> None:
        bl = BulletList(
            children=(
                ListItem(children=(Paragraph(children=(Text(value="item 1"),)),)),
                ListItem(children=(Paragraph(children=(Text(value="item 2"),)),)),
            )
        )
        assert bl.node_type == "bullet_list"
        assert len(bl.children) == 2

    def test_ordered_list(self) -> None:
        ol = OrderedList(
            start=3,
            children=(ListItem(children=(Paragraph(children=(Text(value="third"),)),)),),
        )
        assert ol.node_type == "ordered_list"
        assert ol.start == 3

    def test_ordered_list_default_start(self) -> None:
        ol = OrderedList(children=())
        assert ol.start == 1

    def test_task_list_item(self) -> None:
        item = ListItem(
            checked=True,
            children=(Paragraph(children=(Text(value="done"),)),),
        )
        assert item.checked is True

    def test_nested_list(self) -> None:
        bl = BulletList(
            children=(
                ListItem(
                    children=(
                        Paragraph(children=(Text(value="top"),)),
                        BulletList(
                            children=(
                                ListItem(children=(Paragraph(children=(Text(value="nested"),)),)),
                            )
                        ),
                    )
                ),
            )
        )
        assert len(bl.children) == 1


class TestDefinitionList:
    def test_basic(self) -> None:
        dl = DefinitionList(
            children=(
                DefinitionItem(
                    term=(Text(value="Term"),),
                    definitions=((Paragraph(children=(Text(value="Definition"),)),),),
                ),
            )
        )
        assert dl.node_type == "definition_list"
        assert len(dl.children) == 1

    def test_multiple_definitions(self) -> None:
        di = DefinitionItem(
            term=(Text(value="Word"),),
            definitions=(
                (Paragraph(children=(Text(value="Def 1"),)),),
                (Paragraph(children=(Text(value="Def 2"),)),),
            ),
        )
        assert len(di.definitions) == 2


class TestTable:
    def test_empty(self) -> None:
        t = Table()
        assert t.node_type == "table"
        assert t.head is None
        assert t.bodies == ()

    def test_simple_table(self) -> None:
        t = Table(
            head=TableSection(
                rows=(
                    Row(
                        cells=(
                            Cell(content=(Paragraph(children=(Text(value="Name"),)),)),
                            Cell(content=(Paragraph(children=(Text(value="Age"),)),)),
                        )
                    ),
                )
            ),
            bodies=(
                TableSection(
                    rows=(
                        Row(
                            cells=(
                                Cell(content=(Paragraph(children=(Text(value="Alice"),)),)),
                                Cell(content=(Paragraph(children=(Text(value="30"),)),)),
                            )
                        ),
                    )
                ),
            ),
        )
        assert t.head is not None
        assert len(t.head.rows) == 1
        assert len(t.bodies) == 1

    def test_with_caption_and_colspecs(self) -> None:
        t = Table(
            caption=Caption(body=(Paragraph(children=(Text(value="Table 1"),)),)),
            col_specs=(
                ColSpec(alignment=Alignment.LEFT),
                ColSpec(alignment=Alignment.RIGHT, width=0.3),
            ),
        )
        assert t.caption is not None
        assert len(t.col_specs) == 2

    def test_cell_span(self) -> None:
        cell = Cell(row_span=2, col_span=3, content=())
        assert cell.row_span == 2
        assert cell.col_span == 3


class TestCodeBlock:
    def test_basic(self) -> None:
        cb = CodeBlock(value="print('hi')")
        assert cb.node_type == "codeblock"
        assert cb.language is None

    def test_with_language(self) -> None:
        cb = CodeBlock(language="python", value="x = 1")
        assert cb.language == "python"


class TestThematicBreak:
    def test_basic(self) -> None:
        tb = ThematicBreak()
        assert tb.node_type == "thematic_break"


class TestFigure:
    def test_basic(self) -> None:
        from kaos_content import Image

        fig = Figure(
            children=(Paragraph(children=(Image(src="photo.png", alt="A photo"),)),),
            caption=Caption(body=(Paragraph(children=(Text(value="Figure 1"),)),)),
        )
        assert fig.node_type == "figure"
        assert fig.caption is not None


class TestPageBreak:
    def test_basic(self) -> None:
        pb = PageBreak()
        assert pb.node_type == "page_break"


class TestDiv:
    def test_basic(self) -> None:
        d = Div(children=(Paragraph(children=(Text(value="inside div"),)),))
        assert d.node_type == "div"

    def test_with_domain_attr(self) -> None:
        d = Div(
            attr=Attr(classes=("schedule", "exhibit-a")),
            children=(Paragraph(children=(Text(value="Exhibit content"),)),),
        )
        assert "schedule" in d.attr.classes


class TestRawBlock:
    def test_basic(self) -> None:
        rb = RawBlock(format="html", value="<div>raw</div>")
        assert rb.node_type == "raw_block"
        assert rb.format == "html"


class TestMathBlock:
    def test_basic(self) -> None:
        mb = MathBlock(value="\\int_0^1 x\\,dx")
        assert mb.node_type == "math_block"


class TestAdmonition:
    def test_basic(self) -> None:
        a = Admonition(
            kind="warning",
            title="Caution",
            children=(Paragraph(children=(Text(value="Be careful"),)),),
        )
        assert a.node_type == "admonition"
        assert a.kind == "warning"
        assert a.title == "Caution"


class TestBlockJsonRoundtrip:
    """JSON round-trip for every block type."""

    def test_paragraph(self) -> None:
        node = Paragraph(children=(Text(value="p"),))
        assert Paragraph.model_validate_json(node.model_dump_json()) == node

    def test_heading(self) -> None:
        node = Heading(depth=3, children=(Text(value="h"),))
        assert Heading.model_validate_json(node.model_dump_json()) == node

    def test_blockquote(self) -> None:
        node = BlockQuote(children=(Paragraph(children=(Text(value="q"),)),))
        assert BlockQuote.model_validate_json(node.model_dump_json()) == node

    def test_ordered_list(self) -> None:
        node = OrderedList(children=(ListItem(children=(Paragraph(children=(Text(value="i"),)),)),))
        assert OrderedList.model_validate_json(node.model_dump_json()) == node

    def test_bullet_list(self) -> None:
        node = BulletList(children=(ListItem(children=(Paragraph(children=(Text(value="i"),)),)),))
        assert BulletList.model_validate_json(node.model_dump_json()) == node

    def test_definition_list(self) -> None:
        node = DefinitionList(
            children=(
                DefinitionItem(
                    term=(Text(value="T"),),
                    definitions=((Paragraph(children=(Text(value="D"),)),),),
                ),
            )
        )
        assert DefinitionList.model_validate_json(node.model_dump_json()) == node

    def test_table(self) -> None:
        node = Table(
            head=TableSection(
                rows=(Row(cells=(Cell(content=(Paragraph(children=(Text(value="h"),)),)),)),)
            )
        )
        assert Table.model_validate_json(node.model_dump_json()) == node

    def test_codeblock(self) -> None:
        node = CodeBlock(language="py", value="x=1")
        assert CodeBlock.model_validate_json(node.model_dump_json()) == node

    def test_thematic_break(self) -> None:
        node = ThematicBreak()
        assert ThematicBreak.model_validate_json(node.model_dump_json()) == node

    def test_figure(self) -> None:
        node = Figure(children=())
        assert Figure.model_validate_json(node.model_dump_json()) == node

    def test_page_break(self) -> None:
        node = PageBreak()
        assert PageBreak.model_validate_json(node.model_dump_json()) == node

    def test_div(self) -> None:
        node = Div(children=(Paragraph(children=(Text(value="d"),)),))
        assert Div.model_validate_json(node.model_dump_json()) == node

    def test_raw_block(self) -> None:
        node = RawBlock(format="html", value="<p>")
        assert RawBlock.model_validate_json(node.model_dump_json()) == node

    def test_math_block(self) -> None:
        node = MathBlock(value="x^2")
        assert MathBlock.model_validate_json(node.model_dump_json()) == node

    def test_admonition(self) -> None:
        node = Admonition(kind="note", children=())
        assert Admonition.model_validate_json(node.model_dump_json()) == node
