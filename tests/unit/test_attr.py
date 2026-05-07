"""Tests for Attr, Provenance, BoundingBox, SourceRef, and table component types."""

import pytest
from pydantic import ValidationError

from kaos_content import (
    Alignment,
    Attr,
    BoundingBox,
    Caption,
    Cell,
    ColSpec,
    CoordOrigin,
    Paragraph,
    Provenance,
    Row,
    SourceRef,
    TableSection,
    Text,
)


class TestAttr:
    def test_defaults(self) -> None:
        attr = Attr()
        assert attr.id is None
        assert attr.classes == ()
        assert attr.kv == {}

    def test_full(self) -> None:
        attr = Attr(id="sec-1", classes=("legal", "schedule"), kv={"provision-type": "recital"})
        assert attr.id == "sec-1"
        assert attr.classes == ("legal", "schedule")
        assert attr.kv["provision-type"] == "recital"

    def test_frozen(self) -> None:
        attr = Attr(id="x")
        with pytest.raises(ValidationError):
            attr.id = "y"

    def test_json_roundtrip(self) -> None:
        attr = Attr(id="sec-1", classes=("a", "b"), kv={"k": "v"})
        data = attr.model_dump_json()
        restored = Attr.model_validate_json(data)
        assert restored == attr


class TestSourceRef:
    def test_minimal(self) -> None:
        ref = SourceRef(uri="file:///doc.pdf")
        assert ref.uri == "file:///doc.pdf"
        assert ref.mime_type is None
        assert ref.artifact_id is None

    def test_full(self) -> None:
        ref = SourceRef(
            uri="file:///doc.pdf",
            mime_type="application/pdf",
            artifact_id="art-123",
        )
        assert ref.mime_type == "application/pdf"

    def test_json_roundtrip(self) -> None:
        ref = SourceRef(uri="s3://bucket/key", mime_type="text/plain", artifact_id="a1")
        assert SourceRef.model_validate_json(ref.model_dump_json()) == ref


class TestBoundingBox:
    def test_coords(self) -> None:
        bbox = BoundingBox(left=10.0, top=20.0, right=100.0, bottom=80.0)
        assert bbox.coord_origin == CoordOrigin.TOP_LEFT

    def test_bottom_left_origin(self) -> None:
        bbox = BoundingBox(
            left=0, top=0, right=100, bottom=100, coord_origin=CoordOrigin.BOTTOM_LEFT
        )
        assert bbox.coord_origin == CoordOrigin.BOTTOM_LEFT

    def test_json_roundtrip(self) -> None:
        bbox = BoundingBox(left=1, top=2, right=3, bottom=4)
        assert BoundingBox.model_validate_json(bbox.model_dump_json()) == bbox


class TestProvenance:
    def test_empty(self) -> None:
        p = Provenance()
        assert p.source is None
        assert p.page is None
        assert p.bbox is None
        assert p.char_span is None
        assert p.confidence is None
        assert p.extractor is None

    def test_full(self) -> None:
        p = Provenance(
            source=SourceRef(uri="file:///x.pdf", mime_type="application/pdf"),
            page=7,
            bbox=BoundingBox(left=10, top=20, right=300, bottom=50),
            char_span=(100, 250),
            confidence=0.95,
            extractor="pdfminer",
        )
        assert p.page == 7
        assert p.confidence == 0.95
        assert p.char_span == (100, 250)

    def test_json_roundtrip(self) -> None:
        p = Provenance(
            source=SourceRef(uri="u"),
            page=3,
            bbox=BoundingBox(left=0, top=0, right=1, bottom=1),
            char_span=(0, 10),
            confidence=0.8,
            extractor="doctr",
        )
        assert Provenance.model_validate_json(p.model_dump_json()) == p


class TestColSpec:
    def test_defaults(self) -> None:
        cs = ColSpec()
        assert cs.alignment is None
        assert cs.width is None

    def test_full(self) -> None:
        cs = ColSpec(alignment=Alignment.CENTER, width=0.5)
        assert cs.alignment == Alignment.CENTER


class TestCaption:
    def test_empty(self) -> None:
        c = Caption()
        assert c.short is None
        assert c.body == ()

    def test_with_content(self) -> None:
        c = Caption(
            short=(Text(value="Table 1"),),
            body=(Paragraph(children=(Text(value="A full caption."),)),),
        )
        assert c.short is not None
        assert len(c.body) == 1


class TestTableComponents:
    def test_cell_defaults(self) -> None:
        cell = Cell()
        assert cell.row_span == 1
        assert cell.col_span == 1
        assert cell.content == ()
        assert cell.alignment is None

    def test_cell_with_span(self) -> None:
        cell = Cell(
            row_span=2,
            col_span=3,
            content=(Paragraph(children=(Text(value="merged"),)),),
        )
        assert cell.row_span == 2
        assert cell.col_span == 3

    def test_row(self) -> None:
        row = Row(cells=(Cell(), Cell()))
        assert len(row.cells) == 2

    def test_table_section(self) -> None:
        section = TableSection(
            rows=(Row(cells=(Cell(content=(Paragraph(children=(Text(value="a"),)),)),)),)
        )
        assert len(section.rows) == 1

    def test_json_roundtrip(self) -> None:
        cell = Cell(
            attr=Attr(id="c1"),
            alignment=Alignment.RIGHT,
            row_span=2,
            col_span=1,
            content=(Paragraph(children=(Text(value="data"),)),),
        )
        restored = Cell.model_validate_json(cell.model_dump_json())
        assert restored == cell
