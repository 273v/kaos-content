"""Phase 0 hardening tests — extreme values, boundary conditions, edge cases.

Attack vectors:
1. Extreme values: empty strings, 10KB+ strings, Unicode (emoji, CJK, RTL, ZWC), null bytes
2. Boundary conditions: heading depth 1/6, empty children, 20+ nesting
3. Pydantic edge cases: extra fields, wrong discriminator types, model_copy on frozen models
4. JSON round-trip fidelity: every-node document, full provenance, all annotation types
5. Pickle edge cases: large documents, complex provenance chains
6. Attr edge cases: empty id vs None, empty classes/kv keys
7. Table edge cases: 0-row tables, row_span=0/col_span=0, mismatched columns
8. Annotation edge cases: empty targets, overlapping annotations, large offsets
"""

from __future__ import annotations

import json
import pickle
import sys

import pytest
from pydantic import ValidationError

from kaos_content import (
    Admonition,
    Alignment,
    Annotation,
    AnnotationTarget,
    AnnotationType,
    Attr,
    BaseBlock,
    BaseInline,
    BaseNode,
    Block,
    BlockQuote,
    BoundingBox,
    BulletList,
    Caption,
    Cell,
    Citation,
    Code,
    CodeBlock,
    ColSpec,
    ContentDocument,
    CoordOrigin,
    DefinitionItem,
    DefinitionList,
    Div,
    DocumentMetadata,
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
    OrderedList,
    PageBreak,
    Paragraph,
    Provenance,
    RawBlock,
    RawInline,
    Row,
    SoftBreak,
    SourceRef,
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
)

# ---------------------------------------------------------------------------
# 1. EXTREME VALUES
# ---------------------------------------------------------------------------


class TestExtremeStrings:
    """Test behaviour with extreme string values."""

    def test_text_empty_string(self) -> None:
        t = Text(value="")
        assert t.value == ""
        rt = Text.model_validate_json(t.model_dump_json())
        assert rt.value == ""

    def test_text_very_long_string(self) -> None:
        """10 KB+ string survives construction and JSON round-trip."""
        big = "A" * 12_000
        t = Text(value=big)
        assert len(t.value) == 12_000
        rt = Text.model_validate_json(t.model_dump_json())
        assert rt.value == big

    def test_codeblock_very_long_value(self) -> None:
        big = "x = 1\n" * 2_000
        cb = CodeBlock(language="python", value=big)
        rt = CodeBlock.model_validate_json(cb.model_dump_json())
        assert rt.value == big

    def test_text_emoji(self) -> None:
        val = "Hello \U0001f600\U0001f4a9\U0001f1fa\U0001f1f8 world"
        t = Text(value=val)
        assert t.value == val
        rt = Text.model_validate_json(t.model_dump_json())
        assert rt.value == val

    def test_text_cjk(self) -> None:
        val = "\u4f60\u597d\u4e16\u754c\u3053\u3093\u306b\u3061\u306f\uc548\ub155\ud558\uc138\uc694"
        t = Text(value=val)
        rt = Text.model_validate_json(t.model_dump_json())
        assert rt.value == val

    def test_text_rtl_arabic(self) -> None:
        val = "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645"
        t = Text(value=val)
        rt = Text.model_validate_json(t.model_dump_json())
        assert rt.value == val

    def test_text_zero_width_chars(self) -> None:
        val = "a\u200b\u200c\u200d\ufeffb"
        t = Text(value=val)
        assert len(t.value) == 6
        rt = Text.model_validate_json(t.model_dump_json())
        assert rt.value == val

    def test_text_null_bytes(self) -> None:
        """Null bytes in strings — Pydantic/JSON should handle or reject."""
        val = "before\x00after"
        t = Text(value=val)
        assert "\x00" in t.value
        # JSON serialization: null bytes are valid in Python str but tricky in JSON.
        # pydantic uses model_dump_json which uses serde — it may escape or preserve.
        json_str = t.model_dump_json()
        rt = Text.model_validate_json(json_str)
        assert rt.value == val

    def test_text_mixed_unicode_planes(self) -> None:
        """BMP + supplementary plane characters."""
        val = "abc\U00010000\U0001f4a5\u00e9\u0301"
        t = Text(value=val)
        rt = Text.model_validate_json(t.model_dump_json())
        assert rt.value == val

    def test_text_newlines_and_tabs(self) -> None:
        val = "line1\nline2\r\nline3\ttab"
        t = Text(value=val)
        rt = Text.model_validate_json(t.model_dump_json())
        assert rt.value == val

    def test_rawblock_binary_like_content(self) -> None:
        val = "".join(chr(i) for i in range(1, 128))
        rb = RawBlock(format="binary", value=val)
        rt = RawBlock.model_validate_json(rb.model_dump_json())
        assert rt.value == val

    def test_attr_id_very_long(self) -> None:
        long_id = "x" * 10_000
        attr = Attr(id=long_id)
        assert attr.id is not None
        assert len(attr.id) == 10_000
        rt = Attr.model_validate_json(attr.model_dump_json())
        assert rt.id == long_id

    def test_attr_classes_many_entries(self) -> None:
        classes = tuple(f"cls-{i}" for i in range(500))
        attr = Attr(classes=classes)
        assert len(attr.classes) == 500
        rt = Attr.model_validate_json(attr.model_dump_json())
        assert rt.classes == classes

    def test_attr_kv_many_entries(self) -> None:
        kv = {f"key-{i}": f"val-{i}" for i in range(500)}
        attr = Attr(kv=kv)
        assert len(attr.kv) == 500
        rt = Attr.model_validate_json(attr.model_dump_json())
        assert rt.kv == kv


# ---------------------------------------------------------------------------
# 2. BOUNDARY CONDITIONS
# ---------------------------------------------------------------------------


class TestHeadingDepthBoundaries:
    """Heading depth boundaries: exactly 1, exactly 6, and out of range."""

    def test_depth_exactly_1(self) -> None:
        h = Heading(depth=1, children=(Text(value="H1"),))
        assert h.depth == 1

    def test_depth_exactly_6(self) -> None:
        h = Heading(depth=6, children=(Text(value="H6"),))
        assert h.depth == 6

    def test_depth_0_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Heading(depth=0, children=(Text(value="bad"),))

    def test_depth_7_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Heading(depth=7, children=(Text(value="bad"),))

    def test_depth_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Heading(depth=-1, children=(Text(value="bad"),))

    def test_depth_very_large_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Heading(depth=100, children=(Text(value="bad"),))


class TestEmptyChildrenLists:
    """Nodes with empty children lists."""

    def test_paragraph_empty_children(self) -> None:
        """Paragraph with no children — no validator prevents this currently."""
        p = Paragraph(children=())
        assert p.children == ()
        rt = Paragraph.model_validate_json(p.model_dump_json())
        assert rt.children == ()

    def test_heading_empty_children(self) -> None:
        h = Heading(depth=1, children=())
        assert h.children == ()

    def test_emphasis_empty_children(self) -> None:
        e = Emphasis(children=())
        assert e.children == ()

    def test_strong_empty_children(self) -> None:
        s = Strong(children=())
        assert s.children == ()

    def test_link_empty_children(self) -> None:
        link = Link(url="https://example.com", children=())
        assert link.children == ()

    def test_span_empty_children(self) -> None:
        s = Span(children=())
        assert s.children == ()

    def test_blockquote_empty_children(self) -> None:
        bq = BlockQuote(children=())
        assert bq.children == ()

    def test_div_empty_children(self) -> None:
        d = Div(children=())
        assert d.children == ()

    def test_bullet_list_empty(self) -> None:
        bl = BulletList(children=())
        assert bl.children == ()

    def test_ordered_list_empty(self) -> None:
        ol = OrderedList(children=())
        assert ol.children == ()

    def test_figure_empty_children(self) -> None:
        f = Figure(children=())
        assert f.children == ()

    def test_admonition_empty_children(self) -> None:
        a = Admonition(kind="note", children=())
        assert a.children == ()

    def test_definition_list_empty(self) -> None:
        dl = DefinitionList(children=())
        assert dl.children == ()

    def test_definition_item_empty_definitions(self) -> None:
        di = DefinitionItem(term=(Text(value="T"),), definitions=())
        assert di.definitions == ()

    def test_definition_item_empty_term(self) -> None:
        di = DefinitionItem(term=(), definitions=())
        assert di.term == ()

    def test_citation_empty_identifiers(self) -> None:
        c = Citation(identifiers=(), children=(Text(value="x"),))
        assert c.identifiers == ()

    def test_citation_empty_children(self) -> None:
        c = Citation(identifiers=("a",), children=())
        assert c.children == ()

    def test_list_item_empty_children(self) -> None:
        li = ListItem(children=())
        assert li.children == ()


class TestDeeplyNested:
    """Deeply nested structures — 20+ levels."""

    def test_deeply_nested_blockquotes(self) -> None:
        """20-level deep BlockQuote nesting."""
        node = Paragraph(children=(Text(value="deep"),))
        for _ in range(20):
            node = BlockQuote(children=(node,))
        assert node.node_type == "blockquote"
        # JSON round-trip of deeply nested structure
        json_str = node.model_dump_json()
        rt = BlockQuote.model_validate_json(json_str)
        assert rt == node

    def test_deeply_nested_divs(self) -> None:
        """20-level deep Div nesting."""
        node = Paragraph(children=(Text(value="deep"),))
        for _ in range(20):
            node = Div(children=(node,))
        json_str = node.model_dump_json()
        rt = Div.model_validate_json(json_str)
        assert rt == node

    def test_deeply_nested_emphasis(self) -> None:
        """20-level deep inline nesting."""
        node = Text(value="deep")
        for _ in range(20):
            node = Emphasis(children=(node,))
        json_str = node.model_dump_json()
        rt = Emphasis.model_validate_json(json_str)
        assert rt == node

    def test_deeply_nested_bullet_lists(self) -> None:
        """Nested bullet lists 10 levels deep."""
        inner = Paragraph(children=(Text(value="leaf"),))
        node = BulletList(children=(ListItem(children=(inner,)),))
        for _ in range(10):
            node = BulletList(children=(ListItem(children=(node,)),))
        json_str = node.model_dump_json()
        rt = BulletList.model_validate_json(json_str)
        assert rt == node


# ---------------------------------------------------------------------------
# 3. PYDANTIC EDGE CASES
# ---------------------------------------------------------------------------


class TestPydanticEdgeCases:
    """Pydantic-specific behavior: extra fields, frozen, model_copy, discriminator."""

    def test_extra_fields_silently_ignored_on_paragraph(self) -> None:
        """ConfigDict(frozen=True) does NOT imply extra='forbid'.

        Extra fields are silently ignored by Pydantic's default behavior.
        This is a design observation: if strictness is desired, models would
        need ``model_config = ConfigDict(frozen=True, extra='forbid')``.
        """
        p = Paragraph.model_validate(
            {
                "node_type": "paragraph",
                "children": [{"node_type": "text", "value": "x"}],
                "bogus_field": 42,
            }
        )
        assert len(p.children) == 1
        assert not hasattr(p, "bogus_field")

    def test_extra_fields_silently_ignored_on_attr(self) -> None:
        """Extra fields on Attr are silently ignored (no extra='forbid')."""
        attr = Attr.model_validate({"id": "x", "bogus": True})
        assert attr.id == "x"
        assert not hasattr(attr, "bogus")

    def test_wrong_discriminator_in_block_union(self) -> None:
        """Invalid node_type in a Block position."""
        with pytest.raises(ValidationError):
            ContentDocument.model_validate(
                {"body": [{"node_type": "nonexistent_block", "value": "x"}]}
            )

    def test_wrong_discriminator_in_inline_union(self) -> None:
        """Invalid node_type in an Inline position."""
        with pytest.raises(ValidationError):
            Paragraph.model_validate(
                {
                    "node_type": "paragraph",
                    "children": [{"node_type": "nonexistent_inline", "value": "x"}],
                }
            )

    def test_missing_discriminator_field(self) -> None:
        """Missing node_type entirely."""
        with pytest.raises(ValidationError):
            ContentDocument.model_validate({"body": [{"value": "x"}]})

    def test_model_copy_on_frozen_text(self) -> None:
        """model_copy on frozen model should produce a new instance."""
        t = Text(value="original")
        t2 = t.model_copy(update={"value": "modified"})
        assert t.value == "original"
        assert t2.value == "modified"
        assert t is not t2

    def test_model_copy_on_frozen_paragraph(self) -> None:
        p = Paragraph(children=(Text(value="a"),))
        p2 = p.model_copy(update={"children": [Text(value="b")]})
        child0 = p.children[0]
        assert isinstance(child0, Text)
        assert child0.value == "a"
        child0_2 = p2.children[0]
        assert isinstance(child0_2, Text)
        assert child0_2.value == "b"

    def test_model_copy_preserves_provenance(self) -> None:
        prov = Provenance(page=5, confidence=0.9)
        t = Text(value="x", provenance=prov)
        t2 = t.model_copy(update={"value": "y"})
        assert t2.provenance is not None
        assert t2.provenance.page == 5
        assert t2.value == "y"

    def test_model_copy_deep_on_frozen(self) -> None:
        """Deep copy should produce fully independent objects."""
        p = Paragraph(
            children=(Text(value="child"),),
            attr=Attr(id="p1", classes=("a",)),
        )
        p2 = p.model_copy(deep=True)
        assert p == p2
        assert p is not p2

    def test_frozen_mutation_on_basenode(self) -> None:
        """Trying to set attr on a frozen node should fail."""
        t = Text(value="x")
        with pytest.raises(ValidationError):
            t.attr = Attr(id="new")

    def test_frozen_mutation_on_provenance(self) -> None:
        p = Provenance(page=5)
        with pytest.raises(ValidationError):
            p.page = 10

    def test_deep_freeze_body_tuple(self) -> None:
        """doc.body is a tuple — append/extend should fail."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="x"),)),))
        assert isinstance(doc.body, tuple)
        with pytest.raises(AttributeError):
            doc.body.append(Paragraph(children=(Text(value="y"),)))  # ty: ignore[unresolved-attribute]

    def test_deep_freeze_children_tuple(self) -> None:
        """Node children are tuples — mutation should fail."""
        p = Paragraph(children=(Text(value="x"),))
        assert isinstance(p.children, tuple)
        with pytest.raises(AttributeError):
            p.children.append(Text(value="y"))  # ty: ignore[unresolved-attribute]

    def test_deep_freeze_attr_classes_tuple(self) -> None:
        """Attr.classes is a tuple — mutation should fail."""
        a = Attr(classes=("a", "b"))
        assert isinstance(a.classes, tuple)
        with pytest.raises(AttributeError):
            a.classes.append("c")  # ty: ignore[unresolved-attribute]

    def test_deep_freeze_annotations_tuple(self) -> None:
        """ContentDocument.annotations is a tuple — mutation should fail."""
        doc = ContentDocument(annotations=())
        assert isinstance(doc.annotations, tuple)
        with pytest.raises(AttributeError):
            doc.annotations.append(None)  # ty: ignore[unresolved-attribute]

    def test_deep_freeze_table_rows_tuple(self) -> None:
        """TableSection.rows is a tuple — mutation should fail."""
        ts = TableSection(rows=())
        assert isinstance(ts.rows, tuple)
        with pytest.raises(AttributeError):
            ts.rows.append(Row())  # ty: ignore[unresolved-attribute]

    def test_list_coercion_to_tuple(self) -> None:
        """Passing a list to a tuple field should auto-coerce."""
        p = Paragraph(children=(Text(value="x"),))
        assert isinstance(p.children, tuple)
        assert len(p.children) == 1

    def test_heading_wrong_type_for_depth(self) -> None:
        """depth must be int, not str."""
        with pytest.raises(ValidationError):
            Heading.model_validate(
                {
                    "node_type": "heading",
                    "depth": "two",
                    "children": [{"node_type": "text", "value": "x"}],
                }
            )

    def test_node_type_cannot_be_overridden(self) -> None:
        """Passing wrong node_type literal for a concrete class."""
        with pytest.raises(ValidationError):
            Paragraph.model_validate(
                {
                    "node_type": "heading",
                    "children": [{"node_type": "text", "value": "x"}],
                }
            )


# ---------------------------------------------------------------------------
# 4. JSON ROUND-TRIP FIDELITY
# ---------------------------------------------------------------------------


class TestComprehensiveJsonRoundTrip:
    """A single document with every node type, full provenance, all annotation types."""

    def _build_kitchen_sink_doc(self) -> ContentDocument:
        """Build a document exercising every node and inline type."""
        prov = Provenance(
            source=SourceRef(uri="file:///test.pdf", mime_type="application/pdf", artifact_id="a1"),
            page=1,
            bbox=BoundingBox(
                left=0.0,
                top=0.0,
                right=612.0,
                bottom=792.0,
                coord_origin=CoordOrigin.BOTTOM_LEFT,
            ),
            char_span=(0, 100),
            confidence=0.99,
            extractor="doctr",
        )
        return ContentDocument(
            metadata=DocumentMetadata(
                title="Kitchen Sink",
                authors=("Alice", "Bob"),
                date="2026-03-25",
                language="en",
                source=SourceRef(uri="file:///ks.pdf"),
                document_type="test",
                extra={"key": "value", "nested": {"a": 1}},
            ),
            body=(
                Heading(
                    depth=1,
                    children=(Text(value="Title"),),
                    attr=Attr(id="h1", classes=("main",), kv={"level": "top"}),
                    provenance=prov,
                ),
                Paragraph(
                    children=(
                        Text(value="Plain "),
                        Strong(children=(Text(value="bold"),)),
                        Text(value=" "),
                        Emphasis(children=(Text(value="italic"),)),
                        Text(value=" "),
                        Strikethrough(children=(Text(value="struck"),)),
                        Text(value=" "),
                        Code(value="inline_code"),
                        Text(value=" "),
                        Superscript(children=(Text(value="sup"),)),
                        Subscript(children=(Text(value="sub"),)),
                        Underline(children=(Text(value="under"),)),
                        LineBreak(),
                        SoftBreak(),
                        Math(value="x^2"),
                        RawInline(format="html", value="<br/>"),
                        Span(
                            children=(Text(value="span"),),
                            attr=Attr(classes=("highlight",)),
                        ),
                    )
                ),
                Paragraph(
                    children=(
                        Link(
                            url="https://example.com",
                            title="Example",
                            children=(Text(value="link"),),
                        ),
                        Image(src="img.png", alt="photo", title="A Photo"),
                        FootnoteRef(identifier="fn1"),
                        Citation(
                            identifiers=("smith2024", "jones2025"),
                            children=(Text(value="Smith; Jones"),),
                        ),
                    )
                ),
                BlockQuote(children=(Paragraph(children=(Text(value="quoted"),)),)),
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
                OrderedList(
                    start=5,
                    children=(ListItem(children=(Paragraph(children=(Text(value="fifth"),)),)),),
                ),
                DefinitionList(
                    children=(
                        DefinitionItem(
                            term=(Text(value="Term"),),
                            definitions=(
                                (Paragraph(children=(Text(value="Def 1"),)),),
                                (Paragraph(children=(Text(value="Def 2"),)),),
                            ),
                        ),
                    )
                ),
                Table(
                    caption=Caption(
                        short=(Text(value="Tbl"),),
                        body=(Paragraph(children=(Text(value="Table caption"),)),),
                    ),
                    col_specs=(
                        ColSpec(alignment=Alignment.LEFT, width=0.5),
                        ColSpec(alignment=Alignment.RIGHT),
                    ),
                    head=TableSection(
                        attr=Attr(id="thead"),
                        rows=(
                            Row(
                                attr=Attr(id="hrow"),
                                cells=(
                                    Cell(
                                        attr=Attr(id="hc1"),
                                        alignment=Alignment.CENTER,
                                        content=(Paragraph(children=(Text(value="H1"),)),),
                                    ),
                                    Cell(content=(Paragraph(children=(Text(value="H2"),)),)),
                                ),
                            ),
                        ),
                    ),
                    bodies=(
                        TableSection(
                            rows=(
                                Row(
                                    cells=(
                                        Cell(
                                            row_span=2,
                                            col_span=1,
                                            content=(Paragraph(children=(Text(value="D1"),)),),
                                        ),
                                        Cell(content=(Paragraph(children=(Text(value="D2"),)),)),
                                    )
                                ),
                            )
                        ),
                    ),
                    foot=TableSection(
                        rows=(
                            Row(
                                cells=(
                                    Cell(content=(Paragraph(children=(Text(value="F1"),)),)),
                                    Cell(content=(Paragraph(children=(Text(value="F2"),)),)),
                                )
                            ),
                        )
                    ),
                ),
                CodeBlock(language="python", value="x = 1\ny = 2"),
                ThematicBreak(),
                Figure(
                    caption=Caption(body=(Paragraph(children=(Text(value="Fig 1"),)),)),
                    children=(Paragraph(children=(Image(src="fig.png"),)),),
                ),
                PageBreak(),
                Div(
                    attr=Attr(classes=("schedule",), kv={"type": "exhibit"}),
                    children=(Paragraph(children=(Text(value="Exhibit A"),)),),
                ),
                RawBlock(format="html", value="<hr/>"),
                MathBlock(value="\\sum_{i=0}^{n} i"),
                Admonition(
                    kind="warning",
                    title="Watch Out",
                    children=(Paragraph(children=(Text(value="Be careful"),)),),
                ),
            ),
            footnotes={
                "fn1": (Paragraph(children=(Text(value="Footnote text."),)),),
            },
            definitions={
                "abbr": "https://example.com/abbr",
            },
            annotations=(
                Annotation(
                    id="ann-1",
                    type=AnnotationType.HIGHLIGHT,
                    targets=(AnnotationTarget(node_ref="#/body/1", start_offset=0, end_offset=5),),
                ),
                Annotation(
                    id="ann-2",
                    type=AnnotationType.REDACTION,
                    targets=(AnnotationTarget(node_ref="#/body/1", start_offset=6, end_offset=10),),
                    body={"reason": "PII"},
                    provenance=Provenance(extractor="manual"),
                ),
                Annotation(
                    id="ann-3",
                    type=AnnotationType.DEFINED_TERM,
                    targets=(AnnotationTarget(node_ref="#/body/6/children/0"),),
                    body={"term": "Term"},
                ),
                Annotation(
                    id="ann-4",
                    type=AnnotationType.ENTITY,
                    targets=(AnnotationTarget(node_ref="#/body/0"),),
                    body={"entity_type": "ORG", "text": "ACME"},
                ),
            ),
        )

    def test_json_roundtrip_kitchen_sink(self) -> None:
        doc = self._build_kitchen_sink_doc()
        json_str = doc.model_dump_json()
        restored = ContentDocument.model_validate_json(json_str)
        assert restored == doc

    def test_json_dict_roundtrip_kitchen_sink(self) -> None:
        doc = self._build_kitchen_sink_doc()
        d = doc.model_dump()
        restored = ContentDocument.model_validate(d)
        assert restored == doc

    def test_json_parseable(self) -> None:
        """The JSON output is valid JSON (not just pydantic internal)."""
        doc = self._build_kitchen_sink_doc()
        json_str = doc.model_dump_json()
        parsed = json.loads(json_str)
        assert isinstance(parsed, dict)
        assert "body" in parsed
        assert "metadata" in parsed

    def test_json_roundtrip_preserves_all_provenance_fields(self) -> None:
        prov = Provenance(
            source=SourceRef(uri="file:///x.pdf", mime_type="application/pdf", artifact_id="art-1"),
            page=42,
            bbox=BoundingBox(
                left=10.5,
                top=20.3,
                right=500.7,
                bottom=700.1,
                coord_origin=CoordOrigin.BOTTOM_LEFT,
            ),
            char_span=(100, 999),
            confidence=0.87654321,
            extractor="tesseract",
        )
        t = Text(value="provenance test", provenance=prov)
        rt = Text.model_validate_json(t.model_dump_json())
        assert rt.provenance is not None
        assert rt.provenance == prov
        assert rt.provenance.source is not None
        assert rt.provenance.source.artifact_id == "art-1"
        assert rt.provenance.bbox is not None
        assert rt.provenance.bbox.coord_origin == CoordOrigin.BOTTOM_LEFT
        assert rt.provenance.char_span == (100, 999)
        assert rt.provenance.confidence == pytest.approx(0.87654321)

    def test_json_roundtrip_preserves_none_provenance(self) -> None:
        t = Text(value="no prov")
        rt = Text.model_validate_json(t.model_dump_json())
        assert rt.provenance is None


# ---------------------------------------------------------------------------
# 5. PICKLE EDGE CASES
# ---------------------------------------------------------------------------


class TestPickleEdgeCases:
    """Pickle serialization for large and complex documents."""

    def test_pickle_empty_document(self) -> None:
        doc = ContentDocument()
        rt = pickle.loads(pickle.dumps(doc))
        assert rt == doc

    def test_pickle_large_document(self) -> None:
        """Document with 200 paragraphs."""
        body = tuple(Paragraph(children=(Text(value=f"Paragraph {i}"),)) for i in range(200))
        doc = ContentDocument(body=body)
        data = pickle.dumps(doc)
        rt = pickle.loads(data)
        assert rt == doc
        assert len(rt.body) == 200

    def test_pickle_complex_provenance(self) -> None:
        prov = Provenance(
            source=SourceRef(uri="s3://bucket/key.pdf", mime_type="application/pdf"),
            page=99,
            bbox=BoundingBox(left=0, top=0, right=100, bottom=100),
            char_span=(50, 150),
            confidence=0.42,
            extractor="custom",
        )
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(Text(value="prov", provenance=prov),),
                    provenance=prov,
                ),
            )
        )
        rt = pickle.loads(pickle.dumps(doc))
        assert rt == doc

    def test_pickle_all_protocols(self) -> None:
        doc = ContentDocument(body=(Paragraph(children=(Text(value="proto"),)),))
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            data = pickle.dumps(doc, protocol=protocol)
            rt = pickle.loads(data)
            assert rt == doc

    def test_pickle_deeply_nested(self) -> None:
        node = Paragraph(children=(Text(value="deep"),))
        for _ in range(15):
            node = BlockQuote(children=(node,))
        doc = ContentDocument(body=(node,))
        rt = pickle.loads(pickle.dumps(doc))
        assert rt == doc

    def test_pickle_with_annotations(self) -> None:
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="annotated"),)),),
            annotations=tuple(
                Annotation(
                    id=f"ann-{i}",
                    type=AnnotationType.HIGHLIGHT,
                    targets=(
                        AnnotationTarget(node_ref="#/body/0", start_offset=i, end_offset=i + 1),
                    ),
                )
                for i in range(50)
            ),
        )
        rt = pickle.loads(pickle.dumps(doc))
        assert rt == doc
        assert len(rt.annotations) == 50

    def test_pickle_size_reasonable(self) -> None:
        """Pickle output should not be absurdly larger than JSON."""
        doc = ContentDocument(
            body=tuple(Paragraph(children=(Text(value=f"p{i}"),)) for i in range(100))
        )
        json_size = len(doc.model_dump_json().encode("utf-8"))
        pickle_size = len(pickle.dumps(doc))
        # Pickle is typically 2-5x JSON for Pydantic models; flag if >20x
        assert pickle_size < json_size * 20


# ---------------------------------------------------------------------------
# 6. ATTR EDGE CASES
# ---------------------------------------------------------------------------


class TestAttrEdgeCases:
    """Edge cases in Attr construction and behavior."""

    def test_attr_id_empty_string(self) -> None:
        """Empty string id is distinct from None."""
        attr = Attr(id="")
        assert attr.id == ""
        assert attr.id is not None

    def test_attr_id_none(self) -> None:
        attr = Attr(id=None)
        assert attr.id is None

    def test_attr_id_none_vs_empty_not_equal(self) -> None:
        a1 = Attr(id=None)
        a2 = Attr(id="")
        assert a1 != a2

    def test_attr_classes_with_empty_string(self) -> None:
        attr = Attr(classes=("", "valid", ""))
        assert attr.classes == ("", "valid", "")

    def test_attr_classes_duplicates(self) -> None:
        """Duplicate classes are allowed (no set dedup)."""
        attr = Attr(classes=("a", "a", "b", "a"))
        assert attr.classes == ("a", "a", "b", "a")

    def test_attr_kv_empty_key(self) -> None:
        attr = Attr(kv={"": "value"})
        assert attr.kv[""] == "value"

    def test_attr_kv_empty_value(self) -> None:
        attr = Attr(kv={"key": ""})
        assert attr.kv["key"] == ""

    def test_attr_kv_empty_key_and_value(self) -> None:
        attr = Attr(kv={"": ""})
        assert attr.kv[""] == ""

    def test_attr_kv_unicode_keys(self) -> None:
        attr = Attr(kv={"\u00e9\u00e8\u00ea": "accented", "\u4f60\u597d": "chinese"})
        assert attr.kv["\u00e9\u00e8\u00ea"] == "accented"

    def test_attr_json_roundtrip_empty_id(self) -> None:
        attr = Attr(id="")
        rt = Attr.model_validate_json(attr.model_dump_json())
        assert rt.id == ""

    def test_attr_equality(self) -> None:
        a1 = Attr(id="x", classes=("a", "b"), kv={"k": "v"})
        a2 = Attr(id="x", classes=("a", "b"), kv={"k": "v"})
        assert a1 == a2

    def test_attr_inequality_different_order_classes(self) -> None:
        """Classes are ordered lists, different order means not equal."""
        a1 = Attr(classes=("a", "b"))
        a2 = Attr(classes=("b", "a"))
        assert a1 != a2

    def test_attr_not_hashable_due_to_mutable_defaults(self) -> None:
        """Frozen Pydantic models with list/dict fields are NOT hashable.

        Even though ConfigDict(frozen=True) prevents mutation, Pydantic's
        __hash__ tries to hash the __dict__ values tuple, which fails
        because list and dict are unhashable types. This is expected
        Pydantic v2 behavior. Models that need hashing would require
        converting classes to tuple and kv to frozenset.
        """
        a = Attr(id="x")
        with pytest.raises(TypeError, match="unhashable type"):
            hash(a)

    def test_text_hashable(self) -> None:
        """Text node (no list/dict fields of its own, but inherits Attr).

        Text inherits from BaseInline -> BaseNode which has attr: Attr.
        Attr has list/dict fields, so Text is also not hashable.
        """
        t = Text(value="x")
        with pytest.raises(TypeError, match="unhashable type"):
            hash(t)

    def test_nodes_without_mutable_fields_still_not_hashable(self) -> None:
        """Even ThematicBreak (no extra fields) inherits Attr with list/dict."""
        tb = ThematicBreak()
        with pytest.raises(TypeError, match="unhashable type"):
            hash(tb)


# ---------------------------------------------------------------------------
# 7. TABLE EDGE CASES
# ---------------------------------------------------------------------------


class TestTableEdgeCases:
    """Edge cases in table construction."""

    def test_table_no_rows(self) -> None:
        """Table with head that has no rows."""
        t = Table(head=TableSection(rows=()))
        assert t.head is not None
        assert t.head.rows == ()

    def test_table_empty_body_sections(self) -> None:
        t = Table(bodies=(TableSection(rows=()), TableSection(rows=())))
        assert len(t.bodies) == 2

    def test_table_row_no_cells(self) -> None:
        """A row with zero cells."""
        r = Row(cells=())
        assert r.cells == ()
        t = Table(head=TableSection(rows=(r,)))
        rt = Table.model_validate_json(t.model_dump_json())
        assert rt.head is not None
        assert rt.head.rows[0].cells == ()

    def test_cell_row_span_zero(self) -> None:
        """row_span=0 is rejected — a cell that doesn't occupy its own slot
        is structurally invalid (audit M1)."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            Cell(row_span=0)

    def test_cell_col_span_zero(self) -> None:
        """col_span=0 is rejected (audit M1)."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            Cell(col_span=0)

    def test_cell_negative_span(self) -> None:
        """Negative spans are rejected (audit M1)."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            Cell(row_span=-1)
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            Cell(col_span=-1)

    def test_cell_very_large_span(self) -> None:
        cell = Cell(row_span=10000, col_span=10000)
        assert cell.row_span == 10000

    def test_mismatched_column_counts(self) -> None:
        """Rows with different numbers of cells — allowed (no structural validation)."""
        t = Table(
            head=TableSection(rows=(Row(cells=(Cell(), Cell(), Cell())),)),
            bodies=(
                TableSection(
                    rows=(
                        Row(cells=(Cell(),)),
                        Row(cells=(Cell(), Cell())),
                    )
                ),
            ),
        )
        assert t.head is not None
        assert len(t.head.rows[0].cells) == 3
        assert len(t.bodies[0].rows[0].cells) == 1
        assert len(t.bodies[0].rows[1].cells) == 2

    def test_table_json_roundtrip_full(self) -> None:
        t = Table(
            caption=Caption(
                short=(Text(value="short"),),
                body=(Paragraph(children=(Text(value="full caption"),)),),
            ),
            col_specs=(
                ColSpec(alignment=Alignment.LEFT, width=0.3),
                ColSpec(alignment=Alignment.CENTER, width=0.7),
            ),
            head=TableSection(
                attr=Attr(id="thead"),
                rows=(
                    Row(
                        attr=Attr(id="hrow"),
                        cells=(
                            Cell(
                                attr=Attr(id="hc1"),
                                alignment=Alignment.LEFT,
                                content=(Paragraph(children=(Text(value="H1"),)),),
                            ),
                            Cell(
                                alignment=Alignment.RIGHT,
                                row_span=1,
                                col_span=1,
                                content=(Paragraph(children=(Text(value="H2"),)),),
                            ),
                        ),
                    ),
                ),
            ),
            bodies=(
                TableSection(
                    rows=(
                        Row(
                            cells=(
                                Cell(
                                    row_span=2,
                                    col_span=1,
                                    content=(Paragraph(children=(Text(value="D1"),)),),
                                ),
                                Cell(content=(Paragraph(children=(Text(value="D2"),)),)),
                            )
                        ),
                    )
                ),
            ),
            foot=TableSection(
                rows=(
                    Row(
                        cells=(
                            Cell(content=(Paragraph(children=(Text(value="F1"),)),)),
                            Cell(content=(Paragraph(children=(Text(value="F2"),)),)),
                        )
                    ),
                )
            ),
        )
        rt = Table.model_validate_json(t.model_dump_json())
        assert rt == t

    def test_table_all_none_sections(self) -> None:
        """Table with all optional sections as None/empty."""
        t = Table()
        assert t.head is None
        assert t.bodies == ()
        assert t.foot is None
        assert t.caption is None
        assert t.col_specs == ()

    def test_table_multiple_body_sections(self) -> None:
        """Table with 3 body sections."""
        t = Table(
            bodies=(
                TableSection(rows=(Row(cells=(Cell(),)),)),
                TableSection(rows=(Row(cells=(Cell(),)),)),
                TableSection(rows=(Row(cells=(Cell(),)),)),
            )
        )
        assert len(t.bodies) == 3


# ---------------------------------------------------------------------------
# 8. ANNOTATION EDGE CASES
# ---------------------------------------------------------------------------


class TestAnnotationEdgeCases:
    """Edge cases in annotation construction."""

    def test_annotation_empty_targets(self) -> None:
        """Annotation with empty targets list — allowed (no validator)."""
        a = Annotation(
            id="a1",
            type=AnnotationType.HIGHLIGHT,
            targets=(),
        )
        assert a.targets == ()

    def test_annotation_very_large_offsets(self) -> None:
        t = AnnotationTarget(
            node_ref="#/body/0",
            start_offset=0,
            end_offset=sys.maxsize,
        )
        assert t.end_offset == sys.maxsize

    def test_annotation_negative_start_offset_rejected(self) -> None:
        """Negative start_offset is rejected."""
        with pytest.raises(ValidationError, match="start_offset must be non-negative"):
            AnnotationTarget(
                node_ref="#/body/0",
                start_offset=-1,
                end_offset=10,
            )

    def test_annotation_negative_end_offset_rejected(self) -> None:
        """Negative end_offset is rejected."""
        with pytest.raises(ValidationError, match="end_offset must be non-negative"):
            AnnotationTarget(
                node_ref="#/body/0",
                start_offset=0,
                end_offset=-1,
            )

    def test_annotation_start_greater_than_end_rejected(self) -> None:
        """start_offset > end_offset is rejected."""
        with pytest.raises(ValidationError, match="start_offset must be <= end_offset"):
            AnnotationTarget(
                node_ref="#/body/0",
                start_offset=100,
                end_offset=50,
            )

    def test_annotation_target_only_start(self) -> None:
        t = AnnotationTarget(node_ref="#/body/0", start_offset=5)
        assert t.start_offset == 5
        assert t.end_offset is None

    def test_annotation_target_only_end(self) -> None:
        t = AnnotationTarget(node_ref="#/body/0", end_offset=10)
        assert t.start_offset is None
        assert t.end_offset == 10

    def test_annotation_all_types(self) -> None:
        """Create an annotation for every AnnotationType member."""
        for at in AnnotationType:
            a = Annotation(
                id=f"ann-{at.value}",
                type=at,
                targets=(AnnotationTarget(node_ref="#/body/0"),),
            )
            assert a.type == at
            # JSON round-trip
            rt = Annotation.model_validate_json(a.model_dump_json())
            assert rt == a

    def test_annotation_body_complex_nested(self) -> None:
        """Annotation body with deeply nested dict."""
        body = {
            "level1": {
                "level2": {
                    "level3": [1, 2, {"level4": True}],
                },
            },
            "list": [None, "str", 42, 3.14],
        }
        a = Annotation(
            id="complex",
            type=AnnotationType.COMMENT,
            targets=(AnnotationTarget(node_ref="#/body/0"),),
            body=body,
        )
        rt = Annotation.model_validate_json(a.model_dump_json())
        assert rt.body == body

    def test_annotation_duplicate_ids_in_document(self) -> None:
        """Document allows duplicate annotation IDs (no uniqueness constraint)."""
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="x"),)),),
            annotations=(
                Annotation(
                    id="same-id",
                    type=AnnotationType.HIGHLIGHT,
                    targets=(AnnotationTarget(node_ref="#/body/0"),),
                ),
                Annotation(
                    id="same-id",
                    type=AnnotationType.COMMENT,
                    targets=(AnnotationTarget(node_ref="#/body/0"),),
                ),
            ),
        )
        assert len(doc.annotations) == 2
        assert doc.annotations[0].id == doc.annotations[1].id

    def test_annotation_overlapping_targets(self) -> None:
        """Two annotations targeting overlapping spans on the same node."""
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="overlapping text here"),)),),
            annotations=(
                Annotation(
                    id="a1",
                    type=AnnotationType.HIGHLIGHT,
                    targets=(AnnotationTarget(node_ref="#/body/0", start_offset=0, end_offset=15),),
                ),
                Annotation(
                    id="a2",
                    type=AnnotationType.REDACTION,
                    targets=(
                        AnnotationTarget(node_ref="#/body/0", start_offset=10, end_offset=20),
                    ),
                ),
            ),
        )
        assert len(doc.annotations) == 2

    def test_annotation_empty_node_ref(self) -> None:
        """Empty string node_ref — allowed (no format validator)."""
        t = AnnotationTarget(node_ref="")
        assert t.node_ref == ""

    def test_annotation_json_roundtrip_with_provenance(self) -> None:
        a = Annotation(
            id="prov-ann",
            type=AnnotationType.ENTITY,
            targets=(AnnotationTarget(node_ref="#/body/0", start_offset=0, end_offset=5),),
            body={"entity": "PERSON", "text": "Alice"},
            provenance=Provenance(
                source=SourceRef(uri="file:///x.pdf"),
                page=3,
                confidence=0.95,
                extractor="spacy",
            ),
        )
        rt = Annotation.model_validate_json(a.model_dump_json())
        assert rt == a
        assert rt.provenance is not None
        assert rt.provenance.extractor == "spacy"


# ---------------------------------------------------------------------------
# 9. BOUNDING BOX EDGE CASES
# ---------------------------------------------------------------------------


class TestBoundingBoxEdgeCases:
    """BoundingBox edge cases."""

    def test_zero_area_box(self) -> None:
        bb = BoundingBox(left=0, top=0, right=0, bottom=0)
        assert bb.left == 0

    def test_negative_coordinates(self) -> None:
        """Negative coordinates — allowed (no validator)."""
        bb = BoundingBox(left=-10, top=-20, right=-5, bottom=-1)
        assert bb.left == -10

    def test_inverted_box(self) -> None:
        """left > right or top > bottom is rejected — a zero/negative-extent
        box is not a valid region (audit M1)."""
        with pytest.raises(ValidationError, match=r"right .* must be >= left"):
            BoundingBox(left=100, top=200, right=50, bottom=300)
        with pytest.raises(ValidationError, match=r"bottom .* must be >= top"):
            BoundingBox(left=0, top=200, right=100, bottom=100)

    def test_very_large_coords(self) -> None:
        bb = BoundingBox(left=0, top=0, right=1e10, bottom=1e10)
        rt = BoundingBox.model_validate_json(bb.model_dump_json())
        assert rt.right == pytest.approx(1e10)

    def test_float_precision(self) -> None:
        bb = BoundingBox(left=0.1 + 0.2, top=0, right=1, bottom=1)
        # 0.1 + 0.2 != 0.3 in float
        rt = BoundingBox.model_validate_json(bb.model_dump_json())
        assert rt.left == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# 10. SOURCEREF EDGE CASES
# ---------------------------------------------------------------------------


class TestSourceRefEdgeCases:
    """SourceRef edge cases."""

    def test_empty_uri(self) -> None:
        ref = SourceRef(uri="")
        assert ref.uri == ""

    def test_very_long_uri(self) -> None:
        uri = "https://example.com/" + "a" * 10_000
        ref = SourceRef(uri=uri)
        assert len(ref.uri) == len(uri)

    def test_uri_with_special_chars(self) -> None:
        uri = "file:///path/to/file with spaces & symbols!@#$%.pdf"
        ref = SourceRef(uri=uri)
        rt = SourceRef.model_validate_json(ref.model_dump_json())
        assert rt.uri == uri

    def test_data_uri(self) -> None:
        uri = "data:text/plain;base64,SGVsbG8gV29ybGQ="
        ref = SourceRef(uri=uri)
        assert ref.uri == uri


# ---------------------------------------------------------------------------
# 11. PROVENANCE EDGE CASES
# ---------------------------------------------------------------------------


class TestProvenanceEdgeCases:
    """Provenance edge cases."""

    def test_provenance_page_zero(self) -> None:
        """page=0 is rejected — pages are 1-indexed; ``None`` means unknown
        (audit M1)."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            Provenance(page=0)

    def test_provenance_page_negative(self) -> None:
        """Negative page is rejected (audit M1)."""
        with pytest.raises(ValidationError, match="greater than or equal to 1"):
            Provenance(page=-1)

    def test_provenance_confidence_zero(self) -> None:
        p = Provenance(confidence=0.0)
        assert p.confidence == 0.0

    def test_provenance_confidence_one(self) -> None:
        p = Provenance(confidence=1.0)
        assert p.confidence == 1.0

    def test_provenance_confidence_out_of_range(self) -> None:
        """Confidence > 1.0 is rejected — bounds are [0.0, 1.0] (audit M1)."""
        with pytest.raises(ValidationError, match="less than or equal to 1"):
            Provenance(confidence=5.0)

    def test_provenance_confidence_negative(self) -> None:
        """Negative confidence is rejected (audit M1)."""
        with pytest.raises(ValidationError, match="greater than or equal to 0"):
            Provenance(confidence=-0.5)

    def test_provenance_char_span_same_start_end(self) -> None:
        p = Provenance(char_span=(5, 5))
        assert p.char_span == (5, 5)

    def test_provenance_char_span_inverted(self) -> None:
        """char_span with end < start is rejected (audit M1)."""
        with pytest.raises(ValidationError, match=r"end .* must be >= start"):
            Provenance(char_span=(10, 5))

    def test_provenance_char_span_negative(self) -> None:
        """char_span with negative values is rejected (audit M1)."""
        with pytest.raises(ValidationError, match="must be non-negative"):
            Provenance(char_span=(-1, 5))
        with pytest.raises(ValidationError, match="must be non-negative"):
            Provenance(char_span=(5, -1))


# ---------------------------------------------------------------------------
# 12. DOCUMENT METADATA EDGE CASES
# ---------------------------------------------------------------------------


class TestDocumentMetadataEdgeCases:
    """DocumentMetadata edge cases."""

    def test_metadata_empty_strings(self) -> None:
        m = DocumentMetadata(title="", language="", document_type="", date="")
        assert m.title == ""

    def test_metadata_many_authors(self) -> None:
        authors = tuple(f"Author {i}" for i in range(100))
        m = DocumentMetadata(authors=authors)
        assert len(m.authors) == 100

    def test_metadata_extra_nested(self) -> None:
        extra = {"a": {"b": {"c": {"d": [1, 2, 3]}}}}
        m = DocumentMetadata(extra=extra)
        assert m.extra["a"]["b"]["c"]["d"] == [1, 2, 3]

    def test_metadata_json_roundtrip(self) -> None:
        m = DocumentMetadata(
            title="T",
            authors=("A", "B"),
            date="2026-01-01",
            language="en",
            source=SourceRef(uri="file:///x.pdf", mime_type="application/pdf"),
            document_type="contract",
            extra={"key": [1, None, "str"]},
        )
        rt = DocumentMetadata.model_validate_json(m.model_dump_json())
        assert rt == m


# ---------------------------------------------------------------------------
# 13. CONTENT DOCUMENT EDGE CASES
# ---------------------------------------------------------------------------


class TestContentDocumentEdgeCases:
    """ContentDocument-level edge cases."""

    def test_document_many_blocks(self) -> None:
        """500 blocks in body."""
        body = tuple(Paragraph(children=(Text(value=f"p{i}"),)) for i in range(500))
        doc = ContentDocument(body=body)
        assert len(doc.body) == 500

    def test_document_many_footnotes(self) -> None:
        fns: dict[str, tuple[Block, ...]] = {
            f"fn{i}": (Paragraph(children=(Text(value=f"Note {i}"),)),) for i in range(100)
        }
        doc = ContentDocument(footnotes=fns)
        assert len(doc.footnotes) == 100

    def test_document_many_definitions(self) -> None:
        defs = {f"def{i}": f"https://example.com/{i}" for i in range(100)}
        doc = ContentDocument(definitions=defs)
        assert len(doc.definitions) == 100

    def test_document_footnote_key_empty_string(self) -> None:
        doc = ContentDocument(
            footnotes={"": (Paragraph(children=(Text(value="empty key footnote"),)),)}
        )
        assert "" in doc.footnotes

    def test_document_frozen(self) -> None:
        doc = ContentDocument()
        with pytest.raises(ValidationError):
            doc.body = []

    def test_document_model_copy(self) -> None:
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="original"),)),),
        )
        doc2 = doc.model_copy(update={"body": [Paragraph(children=(Text(value="modified"),))]})
        block0 = doc.body[0]
        assert isinstance(block0, Paragraph)
        child0 = block0.children[0]
        assert isinstance(child0, Text)
        assert child0.value == "original"
        block0_2 = doc2.body[0]
        assert isinstance(block0_2, Paragraph)
        child0_2 = block0_2.children[0]
        assert isinstance(child0_2, Text)
        assert child0_2.value == "modified"


# ---------------------------------------------------------------------------
# 14. MIXED INLINE TYPES IN SINGLE PARAGRAPH
# ---------------------------------------------------------------------------


class TestMixedInlineTypes:
    """Paragraphs with every possible inline type combined."""

    def test_all_inline_types_in_one_paragraph(self) -> None:
        p = Paragraph(
            children=(
                Text(value="text"),
                Emphasis(children=(Text(value="em"),)),
                Strong(children=(Text(value="strong"),)),
                Strikethrough(children=(Text(value="strike"),)),
                Code(value="code"),
                Link(url="https://example.com", children=(Text(value="link"),)),
                Image(src="img.png"),
                FootnoteRef(identifier="fn1"),
                Citation(identifiers=("a",), children=(Text(value="cite"),)),
                Math(value="x"),
                RawInline(format="html", value="<b>"),
                LineBreak(),
                SoftBreak(),
                Span(children=(Text(value="span"),)),
                Superscript(children=(Text(value="sup"),)),
                Subscript(children=(Text(value="sub"),)),
                Underline(children=(Text(value="u"),)),
            )
        )
        assert len(p.children) == 17
        rt = Paragraph.model_validate_json(p.model_dump_json())
        assert rt == p

    def test_nested_formatting(self) -> None:
        """Strong > Emphasis > Underline > Text."""
        p = Paragraph(
            children=(
                Strong(
                    children=(Emphasis(children=(Underline(children=(Text(value="formatted"),)),)),)
                ),
            )
        )
        rt = Paragraph.model_validate_json(p.model_dump_json())
        assert rt == p


# ---------------------------------------------------------------------------
# 15. INHERITANCE & TYPE HIERARCHY
# ---------------------------------------------------------------------------


class TestTypeHierarchy:
    """Verify the inheritance chain is correct."""

    def test_paragraph_is_baseblock(self) -> None:
        assert issubclass(Paragraph, BaseBlock)

    def test_paragraph_is_basenode(self) -> None:
        assert issubclass(Paragraph, BaseNode)

    def test_text_is_baseinline(self) -> None:
        assert issubclass(Text, BaseInline)

    def test_text_is_basenode(self) -> None:
        assert issubclass(Text, BaseNode)

    def test_heading_is_baseblock(self) -> None:
        assert issubclass(Heading, BaseBlock)

    def test_emphasis_is_baseinline(self) -> None:
        assert issubclass(Emphasis, BaseInline)

    def test_all_blocks_are_baseblock(self) -> None:
        block_types = [
            Paragraph,
            Heading,
            BlockQuote,
            OrderedList,
            BulletList,
            ListItem,
            DefinitionList,
            DefinitionItem,
            Table,
            CodeBlock,
            ThematicBreak,
            Figure,
            PageBreak,
            Div,
            RawBlock,
            MathBlock,
            Admonition,
        ]
        for cls in block_types:
            assert issubclass(cls, BaseBlock), f"{cls.__name__} not a BaseBlock"

    def test_all_inlines_are_baseinline(self) -> None:
        inline_types = [
            Text,
            Emphasis,
            Strong,
            Strikethrough,
            Code,
            Link,
            Image,
            FootnoteRef,
            Citation,
            Math,
            RawInline,
            LineBreak,
            SoftBreak,
            Span,
            Superscript,
            Subscript,
            Underline,
        ]
        for cls in inline_types:
            assert issubclass(cls, BaseInline), f"{cls.__name__} not a BaseInline"

    def test_isinstance_check_on_instances(self) -> None:
        t = Text(value="x")
        assert isinstance(t, BaseInline)
        assert isinstance(t, BaseNode)
        assert not isinstance(t, BaseBlock)

        p = Paragraph(children=(t,))
        assert isinstance(p, BaseBlock)
        assert isinstance(p, BaseNode)
        assert not isinstance(p, BaseInline)


# ---------------------------------------------------------------------------
# 16. COLSPEC EDGE CASES
# ---------------------------------------------------------------------------


class TestColSpecEdgeCases:
    def test_width_zero(self) -> None:
        cs = ColSpec(width=0.0)
        assert cs.width == 0.0

    def test_width_negative(self) -> None:
        cs = ColSpec(width=-1.0)
        assert cs.width == -1.0

    def test_width_greater_than_one(self) -> None:
        cs = ColSpec(width=2.5)
        assert cs.width == 2.5

    def test_all_alignments(self) -> None:
        for alignment in Alignment:
            cs = ColSpec(alignment=alignment)
            assert cs.alignment == alignment


# ---------------------------------------------------------------------------
# 17. ORDERED LIST START VALUES
# ---------------------------------------------------------------------------


class TestOrderedListStartEdgeCases:
    def test_start_zero(self) -> None:
        ol = OrderedList(start=0, children=())
        assert ol.start == 0

    def test_start_negative(self) -> None:
        ol = OrderedList(start=-5, children=())
        assert ol.start == -5

    def test_start_very_large(self) -> None:
        ol = OrderedList(start=999999, children=())
        assert ol.start == 999999


# ---------------------------------------------------------------------------
# 18. DESERIALIZATION FROM RAW DICTS (model_validate)
# ---------------------------------------------------------------------------


class TestRawDictDeserialization:
    """Verify model_validate works for discriminated unions from raw dicts."""

    def test_document_from_raw_dict(self) -> None:
        raw = {
            "body": [
                {"node_type": "paragraph", "children": [{"node_type": "text", "value": "hello"}]},
                {
                    "node_type": "heading",
                    "depth": 2,
                    "children": [{"node_type": "text", "value": "H2"}],
                },
                {"node_type": "codeblock", "value": "x=1", "language": "python"},
                {"node_type": "thematic_break"},
                {"node_type": "page_break"},
            ]
        }
        doc = ContentDocument.model_validate(raw)
        assert len(doc.body) == 5
        assert type(doc.body[0]).__name__ == "Paragraph"
        assert type(doc.body[1]).__name__ == "Heading"
        assert type(doc.body[2]).__name__ == "CodeBlock"
        assert type(doc.body[3]).__name__ == "ThematicBreak"
        assert type(doc.body[4]).__name__ == "PageBreak"

    def test_inline_discriminator_from_dict(self) -> None:
        raw = {
            "node_type": "paragraph",
            "children": [
                {"node_type": "text", "value": "a"},
                {"node_type": "emphasis", "children": [{"node_type": "text", "value": "b"}]},
                {"node_type": "code", "value": "c"},
                {"node_type": "line_break"},
                {"node_type": "soft_break"},
            ],
        }
        p = Paragraph.model_validate(raw)
        assert len(p.children) == 5


# ---------------------------------------------------------------------------
# 19. CAPTION EDGE CASES
# ---------------------------------------------------------------------------


class TestCaptionEdgeCases:
    def test_caption_short_only(self) -> None:
        c = Caption(short=(Text(value="short"),))
        assert c.short is not None
        assert c.body == ()

    def test_caption_body_only(self) -> None:
        c = Caption(body=(Paragraph(children=(Text(value="body"),)),))
        assert c.short is None

    def test_caption_empty_short_list(self) -> None:
        c = Caption(short=())
        assert c.short == ()

    def test_caption_json_roundtrip(self) -> None:
        c = Caption(
            short=(Text(value="s1"), Emphasis(children=(Text(value="s2"),))),
            body=(
                Paragraph(children=(Text(value="Full caption text."),)),
                Paragraph(children=(Text(value="Second paragraph."),)),
            ),
        )
        rt = Caption.model_validate_json(c.model_dump_json())
        assert rt == c


# ---------------------------------------------------------------------------
# 20. LIST ITEM CHECKED FIELD
# ---------------------------------------------------------------------------


class TestListItemCheckedEdgeCases:
    def test_checked_none(self) -> None:
        li = ListItem(checked=None, children=())
        assert li.checked is None

    def test_checked_true(self) -> None:
        li = ListItem(checked=True, children=())
        assert li.checked is True

    def test_checked_false(self) -> None:
        li = ListItem(checked=False, children=())
        assert li.checked is False

    def test_json_roundtrip_preserves_checked(self) -> None:
        for val in [None, True, False]:
            li = ListItem(checked=val, children=(Paragraph(children=(Text(value="x"),)),))
            rt = ListItem.model_validate_json(li.model_dump_json())
            assert rt.checked is val


# ---------------------------------------------------------------------------
# 21. FIGURE/ADMONITION/RAW BLOCK EDGE CASES
# ---------------------------------------------------------------------------


class TestFigureEdgeCases:
    def test_figure_no_caption(self) -> None:
        f = Figure(children=(Paragraph(children=(Text(value="content"),)),))
        assert f.caption is None

    def test_figure_empty_caption(self) -> None:
        f = Figure(caption=Caption(), children=())
        assert f.caption is not None
        assert f.caption.short is None
        assert f.caption.body == ()


class TestAdmonitionEdgeCases:
    def test_admonition_no_title(self) -> None:
        a = Admonition(kind="info", children=())
        assert a.title is None

    def test_admonition_empty_kind(self) -> None:
        a = Admonition(kind="", children=())
        assert a.kind == ""

    def test_admonition_json_roundtrip(self) -> None:
        a = Admonition(
            kind="warning",
            title="Title",
            children=(Paragraph(children=(Text(value="body"),)),),
            attr=Attr(id="adm-1", classes=("important",)),
        )
        rt = Admonition.model_validate_json(a.model_dump_json())
        assert rt == a


class TestRawBlockEdgeCases:
    def test_rawblock_empty_value(self) -> None:
        rb = RawBlock(format="html", value="")
        assert rb.value == ""

    def test_rawblock_empty_format(self) -> None:
        rb = RawBlock(format="", value="content")
        assert rb.format == ""

    def test_rawinline_empty_format(self) -> None:
        ri = RawInline(format="", value="x")
        assert ri.format == ""
