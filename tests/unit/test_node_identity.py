"""Tests for node UUID identity and content hashing."""

from __future__ import annotations

import json
import pickle
import re

from kaos_content import (
    BulletList,
    Cell,
    CodeBlock,
    ContentDocument,
    Emphasis,
    Heading,
    ListItem,
    NodeIndex,
    Paragraph,
    Row,
    Strong,
    Table,
    TableSection,
    Text,
    content_hash,
)


class TestNodeUUID:
    """Every BaseNode gets a UUID v7 id."""

    def test_text_has_id(self) -> None:
        t = Text(value="hello")
        assert t.id
        assert len(t.id) == 32
        assert re.fullmatch(r"[0-9a-f]{32}", t.id)

    def test_paragraph_has_id(self) -> None:
        p = Paragraph(children=(Text(value="x"),))
        assert p.id
        assert len(p.id) == 32

    def test_all_nodes_unique_ids(self) -> None:
        """Every node in a document gets a distinct id."""
        t1 = Text(value="a")
        t2 = Text(value="b")
        p = Paragraph(children=(t1, t2))
        # At minimum t1, t2, p should all be different
        assert len({t1.id, t2.id, p.id}) == 3

    def test_same_content_different_ids(self) -> None:
        """Two nodes with identical content get different UUIDs."""
        t1 = Text(value="same")
        t2 = Text(value="same")
        assert t1.id != t2.id

    def test_id_is_frozen(self) -> None:
        """id cannot be reassigned on a frozen model."""
        import pytest
        from pydantic import ValidationError

        t = Text(value="x")
        with pytest.raises(ValidationError):
            t.id = "new-id"

    def test_json_roundtrip_preserves_id(self) -> None:
        t = Text(value="hello")
        original_id = t.id
        j = t.model_dump_json()
        t2 = Text.model_validate_json(j)
        assert t2.id == original_id

    def test_json_roundtrip_document_preserves_ids(self) -> None:
        t = Text(value="x")
        p = Paragraph(children=(t,))
        doc = ContentDocument(body=(p,))
        j = doc.model_dump_json()
        doc2 = ContentDocument.model_validate_json(j)
        assert doc2.body[0].id == p.id

    def test_pickle_roundtrip_preserves_id(self) -> None:
        t = Text(value="hello")
        original_id = t.id
        t2 = pickle.loads(pickle.dumps(t))
        assert t2.id == original_id

    def test_explicit_id(self) -> None:
        """User can supply a specific id."""
        t = Text(value="x", id="custom-id-123")
        assert t.id == "custom-id-123"

    def test_id_in_json(self) -> None:
        """id appears in JSON serialization."""
        t = Text(value="x")
        data = json.loads(t.model_dump_json())
        assert "id" in data
        assert data["id"] == t.id

    def test_table_components_have_ids(self) -> None:
        """Cell, Row, TableSection all get UUIDs."""
        cell = Cell(content=(Paragraph(children=(Text(value="x"),)),))
        row = Row(cells=(cell,))
        section = TableSection(rows=(row,))
        assert cell.id
        assert row.id
        assert section.id
        assert len({cell.id, row.id, section.id}) == 3

    def test_uuid_v7_time_ordered(self) -> None:
        """UUIDs generated in sequence should sort in creation order."""
        ids = [Text(value=str(i)).id for i in range(100)]
        assert ids == sorted(ids)


class TestContentHash:
    """content_hash produces deterministic hashes based on content."""

    def test_same_content_same_hash(self) -> None:
        t1 = Text(value="hello")
        t2 = Text(value="hello")
        assert content_hash(t1) == content_hash(t2)

    def test_different_content_different_hash(self) -> None:
        t1 = Text(value="hello")
        t2 = Text(value="world")
        assert content_hash(t1) != content_hash(t2)

    def test_hash_is_hex_sha256(self) -> None:
        t = Text(value="x")
        h = content_hash(t)
        assert len(h) == 64
        assert re.fullmatch(r"[0-9a-f]{64}", h)

    def test_hash_ignores_id(self) -> None:
        """Two nodes with same content but different ids have the same hash."""
        t1 = Text(value="x", id="aaa")
        t2 = Text(value="x", id="bbb")
        assert content_hash(t1) == content_hash(t2)

    def test_hash_ignores_provenance(self) -> None:
        from kaos_content import Provenance

        t1 = Text(value="x")
        t2 = Text(value="x", provenance=Provenance(page=5))
        assert content_hash(t1) == content_hash(t2)

    def test_hash_ignores_attr(self) -> None:
        from kaos_content import Attr

        t1 = Text(value="x")
        t2 = Text(value="x", attr=Attr(id="sec-1", classes=("legal",)))
        assert content_hash(t1) == content_hash(t2)

    def test_hash_includes_structure(self) -> None:
        """A paragraph containing 'hello' hashes differently from bare Text('hello')."""
        t = Text(value="hello")
        p = Paragraph(children=(Text(value="hello"),))
        assert content_hash(t) != content_hash(p)

    def test_hash_includes_heading_depth(self) -> None:
        h1 = Heading(depth=1, children=(Text(value="Title"),))
        h2 = Heading(depth=2, children=(Text(value="Title"),))
        assert content_hash(h1) != content_hash(h2)

    def test_hash_includes_code_language(self) -> None:
        c1 = CodeBlock(value="x = 1", language="python")
        c2 = CodeBlock(value="x = 1", language="javascript")
        assert content_hash(c1) != content_hash(c2)

    def test_hash_nested_document(self) -> None:
        """Hash of a complex document is deterministic."""

        def make_doc() -> ContentDocument:
            return ContentDocument(
                body=(
                    Heading(depth=1, children=(Text(value="Title"),)),
                    Paragraph(
                        children=(
                            Text(value="Some "),
                            Strong(children=(Text(value="bold"),)),
                            Text(value=" text."),
                        )
                    ),
                    BulletList(
                        children=(
                            ListItem(children=(Paragraph(children=(Text(value="item 1"),)),)),
                            ListItem(children=(Paragraph(children=(Text(value="item 2"),)),)),
                        )
                    ),
                )
            )

        doc1 = make_doc()
        doc2 = make_doc()
        # Hash each block — same structure should give same hash
        for b1, b2 in zip(doc1.body, doc2.body, strict=True):
            assert content_hash(b1) == content_hash(b2)

    def test_hash_emphasis_vs_strong(self) -> None:
        """Emphasis and Strong with same text hash differently."""
        em = Emphasis(children=(Text(value="x"),))
        st = Strong(children=(Text(value="x"),))
        assert content_hash(em) != content_hash(st)


class TestNodeIndexById:
    """NodeIndex supports id-based lookup."""

    def test_get_by_id(self) -> None:
        t = Text(value="hello")
        p = Paragraph(children=(t,))
        doc = ContentDocument(body=(p,))
        index = NodeIndex(doc)
        assert index.get_by_id(p.id) is p
        assert index.get_by_id(t.id) is t

    def test_get_by_id_not_found(self) -> None:
        doc = ContentDocument(body=(Paragraph(children=(Text(value="x"),)),))
        index = NodeIndex(doc)
        assert index.get_by_id("nonexistent") is None

    def test_get_by_id_table_components(self) -> None:
        """Cell, Row, TableSection are findable by id."""
        cell = Cell(content=(Paragraph(children=(Text(value="data"),)),))
        row = Row(cells=(cell,))
        section = TableSection(rows=(row,))
        table = Table(bodies=(section,))
        doc = ContentDocument(body=(table,))
        index = NodeIndex(doc)
        assert index.get_by_id(table.id) is table
        assert index.get_by_id(section.id) is section
        assert index.get_by_id(row.id) is row
        assert index.get_by_id(cell.id) is cell

    def test_get_by_id_footnote_nodes(self) -> None:
        t = Text(value="fn content")
        p = Paragraph(children=(t,))
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="body"),)),),
            footnotes={"fn1": (p,)},
        )
        index = NodeIndex(doc)
        assert index.get_by_id(p.id) is p
        assert index.get_by_id(t.id) is t
