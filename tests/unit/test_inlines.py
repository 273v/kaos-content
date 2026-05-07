"""Tests for inline AST node types."""

import pytest
from pydantic import ValidationError

from kaos_content import (
    Attr,
    Citation,
    Code,
    Emphasis,
    FootnoteRef,
    Image,
    LineBreak,
    Link,
    Math,
    Provenance,
    RawInline,
    SoftBreak,
    Span,
    Strikethrough,
    Strong,
    Subscript,
    Superscript,
    Text,
    Underline,
)


class TestText:
    def test_basic(self) -> None:
        t = Text(value="hello")
        assert t.value == "hello"
        assert t.node_type == "text"

    def test_empty_string(self) -> None:
        t = Text(value="")
        assert t.value == ""

    def test_frozen(self) -> None:
        t = Text(value="x")
        with pytest.raises(ValidationError):
            t.value = "y"

    def test_with_attr(self) -> None:
        t = Text(value="term", attr=Attr(kv={"defined-term": "true"}))
        assert t.attr.kv["defined-term"] == "true"

    def test_with_provenance(self) -> None:
        t = Text(value="x", provenance=Provenance(page=3))
        assert t.provenance is not None
        assert t.provenance.page == 3


class TestEmphasis:
    def test_basic(self) -> None:
        e = Emphasis(children=(Text(value="italic"),))
        assert e.node_type == "emphasis"
        assert len(e.children) == 1

    def test_nested(self) -> None:
        e = Emphasis(children=(Strong(children=(Text(value="bold-italic"),)),))
        assert len(e.children) == 1


class TestStrong:
    def test_basic(self) -> None:
        s = Strong(children=(Text(value="bold"),))
        assert s.node_type == "strong"


class TestStrikethrough:
    def test_basic(self) -> None:
        s = Strikethrough(children=(Text(value="struck"),))
        assert s.node_type == "strikethrough"


class TestCode:
    def test_basic(self) -> None:
        c = Code(value="x = 1")
        assert c.node_type == "code"
        assert c.value == "x = 1"


class TestLink:
    def test_basic(self) -> None:
        link = Link(url="https://example.com", children=(Text(value="click"),))
        assert link.url == "https://example.com"
        assert link.title is None

    def test_with_title(self) -> None:
        link = Link(url="u", title="A link", children=(Text(value="t"),))
        assert link.title == "A link"


class TestImage:
    def test_basic(self) -> None:
        img = Image(src="img.png")
        assert img.node_type == "image"
        assert img.alt is None

    def test_full(self) -> None:
        img = Image(src="img.png", alt="A photo", title="Photo title")
        assert img.alt == "A photo"


class TestFootnoteRef:
    def test_basic(self) -> None:
        ref = FootnoteRef(identifier="fn1")
        assert ref.node_type == "footnote_ref"
        assert ref.identifier == "fn1"


class TestCitation:
    def test_basic(self) -> None:
        c = Citation(identifiers=("smith2024",), children=(Text(value="Smith (2024)"),))
        assert c.identifiers == ("smith2024",)

    def test_multiple(self) -> None:
        c = Citation(
            identifiers=("a", "b"),
            children=(Text(value="A; B"),),
        )
        assert len(c.identifiers) == 2


class TestMath:
    def test_basic(self) -> None:
        m = Math(value="E = mc^2")
        assert m.node_type == "math"


class TestRawInline:
    def test_basic(self) -> None:
        r = RawInline(format="html", value="<br/>")
        assert r.format == "html"


class TestLineBreak:
    def test_basic(self) -> None:
        lb = LineBreak()
        assert lb.node_type == "line_break"


class TestSoftBreak:
    def test_basic(self) -> None:
        sb = SoftBreak()
        assert sb.node_type == "soft_break"


class TestSpan:
    def test_basic(self) -> None:
        s = Span(children=(Text(value="inner"),))
        assert s.node_type == "span"

    def test_with_domain_attr(self) -> None:
        s = Span(
            attr=Attr(classes=("defined-term",), kv={"term-id": "dt-001"}),
            children=(Text(value="Force Majeure"),),
        )
        assert "defined-term" in s.attr.classes


class TestSuperscript:
    def test_basic(self) -> None:
        s = Superscript(children=(Text(value="2"),))
        assert s.node_type == "superscript"


class TestSubscript:
    def test_basic(self) -> None:
        s = Subscript(children=(Text(value="i"),))
        assert s.node_type == "subscript"


class TestUnderline:
    def test_basic(self) -> None:
        u = Underline(children=(Text(value="underlined"),))
        assert u.node_type == "underline"


class TestInlineJsonRoundtrip:
    """JSON round-trip for every inline type."""

    def test_text(self) -> None:
        node = Text(value="hello")
        assert Text.model_validate_json(node.model_dump_json()) == node

    def test_emphasis(self) -> None:
        node = Emphasis(children=(Text(value="em"),))
        assert Emphasis.model_validate_json(node.model_dump_json()) == node

    def test_strong(self) -> None:
        node = Strong(children=(Text(value="b"),))
        assert Strong.model_validate_json(node.model_dump_json()) == node

    def test_strikethrough(self) -> None:
        node = Strikethrough(children=(Text(value="s"),))
        assert Strikethrough.model_validate_json(node.model_dump_json()) == node

    def test_code(self) -> None:
        node = Code(value="x")
        assert Code.model_validate_json(node.model_dump_json()) == node

    def test_link(self) -> None:
        node = Link(url="u", children=(Text(value="t"),))
        assert Link.model_validate_json(node.model_dump_json()) == node

    def test_image(self) -> None:
        node = Image(src="s", alt="a")
        assert Image.model_validate_json(node.model_dump_json()) == node

    def test_footnote_ref(self) -> None:
        node = FootnoteRef(identifier="fn1")
        assert FootnoteRef.model_validate_json(node.model_dump_json()) == node

    def test_citation(self) -> None:
        node = Citation(identifiers=("a",), children=(Text(value="A"),))
        assert Citation.model_validate_json(node.model_dump_json()) == node

    def test_math(self) -> None:
        node = Math(value="x^2")
        assert Math.model_validate_json(node.model_dump_json()) == node

    def test_raw_inline(self) -> None:
        node = RawInline(format="html", value="<b>")
        assert RawInline.model_validate_json(node.model_dump_json()) == node

    def test_line_break(self) -> None:
        node = LineBreak()
        assert LineBreak.model_validate_json(node.model_dump_json()) == node

    def test_soft_break(self) -> None:
        node = SoftBreak()
        assert SoftBreak.model_validate_json(node.model_dump_json()) == node

    def test_span(self) -> None:
        node = Span(children=(Text(value="x"),))
        assert Span.model_validate_json(node.model_dump_json()) == node

    def test_superscript(self) -> None:
        node = Superscript(children=(Text(value="2"),))
        assert Superscript.model_validate_json(node.model_dump_json()) == node

    def test_subscript(self) -> None:
        node = Subscript(children=(Text(value="i"),))
        assert Subscript.model_validate_json(node.model_dump_json()) == node

    def test_underline(self) -> None:
        node = Underline(children=(Text(value="u"),))
        assert Underline.model_validate_json(node.model_dump_json()) == node
