"""Tests for Inline XBRL (iXBRL) handling in ``kaos_content.parsers.html``.

Inline XBRL documents are XHTML with ``ix:`` elements that tag facts in
the visible text and an ``ix:header`` block (usually inside a
``display:none`` container) that carries hidden facts, contexts and
units. Parsing must keep every visible character -- including the text
of ``ix:nonNumeric`` / ``ix:nonFraction`` / ``ix:continuation`` -- and
drop everything that a browser would not render.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from kaos_content.model.blocks import Table
from kaos_content.parsers.html import looks_like_xbrl, parse_html, strip_inline_xbrl
from kaos_content.serializers.text import serialize_text

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "ixbrl" / "synthetic_annual_report.htm"

_NS = (
    'xmlns="http://www.w3.org/1999/xhtml" '
    'xmlns:ix="http://www.xbrl.org/2013/inlineXBRL" '
    'xmlns:xbrli="http://www.xbrl.org/2003/instance"'
)


def _ixbrl(body: str, *, preamble: str = "") -> str:
    return (
        "<?xml version='1.0' encoding='ASCII'?>\n"
        f"{preamble}<html {_NS}><head><title>t</title></head><body>{body}</body></html>"
    )


def _text(html: str, *, strip_xbrl: bool | None = None) -> str:
    return serialize_text(parse_html(html, strip_xbrl=strip_xbrl))


@pytest.fixture(scope="module")
def fixture_html() -> str:
    return FIXTURE.read_text(encoding="utf-8")


class TestFixtureDocument:
    """A full synthetic annual report keeps its body and drops hidden facts."""

    def test_detected(self, fixture_html: str) -> None:
        assert looks_like_xbrl(fixture_html)

    def test_body_text_survives(self, fixture_html: str) -> None:
        text = _text(fixture_html)
        assert "Item 1A." in text
        assert "Risk Factors" in text
        assert "A disruption at a single-source supplier" in text
        assert "Failures of information technology systems could disrupt operations." in text
        # Cover page facts are visible and must be kept.
        assert "FORM 10-K" in text
        assert "Example Widget Corporation" in text
        # Text-block facts and their continuation are visible body text.
        assert "Summary of Significant Accounting Policies" in text
        assert "Inventories are stated at the lower of cost" in text

    def test_numeric_facts_keep_their_display_text(self, fixture_html: str) -> None:
        text = _text(fixture_html)
        assert "Net sales for the year were $1,234 million, an increase of 7%" in text
        doc = parse_html(fixture_html)
        tables = [b for b in doc.body if isinstance(b, Table)]
        assert tables, "financial table should survive as a Table block"

    def test_hidden_facts_do_not_leak(self, fixture_html: str) -> None:
        text = _text(fixture_html)
        assert "0009999999" not in text  # hidden dei:EntityCentralIndexKey + context id
        assert "HIDDENFACTFY" not in text  # ix:hidden fact
        assert "2025-01-01" not in text  # context period
        assert "iso4217:USD" not in text  # unit measure
        assert "http://www.sec.gov/CIK" not in text

    def test_most_visible_text_is_kept(self, fixture_html: str) -> None:
        from lxml import html as lxml_html

        # Browser-visible text: everything in <body> except the hidden block.
        visible = lxml_html.document_fromstring(fixture_html.encode("ascii"))
        for el in visible.xpath('//*[contains(@style, "display:none")]'):
            el.drop_tree()
        visible_words = visible.body.text_content().split()
        parsed_words = _text(fixture_html).split()
        assert len(parsed_words) >= 0.95 * len(visible_words)

    def test_explicit_opt_out_is_honoured(self) -> None:
        # strip_xbrl=False skips preprocessing even when iXBRL is detected.
        html = (
            f"<html {_NS}><body><div style='display:none'>HIDDEN</div>"
            '<p><ix:nonNumeric name="a">Fact</ix:nonNumeric></p></body></html>'
        )
        assert "HIDDEN" in _text(html, strip_xbrl=False)
        assert "HIDDEN" not in _text(html)


class TestHiddenContent:
    """Hidden-block removal is DOM-based, not a lazy regex."""

    def test_nested_divs_inside_hidden_block(self) -> None:
        html = _ixbrl(
            '<div style="display:none"><div>HIDDEN-ONE</div><div>HIDDEN-TWO</div></div>'
            "<p>Visible paragraph.</p>"
        )
        text = _text(html)
        assert "HIDDEN-ONE" not in text
        assert "HIDDEN-TWO" not in text
        assert "Visible paragraph." in text

    def test_single_quoted_style_and_whitespace(self) -> None:
        html = _ixbrl("<div style='DISPLAY : None'>HIDDEN</div><p>Shown.</p>")
        text = _text(html)
        assert "HIDDEN" not in text
        assert "Shown." in text

    def test_hidden_non_div_element(self) -> None:
        html = _ixbrl('<p>Before <span style="display:none">HIDDEN</span> after.</p>')
        text = _text(html)
        assert "HIDDEN" not in text
        assert "Before" in text
        assert "after." in text

    def test_tail_text_after_hidden_element_is_kept(self) -> None:
        html = _ixbrl('<p>Lead <span style="display:none">HIDDEN</span>trailing words.</p>')
        assert "trailing words." in _text(html)

    def test_header_outside_hidden_div_is_removed(self) -> None:
        html = _ixbrl(
            "<ix:header><ix:hidden>"
            '<ix:nonNumeric name="dei:AmendmentFlag" contextRef="c">HIDDEN-FLAG</ix:nonNumeric>'
            "</ix:hidden><ix:resources><xbrli:context id='c'>CTX-TEXT</xbrli:context>"
            "</ix:resources></ix:header><p>Body.</p>"
        )
        text = _text(html)
        assert "HIDDEN-FLAG" not in text
        assert "CTX-TEXT" not in text
        assert "Body." in text

    def test_visible_display_block_is_kept(self) -> None:
        html = _ixbrl('<div style="display:block">Shown block.</div>')
        assert "Shown block." in _text(html)


class TestVisibleFacts:
    """Visible ``ix:`` wrappers are unwrapped, keeping text and structure."""

    def test_nested_fact_wrappers(self) -> None:
        html = _ixbrl(
            '<p><ix:nonNumeric name="a" contextRef="c">Outer '
            '<ix:nonFraction name="b" contextRef="c" unitRef="u">42</ix:nonFraction>'
            " tail</ix:nonNumeric> end.</p>"
        )
        assert "Outer 42 tail end." in _text(html)

    def test_exclude_and_continuation_text_is_visible(self) -> None:
        html = _ixbrl(
            '<ix:nonNumeric name="a" contextRef="c" continuedAt="k">'
            "<p>Part one <ix:exclude>page 7</ix:exclude>.</p></ix:nonNumeric>"
            '<ix:continuation id="k"><p>Part two.</p></ix:continuation>'
        )
        text = _text(html)
        assert "Part one page 7." in text
        assert "Part two." in text

    def test_block_structure_inside_fact_is_preserved(self) -> None:
        html = _ixbrl(
            '<ix:nonNumeric name="a" contextRef="c"><h2>Note 1</h2><p>Policy text.</p>'
            "</ix:nonNumeric>"
        )
        doc = parse_html(html)
        kinds = [b.node_type for b in doc.body]
        assert kinds == ["heading", "paragraph"]

    def test_non_ascii_and_entities(self) -> None:
        html = _ixbrl(
            '<p>Caf\u00e9 \u2014 <ix:nonNumeric name="a" contextRef="c">S\u00e3o Paulo &#8217;26'
            "</ix:nonNumeric> \U0001f4c8</p>"
        )
        assert "Caf\u00e9 \u2014 S\u00e3o Paulo \u201926 \U0001f4c8" in _text(html)

    def test_custom_prefix_bound_to_inline_namespace(self) -> None:
        html = (
            '<html xmlns:inl="http://www.xbrl.org/2013/inlineXBRL"><body>'
            '<inl:header><inl:hidden><inl:nonNumeric name="a">HIDDEN</inl:nonNumeric>'
            "</inl:hidden></inl:header>"
            '<p>Value <inl:nonFraction name="b">9</inl:nonFraction>.</p></body></html>'
        )
        text = _text(html)
        assert "HIDDEN" not in text
        assert "Value 9." in text


class TestDetection:
    """``looks_like_xbrl`` looks for real Inline XBRL markup anywhere."""

    def test_namespace_after_long_preamble(self) -> None:
        preamble = "<!--" + ("x" * 5000) + "-->\n"
        html = _ixbrl('<p>Rev <ix:nonFraction name="b">5</ix:nonFraction>.</p>', preamble=preamble)
        assert looks_like_xbrl(html)
        assert "Rev 5." in _text(html)

    def test_fact_tag_without_namespace_declaration(self) -> None:
        # A fragment of a filing, with the xmlns-bearing root cut off.
        assert looks_like_xbrl('<div><ix:nonNumeric name="a">x</ix:nonNumeric></div>')

    @pytest.mark.parametrize(
        "html",
        [
            "<html><head><title>Linux: a guide</title></head><body><p>Fix: it.</p></body></html>",
            "<html><body><h1>What is XBRL?</h1><p>XBRL is a reporting format.</p></body></html>",
            "<html><body><p>The inlineXBRL spec is long.</p></body></html>",
            "<html><body><p>&lt;ix:header&gt; is escaped text</p></body></html>",
            "",
        ],
    )
    def test_plain_html_is_not_detected(self, html: str) -> None:
        assert not looks_like_xbrl(html)

    def test_plain_html_with_hidden_div_is_untouched_by_default(self) -> None:
        # Auto-detection must not apply iXBRL preprocessing to ordinary pages.
        html = "<html><body><p>Fix: things.</p><div style='display:none'>kept</div></body></html>"
        assert "kept" in _text(html)


class TestStripInlineXbrl:
    """The string-level helper returns plain HTML with no ``ix:`` markup."""

    def test_returns_plain_html(self, fixture_html: str) -> None:
        out = strip_inline_xbrl(fixture_html)
        lowered = out.lower()
        assert "<ix:" not in lowered
        assert "</ix:" not in lowered
        assert "<?xml" not in lowered
        assert "xmlns:" not in lowered
        assert "0009999999" not in out
        assert "Risk Factors" in out

    def test_empty_input(self) -> None:
        assert strip_inline_xbrl("") == ""
        assert strip_inline_xbrl("   ") == "   "

    def test_output_reparses_to_same_text(self, fixture_html: str) -> None:
        stripped = strip_inline_xbrl(fixture_html)
        assert _text(stripped, strip_xbrl=False) == _text(fixture_html)
