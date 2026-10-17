"""Whitespace at inline element boundaries in ``parse_html``.

Words must never be glued across element boundaries that a browser
separates: whitespace-only text and tails (including ``&nbsp;``) between
inline elements are one space, block-level elements inside inline content
end the line, ``<br>`` is a line break, and paragraphs inside a table cell
stay apart. Text that the page itself joins (``<b>W</b>ord``) stays joined.

The markup snippets are reduced from public U.S. government pages where
the glue was observed (OSHRC decisions, FTC, USAID, SEC, HHS).
"""

from __future__ import annotations

import pytest
from lxml import html as lxml_html

from kaos_content import serialize_markdown, serialize_text
from kaos_content.model.blocks import Paragraph
from kaos_content.model.inlines import LineBreak, Strong, Text
from kaos_content.parsers.html import collapse_whitespace, parse_html, process_inlines


def text_of(html: str) -> str:
    return serialize_text(parse_html(html)).strip()


class TestWhitespaceOnlyNodes:
    def test_whitespace_only_span_between_words(self) -> None:
        # OSHRC decision exported from Word: the word spaces live in spans.
        html = (
            '<p><span style="font-size:12pt">For<span>\r\n\t\t</span></span>'
            '<span style="font-size:12pt">the<span>\r\n\t\t</span></span>'
            '<span style="font-size:12pt">Respondent</span></p>'
        )
        assert text_of(html) == "For the Respondent"

    def test_whitespace_tail_between_formatting_elements(self) -> None:
        assert text_of("<p>For <b>the</b> <i>Respondent</i></p>") == "For the Respondent"

    def test_whitespace_tail_after_span_before_text(self) -> None:
        html = "<p>(Birdsboro or Respondent)<span>\r\n\t</span>operates a poultry facility</p>"
        assert text_of(html) == "(Birdsboro or Respondent) operates a poultry facility"

    @pytest.mark.parametrize("space", ["\xa0", "&nbsp;", " \xa0 "])
    def test_nbsp_between_strong_elements(self, space: str) -> None:
        # FTC: <strong>...problems.</strong>&nbsp;<strong>(It won't.)</strong>
        html = f"<p><strong>health problems.</strong>{space}<strong>(It won't.)</strong></p>"
        assert text_of(html) == "health problems. (It won't.)"

    def test_nbsp_inside_strong_before_text(self) -> None:
        # USAID transcript: <strong>NAME:&nbsp;</strong>So, ...
        html = "<p><strong>RANEEM BOU KHZAM:&nbsp;</strong>So, thank you.</p>"
        assert text_of(html) == "RANEEM BOU KHZAM: So, thank you."

    def test_whitespace_only_formatting_element(self) -> None:
        assert text_of("<p>For<b> </b>the</p>") == "For the"

    def test_adjacent_inline_text_without_whitespace_stays_joined(self) -> None:
        assert text_of("<p><b>W</b>ord and <i>Bold</i><b>Ital</b></p>") == "Word and BoldItal"

    def test_spaces_collapse_across_boundaries(self) -> None:
        assert text_of("<p>a <b> b </b> <i> c</i></p>") == "a b c"

    def test_collapse_whitespace_keeps_a_space(self) -> None:
        assert collapse_whitespace("\r\n\t") == " "
        assert collapse_whitespace("\xa0") == " "
        assert collapse_whitespace("foo\xa0") == "foo "
        assert collapse_whitespace("") == ""


class TestEdgeWhitespaceHoisted:
    def test_strong_does_not_carry_edge_spaces(self) -> None:
        el = lxml_html.fragment_fromstring("<p>a<b> bold </b>c</p>")
        nodes = process_inlines(el, "")
        assert [type(n) for n in nodes] == [Text, Strong, Text]
        strong = nodes[1]
        assert isinstance(strong, Strong)
        child = strong.children[0]
        assert isinstance(child, Text)
        assert child.value == "bold"
        assert serialize_text(parse_html("<p>a<b> bold </b>c</p>")).strip() == "a bold c"

    def test_markdown_emphasis_has_no_inner_spaces(self) -> None:
        md = serialize_markdown(parse_html("<p>see<b> this </b>now</p>"))
        assert "see **this** now" in md

    def test_paragraph_edges_trimmed(self) -> None:
        doc = parse_html("<p>  <b>x</b>  </p>")
        para = doc.body[0]
        assert isinstance(para, Paragraph)
        assert serialize_text(doc) == "x\n"


class TestBlockBoundariesInInlineContent:
    def test_divs_inside_link_are_lines(self) -> None:
        html = '<p><a href="/staff"><div>Michael Sarich</div><div>Eric Stein</div></a></p>'
        assert text_of(html) == "Michael Sarich\nEric Stein"

    def test_display_block_style_is_a_boundary(self) -> None:
        html = '<p>Name<span style="display: block">Title</span>Office</p>'
        assert text_of(html) == "Name\nTitle\nOffice"

    def test_display_inline_block_is_not_a_boundary(self) -> None:
        html = '<p>Name <span style="display:inline-block">Title</span></p>'
        assert text_of(html) == "Name Title"

    def test_br_is_a_line_break_without_edge_spaces(self) -> None:
        doc = parse_html("<p>line one <br> line two</p>")
        para = doc.body[0]
        assert isinstance(para, Paragraph)
        assert any(isinstance(c, LineBreak) for c in para.children)
        assert serialize_text(doc) == "line one\nline two\n"

    def test_unknown_inline_tag_keeps_inner_spacing(self) -> None:
        html = "<p><cite><b>Smith</b> <i>v.</i> Jones</cite>, 1 U.S. 1</p>"
        assert text_of(html) == "Smith v. Jones, 1 U.S. 1"


class TestInlineRunsInBlockContainers:
    def test_mixed_inline_content_is_one_paragraph(self) -> None:
        assert text_of("<blockquote>quoted <b>x</b> <i>y</i></blockquote>") == "quoted x y"
        assert text_of("<div>Lead text <a href='/a'>link</a> tail.</div>") == "Lead text link tail."

    def test_touching_inline_elements_at_block_level_are_separated(self) -> None:
        # Nav-like rows: anchors laid out apart by a stylesheet.
        html = '<div><a href="/1">6. CCWIS</a><a href="/2">7. Title IV-B</a></div>'
        assert text_of(html) == "6. CCWIS\n7. Title IV-B"

    def test_text_then_inline_element_at_block_level_joins(self) -> None:
        assert text_of("<div>Note<sup>1</sup> here</div>") == "Note1 here"

    def test_span_wrapping_paragraphs_keeps_paragraphs(self) -> None:
        doc = parse_html("<div><span><p>First.</p><p>Second.</p></span></div>")
        assert [type(b) for b in doc.body] == [Paragraph, Paragraph]

    def test_blocks_split_the_run(self) -> None:
        doc = parse_html("<div>before<p>middle</p>after</div>")
        assert serialize_text(doc) == "before\n\nmiddle\n\nafter\n"


class TestTableCells:
    def test_paragraphs_in_a_cell_do_not_run_together(self) -> None:
        # SEC litigation release index: several <p> per cell.
        html = (
            "<table><tr><td><p>Respondents:</p><p>Agri-Fintech Holdings, Inc.</p></td>"
            "<td>Release No.<br>34-98921</td></tr></table>"
        )
        text = text_of(html)
        assert "Respondents:Agri" not in text
        assert "Respondents:\nAgri-Fintech Holdings, Inc." in text
        assert "Release No.\n34-98921" in text

    def test_single_block_cell_unchanged(self) -> None:
        assert text_of("<table><tr><td>x</td><td>y</td></tr></table>") == "x | y"


class TestLinkListsAtBlockLevel:
    def test_links_separated_by_whitespace_are_lines(self) -> None:
        html = (
            '<div><a href="/t">Twitter</a>\n<a href="/f">Facebook</a>\n'
            '<span><a href="/e">E-mail</a></span></div>'
        )
        assert text_of(html) == "Twitter\nFacebook\nE-mail"

    def test_link_inside_prose_stays_inline(self) -> None:
        html = '<div>See <a href="/a">the rule</a> and <a href="/b">the order</a>.</div>'
        assert text_of(html) == "See the rule and the order."

    def test_formatting_separated_by_whitespace_stays_inline(self) -> None:
        assert text_of("<div><b>Note:</b> <i>read this</i></div>") == "Note: read this"


class TestPageWrapperForm:
    def test_aspnet_form_wrapping_the_page_is_content(self) -> None:
        html = (
            '<body><form id="aspnetForm" method="post"><input type="hidden" name="__VIEWSTATE">'
            "<div><h1>Oral Arguments</h1><p>The Court holds oral argument in about 70-80 cases.</p>"
            "</div></form></body>"
        )
        text = text_of(html)
        assert text.startswith("Oral Arguments\n")
        assert text.endswith("\nThe Court holds oral argument in about 70-80 cases.")

    def test_plain_form_is_skipped(self) -> None:
        html = (
            "<body><p>Body text.</p><form><label>Search</label><input name='q'>"
            "<select><option>Alabama</option></select></form></body>"
        )
        assert text_of(html) == "Body text."

    def test_select_options_in_page_form_are_not_text(self) -> None:
        html = (
            "<body><form><p>Pick a state.</p><select><option>Alabama</option>"
            "<option>Alaska</option></select></form></body>"
        )
        assert text_of(html) == "Pick a state."


class TestMisnestedChrome:
    def test_main_inside_unclosed_nav_is_kept(self) -> None:
        # An unclosed <nav> makes lxml nest the rest of the page inside it.
        html = (
            "<body><header><nav><ul><li><a href='/'>Home</a></li></ul>"
            "<main><h1>Photonics Research Center</h1><p>About the Center.</p></main>"
            "</nav></header></body>"
        )
        text = text_of(html)
        assert "About the Center." in text
        assert "Photonics Research Center" in text

    def test_plain_nav_and_footer_are_skipped(self) -> None:
        html = "<body><nav><a href='/'>Home</a></nav><p>Body.</p><footer>Contact</footer></body>"
        assert text_of(html) == "Body."
