"""CommonMark-spec escaping correctness tests for the markdown serializer.

Each test verifies that the serializer output would be interpreted correctly
by a CommonMark parser — not just that it doesn't crash.
"""

from kaos_content import (
    Cell,
    Code,
    ContentDocument,
    Emphasis,
    Link,
    Paragraph,
    Row,
    SoftBreak,
    Strong,
    Table,
    TableSection,
    Text,
    serialize_markdown,
)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. LINE-START ESCAPING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLineStartEscaping:
    """Text at the start of a line that resembles block syntax MUST be escaped.
    The same characters in the middle of a line must NOT be escaped."""

    def test_heading_at_line_start_escaped(self) -> None:
        """'# Heading' at paragraph start must be escaped."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="# Heading"),)),))
        result = serialize_markdown(doc)
        assert "\\# Heading" in result

    def test_hash_mid_line_not_escaped(self) -> None:
        """'# ' in the middle of a line must NOT be escaped."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="see section # Foo"),)),))
        result = serialize_markdown(doc)
        assert "see section # Foo" in result
        # The # should NOT be escaped when not at line start
        assert "\\#" not in result

    def test_h2_at_line_start_escaped(self) -> None:
        """'## ' at line start must be escaped."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="## Sub"),)),))
        result = serialize_markdown(doc)
        assert "\\## Sub" in result

    def test_dash_list_at_line_start_escaped(self) -> None:
        """'- item' at paragraph start must be escaped."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="- item"),)),))
        result = serialize_markdown(doc)
        assert "\\- item" in result

    def test_dash_mid_line_not_escaped(self) -> None:
        """'- ' in the middle of a line must NOT be escaped."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="range: 1- 5"),)),))
        result = serialize_markdown(doc)
        # The dash is not at line start, so no escaping
        assert "range: 1- 5" in result
        assert "\\-" not in result

    def test_blockquote_at_line_start_escaped(self) -> None:
        """'> quote' at paragraph start must be escaped."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="> quote"),)),))
        result = serialize_markdown(doc)
        assert "\\> quote" in result

    def test_gt_mid_line_not_escaped(self) -> None:
        """'>' in the middle of a line is not block syntax."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="x > y"),)),))
        result = serialize_markdown(doc)
        assert "x > y" in result
        # > should NOT be backslash-escaped when mid-line
        assert "x \\> y" not in result

    def test_ordered_list_at_line_start_escaped(self) -> None:
        """'1. item' at paragraph start must be escaped."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="1. item"),)),))
        result = serialize_markdown(doc)
        assert "\\1. item" in result

    def test_ordered_list_mid_line_not_escaped(self) -> None:
        """'1. ' in the middle of a line must NOT be escaped."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="see item 1. above"),)),))
        result = serialize_markdown(doc)
        assert "see item 1. above" in result
        assert "\\1" not in result

    def test_star_list_at_line_start_escaped(self) -> None:
        """'* item' at paragraph start must be escaped.

        The * is escaped by inline escaping (always), so the result is '\\* item'.
        This prevents it from being parsed as a list marker.
        """
        doc = ContentDocument(body=(Paragraph(children=(Text(value="* item"),)),))
        result = serialize_markdown(doc)
        assert "\\* item" in result

    def test_plus_list_at_line_start_escaped(self) -> None:
        """'+ item' at paragraph start must be escaped."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="+ item"),)),))
        result = serialize_markdown(doc)
        assert "\\+ item" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. EMPHASIS BOUNDARY WHITESPACE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestEmphasisWhitespace:
    """Per CommonMark, emphasis delimiters must be flanking — no space
    between delimiter and content. The serializer must expel whitespace."""

    def test_emphasis_with_surrounding_spaces(self) -> None:
        """Emphasis([Text(' hello ')]) expels whitespace outside delimiters."""
        # Embed in surrounding text so the expelled whitespace is visible
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Text(value="A"),
                        Emphasis(children=(Text(value=" hello "),)),
                        Text(value="B"),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        # Whitespace must be outside the delimiters: A *hello* B
        assert "A *hello* B" in result
        # Invalid: * hello * would not parse as emphasis
        assert "* hello *" not in result

    def test_strong_with_surrounding_spaces(self) -> None:
        """Strong([Text(' bold ')]) expels whitespace outside delimiters."""
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Text(value="A"),
                        Strong(children=(Text(value=" bold "),)),
                        Text(value="B"),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "A **bold** B" in result
        assert "** bold **" not in result

    def test_emphasis_whitespace_only(self) -> None:
        """Emphasis([Text(' ')]) -> just the whitespace, no delimiters."""
        doc = ContentDocument(body=(Paragraph(children=(Emphasis(children=(Text(value=" "),)),)),))
        result = serialize_markdown(doc)
        # With only whitespace, there's nothing to emphasize — just emit the space
        assert "*" not in result

    def test_emphasis_leading_space_only(self) -> None:
        """Emphasis([Text(' hello')]) expels leading space outside delimiter."""
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Text(value="A"),
                        Emphasis(children=(Text(value=" hello"),)),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "A *hello*" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. NESTED EMPHASIS DELIMITER ALTERNATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestNestedEmphasisDelimiterAlternation:
    """Nested emphasis must alternate between * and _ delimiters to avoid
    ambiguity per CommonMark rules."""

    def test_emphasis_uses_star(self) -> None:
        """Top-level emphasis uses *."""
        doc = ContentDocument(
            body=(Paragraph(children=(Emphasis(children=(Text(value="word"),)),)),)
        )
        result = serialize_markdown(doc)
        assert "*word*" in result

    def test_strong_uses_double_star(self) -> None:
        """Top-level strong uses **."""
        doc = ContentDocument(body=(Paragraph(children=(Strong(children=(Text(value="word"),)),)),))
        result = serialize_markdown(doc)
        assert "**word**" in result

    def test_strong_wrapping_emphasis_alternates(self) -> None:
        """Strong([Emphasis([Text('word')])]) -> **_word_**."""
        doc = ContentDocument(
            body=(
                Paragraph(children=(Strong(children=(Emphasis(children=(Text(value="word"),)),)),)),
            )
        )
        result = serialize_markdown(doc)
        assert "**_word_**" in result

    def test_emphasis_wrapping_strong_alternates(self) -> None:
        """Emphasis([Strong([Text('word')])]) -> *__word__*."""
        doc = ContentDocument(
            body=(
                Paragraph(children=(Emphasis(children=(Strong(children=(Text(value="word"),)),)),)),
            )
        )
        result = serialize_markdown(doc)
        assert "*__word__*" in result

    def test_emphasis_with_mixed_strong_text(self) -> None:
        """Emphasis([Text('a '), Strong([Text('b')]), Text(' c')]) -> *a __b__ c*."""
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Emphasis(
                            children=(
                                Text(value="a "),
                                Strong(children=(Text(value="b"),)),
                                Text(value=" c"),
                            )
                        ),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "*a __b__ c*" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. INLINE CODE BACKTICK SELECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestInlineCodeBacktickSelection:
    """Per CommonMark, backtick delimiter must not appear as a run in the value.
    When value starts/ends with backtick, space padding is required."""

    def test_no_backticks_in_value(self) -> None:
        """Code('hello') -> `hello`."""
        doc = ContentDocument(body=(Paragraph(children=(Code(value="hello"),)),))
        result = serialize_markdown(doc)
        assert "`hello`" in result

    def test_single_backtick_in_value(self) -> None:
        """Code('a`b') -> ``a`b`` (no spaces, value doesn't start/end with `)."""
        doc = ContentDocument(body=(Paragraph(children=(Code(value="a`b"),)),))
        result = serialize_markdown(doc)
        assert "``a`b``" in result

    def test_value_starts_and_ends_with_backtick(self) -> None:
        """Code('`hello`') -> `` `hello` `` (spaces needed around value)."""
        doc = ContentDocument(body=(Paragraph(children=(Code(value="`hello`"),)),))
        result = serialize_markdown(doc)
        assert "`` `hello` ``" in result

    def test_single_backtick_value(self) -> None:
        """Code('`') -> `` ` `` (single backtick needs double delimiter + spaces)."""
        doc = ContentDocument(body=(Paragraph(children=(Code(value="`"),)),))
        result = serialize_markdown(doc)
        assert "`` ` ``" in result

    def test_double_backtick_value(self) -> None:
        """Code('``') -> ``` `` ``` (double backtick needs triple delimiter + spaces)."""
        doc = ContentDocument(body=(Paragraph(children=(Code(value="``"),)),))
        result = serialize_markdown(doc)
        # The value is ``, which has a max run of 2. Delimiter must be ``` (3).
        # Value starts and ends with `, so space padding is needed.
        assert "``` `` ```" in result

    def test_backtick_only_at_start(self) -> None:
        """Code('`hello') -> `` `hello `` (space padding because value starts with `)."""
        doc = ContentDocument(body=(Paragraph(children=(Code(value="`hello"),)),))
        result = serialize_markdown(doc)
        assert "`` `hello ``" in result

    def test_backtick_only_at_end(self) -> None:
        """Code('hello`') -> `` hello` `` (space padding because value ends with `)."""
        doc = ContentDocument(body=(Paragraph(children=(Code(value="hello`"),)),))
        result = serialize_markdown(doc)
        assert "`` hello` ``" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. LINK URL PARENTHESES
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestLinkUrlParentheses:
    """All parens in link URLs are backslash-escaped per CommonMark
    inline-destination escape rules.

    Sec-2 (security finding #1) replaced the prior balanced/unbalanced
    + ``<>``-wrap heuristic with unconditional backslash-escaping of
    ``\\``, ``(``, ``)``, ``<``, ``>``. The new output is always
    parser-stable and immune to the parens-balancing breakout XSS;
    the AST round-trip is unchanged because CommonMark treats ``\\(``
    in a destination as a literal ``(``.
    """

    def test_balanced_parens_in_url(self) -> None:
        """URL with balanced parens — escapes apply."""
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Link(
                            url="https://en.wikipedia.org/wiki/Foo_(bar)",
                            children=(Text(value="text"),),
                        ),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "[text](https://en.wikipedia.org/wiki/Foo_\\(bar\\))" in result

    def test_unbalanced_parens_in_url(self) -> None:
        """URL with unbalanced parens — escapes apply (no <> wrap)."""
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Link(
                            url="https://example.com/a)",
                            children=(Text(value="text"),),
                        ),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "[text](https://example.com/a\\))" in result

    def test_no_parens_in_url(self) -> None:
        """Simple URL without parens — emitted as-is."""
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Link(
                            url="https://example.com/path",
                            children=(Text(value="text"),),
                        ),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "[text](https://example.com/path)" in result

    def test_multiple_balanced_parens(self) -> None:
        """URL with multiple parens pairs — every paren escaped."""
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Link(
                            url="https://example.com/a(b)(c)",
                            children=(Text(value="text"),),
                        ),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "[text](https://example.com/a\\(b\\)\\(c\\))" in result

    def test_unbalanced_open_paren(self) -> None:
        """URL with more ( than ) — escapes apply."""
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Link(
                            url="https://example.com/a(b",
                            children=(Text(value="text"),),
                        ),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "[text](https://example.com/a\\(b)" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 6. TABLE CELL PIPE ESCAPING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestTableCellPipeEscaping:
    """Pipe characters in table cells must be escaped. In regular paragraph
    text, pipes must NOT be escaped."""

    def test_pipe_in_table_cell_escaped(self) -> None:
        """Cell text containing | must have it escaped."""
        doc = ContentDocument(
            body=(
                Table(
                    head=TableSection(
                        rows=(
                            Row(
                                cells=(Cell(content=(Paragraph(children=(Text(value="A | B"),)),)),)
                            ),
                        )
                    ),
                    bodies=(
                        TableSection(
                            rows=(
                                Row(
                                    cells=(
                                        Cell(content=(Paragraph(children=(Text(value="C | D"),)),)),
                                    )
                                ),
                            )
                        ),
                    ),
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "A \\| B" in result
        assert "C \\| D" in result

    def test_pipe_in_paragraph_not_escaped(self) -> None:
        """Pipe in regular paragraph text must NOT be escaped."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="A | B"),)),))
        result = serialize_markdown(doc)
        assert "A | B" in result
        assert "\\|" not in result

    def test_multiple_pipes_in_cell(self) -> None:
        """Multiple pipes in a single cell must all be escaped."""
        doc = ContentDocument(
            body=(
                Table(
                    head=TableSection(
                        rows=(
                            Row(
                                cells=(Cell(content=(Paragraph(children=(Text(value="a|b|c"),)),)),)
                            ),
                        )
                    ),
                    bodies=(TableSection(rows=()),),
                ),
            )
        )
        result = serialize_markdown(doc)
        assert "a\\|b\\|c" in result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# 7. MULTI-LINE TEXT LINE-START ESCAPING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class TestMultiLineTextEscaping:
    """Text nodes containing newlines must apply line-start escaping
    to every line, not just the first."""

    def test_heading_on_second_line_escaped(self) -> None:
        """A text node 'hello\\n# Heading' must escape the second line."""
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Text(value="hello"),
                        SoftBreak(),
                        Text(value="# Heading"),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        lines = result.strip().splitlines()
        # The second line should have the # escaped
        assert len(lines) >= 2
        assert "\\# Heading" in lines[1]

    def test_dash_list_on_second_line_escaped(self) -> None:
        """A text node 'hello\\n- item' must escape the dash on the second line."""
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Text(value="hello"),
                        SoftBreak(),
                        Text(value="- item"),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        lines = result.strip().splitlines()
        assert len(lines) >= 2
        assert "\\- item" in lines[1]

    def test_blockquote_on_second_line_escaped(self) -> None:
        """Blockquote marker on second line must be escaped."""
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Text(value="hello"),
                        SoftBreak(),
                        Text(value="> quote"),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        lines = result.strip().splitlines()
        assert len(lines) >= 2
        assert "\\> quote" in lines[1]

    def test_ordered_list_on_second_line_escaped(self) -> None:
        """Ordered list marker on second line must be escaped."""
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Text(value="hello"),
                        SoftBreak(),
                        Text(value="1. item"),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        lines = result.strip().splitlines()
        assert len(lines) >= 2
        assert "\\1. item" in lines[1]

    def test_multiline_in_single_text_node(self) -> None:
        """A single Text node with embedded newline must escape per-line."""
        doc = ContentDocument(body=(Paragraph(children=(Text(value="hello\n# Heading"),)),))
        result = serialize_markdown(doc)
        lines = result.strip().splitlines()
        assert len(lines) >= 2
        assert "\\# Heading" in lines[1]

    def test_safe_text_on_second_line_not_escaped(self) -> None:
        """Regular text on the second line must not be gratuitously escaped."""
        doc = ContentDocument(
            body=(
                Paragraph(
                    children=(
                        Text(value="hello"),
                        SoftBreak(),
                        Text(value="world"),
                    )
                ),
            )
        )
        result = serialize_markdown(doc)
        lines = result.strip().splitlines()
        assert len(lines) >= 2
        assert lines[1] == "world"
