"""Tests for cross-type content model violations.

Backfill from Phase 0 gap: verify that Pydantic's discriminated union
rejects invalid tree structures at construction time.
"""

import pytest
from pydantic import ValidationError

from kaos_content import (
    BlockQuote,
    BulletList,
    CodeBlock,
    Div,
    Heading,
    ListItem,
    OrderedList,
    Paragraph,
    Table,
    Text,
)


class TestBlockInInlinePosition:
    """Blocks must not appear where inlines are expected."""

    def test_paragraph_rejects_block_child(self) -> None:
        with pytest.raises(ValidationError):
            Paragraph(children=(Paragraph(children=(Text(value="nested"),)),))

    def test_paragraph_rejects_heading(self) -> None:
        with pytest.raises(ValidationError):
            Paragraph(children=(Heading(depth=1, children=(Text(value="h"),)),))

    def test_heading_rejects_block_child(self) -> None:
        with pytest.raises(ValidationError):
            Heading(depth=1, children=(Table(),))

    def test_heading_rejects_codeblock(self) -> None:
        with pytest.raises(ValidationError):
            Heading(depth=1, children=(CodeBlock(value="x"),))


class TestInlineInBlockPosition:
    """Inlines must not appear where blocks are expected."""

    def test_blockquote_rejects_text(self) -> None:
        with pytest.raises(ValidationError):
            BlockQuote(children=(Text(value="text"),))

    def test_div_rejects_text(self) -> None:
        with pytest.raises(ValidationError):
            Div(children=(Text(value="text"),))

    def test_list_item_rejects_text(self) -> None:
        with pytest.raises(ValidationError):
            ListItem(children=(Text(value="text"),))


class TestListTypeConstraints:
    """OrderedList and BulletList only accept ListItem children."""

    def test_ordered_list_rejects_paragraph(self) -> None:
        with pytest.raises(ValidationError):
            OrderedList(children=(Paragraph(children=(Text(value="x"),)),))

    def test_bullet_list_rejects_paragraph(self) -> None:
        with pytest.raises(ValidationError):
            BulletList(children=(Paragraph(children=(Text(value="x"),)),))


class TestValidConstructions:
    """Verify that valid constructions still work (sanity check)."""

    def test_paragraph_with_text(self) -> None:
        p = Paragraph(children=(Text(value="ok"),))
        assert len(p.children) == 1

    def test_blockquote_with_paragraph(self) -> None:
        bq = BlockQuote(children=(Paragraph(children=(Text(value="ok"),)),))
        assert len(bq.children) == 1

    def test_list_with_items(self) -> None:
        bl = BulletList(children=(ListItem(children=(Paragraph(children=(Text(value="ok"),)),)),))
        assert len(bl.children) == 1
