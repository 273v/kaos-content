"""Tests for clean_extracted_text: noncharacters, soft hyphens, NFKC."""

from __future__ import annotations

import pytest

from kaos_content import clean_extracted_text, is_noncharacter


@pytest.mark.parametrize("cp", [0xFFFE, 0xFFFF, 0xFDD0, 0xFDEF, 0x1FFFE, 0x10FFFF])
def test_noncharacters(cp: int) -> None:
    assert is_noncharacter(chr(cp))


@pytest.mark.parametrize("cp", [0xFFFD, 0xFDCF, 0xFDF0, 0x00AD, 0x0041])
def test_not_noncharacters(cp: int) -> None:
    assert not is_noncharacter(chr(cp))


def test_line_end_soft_hyphen_joins_word() -> None:
    assert clean_extracted_text("the repre­\nsentatives met") == "the representatives met"


def test_line_end_legacy_pdfium_marker_joins_word() -> None:
    assert clean_extracted_text("appro￾\n  priations act") == "appropriations act"


def test_crlf_line_end_joins() -> None:
    assert clean_extracted_text("commit­\r\ntee") == "committee"


def test_uppercase_continuation_keeps_break() -> None:
    assert clean_extracted_text("end of line­\nNew sentence") == "end of line\nNew sentence"


def test_mid_line_soft_hyphen_removed() -> None:
    assert clean_extracted_text("hyphen­ation") == "hyphenation"


def test_mid_line_noncharacter_removed() -> None:
    assert clean_extracted_text("a￾b￿c") == "abc"


def test_marker_after_digit_does_not_join() -> None:
    assert clean_extracted_text("page 12­\ncontinued") == "page 12\ncontinued"


def test_nfkc_applied_and_optional() -> None:
    assert clean_extracted_text("ﬁle ™") == "file TM"
    assert clean_extracted_text("ﬁle", unicode_nfkc=False) == "ﬁle"


def test_plain_text_unchanged() -> None:
    s = "Section 1.\n\nThe Secretary shall\tissue rules.\n"
    assert clean_extracted_text(s) == s
