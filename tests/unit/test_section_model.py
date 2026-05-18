"""Tests for the ``Section`` model + ``ContentDocument.sections`` field.

Sections describe page-layout regions within a document. This file
pins down the invariants a ``Section`` tuple must satisfy so that
round-tripping through a format (DOCX / future HTML @page / PDF) stays
lossless and deterministic.
"""

from __future__ import annotations

import pytest

from kaos_content.model.blocks import Paragraph
from kaos_content.model.document import ContentDocument
from kaos_content.model.inlines import Text
from kaos_content.model.metadata import PageSetup, Section


class TestSectionDefaults:
    def test_break_type_defaults_to_next_page(self) -> None:
        s = Section(end_block_index=5)
        assert s.break_type == "nextPage"
        assert s.page_setup is None

    def test_frozen(self) -> None:
        from pydantic import ValidationError

        s = Section(end_block_index=5)
        with pytest.raises(ValidationError):
            s.end_block_index = 10


class TestSectionOnDocument:
    def test_default_sections_is_empty(self) -> None:
        doc = ContentDocument()
        assert doc.sections == ()

    def test_single_section_round_trip(self) -> None:
        section = Section(
            end_block_index=1,
            page_setup=PageSetup(page_width_pt=612.0, page_height_pt=792.0),
        )
        doc = ContentDocument(
            body=(Paragraph(children=(Text(value="hi"),)),),
            sections=(section,),
        )
        # pydantic.model_dump / model_validate round-trip.
        roundtripped = ContentDocument.model_validate(doc.model_dump())
        assert roundtripped.sections == doc.sections

    def test_multi_section(self) -> None:
        portrait = Section(
            end_block_index=2,
            page_setup=PageSetup(page_width_pt=612.0, page_height_pt=792.0),
        )
        landscape = Section(
            end_block_index=4,
            page_setup=PageSetup(page_width_pt=792.0, page_height_pt=612.0),
        )
        doc = ContentDocument(
            body=tuple(Paragraph(children=(Text(value=f"p{i}"),)) for i in range(4)),
            sections=(portrait, landscape),
        )
        assert len(doc.sections) == 2
        # Section ranges: body[0:2] then body[2:4]. Last section's
        # end_block_index equals len(body).
        assert doc.sections[-1].end_block_index == len(doc.body)


class TestSectionBreakType:
    def test_all_break_types_accepted(self) -> None:
        for bt in ("continuous", "nextPage", "nextColumn", "evenPage", "oddPage"):
            s = Section(end_block_index=1, break_type=bt)
            assert s.break_type == bt

    def test_unknown_break_type_rejected(self) -> None:
        from pydantic import ValidationError

        # Literal["..."] is enforced at validation time.
        with pytest.raises(ValidationError):
            Section(end_block_index=1, break_type="bogus")  # ty: ignore[invalid-argument-type]
