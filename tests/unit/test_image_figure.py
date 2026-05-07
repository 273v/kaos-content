"""Tests for Image dimensions and Figure caption."""

from __future__ import annotations

from kaos_content.model.attr import Caption
from kaos_content.model.blocks import Figure, Paragraph
from kaos_content.model.inlines import Image, Text


class TestImageDimensions:
    def test_image_defaults(self) -> None:
        img = Image(src="x.png")
        assert img.width is None
        assert img.height is None
        assert img.alt is None

    def test_image_with_dimensions(self) -> None:
        img = Image(src="x.png", width=100.0, height=50.0, alt="test")
        assert img.width == 100.0
        assert img.height == 50.0
        assert img.alt == "test"

    def test_image_accepts_int_and_float(self) -> None:
        """Pydantic coerces int to float."""
        img = Image(src="x.png", width=100, height=50)
        assert img.width == 100.0
        assert img.height == 50.0

    def test_image_json_roundtrip(self) -> None:
        img = Image(src="x.png", width=100.0, height=50.0, alt="a")
        restored = Image.model_validate_json(img.model_dump_json())
        assert restored == img

    def test_image_shortcut_with_dimensions(self) -> None:
        from kaos_content.shortcuts import image

        img = image("x.png", "alt", width=200, height=100)
        assert img.width == 200.0
        assert img.height == 100.0


class TestFigureCaption:
    def test_figure_without_caption(self) -> None:
        fig = Figure(children=(Paragraph(children=(Text(value="x"),)),))
        assert fig.caption is None

    def test_figure_with_caption(self) -> None:
        cap = Caption(body=(Paragraph(children=(Text(value="Fig. 1"),)),))
        fig = Figure(
            children=(Paragraph(children=(Text(value="content"),)),),
            caption=cap,
        )
        assert fig.caption is not None
        assert len(fig.caption.body) == 1

    def test_figure_json_roundtrip_with_caption(self) -> None:
        cap = Caption(body=(Paragraph(children=(Text(value="Caption text"),)),))
        fig = Figure(
            children=(Paragraph(children=(Text(value="body"),)),),
            caption=cap,
        )
        data = fig.model_dump_json()
        restored = Figure.model_validate_json(data)
        assert restored.caption is not None
        caption_block = restored.caption.body[0]
        assert isinstance(caption_block, Paragraph)
        caption_text = caption_block.children[0]
        assert isinstance(caption_text, Text)
        assert caption_text.value == "Caption text"
