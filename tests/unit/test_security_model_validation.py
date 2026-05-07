"""Model field-constraint regression tests (audit M1).

Pins the validation contract introduced in 0.1.0a1:

- ``BoundingBox`` rejects right < left or (top_left) bottom < top — a
  zero/negative-extent box is not a valid region.
- ``Provenance.page`` is 1-indexed (``ge=1``); ``0`` is rejected.
- ``Provenance.confidence`` is bounded to ``[0.0, 1.0]``.
- ``Provenance.char_span`` is non-negative and ``end >= start``.
- ``Cell.row_span`` and ``Cell.col_span`` are ``>= 1``.
- ``Image.width`` and ``Image.height`` are ``> 0`` when set.

Audit findings addressed: M1 (model accepts negative image sizes, zero
spans, confidence > 1, etc.).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from kaos_content.model.attr import BoundingBox, CoordOrigin, Provenance
from kaos_content.model.inlines import Image
from kaos_content.model.table import Cell

# ────────────────────────────────────────────────────────────────────
# BoundingBox extents
# ────────────────────────────────────────────────────────────────────


def test_bounding_box_valid_top_left() -> None:
    """A normal box passes."""
    bb = BoundingBox(left=10, top=20, right=100, bottom=200)
    assert bb.right == 100


def test_bounding_box_valid_zero_extent() -> None:
    """A degenerate (zero-area) box at a single point is allowed —
    only strictly inverted boxes are rejected."""
    bb = BoundingBox(left=10, top=20, right=10, bottom=20)
    assert bb.left == bb.right == 10


def test_bounding_box_rejects_inverted_horizontal() -> None:
    """right < left is rejected for any origin."""
    with pytest.raises(ValidationError, match=r"right .* must be >= left"):
        BoundingBox(left=100, top=0, right=50, bottom=200)


def test_bounding_box_rejects_inverted_vertical_top_left() -> None:
    """In top_left origin, bottom < top is rejected."""
    with pytest.raises(ValidationError, match=r"bottom .* must be >= top"):
        BoundingBox(left=0, top=200, right=100, bottom=100)


def test_bounding_box_allows_inverted_vertical_bottom_left() -> None:
    """In bottom_left (PDF) origin, bottom < top is the convention."""
    bb = BoundingBox(
        left=0,
        top=200,
        right=100,
        bottom=100,
        coord_origin=CoordOrigin.BOTTOM_LEFT,
    )
    assert bb.bottom == 100 and bb.top == 200


def test_bounding_box_negative_coords_allowed() -> None:
    """Negative coordinates are allowed — translated pages may go negative.
    We only enforce extents, not sign."""
    bb = BoundingBox(left=-10, top=-20, right=-5, bottom=-1)
    assert bb.left == -10


# ────────────────────────────────────────────────────────────────────
# Provenance.page (1-indexed)
# ────────────────────────────────────────────────────────────────────


def test_provenance_page_valid() -> None:
    p = Provenance(page=1)
    assert p.page == 1


def test_provenance_page_none_means_unknown() -> None:
    """page=None is the canonical 'unknown' marker."""
    p = Provenance(page=None)
    assert p.page is None


def test_provenance_page_zero_rejected() -> None:
    """0 is the historical 'unknown' sentinel — callers must use None."""
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        Provenance(page=0)


def test_provenance_page_negative_rejected() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        Provenance(page=-3)


# ────────────────────────────────────────────────────────────────────
# Provenance.confidence (bounded [0.0, 1.0])
# ────────────────────────────────────────────────────────────────────


def test_provenance_confidence_valid_endpoints() -> None:
    Provenance(confidence=0.0)
    Provenance(confidence=1.0)
    Provenance(confidence=0.5)


def test_provenance_confidence_above_one_rejected() -> None:
    """Out-of-range confidence used to silently break ranking logic."""
    with pytest.raises(ValidationError, match="less than or equal to 1"):
        Provenance(confidence=1.5)


def test_provenance_confidence_negative_rejected() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 0"):
        Provenance(confidence=-0.1)


# ────────────────────────────────────────────────────────────────────
# Provenance.char_span (non-negative, end >= start)
# ────────────────────────────────────────────────────────────────────


def test_provenance_char_span_valid() -> None:
    p = Provenance(char_span=(5, 10))
    assert p.char_span == (5, 10)


def test_provenance_char_span_zero_length_allowed() -> None:
    """A zero-length span (insertion point) is valid."""
    p = Provenance(char_span=(7, 7))
    assert p.char_span == (7, 7)


def test_provenance_char_span_inverted_rejected() -> None:
    with pytest.raises(ValidationError, match=r"end .* must be >= start"):
        Provenance(char_span=(10, 5))


def test_provenance_char_span_negative_start_rejected() -> None:
    with pytest.raises(ValidationError, match="must be non-negative"):
        Provenance(char_span=(-1, 5))


def test_provenance_char_span_negative_end_rejected() -> None:
    with pytest.raises(ValidationError, match="must be non-negative"):
        Provenance(char_span=(5, -1))


# ────────────────────────────────────────────────────────────────────
# Cell row_span / col_span (>= 1)
# ────────────────────────────────────────────────────────────────────


def test_cell_default_spans() -> None:
    cell = Cell()
    assert cell.row_span == 1
    assert cell.col_span == 1


def test_cell_valid_large_span() -> None:
    cell = Cell(row_span=5, col_span=3)
    assert cell.row_span == 5


def test_cell_row_span_zero_rejected() -> None:
    """A cell that doesn't occupy its own slot is structurally invalid."""
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        Cell(row_span=0)


def test_cell_col_span_zero_rejected() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        Cell(col_span=0)


def test_cell_row_span_negative_rejected() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        Cell(row_span=-1)


def test_cell_col_span_negative_rejected() -> None:
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        Cell(col_span=-1)


# ────────────────────────────────────────────────────────────────────
# Image.width / Image.height (> 0 when set)
# ────────────────────────────────────────────────────────────────────


def test_image_no_dimensions_allowed() -> None:
    """Width/height are optional."""
    img = Image(src="x.png")
    assert img.width is None
    assert img.height is None


def test_image_positive_dimensions_allowed() -> None:
    img = Image(src="x.png", width=100.0, height=200.0)
    assert img.width == 100.0


def test_image_zero_width_rejected() -> None:
    """A zero-width image renders nothing — structurally invalid."""
    with pytest.raises(ValidationError, match="greater than 0"):
        Image(src="x.png", width=0)


def test_image_zero_height_rejected() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        Image(src="x.png", height=0)


def test_image_negative_width_rejected() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        Image(src="x.png", width=-100)


def test_image_negative_height_rejected() -> None:
    with pytest.raises(ValidationError, match="greater than 0"):
        Image(src="x.png", height=-50)
