"""Document segmentation via recursive X-Y cut.

The X-Y cut algorithm (Nagy & Seth, 1984) recursively splits a page at
the widest whitespace gaps, alternating between horizontal and vertical
cuts. This produces a hierarchical segmentation of the page into reading-
order regions.

This is the standard approach for document layout analysis when no
explicit structure (e.g., HTML DOM, heading hierarchy) is available.
"""

from __future__ import annotations

from kaos_content.layout.profiles import (
    find_widest_valley,
    projection_profile,
)
from kaos_content.layout.types import TextBlock


def xy_cut(
    blocks: list[TextBlock],
    bbox: tuple[float, float, float, float],
    *,
    min_gap: float = 5.0,
    min_region_blocks: int = 1,
    resolution: float = 1.0,
    max_depth: int = 20,
) -> list[list[TextBlock]]:
    """Recursive X-Y cut document segmentation.

    Alternately attempts horizontal (Y-axis) and vertical (X-axis) cuts
    at the widest whitespace gap. Recursion stops when no qualifying gap
    is found or the region is too small.

    Args:
        blocks: Text blocks to segment.
        bbox: Bounding box of the region ``(left, top, right, bottom)``.
        min_gap: Minimum gap width (in coordinate units) to make a cut.
        min_region_blocks: Minimum blocks in a region to attempt further cuts.
        resolution: Profile resolution for gap detection.
        max_depth: Maximum recursion depth (safety limit).

    Returns:
        List of block groups in reading order (top-to-bottom, left-to-right).

    Complexity: O(n * D * W/resolution) where D = depth, W = page dimension.
    """
    if not blocks or max_depth <= 0:
        return [blocks] if blocks else []

    if len(blocks) <= min_region_blocks:
        return [blocks]

    left, top, right, bottom = bbox
    width = right - left
    height = bottom - top

    if width <= 0 or height <= 0:
        return [blocks]

    min_gap_bins = max(1, int(min_gap / resolution))

    # Try Y-cut first (horizontal split) — more natural for documents
    y_profile = projection_profile(
        blocks,
        "y",
        resolution=resolution,
        bounds=(top, bottom),
    )
    y_valley = find_widest_valley(y_profile, min_width=min_gap_bins)

    if y_valley is not None and y_valley.width >= min_gap_bins:
        cut_y = top + y_valley.center * resolution
        top_blocks = [b for b in blocks if b.y_center < cut_y]
        bottom_blocks = [b for b in blocks if b.y_center >= cut_y]

        if top_blocks and bottom_blocks:
            top_bbox = (left, top, right, cut_y)
            bottom_bbox = (left, cut_y, right, bottom)
            return xy_cut(
                top_blocks,
                top_bbox,
                min_gap=min_gap,
                min_region_blocks=min_region_blocks,
                resolution=resolution,
                max_depth=max_depth - 1,
            ) + xy_cut(
                bottom_blocks,
                bottom_bbox,
                min_gap=min_gap,
                min_region_blocks=min_region_blocks,
                resolution=resolution,
                max_depth=max_depth - 1,
            )

    # Try X-cut (vertical split) — for multi-column layouts
    x_profile = projection_profile(
        blocks,
        "x",
        resolution=resolution,
        bounds=(left, right),
    )
    x_valley = find_widest_valley(x_profile, min_width=min_gap_bins)

    if x_valley is not None and x_valley.width >= min_gap_bins:
        cut_x = left + x_valley.center * resolution
        left_blocks = [b for b in blocks if b.x_center < cut_x]
        right_blocks = [b for b in blocks if b.x_center >= cut_x]

        if left_blocks and right_blocks:
            left_bbox = (left, top, cut_x, bottom)
            right_bbox = (cut_x, top, right, bottom)
            return xy_cut(
                left_blocks,
                left_bbox,
                min_gap=min_gap,
                min_region_blocks=min_region_blocks,
                resolution=resolution,
                max_depth=max_depth - 1,
            ) + xy_cut(
                right_blocks,
                right_bbox,
                min_gap=min_gap,
                min_region_blocks=min_region_blocks,
                resolution=resolution,
                max_depth=max_depth - 1,
            )

    # No viable cut — this is a leaf region
    # Sort blocks in reading order: top-to-bottom, then left-to-right
    return [sorted(blocks, key=lambda b: (b.y_center, b.x_center))]
