"""Projection profiles — reduce 2D positioned blocks to 1D density signals.

A projection profile counts how many blocks (or how much area/length) overlap
each position along an axis. This reduces the 2D layout analysis problem to
1D signal processing: valleys in the horizontal profile indicate column
gutters; valleys in the vertical profile indicate paragraph/section gaps.

Requires numpy for efficient array operations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.typing import NDArray

from kaos_content.layout.types import TextBlock, Valley

if TYPE_CHECKING:
    pass


def projection_profile(
    blocks: list[TextBlock],
    axis: Literal["x", "y"],
    *,
    resolution: float = 1.0,
    weight: Literal["count", "area", "length"] = "count",
    bounds: tuple[float, float] | None = None,
) -> NDArray[np.float64]:
    """Compute a projection profile along one axis.

    Projects block extents onto the chosen axis and accumulates a density
    signal. Each position in the output array represents one ``resolution``-
    wide bin.

    Args:
        blocks: Input text blocks.
        axis: ``"x"`` for horizontal profile (finding column gutters),
              ``"y"`` for vertical profile (finding row/paragraph gaps).
        resolution: Bin width in coordinate units. Smaller = more precise,
                    larger = smoother. Default 1.0.
        weight: How to weight each block's contribution:
            - ``"count"``: Each bin covered by a block gets +1.
            - ``"area"``: Weighted by the perpendicular dimension.
            - ``"length"``: Weighted by the text length.
        bounds: Optional ``(min, max)`` to fix the profile range. If None,
                computed from block extents.

    Returns:
        1D numpy float64 array of length ``ceil((max - min) / resolution)``.

    Complexity: O(n *W/resolution) where W = mean block extent.
    """
    if not blocks:
        return np.array([], dtype=np.float64)

    # Determine axis extents
    if axis == "x":
        starts = [b.left for b in blocks]
        ends = [b.right for b in blocks]
        perp_sizes = [b.height for b in blocks]
    else:
        starts = [b.top for b in blocks]
        ends = [b.bottom for b in blocks]
        perp_sizes = [b.width for b in blocks]

    if bounds is not None:
        lo, hi = bounds
    else:
        lo = min(starts)
        hi = max(ends)

    if hi <= lo:
        return np.array([0.0], dtype=np.float64)

    n_bins = max(1, int(np.ceil((hi - lo) / resolution)))
    profile = np.zeros(n_bins, dtype=np.float64)

    for i, block in enumerate(blocks):
        s = starts[i]
        e = ends[i]
        bin_start = max(0, int((s - lo) / resolution))
        bin_end = min(n_bins, int(np.ceil((e - lo) / resolution)))

        if weight == "count":
            w = 1.0
        elif weight == "area":
            w = perp_sizes[i]
        else:  # length
            w = float(len(block.text))

        profile[bin_start:bin_end] += w

    return profile


def find_valleys(
    profile: NDArray[np.float64],
    *,
    min_width: int = 1,
    threshold: float = 0.0,
) -> list[Valley]:
    """Find valleys (gaps) in a projection profile.

    A valley is a contiguous run of bins at or below the threshold.
    Returns valleys sorted by width (widest first).

    Args:
        profile: 1D array from ``projection_profile``.
        min_width: Minimum valley width in bins to report.
        threshold: Values at or below this are considered "empty".

    Returns:
        List of Valley objects sorted by width descending.

    Complexity: O(W) where W = profile length.
    """
    if len(profile) == 0:
        return []

    valleys: list[Valley] = []
    in_valley = False
    start = 0

    for i in range(len(profile)):
        if profile[i] <= threshold:
            if not in_valley:
                start = i
                in_valley = True
        else:
            if in_valley:
                width = i - start
                if width >= min_width:
                    valleys.append(
                        Valley(
                            start=start,
                            end=i,
                            width=width,
                            center=(start + i) // 2,
                        )
                    )
                in_valley = False

    # Handle valley at end of profile
    if in_valley:
        width = len(profile) - start
        if width >= min_width:
            valleys.append(
                Valley(
                    start=start,
                    end=len(profile),
                    width=width,
                    center=(start + len(profile)) // 2,
                )
            )

    # Sort widest first
    valleys.sort(key=lambda v: v.width, reverse=True)
    return valleys


def find_widest_valley(
    profile: NDArray[np.float64],
    *,
    min_width: int = 1,
    threshold: float = 0.0,
) -> Valley | None:
    """Find the single widest valley in a profile.

    Convenience wrapper around ``find_valleys`` — returns the widest gap,
    or None if no qualifying valley exists.

    Args:
        profile: 1D array from ``projection_profile``.
        min_width: Minimum valley width in bins.
        threshold: Values at or below this are considered "empty".

    Returns:
        The widest Valley, or None.
    """
    valleys = find_valleys(profile, min_width=min_width, threshold=threshold)
    return valleys[0] if valleys else None
