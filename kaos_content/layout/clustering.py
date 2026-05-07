"""1D clustering primitives — the foundation of all layout analysis.

Every layout detection task reduces to clustering 1D distributions:
font sizes, Y-gaps, X-positions, line spacings, etc. These functions
operate on plain ``list[float]`` and return structured results with
full diagnostic metadata.

Algorithms implemented:
- **cluster_1d**: Tolerance-based grouping (the pdfplumber pattern)
- **otsu_threshold**: Optimal binary split (maximize between-class variance)
- **jenks_breaks**: Fisher-Jenks natural breaks (minimize within-class variance)
- **find_modes**: Histogram peak detection

All are O(n log n) or better except Jenks which is O(n²k) —
acceptable for document-scale inputs (n < 10K).
"""

from __future__ import annotations

import math

from kaos_content.layout.types import (
    BreaksResult,
    ClusterResult,
    ModeResult,
    ThresholdResult,
)


def cluster_1d(
    values: list[float],
    tolerance: float,
    *,
    presorted: bool = False,
) -> ClusterResult:
    """Tolerance-based 1D clustering: group values within tolerance of neighbors.

    This is the pdfplumber pattern — simple, fast, and predictable.
    Values are sorted, then adjacent values within ``tolerance`` of each
    other are grouped together.

    Args:
        values: Input values to cluster.
        tolerance: Maximum gap between adjacent values in the same cluster.
        presorted: If True, skip internal sort (caller guarantees ascending order).

    Returns:
        ClusterResult with groups (index lists), centroids, and sizes.

    Complexity: O(n log n) — dominated by sort. O(n) if presorted.
    """
    n = len(values)
    if n == 0:
        return ClusterResult(groups=[], centroids=[], sizes=[])

    # Sort and track original indices
    if presorted:
        indexed = list(enumerate(values))
    else:
        indexed = sorted(enumerate(values), key=lambda iv: iv[1])

    groups: list[list[int]] = []
    centroids: list[float] = []
    current_group: list[int] = [indexed[0][0]]
    current_sum = indexed[0][1]

    for i in range(1, n):
        idx, val = indexed[i]
        prev_val = indexed[i - 1][1]

        if val - prev_val <= tolerance:
            current_group.append(idx)
            current_sum += val
        else:
            groups.append(current_group)
            centroids.append(current_sum / len(current_group))
            current_group = [idx]
            current_sum = val

    # Flush last group
    groups.append(current_group)
    centroids.append(current_sum / len(current_group))

    return ClusterResult(
        groups=groups,
        centroids=centroids,
        sizes=[len(g) for g in groups],
    )


def otsu_threshold(values: list[float]) -> ThresholdResult:
    """Find the optimal binary split threshold (maximize between-class variance).

    Otsu's method applied to arbitrary float distributions. Sweeps over
    the midpoints between consecutive distinct sorted values and picks the
    split that maximizes between-class variance.

    This is the workhorse for binary decisions: line-gap vs paragraph-gap,
    body-font vs heading-font, column-gutter vs word-spacing.

    Args:
        values: Input values. Must have at least 2 distinct values.

    Returns:
        ThresholdResult with threshold, variance, and class statistics.

    Complexity: O(n log n) — dominated by sort.
    """
    n = len(values)
    if n == 0:
        return ThresholdResult(
            threshold=0.0,
            variance=0.0,
            below_count=0,
            above_count=0,
            below_mean=0.0,
            above_mean=0.0,
        )
    if n == 1:
        return ThresholdResult(
            threshold=values[0],
            variance=0.0,
            below_count=0,
            above_count=1,
            below_mean=0.0,
            above_mean=values[0],
        )

    sorted_vals = sorted(values)

    # Degenerate case: all values identical
    if sorted_vals[0] == sorted_vals[-1]:
        return ThresholdResult(
            threshold=sorted_vals[0],
            variance=0.0,
            below_count=0,
            above_count=n,
            below_mean=0.0,
            above_mean=sorted_vals[0],
        )

    # Sweep: try splitting between every pair of consecutive distinct values
    total_sum = sum(sorted_vals)
    best_variance = -1.0
    best_threshold = sorted_vals[0]
    best_split_idx = 0  # number of values in the "below" class

    left_sum = 0.0
    left_count = 0

    for i in range(n - 1):
        left_sum += sorted_vals[i]
        left_count += 1

        # Only consider splits between distinct values
        if sorted_vals[i] == sorted_vals[i + 1]:
            continue

        right_count = n - left_count
        right_sum = total_sum - left_sum

        left_mean = left_sum / left_count
        right_mean = right_sum / right_count

        variance = left_count * right_count * (left_mean - right_mean) ** 2

        if variance > best_variance:
            best_variance = variance
            best_threshold = (sorted_vals[i] + sorted_vals[i + 1]) / 2.0
            best_split_idx = left_count

    # Compute class statistics at the chosen threshold
    below = sorted_vals[:best_split_idx]
    above = sorted_vals[best_split_idx:]

    return ThresholdResult(
        threshold=best_threshold,
        variance=best_variance / (n * n) if n > 0 else 0.0,
        below_count=len(below),
        above_count=len(above),
        below_mean=sum(below) / len(below) if below else 0.0,
        above_mean=sum(above) / len(above) if above else 0.0,
    )


def jenks_breaks(values: list[float], k: int) -> BreaksResult:
    """Fisher-Jenks natural breaks optimization (minimize within-class variance).

    Finds k-1 break points that divide the data into k classes, minimizing
    the sum of squared deviations within each class. This is the gold standard
    for choropleth map classification and works perfectly for font-size
    clustering, spacing analysis, and any multi-class 1D segmentation.

    Uses the Fisher-Jenks DP algorithm (not the iterative heuristic).

    Args:
        values: Input values. Must have at least k distinct values for
                meaningful results.
        k: Number of classes (2-10 typical for documents).

    Returns:
        BreaksResult with break points, per-value labels, and Goodness
        of Variance Fit (GVF).

    Complexity: O(n² * k) time, O(n * k) space. For n=5000, k=3: ~25ms.
    """
    n = len(values)
    if n == 0:
        return BreaksResult(breaks=[], labels=[], gvf=1.0)
    if k <= 1 or n <= 1:
        return BreaksResult(breaks=[], labels=[0] * n, gvf=1.0)

    # Sort and track original indices
    indexed = sorted(enumerate(values), key=lambda iv: iv[1])
    sorted_vals = [iv[1] for iv in indexed]
    original_indices = [iv[0] for iv in indexed]

    # Clamp k to number of distinct values
    distinct = len(set(sorted_vals))
    if k > distinct:
        k = distinct
    if k <= 1:
        return BreaksResult(breaks=[], labels=[0] * n, gvf=1.0)

    # Precompute cumulative sums for O(1) range-sum queries
    cum_sum = [0.0] * (n + 1)
    cum_sum_sq = [0.0] * (n + 1)
    for i in range(n):
        cum_sum[i + 1] = cum_sum[i] + sorted_vals[i]
        cum_sum_sq[i + 1] = cum_sum_sq[i] + sorted_vals[i] ** 2

    def ssdev(start: int, end: int) -> float:
        """Sum of squared deviations for sorted_vals[start..end] (inclusive)."""
        count = end - start + 1
        s = cum_sum[end + 1] - cum_sum[start]
        sq = cum_sum_sq[end + 1] - cum_sum_sq[start]
        return sq - (s * s) / count

    # DP tables: lower_class_limits[i][j] = optimal last break before class j
    #            variance_combinations[i][j] = min SDAM for first j classes on first i items
    INF = float("inf")
    lower_class_limits = [[0] * (k + 1) for _ in range(n + 1)]
    variance_combinations = [[INF] * (k + 1) for _ in range(n + 1)]

    # Base case: one class
    for i in range(1, n + 1):
        variance_combinations[i][1] = ssdev(0, i - 1)
        lower_class_limits[i][1] = 0

    # Fill DP
    for j in range(2, k + 1):
        for i in range(j, n + 1):
            for m in range(j - 1, i):
                cost = variance_combinations[m][j - 1] + ssdev(m, i - 1)
                if cost < variance_combinations[i][j]:
                    variance_combinations[i][j] = cost
                    lower_class_limits[i][j] = m

    # Backtrack to find break indices
    break_indices = [0] * (k - 1)
    idx = n
    for j in range(k, 1, -1):
        idx = lower_class_limits[idx][j]
        break_indices[j - 2] = idx

    # Convert to break values (midpoint between adjacent sorted values)
    breaks: list[float] = []
    for bi in break_indices:
        if bi > 0 and bi < n:
            breaks.append((sorted_vals[bi - 1] + sorted_vals[bi]) / 2.0)
        elif bi == 0:
            breaks.append(sorted_vals[0])
        else:
            breaks.append(sorted_vals[-1])

    # Assign labels based on breaks
    sorted_labels = [0] * n
    for i in range(n):
        label = 0
        for b in breaks:
            if sorted_vals[i] >= b:
                label += 1
        sorted_labels[i] = label

    # Map back to original order
    labels = [0] * n
    for i in range(n):
        labels[original_indices[i]] = sorted_labels[i]

    # Compute Goodness of Variance Fit
    total_ssdev = ssdev(0, n - 1)
    within_ssdev = variance_combinations[n][k]
    gvf = 1.0 - (within_ssdev / total_ssdev) if total_ssdev > 0 else 1.0

    return BreaksResult(breaks=breaks, labels=labels, gvf=gvf)


def find_modes(
    values: list[float],
    bin_width: float,
    *,
    min_count: int = 1,
) -> ModeResult:
    """Find modal values (peaks in a histogram).

    Bins the values and returns centers of bins that are local maxima
    (higher count than both neighbors). Useful for finding the most common
    font size (body text), most common line spacing, etc.

    Args:
        values: Input values.
        bin_width: Width of each histogram bin. Should be chosen based on
                   the precision of the input data (e.g., 0.5 for font sizes
                   in points, 1.0 for pixel positions).
        min_count: Minimum count in a bin to qualify as a mode.

    Returns:
        ModeResult with modal values, counts, and the bin_width used.

    Complexity: O(n + B) where B = number of bins.
    """
    if not values:
        return ModeResult(modes=[], counts=[], bin_width=bin_width)

    v_min = min(values)
    v_max = max(values)

    if bin_width <= 0:
        bin_width = 1.0

    n_bins = max(1, math.ceil((v_max - v_min) / bin_width) + 1)
    histogram = [0] * n_bins

    for v in values:
        b = int((v - v_min) / bin_width)
        if b >= n_bins:
            b = n_bins - 1
        histogram[b] += 1

    # Find local maxima
    modes: list[float] = []
    counts: list[int] = []

    for i in range(n_bins):
        if histogram[i] < min_count:
            continue
        left = histogram[i - 1] if i > 0 else 0
        right = histogram[i + 1] if i < n_bins - 1 else 0
        if histogram[i] >= left and histogram[i] >= right:
            center = v_min + (i + 0.5) * bin_width
            modes.append(center)
            counts.append(histogram[i])

    return ModeResult(modes=modes, counts=counts, bin_width=bin_width)
