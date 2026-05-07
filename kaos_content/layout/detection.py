"""Layout detection — compositions of clustering and projection primitives.

Each function in this module composes the Tier 1 (clustering) and Tier 2
(projection) primitives to detect specific layout features. They all take
``list[TextBlock]`` and return structured results.

Functions:
- ``group_into_lines``: Group blocks sharing the same visual line
- ``detect_columns``: Find column boundaries from horizontal profiles
- ``detect_paragraph_breaks``: Find paragraph gaps via vertical spacing analysis
- ``classify_font_sizes``: Classify blocks by font size (heading/body/footnote)
- ``detect_headers_footers``: Find repeated content across pages
- ``detect_table_regions``: Find grid-aligned text regions
"""

from __future__ import annotations

from collections import Counter
from typing import Literal

from kaos_content.layout.clustering import (
    cluster_1d,
    jenks_breaks,
    otsu_threshold,
)
from kaos_content.layout.profiles import (
    find_valleys,
    projection_profile,
)
from kaos_content.layout.types import (
    ColumnResult,
    FontSizeClassification,
    LineGroup,
    TextBlock,
)


def group_into_lines(
    blocks: list[TextBlock],
    y_tolerance: float = 2.0,
) -> list[LineGroup]:
    """Group blocks into horizontal lines by Y-coordinate proximity.

    This is the most fundamental layout operation — used by nearly every
    other detection function. Blocks whose Y-centers are within
    ``y_tolerance`` of each other are grouped into the same line.

    Within each line, blocks are sorted left-to-right by ``left`` coordinate.

    Args:
        blocks: Input text blocks.
        y_tolerance: Maximum vertical distance between Y-centers to
                     consider blocks on the same line.

    Returns:
        List of LineGroup objects, sorted top-to-bottom by Y-center.

    Complexity: O(n log n) — dominated by sorting.
    """
    if not blocks:
        return []

    # Sort by Y-center
    indexed = sorted(enumerate(blocks), key=lambda ib: ib[1].y_center)

    lines: list[LineGroup] = []
    current_indices: list[int] = [indexed[0][0]]
    current_y_sum = indexed[0][1].y_center

    for i in range(1, len(indexed)):
        idx, block = indexed[i]
        current_mean_y = current_y_sum / len(current_indices)

        if abs(block.y_center - current_mean_y) <= y_tolerance:
            current_indices.append(idx)
            current_y_sum += block.y_center
        else:
            # Flush current line
            line_blocks = [blocks[j] for j in current_indices]
            # Sort within line by x position
            pairs = sorted(zip(current_indices, line_blocks, strict=True), key=lambda p: p[1].left)
            sorted_indices = [p[0] for p in pairs]
            sorted_blocks = [p[1] for p in pairs]
            y_center = current_y_sum / len(current_indices)
            lines.append(
                LineGroup(
                    blocks=sorted_blocks,
                    indices=sorted_indices,
                    y_center=y_center,
                    top=min(b.top for b in sorted_blocks),
                    bottom=max(b.bottom for b in sorted_blocks),
                )
            )
            current_indices = [idx]
            current_y_sum = block.y_center

    # Flush last line
    if current_indices:
        line_blocks = [blocks[j] for j in current_indices]
        pairs = sorted(zip(current_indices, line_blocks, strict=True), key=lambda p: p[1].left)
        sorted_indices = [p[0] for p in pairs]
        sorted_blocks = [p[1] for p in pairs]
        y_center = current_y_sum / len(current_indices)
        lines.append(
            LineGroup(
                blocks=sorted_blocks,
                indices=sorted_indices,
                y_center=y_center,
                top=min(b.top for b in sorted_blocks),
                bottom=max(b.bottom for b in sorted_blocks),
            )
        )

    # Sort lines top-to-bottom
    lines.sort(key=lambda lg: lg.y_center)
    return lines


def detect_columns(
    blocks: list[TextBlock],
    *,
    page_width: float | None = None,
    min_gutter_width: float = 10.0,
    resolution: float = 1.0,
    threshold_ratio: float = 0.1,
) -> ColumnResult:
    """Detect column boundaries from block positions.

    Uses horizontal projection profile + valley detection. Valleys wider
    than ``min_gutter_width`` that drop below the threshold are identified
    as column gutters.

    Args:
        blocks: Input text blocks.
        page_width: Page width for profile bounds. If None, computed from blocks.
        min_gutter_width: Minimum gutter width (in coordinate units).
        resolution: Profile resolution (bin width).
        threshold_ratio: Fraction of peak density below which a valley qualifies.

    Returns:
        ColumnResult with column boundaries and gutter locations.
    """
    if not blocks:
        return ColumnResult(columns=[], gutters=[])

    bounds = (0.0, page_width) if page_width is not None else None
    profile = projection_profile(blocks, "x", resolution=resolution, bounds=bounds)

    if len(profile) == 0:
        return ColumnResult(columns=[], gutters=[])

    peak = float(profile.max())
    threshold = peak * threshold_ratio
    min_bins = max(1, int(min_gutter_width / resolution))

    valleys = find_valleys(profile, min_width=min_bins, threshold=threshold)

    if not valleys:
        # Single column: entire width
        lo = min(b.left for b in blocks) if bounds is None else 0.0
        hi = max(b.right for b in blocks) if bounds is None else (page_width or 0.0)
        return ColumnResult(columns=[(lo, hi)], gutters=[])

    # Sort valleys by position (left to right)
    valleys_sorted = sorted(valleys, key=lambda v: v.start)

    # Build column ranges between valleys
    lo = min(b.left for b in blocks) if bounds is None else 0.0
    hi = max(b.right for b in blocks) if bounds is None else (page_width or 0.0)

    columns: list[tuple[float, float]] = []
    prev_end = lo

    for valley in valleys_sorted:
        col_left = prev_end
        col_right = lo + valley.start * resolution
        if col_right > col_left:
            columns.append((col_left, col_right))
        prev_end = lo + valley.end * resolution

    # Last column
    if prev_end < hi:
        columns.append((prev_end, hi))

    return ColumnResult(columns=columns, gutters=valleys_sorted)


def detect_paragraph_breaks(
    blocks: list[TextBlock],
    *,
    method: Literal["otsu", "relative", "jenks"] = "otsu",
    y_tolerance: float = 2.0,
    relative_threshold: float = 1.5,
) -> list[int]:
    """Find paragraph boundaries from vertical spacing analysis.

    Groups blocks into lines, computes inter-line Y-gaps, then classifies
    gaps as "line spacing" or "paragraph spacing" using the chosen method.

    Args:
        blocks: Input text blocks.
        method: Classification method for gaps:
            - ``"otsu"``: Binary split at optimal threshold (recommended).
            - ``"relative"``: Gaps > ``relative_threshold`` xmedian are paragraph breaks.
            - ``"jenks"``: Jenks natural breaks with k=2.
        y_tolerance: Y-tolerance for line grouping.
        relative_threshold: Multiplier for the "relative" method.

    Returns:
        Indices into the ``blocks`` list where paragraph breaks occur.
        A break at index ``i`` means the block at ``i`` starts a new paragraph.
    """
    lines = group_into_lines(blocks, y_tolerance=y_tolerance)

    if len(lines) < 2:
        return []

    # Compute inter-line gaps
    gaps: list[float] = []
    for i in range(1, len(lines)):
        gap = lines[i].top - lines[i - 1].bottom
        gaps.append(gap)

    if not gaps or max(gaps) <= 0:
        return []

    # Guard: if all gaps are nearly identical, there are no paragraph breaks.
    # Check whether the largest gap is meaningfully bigger than the median.
    sorted_gaps = sorted(gaps)
    median_gap = sorted_gaps[len(sorted_gaps) // 2]
    if median_gap > 0 and max(gaps) / median_gap < 1.3:
        return []

    # Classify gaps
    if method == "otsu":
        result = otsu_threshold(gaps)
        # Additional guard: if the Otsu split puts >80% of gaps in the "above" class,
        # the data doesn't have a meaningful bimodal split.
        if result.above_count > 0.8 * len(gaps):
            return []
        is_para_break = [g >= result.threshold for g in gaps]
    elif method == "relative":
        sorted_gaps = sorted(gaps)
        median_gap = sorted_gaps[len(sorted_gaps) // 2]
        para_threshold = median_gap * relative_threshold
        is_para_break = [g >= para_threshold for g in gaps]
    else:  # jenks
        result = jenks_breaks(gaps, k=2)
        is_para_break = [label > 0 for label in result.labels]

    # Convert gap indices to block indices
    # Gap i is between line i and line i+1.
    # A paragraph break at gap i means the first block of line i+1 starts a new paragraph.
    break_indices: list[int] = []
    for i, is_break in enumerate(is_para_break):
        if is_break:
            # First block index of line i+1
            break_indices.append(lines[i + 1].indices[0])

    return break_indices


def classify_font_sizes(
    blocks: list[TextBlock],
    *,
    k: int = 0,
    min_distinct_sizes: int = 2,
    size_tolerance: float = 0.5,
) -> FontSizeClassification:
    """Classify blocks by font size into named groups.

    Groups blocks into font-size classes (heading, body, footnote, etc.)
    using statistical clustering.

    Args:
        blocks: Input text blocks (must have font_size > 0).
        k: Number of classes. ``0`` = auto-detect:
           - 2 distinct sizes → Otsu (body vs heading)
           - 3+ distinct sizes → Jenks with k=min(distinct, 4)
        min_distinct_sizes: Minimum distinct font sizes to attempt
                           classification. Below this, all blocks → "body".
        size_tolerance: Font sizes within this tolerance are considered identical.

    Returns:
        FontSizeClassification with named classes, thresholds, and method used.
    """
    if not blocks:
        return FontSizeClassification(classes={"body": []}, thresholds=[], method="none")

    # Filter blocks with valid font sizes
    valid = [(i, b.font_size) for i, b in enumerate(blocks) if b.font_size > 0]
    if not valid:
        return FontSizeClassification(
            classes={"body": list(range(len(blocks)))},
            thresholds=[],
            method="none",
        )

    indices, sizes = zip(*valid, strict=False)
    indices = list(indices)
    sizes = list(sizes)

    # Count distinct sizes (using tolerance)
    distinct_sizes = cluster_1d(sizes, size_tolerance)
    n_distinct = distinct_sizes.k

    if n_distinct < min_distinct_sizes:
        return FontSizeClassification(
            classes={"body": list(range(len(blocks)))},
            thresholds=[],
            method="none",
        )

    # Determine k if auto
    if k == 0:
        k = min(n_distinct, 4)

    # Classify
    if k == 2:
        result = otsu_threshold(sizes)
        thresholds = [result.threshold]
        method = "otsu"
        # Below threshold = body (most common), above = heading
        labels = [0 if s < result.threshold else 1 for s in sizes]
    else:
        result = jenks_breaks(sizes, k=k)
        thresholds = result.breaks
        method = "jenks"
        labels = result.labels

    # Name the classes (sorted by ascending font size)
    # Lowest = footnote (if k>=3), middle = body, highest = heading
    class_names: list[str]
    if k == 2:
        class_names = ["body", "heading"]
    elif k == 3:
        class_names = ["footnote", "body", "heading"]
    elif k == 4:
        class_names = ["footnote", "body", "subheading", "heading"]
    else:
        class_names = [f"class_{i}" for i in range(k)]

    classes: dict[str, list[int]] = {name: [] for name in class_names}

    for i, label in enumerate(labels):
        orig_idx = indices[i]
        name = class_names[min(label, len(class_names) - 1)]
        classes[name].append(orig_idx)

    # Add blocks without valid font sizes to "body"
    classified = set(indices)
    for i in range(len(blocks)):
        if i not in classified:
            classes["body"].append(i)

    return FontSizeClassification(classes=classes, thresholds=thresholds, method=method)


def detect_headers_footers(
    pages: list[list[TextBlock]],
    *,
    zone_fraction: float = 0.1,
    min_occurrence: float = 0.5,
) -> tuple[list[int], list[int]]:
    """Find repeated header/footer elements across pages.

    Extracts text from the top/bottom zones of each page and identifies
    content that appears on a sufficient fraction of pages. Returns indices
    into the flattened block list.

    Args:
        pages: List of per-page block lists.
        zone_fraction: Fraction of page height to consider as header/footer zone.
        min_occurrence: Minimum fraction of pages a text must appear on.

    Returns:
        Tuple of (header_indices, footer_indices) into a flattened
        concatenation of all pages.
    """
    if len(pages) < 2:
        return ([], [])

    min_pages = max(2, int(len(pages) * min_occurrence))

    # For each page, find page extent and extract zone blocks
    header_texts: list[dict[str, list[int]]] = []  # per-page: text -> [flat_indices]
    footer_texts: list[dict[str, list[int]]] = []

    flat_offset = 0
    for page_blocks in pages:
        if not page_blocks:
            header_texts.append({})
            footer_texts.append({})
            flat_offset += len(page_blocks)
            continue

        page_top = min(b.top for b in page_blocks)
        page_bottom = max(b.bottom for b in page_blocks)
        page_height = page_bottom - page_top

        if page_height <= 0:
            header_texts.append({})
            footer_texts.append({})
            flat_offset += len(page_blocks)
            continue

        header_zone = page_top + page_height * zone_fraction
        footer_zone = page_bottom - page_height * zone_fraction

        h_texts: dict[str, list[int]] = {}
        f_texts: dict[str, list[int]] = {}

        for i, block in enumerate(page_blocks):
            flat_idx = flat_offset + i
            normalized = block.text.strip().lower()
            if not normalized:
                continue

            if block.y_center < header_zone:
                h_texts.setdefault(normalized, []).append(flat_idx)
            elif block.y_center > footer_zone:
                f_texts.setdefault(normalized, []).append(flat_idx)

        header_texts.append(h_texts)
        footer_texts.append(f_texts)
        flat_offset += len(page_blocks)

    # Count occurrences across pages
    def find_repeated(
        zone_texts: list[dict[str, list[int]]],
    ) -> list[int]:
        text_pages: Counter[str] = Counter()
        for page_dict in zone_texts:
            for text in page_dict:
                text_pages[text] += 1

        # Collect indices of texts appearing on enough pages
        repeated_texts = {t for t, count in text_pages.items() if count >= min_pages}
        result_indices: list[int] = []
        for page_dict in zone_texts:
            for text, indices in page_dict.items():
                if text in repeated_texts:
                    result_indices.extend(indices)
        return sorted(result_indices)

    return (find_repeated(header_texts), find_repeated(footer_texts))


def detect_table_regions(
    blocks: list[TextBlock],
    *,
    min_cols: int = 2,
    min_rows: int = 2,
    alignment_tolerance: float = 3.0,
    y_tolerance: float = 2.0,
) -> list[tuple[float, float, float, float]]:
    """Detect table regions using text alignment patterns (stream strategy).

    Finds regions where text blocks align into a grid pattern — multiple
    blocks share similar X-positions across multiple lines. This is the
    "stream" table detection strategy (as opposed to "lattice" which uses
    drawn lines).

    Args:
        blocks: Input text blocks.
        min_cols: Minimum columns in a table.
        min_rows: Minimum rows in a table.
        alignment_tolerance: X-position tolerance for considering blocks
                            to be in the same column.
        y_tolerance: Y-tolerance for line grouping.

    Returns:
        List of bounding boxes ``(left, top, right, bottom)`` of detected
        table regions.
    """
    if not blocks:
        return []

    lines = group_into_lines(blocks, y_tolerance=y_tolerance)
    if len(lines) < min_rows:
        return []

    # Find column positions: cluster the left-edge X positions of all blocks
    all_lefts: list[float] = []
    for line in lines:
        for block in line.blocks:
            all_lefts.append(block.left)

    col_clusters = cluster_1d(all_lefts, alignment_tolerance)

    # Only keep column positions that appear in multiple lines
    col_positions: list[float] = []
    for centroid, group in zip(col_clusters.centroids, col_clusters.groups, strict=True):
        # Count how many distinct lines contribute to this column
        line_set: set[int] = set()
        flat_idx = 0
        for line_idx, line in enumerate(lines):
            for _ in line.blocks:
                if flat_idx in group:
                    line_set.add(line_idx)
                flat_idx += 1
        if len(line_set) >= min_rows:
            col_positions.append(centroid)

    if len(col_positions) < min_cols:
        return []

    col_positions.sort()

    # Find contiguous runs of lines that have blocks at enough column positions
    table_regions: list[tuple[float, float, float, float]] = []
    run_start: int | None = None

    for line_idx, line in enumerate(lines):
        # Count how many column positions this line has blocks near
        matched_cols = 0
        for col_x in col_positions:
            for block in line.blocks:
                if abs(block.left - col_x) <= alignment_tolerance:
                    matched_cols += 1
                    break

        if matched_cols >= min_cols:
            if run_start is None:
                run_start = line_idx
        else:
            if run_start is not None and line_idx - run_start >= min_rows:
                # Emit table region
                region_lines = lines[run_start:line_idx]
                region_blocks = [b for lg in region_lines for b in lg.blocks]
                table_regions.append(
                    (
                        min(b.left for b in region_blocks),
                        min(b.top for b in region_blocks),
                        max(b.right for b in region_blocks),
                        max(b.bottom for b in region_blocks),
                    )
                )
            run_start = None

    # Flush trailing run
    if run_start is not None and len(lines) - run_start >= min_rows:
        region_lines = lines[run_start:]
        region_blocks = [b for lg in region_lines for b in lg.blocks]
        table_regions.append(
            (
                min(b.left for b in region_blocks),
                min(b.top for b in region_blocks),
                max(b.right for b in region_blocks),
                max(b.bottom for b in region_blocks),
            )
        )

    return table_regions
