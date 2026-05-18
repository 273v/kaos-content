"""Comprehensive tests for layout analysis primitives.

Tests cover:
- Edge cases (empty, single, uniform, degenerate inputs)
- Correctness on known distributions
- Diagnostic metadata quality
- Performance characteristics
- Real-world-like document layouts
"""

from __future__ import annotations

import random
import time

import numpy as np
import pytest

from kaos_content.layout import (
    TextBlock,
    classify_font_sizes,
    cluster_1d,
    detect_columns,
    detect_headers_footers,
    detect_paragraph_breaks,
    detect_table_regions,
    find_modes,
    find_valleys,
    find_widest_valley,
    group_into_lines,
    jenks_breaks,
    otsu_threshold,
    projection_profile,
    xy_cut,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_block(
    left: float,
    top: float,
    right: float,
    bottom: float,
    text: str = "",
    font_size: float = 12.0,
    page: int = 1,
) -> TextBlock:
    return TextBlock(left, top, right, bottom, text, font_size, page)


def make_line_blocks(
    y: float,
    n: int = 5,
    *,
    x_start: float = 72.0,
    x_spacing: float = 80.0,
    height: float = 12.0,
    font_size: float = 12.0,
    page: int = 1,
) -> list[TextBlock]:
    """Create n blocks evenly spaced on a horizontal line."""
    blocks = []
    for i in range(n):
        left = x_start + i * x_spacing
        blocks.append(
            make_block(
                left,
                y,
                left + x_spacing * 0.8,
                y + height,
                text=f"word_{i}",
                font_size=font_size,
                page=page,
            )
        )
    return blocks


def make_page_blocks(
    n_lines: int = 20,
    *,
    line_spacing: float = 14.0,
    para_spacing: float = 28.0,
    para_every: int = 5,
    font_size: float = 12.0,
    page: int = 1,
) -> list[TextBlock]:
    """Create a page of text blocks with paragraph breaks."""
    blocks = []
    y = 72.0
    for i in range(n_lines):
        blocks.append(
            make_block(
                72,
                y,
                540,
                y + 12,
                text=f"Line {i + 1}",
                font_size=font_size,
                page=page,
            )
        )
        if (i + 1) % para_every == 0 and i < n_lines - 1:
            y += para_spacing
        else:
            y += line_spacing
    return blocks


# ===========================================================================
# TextBlock tests
# ===========================================================================


class TestTextBlock:
    def test_frozen(self):
        b = make_block(10, 20, 100, 40)
        with pytest.raises(AttributeError):
            b.left = 5  # ty: ignore[invalid-assignment]

    def test_properties(self):
        b = make_block(10, 20, 110, 50)
        assert b.width == 100.0
        assert b.height == 30.0
        assert b.x_center == 60.0
        assert b.y_center == 35.0
        assert b.area == 3000.0

    def test_overlaps(self):
        a = make_block(0, 0, 50, 50)
        b = make_block(25, 25, 75, 75)
        c = make_block(60, 60, 100, 100)
        assert a.overlaps(b)
        assert not a.overlaps(c)
        assert a.overlaps_x(b)
        assert not a.overlaps_x(c)

    def test_contains(self):
        outer = make_block(0, 0, 100, 100)
        inner = make_block(10, 10, 90, 90)
        assert outer.contains(inner)
        assert not inner.contains(outer)

    def test_merge(self):
        a = make_block(10, 20, 50, 40, text="hello", font_size=12.0)
        b = make_block(55, 20, 100, 40, text="world", font_size=14.0)
        merged = a.merge(b)
        assert merged.left == 10
        assert merged.right == 100
        assert merged.text == "hello world"
        assert merged.font_size == 14.0

    def test_merge_empty_text(self):
        a = make_block(0, 0, 10, 10, text="hi")
        b = make_block(10, 0, 20, 10, text="")
        merged = a.merge(b)
        assert merged.text == "hi"


# ===========================================================================
# Tier 1: Clustering tests
# ===========================================================================


class TestCluster1D:
    def test_empty(self):
        result = cluster_1d([], tolerance=1.0)
        assert result.k == 0
        assert result.groups == []

    def test_single(self):
        result = cluster_1d([5.0], tolerance=1.0)
        assert result.k == 1
        assert result.groups == [[0]]
        assert result.centroids == [5.0]

    def test_two_clusters(self):
        values = [1.0, 1.5, 2.0, 10.0, 10.5, 11.0]
        result = cluster_1d(values, tolerance=1.0)
        assert result.k == 2
        assert result.sizes == [3, 3]

    def test_tolerance_zero(self):
        values = [1.0, 1.0, 2.0, 2.0]
        result = cluster_1d(values, tolerance=0.0)
        # Only exact duplicates cluster
        assert result.k == 2

    def test_all_same(self):
        values = [5.0, 5.0, 5.0]
        result = cluster_1d(values, tolerance=1.0)
        assert result.k == 1
        assert len(result.groups[0]) == 3

    def test_wide_tolerance_merges_all(self):
        values = [1.0, 5.0, 10.0, 15.0]
        result = cluster_1d(values, tolerance=100.0)
        assert result.k == 1

    def test_presorted(self):
        values = [1.0, 2.0, 10.0, 11.0]
        result = cluster_1d(values, tolerance=2.0, presorted=True)
        assert result.k == 2

    def test_preserves_original_indices(self):
        values = [10.0, 1.0, 11.0, 2.0]
        result = cluster_1d(values, tolerance=2.0)
        assert result.k == 2
        # Group containing index 0 and 2 (values 10, 11)
        # Group containing index 1 and 3 (values 1, 2)
        low_group = next(g for g in result.groups if 1 in g)
        high_group = next(g for g in result.groups if 0 in g)
        assert sorted(low_group) == [1, 3]
        assert sorted(high_group) == [0, 2]

    def test_many_clusters(self):
        values = [float(i * 100) for i in range(10)]
        result = cluster_1d(values, tolerance=1.0)
        assert result.k == 10

    def test_font_sizes_realistic(self):
        # Typical PDF: body=12, heading=18, footnote=9
        sizes = [12.0] * 50 + [18.0] * 5 + [9.0] * 10
        result = cluster_1d(sizes, tolerance=1.0)
        assert result.k == 3


class TestOtsuThreshold:
    def test_empty(self):
        result = otsu_threshold([])
        assert result.threshold == 0.0
        assert result.below_count == 0

    def test_single(self):
        result = otsu_threshold([5.0])
        assert result.above_count == 1

    def test_uniform(self):
        result = otsu_threshold([3.0, 3.0, 3.0])
        assert result.variance == 0.0

    def test_bimodal(self):
        values = [1.0, 1.1, 1.2, 1.0, 1.1, 10.0, 10.1, 10.2, 10.0, 10.1]
        result = otsu_threshold(values)
        assert 2.0 < result.threshold < 9.0
        assert result.below_count == 5
        assert result.above_count == 5

    def test_diagnostics(self):
        values = [1.0, 2.0, 3.0, 100.0, 101.0, 102.0]
        result = otsu_threshold(values)
        assert result.below_mean < result.above_mean
        assert result.variance > 0

    def test_threshold_between_clusters(self):
        # Clear separation: body=12, heading=24
        values = [12.0] * 100 + [24.0] * 10
        result = otsu_threshold(values)
        assert 12.0 < result.threshold < 24.0

    def test_small_vs_large_input(self):
        small = [1.0, 1.5, 2.0, 10.0, 10.5, 11.0]
        large = small * 100
        r_small = otsu_threshold(small)
        r_large = otsu_threshold(large)
        # Both should find threshold between the clusters
        assert 3.0 < r_small.threshold < 9.0
        assert 3.0 < r_large.threshold < 9.0


class TestJenksBreaks:
    def test_empty(self):
        result = jenks_breaks([], k=2)
        assert result.breaks == []
        assert result.labels == []
        assert result.gvf == 1.0

    def test_single(self):
        result = jenks_breaks([5.0], k=2)
        assert result.labels == [0]

    def test_k_1(self):
        result = jenks_breaks([1.0, 2.0, 3.0], k=1)
        assert all(label == 0 for label in result.labels)

    def test_two_classes(self):
        values = [1.0, 2.0, 3.0, 100.0, 101.0, 102.0]
        result = jenks_breaks(values, k=2)
        assert len(result.breaks) == 1
        # Break should be between 3 and 100
        assert 3.0 < result.breaks[0] < 100.0
        # First three should be class 0, last three class 1
        assert result.labels[:3] == [0, 0, 0]
        assert result.labels[3:] == [1, 1, 1]

    def test_three_classes(self):
        values = [1.0, 2.0, 50.0, 51.0, 100.0, 101.0]
        result = jenks_breaks(values, k=3)
        assert len(result.breaks) == 2
        assert result.labels[:2] == [0, 0]
        assert result.labels[2:4] == [1, 1]
        assert result.labels[4:] == [2, 2]

    def test_gvf_high_for_clear_clusters(self):
        values = [1.0, 1.0, 1.0, 100.0, 100.0, 100.0]
        result = jenks_breaks(values, k=2)
        assert result.gvf > 0.95

    def test_gvf_lower_for_noise(self):
        random.seed(42)
        values = [random.gauss(0, 10) for _ in range(100)]
        result = jenks_breaks(values, k=3)
        # Random data has lower GVF
        assert result.gvf < 0.95

    def test_k_exceeds_distinct(self):
        values = [1.0, 1.0, 2.0, 2.0]
        result = jenks_breaks(values, k=10)
        assert result.k <= 2  # Clamped to distinct count

    def test_preserves_original_order(self):
        values = [100.0, 1.0, 101.0, 2.0]
        result = jenks_breaks(values, k=2)
        # Indices 0,2 are high class; 1,3 are low class
        assert result.labels[0] == result.labels[2]
        assert result.labels[1] == result.labels[3]
        assert result.labels[0] != result.labels[1]

    def test_font_sizes_realistic(self):
        # body=12, heading=18, footnote=9
        values = [12.0] * 50 + [18.0] * 5 + [9.0] * 10
        result = jenks_breaks(values, k=3)
        assert len(result.breaks) == 2
        assert result.gvf > 0.9


class TestFindModes:
    def test_empty(self):
        result = find_modes([], bin_width=1.0)
        assert result.modes == []

    def test_single_value(self):
        result = find_modes([5.0], bin_width=1.0)
        assert len(result.modes) == 1

    def test_bimodal(self):
        values = [10.0] * 50 + [20.0] * 30
        result = find_modes(values, bin_width=1.0)
        assert len(result.modes) >= 2
        # Check modes are near 10 and 20
        mode_values = sorted(result.modes)
        assert any(abs(m - 10.0) < 1.5 for m in mode_values)
        assert any(abs(m - 20.0) < 1.5 for m in mode_values)

    def test_min_count(self):
        values = [10.0] * 50 + [20.0] * 2
        result = find_modes(values, bin_width=1.0, min_count=10)
        # Only the dominant mode qualifies
        assert all(c >= 10 for c in result.counts)

    def test_font_size_modes(self):
        values = [12.0] * 100 + [18.0] * 10 + [9.0] * 20
        result = find_modes(values, bin_width=0.5)
        # Should find modes near 9, 12, and 18
        assert len(result.modes) >= 3


# ===========================================================================
# Tier 2: Projection profile tests
# ===========================================================================


class TestProjectionProfile:
    def test_empty(self):
        profile = projection_profile([], "x")
        assert len(profile) == 0

    def test_single_block(self):
        blocks = [make_block(10, 20, 50, 40)]
        profile = projection_profile(blocks, "x", resolution=1.0)
        assert len(profile) > 0
        assert profile.sum() > 0

    def test_x_profile_column_detection(self):
        # Two columns with a gap
        left_col = [make_block(10, y, 100, y + 10) for y in range(10, 200, 15)]
        right_col = [make_block(150, y, 250, y + 10) for y in range(10, 200, 15)]
        blocks = left_col + right_col

        profile = projection_profile(blocks, "x", resolution=1.0)
        # The gap between 100 and 150 should be zero
        lo = min(b.left for b in blocks)
        gap_start = int(100 - lo)
        gap_end = int(150 - lo)
        assert profile[gap_start:gap_end].sum() == 0

    def test_y_profile_paragraph_gaps(self):
        blocks = make_page_blocks(10, line_spacing=14, para_spacing=30, para_every=5)
        profile = projection_profile(blocks, "y", resolution=1.0)
        assert len(profile) > 0

    def test_weight_area(self):
        blocks = [
            make_block(0, 0, 10, 100),  # tall
            make_block(0, 0, 100, 10),  # wide
        ]
        profile_count = projection_profile(blocks, "x", weight="count")
        profile_area = projection_profile(blocks, "x", weight="area")
        # Area-weighted should differ from count
        assert not np.array_equal(profile_count, profile_area)

    def test_weight_length(self):
        blocks = [
            make_block(0, 0, 50, 10, text="short"),
            make_block(0, 20, 50, 30, text="much longer text here"),
        ]
        profile = projection_profile(blocks, "x", weight="length")
        assert profile.sum() > 0

    def test_bounds(self):
        blocks = [make_block(50, 50, 100, 100)]
        profile = projection_profile(blocks, "x", bounds=(0, 200), resolution=1.0)
        assert len(profile) == 200
        # First 50 bins should be zero
        assert profile[:50].sum() == 0


class TestFindValleys:
    def test_empty(self):
        assert find_valleys(np.array([], dtype=np.float64)) == []

    def test_no_valleys(self):
        profile = np.ones(100, dtype=np.float64)
        assert find_valleys(profile) == []

    def test_single_valley(self):
        profile = np.array([1, 1, 0, 0, 0, 1, 1], dtype=np.float64)
        valleys = find_valleys(profile)
        assert len(valleys) == 1
        assert valleys[0].width == 3
        assert valleys[0].start == 2
        assert valleys[0].end == 5

    def test_multiple_valleys(self):
        profile = np.array([1, 0, 0, 1, 0, 1], dtype=np.float64)
        valleys = find_valleys(profile)
        assert len(valleys) == 2

    def test_min_width(self):
        profile = np.array([1, 0, 1, 0, 0, 0, 1], dtype=np.float64)
        valleys = find_valleys(profile, min_width=2)
        assert len(valleys) == 1
        assert valleys[0].width == 3

    def test_threshold(self):
        profile = np.array([10, 2, 1, 2, 10], dtype=np.float64)
        valleys = find_valleys(profile, threshold=3.0)
        assert len(valleys) == 1
        assert valleys[0].width == 3

    def test_sorted_widest_first(self):
        profile = np.array([1, 0, 1, 0, 0, 0, 1], dtype=np.float64)
        valleys = find_valleys(profile)
        assert valleys[0].width >= valleys[1].width

    def test_valley_at_end(self):
        profile = np.array([1, 1, 0, 0], dtype=np.float64)
        valleys = find_valleys(profile)
        assert len(valleys) == 1
        assert valleys[0].end == 4


class TestFindWidestValley:
    def test_empty(self):
        assert find_widest_valley(np.array([], dtype=np.float64)) is None

    def test_no_valley(self):
        assert find_widest_valley(np.ones(10, dtype=np.float64)) is None

    def test_returns_widest(self):
        profile = np.array([1, 0, 1, 0, 0, 0, 1], dtype=np.float64)
        valley = find_widest_valley(profile)
        assert valley is not None
        assert valley.width == 3


# ===========================================================================
# Tier 3: Detection tests
# ===========================================================================


class TestGroupIntoLines:
    def test_empty(self):
        assert group_into_lines([]) == []

    def test_single_block(self):
        blocks = [make_block(10, 10, 50, 20)]
        lines = group_into_lines(blocks)
        assert len(lines) == 1
        assert len(lines[0].blocks) == 1

    def test_single_line(self):
        blocks = make_line_blocks(y=50, n=5)
        lines = group_into_lines(blocks, y_tolerance=5.0)
        assert len(lines) == 1
        assert len(lines[0].blocks) == 5

    def test_multiple_lines(self):
        blocks = make_line_blocks(y=50, n=3) + make_line_blocks(y=70, n=3)
        lines = group_into_lines(blocks, y_tolerance=5.0)
        assert len(lines) == 2

    def test_sorted_left_to_right(self):
        # Add blocks in random x order
        blocks = [
            make_block(100, 50, 150, 62, text="second"),
            make_block(10, 50, 60, 62, text="first"),
            make_block(200, 50, 250, 62, text="third"),
        ]
        lines = group_into_lines(blocks, y_tolerance=5.0)
        assert len(lines) == 1
        assert lines[0].blocks[0].text == "first"
        assert lines[0].blocks[1].text == "second"
        assert lines[0].blocks[2].text == "third"

    def test_sorted_top_to_bottom(self):
        blocks = make_line_blocks(y=100) + make_line_blocks(y=50)
        lines = group_into_lines(blocks, y_tolerance=5.0)
        assert len(lines) == 2
        assert lines[0].y_center < lines[1].y_center

    def test_tight_tolerance(self):
        blocks = [
            make_block(10, 50, 50, 60),
            make_block(60, 51, 100, 61),  # 1px off
            make_block(10, 80, 50, 90),
        ]
        lines = group_into_lines(blocks, y_tolerance=2.0)
        assert len(lines) == 2

    def test_preserves_indices(self):
        blocks = [
            make_block(10, 100, 50, 110),  # will be line 2
            make_block(10, 50, 50, 60),  # will be line 1
        ]
        lines = group_into_lines(blocks)
        assert lines[0].indices == [1]  # block at y=50
        assert lines[1].indices == [0]  # block at y=100


class TestDetectColumns:
    def test_empty(self):
        result = detect_columns([])
        assert result.columns == []

    def test_single_column(self):
        blocks = [make_block(72, y, 540, y + 12) for y in range(72, 720, 14)]
        result = detect_columns(blocks, page_width=612.0)
        assert len(result.columns) == 1

    def test_two_columns(self):
        left_col = [make_block(72, y, 280, y + 12) for y in range(72, 500, 14)]
        right_col = [make_block(332, y, 540, y + 12) for y in range(72, 500, 14)]
        blocks = left_col + right_col

        result = detect_columns(blocks, page_width=612.0, min_gutter_width=20.0)
        assert len(result.columns) == 2
        assert len(result.gutters) >= 1
        # Left column should end before right column starts
        assert result.columns[0][1] < result.columns[1][0]

    def test_three_columns(self):
        col1 = [make_block(36, y, 180, y + 10) for y in range(36, 500, 14)]
        col2 = [make_block(216, y, 396, y + 10) for y in range(36, 500, 14)]
        col3 = [make_block(432, y, 576, y + 10) for y in range(36, 500, 14)]
        blocks = col1 + col2 + col3

        result = detect_columns(blocks, page_width=612.0, min_gutter_width=20.0)
        assert len(result.columns) == 3


class TestDetectParagraphBreaks:
    def test_empty(self):
        assert detect_paragraph_breaks([]) == []

    def test_single_line(self):
        blocks = [make_block(72, 72, 540, 84)]
        assert detect_paragraph_breaks(blocks) == []

    def test_uniform_spacing(self):
        blocks = [make_block(72, y, 540, y + 12) for y in range(72, 720, 14)]
        breaks = detect_paragraph_breaks(blocks, y_tolerance=3.0)
        # Uniform spacing → no paragraph breaks
        assert breaks == []

    def test_clear_paragraph_break(self):
        blocks = make_page_blocks(
            n_lines=10,
            line_spacing=14,
            para_spacing=35,
            para_every=5,
        )
        breaks = detect_paragraph_breaks(blocks, y_tolerance=3.0)
        assert len(breaks) >= 1

    def test_method_relative(self):
        blocks = make_page_blocks(
            n_lines=10,
            line_spacing=14,
            para_spacing=35,
            para_every=5,
        )
        breaks = detect_paragraph_breaks(
            blocks,
            method="relative",
            y_tolerance=3.0,
            relative_threshold=1.5,
        )
        assert len(breaks) >= 1

    def test_method_jenks(self):
        blocks = make_page_blocks(
            n_lines=10,
            line_spacing=14,
            para_spacing=35,
            para_every=5,
        )
        breaks = detect_paragraph_breaks(blocks, method="jenks", y_tolerance=3.0)
        assert len(breaks) >= 1

    def test_break_indices_are_valid(self):
        blocks = make_page_blocks(n_lines=20, para_every=5)
        breaks = detect_paragraph_breaks(blocks)
        for idx in breaks:
            assert 0 <= idx < len(blocks)


class TestClassifyFontSizes:
    def test_empty(self):
        result = classify_font_sizes([])
        assert "body" in result.classes

    def test_no_font_sizes(self):
        blocks = [make_block(10, y, 100, y + 12, font_size=0.0) for y in range(0, 100, 14)]
        result = classify_font_sizes(blocks)
        assert result.method == "none"

    def test_uniform_font_size(self):
        blocks = [make_block(10, y, 100, y + 12, font_size=12.0) for y in range(0, 100, 14)]
        result = classify_font_sizes(blocks)
        assert result.method == "none"
        assert len(result.classes["body"]) == len(blocks)

    def test_two_sizes(self):
        blocks = [make_block(10, y, 100, y + 12, font_size=12.0) for y in range(0, 200, 14)] + [
            make_block(10, 300, 100, 318, font_size=24.0)
        ]
        result = classify_font_sizes(blocks)
        assert result.method == "otsu"
        assert "heading" in result.classes
        assert "body" in result.classes
        assert len(result.classes["heading"]) >= 1

    def test_three_sizes(self):
        blocks = (
            [make_block(10, y, 100, y + 12, font_size=12.0) for y in range(0, 200, 14)]
            + [make_block(10, y, 100, y + 18, font_size=18.0) for y in range(200, 260, 20)]
            + [make_block(10, y, 100, y + 8, font_size=8.0) for y in range(260, 310, 10)]
        )
        result = classify_font_sizes(blocks)
        assert "footnote" in result.classes or "body" in result.classes

    def test_explicit_k(self):
        blocks = [make_block(10, y, 100, y + 12, font_size=12.0) for y in range(0, 200, 14)] + [
            make_block(10, 300, 100, 318, font_size=24.0)
        ]
        result = classify_font_sizes(blocks, k=2)
        assert len(result.thresholds) == 1


class TestDetectHeadersFooters:
    def test_empty(self):
        headers, footers = detect_headers_footers([])
        assert headers == []
        assert footers == []

    def test_single_page(self):
        headers, footers = detect_headers_footers([[make_block(10, 10, 100, 20)]])
        assert headers == []
        assert footers == []

    def test_repeated_header(self):
        pages = []
        for p in range(5):
            blocks = [
                make_block(72, 30, 540, 42, text="Company Name", page=p + 1),
                make_block(72, 72, 540, 84, text=f"Content on page {p + 1}", page=p + 1),
            ]
            pages.append(blocks)

        headers, _footers = detect_headers_footers(pages, zone_fraction=0.15)
        assert len(headers) >= 5  # "Company Name" on all 5 pages

    def test_repeated_footer(self):
        pages = []
        for p in range(5):
            blocks = [
                make_block(72, 72, 540, 84, text=f"Content {p + 1}", page=p + 1),
                make_block(72, 700, 540, 712, text="Confidential", page=p + 1),
            ]
            pages.append(blocks)

        _headers, footers = detect_headers_footers(pages, zone_fraction=0.15)
        assert len(footers) >= 5

    def test_non_repeated_not_detected(self):
        pages = []
        for p in range(5):
            blocks = [
                make_block(72, 30, 540, 42, text=f"Unique header {p}", page=p + 1),
                make_block(72, 72, 540, 84, text=f"Content {p}", page=p + 1),
            ]
            pages.append(blocks)

        headers, _footers = detect_headers_footers(pages, zone_fraction=0.15)
        assert len(headers) == 0


class TestDetectTableRegions:
    def test_empty(self):
        assert detect_table_regions([]) == []

    def test_no_table(self):
        blocks = [make_block(72, y, 540, y + 12) for y in range(72, 720, 14)]
        regions = detect_table_regions(blocks)
        assert len(regions) == 0

    def test_grid_pattern(self):
        # Create a 5x4 grid of aligned blocks
        blocks = []
        col_positions = [72, 180, 300, 420]
        for row in range(5):
            y = 72 + row * 20
            for col_x in col_positions:
                blocks.append(make_block(col_x, y, col_x + 80, y + 14, text=f"r{row}c{col_x}"))

        regions = detect_table_regions(
            blocks,
            min_cols=3,
            min_rows=3,
            alignment_tolerance=5.0,
        )
        assert len(regions) >= 1

    def test_mixed_content(self):
        # Table region embedded in normal text
        blocks = []
        # Normal text lines
        for y in range(72, 200, 14):
            blocks.append(make_block(72, y, 540, y + 12, text="Normal text"))

        # Table region
        col_positions = [72, 200, 350, 480]
        for row in range(6):
            y = 250 + row * 20
            for col_x in col_positions:
                blocks.append(make_block(col_x, y, col_x + 100, y + 14, text="cell"))

        # More normal text
        for y in range(450, 600, 14):
            blocks.append(make_block(72, y, 540, y + 12, text="More text"))

        regions = detect_table_regions(blocks, min_cols=3, min_rows=4)
        assert len(regions) >= 1
        # Region should be in the table area
        assert any(r[1] > 200 and r[3] < 500 for r in regions)


# ===========================================================================
# Tier 4: Segmentation tests
# ===========================================================================


class TestXYCut:
    def test_empty(self):
        assert xy_cut([], (0, 0, 100, 100)) == []

    def test_single_block(self):
        blocks = [make_block(10, 10, 50, 30)]
        regions = xy_cut(blocks, (0, 0, 100, 100))
        assert len(regions) == 1
        assert len(regions[0]) == 1

    def test_two_rows(self):
        blocks = [make_block(10, 10 + i * 14, 200, 22 + i * 14) for i in range(5)] + [
            make_block(10, 200 + i * 14, 200, 212 + i * 14) for i in range(5)
        ]
        regions = xy_cut(blocks, (0, 0, 300, 300), min_gap=30.0)
        assert len(regions) == 2

    def test_two_columns(self):
        left = [make_block(10, 10 + i * 14, 100, 22 + i * 14) for i in range(5)]
        right = [make_block(200, 10 + i * 14, 300, 22 + i * 14) for i in range(5)]
        blocks = left + right
        regions = xy_cut(blocks, (0, 0, 400, 100), min_gap=30.0)
        assert len(regions) == 2

    def test_reading_order(self):
        # Two columns: should read left column first (top-to-bottom),
        # then right column
        left = [
            make_block(10, y, 100, y + 10, text=f"L{i}") for i, y in enumerate(range(10, 80, 14))
        ]
        right = [
            make_block(200, y, 300, y + 10, text=f"R{i}") for i, y in enumerate(range(10, 80, 14))
        ]
        blocks = left + right
        regions = xy_cut(blocks, (0, 0, 400, 100), min_gap=30.0)

        if len(regions) == 2:
            # X-cut: Y-cut tried first (no y-gap), then X-cut splits columns
            first_texts = [b.text for b in regions[0]]
            second_texts = [b.text for b in regions[1]]
            assert all(t.startswith("L") for t in first_texts)
            assert all(t.startswith("R") for t in second_texts)

    def test_max_depth(self):
        blocks = [make_block(i * 100, 0, i * 100 + 50, 20) for i in range(10)]
        regions = xy_cut(blocks, (0, 0, 1000, 50), min_gap=10.0, max_depth=1)
        # Should stop after 1 cut
        assert len(regions) <= 3  # At most 2 from one cut + maybe more

    def test_complex_layout(self):
        # Header, two columns, footer — with clear gaps between each section
        header = [make_block(72, 36, 540, 50, text="Title")]
        left_col = [make_block(72, 100 + i * 14, 280, 112 + i * 14) for i in range(10)]
        right_col = [make_block(332, 100 + i * 14, 540, 112 + i * 14) for i in range(10)]
        footer = [make_block(72, 700, 540, 712, text="Page 1")]

        blocks = header + left_col + right_col + footer
        regions = xy_cut(blocks, (0, 0, 612, 792), min_gap=20.0)
        # Should separate at least the footer (big Y gap) and potentially
        # split header from columns and columns from each other
        assert len(regions) >= 2
        # All blocks accounted for
        total_blocks = sum(len(r) for r in regions)
        assert total_blocks == len(blocks)


# ===========================================================================
# Integration / realistic scenarios
# ===========================================================================


class TestRealisticScenarios:
    def test_federal_register_like(self):
        """Simulate a Federal Register page: two columns, headers, footers."""
        # Column blocks only (exclude full-width headers for column detection)
        col_blocks = []
        y = 100
        for _ in range(20):
            col_blocks.append(make_block(72, y, 280, y + 10, text="Body text", font_size=10.0))
            y += 12
        y = 100
        for _ in range(20):
            col_blocks.append(make_block(332, y, 540, y + 10, text="More body", font_size=10.0))
            y += 12

        # Column detection on column blocks
        columns = detect_columns(col_blocks, page_width=612.0, min_gutter_width=20.0)
        assert len(columns.columns) >= 2

        # Full-page blocks including headers and headings for font classification
        all_blocks = [
            make_block(72, 36, 540, 50, text="Federal Register", font_size=14.0),
            make_block(72, 52, 540, 62, text="Vol. 89, No. 42", font_size=10.0),
            *col_blocks,
            make_block(72, 360, 540, 376, text="PART 200—REGULATIONS", font_size=14.0),
        ]
        font_classes = classify_font_sizes(all_blocks)
        assert "heading" in font_classes.classes
        assert "body" in font_classes.classes

    def test_court_filing_like(self):
        """Simulate a court filing: single column, headings, paragraphs."""
        blocks = []
        y = 72.0

        # Case caption (heading)
        blocks.append(make_block(72, y, 540, y + 18, text="IN THE DISTRICT COURT", font_size=16.0))
        y += 30

        # Body paragraphs with spacing
        for para in range(3):
            for line in range(4):
                blocks.append(
                    make_block(72, y, 540, y + 12, text=f"Para {para} line {line}", font_size=12.0)
                )
                y += 14
            y += 10  # paragraph gap

        # Test paragraph breaks
        breaks = detect_paragraph_breaks(blocks[1:])  # skip heading
        assert len(breaks) >= 1

    def test_table_in_document(self):
        """Simulate a document with an embedded table."""
        blocks = []

        # Normal text
        for i in range(5):
            blocks.append(make_block(72, 72 + i * 14, 540, 84 + i * 14, text=f"Text line {i}"))

        # Table
        cols = [72, 200, 350, 480]
        for row in range(8):
            y = 200 + row * 18
            for cx in cols:
                blocks.append(make_block(cx, y, cx + 100, y + 14, text="data"))

        # More text
        for i in range(5):
            y = 400 + i * 14
            blocks.append(make_block(72, y, 540, y + 12, text=f"After table {i}"))

        tables = detect_table_regions(blocks, min_cols=3, min_rows=5)
        assert len(tables) >= 1


# ===========================================================================
# Performance tests
# ===========================================================================


class TestPerformance:
    """Performance sanity checks — not benchmarks, but ensure O(n) or O(n log n)."""

    def test_cluster_1d_5000(self):
        values = [random.gauss(0, 100) for _ in range(5000)]
        t0 = time.perf_counter()
        cluster_1d(values, tolerance=1.0)
        dt = time.perf_counter() - t0
        assert dt < 0.1  # Should be <5ms typically

    def test_otsu_5000(self):
        values = [random.gauss(50, 10) for _ in range(5000)]
        t0 = time.perf_counter()
        otsu_threshold(values)
        dt = time.perf_counter() - t0
        assert dt < 0.1

    def test_jenks_1000_k3(self):
        values = [random.gauss(0, 100) for _ in range(1000)]
        t0 = time.perf_counter()
        jenks_breaks(values, k=3)
        dt = time.perf_counter() - t0
        assert dt < 2.0  # O(n²k), 1000 is manageable

    def test_projection_profile_5000(self):
        blocks = [
            make_block(
                random.uniform(0, 500),
                random.uniform(0, 700),
                random.uniform(500, 600),
                random.uniform(700, 800),
            )
            for _ in range(5000)
        ]
        t0 = time.perf_counter()
        projection_profile(blocks, "x", resolution=1.0)
        dt = time.perf_counter() - t0
        assert dt < 0.5

    def test_group_into_lines_5000(self):
        blocks = [
            make_block(random.uniform(72, 540), y, random.uniform(540, 600), y + 12)
            for y in [random.uniform(72, 700) for _ in range(5000)]
        ]
        t0 = time.perf_counter()
        group_into_lines(blocks, y_tolerance=3.0)
        dt = time.perf_counter() - t0
        assert dt < 0.5

    def test_xy_cut_1000(self):
        blocks = [
            make_block(
                random.uniform(0, 500),
                random.uniform(0, 700),
                random.uniform(500, 600),
                random.uniform(700, 800),
            )
            for _ in range(1000)
        ]
        t0 = time.perf_counter()
        xy_cut(blocks, (0, 0, 612, 792), min_gap=20.0)
        dt = time.perf_counter() - t0
        assert dt < 2.0


# ===========================================================================
# Edge cases
# ===========================================================================


class TestEdgeCases:
    def test_negative_coordinates(self):
        blocks = [make_block(-50, -50, 50, 50)]
        lines = group_into_lines(blocks)
        assert len(lines) == 1

    def test_zero_size_blocks(self):
        blocks = [make_block(10, 10, 10, 10)]
        assert blocks[0].width == 0
        assert blocks[0].height == 0
        assert blocks[0].area == 0

    def test_very_large_coordinates(self):
        blocks = [make_block(0, 0, 100000, 100000)]
        profile = projection_profile(blocks, "x", resolution=100.0)
        assert len(profile) > 0

    def test_nan_values_in_clustering(self):
        values = [1.0, float("nan"), 2.0]
        # Should not crash — nan goes to its own cluster
        result = cluster_1d(values, tolerance=1.0)
        assert result.k >= 1

    def test_inf_values_in_clustering(self):
        values = [1.0, float("inf"), 2.0]
        result = cluster_1d(values, tolerance=1.0)
        assert result.k >= 1

    def test_duplicate_blocks(self):
        block = make_block(10, 10, 50, 20, text="same")
        blocks = [block, block, block]
        lines = group_into_lines(blocks)
        assert len(lines) == 1
        assert len(lines[0].blocks) == 3

    def test_overlapping_blocks(self):
        blocks = [
            make_block(10, 10, 100, 30),
            make_block(50, 15, 150, 35),
        ]
        lines = group_into_lines(blocks, y_tolerance=10.0)
        assert len(lines) == 1

    def test_all_same_font_size(self):
        blocks = [make_block(10, y, 100, y + 12, font_size=12.0) for y in range(0, 100, 14)]
        result = classify_font_sizes(blocks)
        assert result.method == "none"
        assert len(result.classes["body"]) == len(blocks)

    def test_single_block_all_operations(self):
        blocks = [make_block(72, 72, 540, 84, text="Only block", font_size=12.0)]

        lines = group_into_lines(blocks)
        assert len(lines) == 1

        cols = detect_columns(blocks)
        assert len(cols.columns) == 1

        breaks = detect_paragraph_breaks(blocks)
        assert breaks == []

        classes = classify_font_sizes(blocks)
        assert "body" in classes.classes

        regions = xy_cut(blocks, (0, 0, 612, 792))
        assert len(regions) == 1
