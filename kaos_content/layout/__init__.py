"""Layout analysis primitives for positioned text.

Composable, source-agnostic primitives for document structure detection.
Works with any positioned text (PDF, Word, OCR, HTML).

Tier 1 — 1D Clustering (foundation):
    cluster_1d, otsu_threshold, jenks_breaks, find_modes

Tier 2 — Projection Profiles:
    projection_profile, find_valleys, find_widest_valley

Tier 3 — Layout Detection (compositions):
    group_into_lines, detect_columns, detect_paragraph_breaks,
    classify_font_sizes, detect_headers_footers, detect_table_regions

Tier 4 — Document Segmentation:
    xy_cut
"""

from kaos_content.layout.clustering import (
    cluster_1d,
    find_modes,
    jenks_breaks,
    otsu_threshold,
)
from kaos_content.layout.detection import (
    classify_font_sizes,
    detect_columns,
    detect_headers_footers,
    detect_paragraph_breaks,
    detect_table_regions,
    group_into_lines,
)
from kaos_content.layout.profiles import (
    find_valleys,
    find_widest_valley,
    projection_profile,
)
from kaos_content.layout.segmentation import xy_cut
from kaos_content.layout.types import (
    BreaksResult,
    ClusterResult,
    ColumnResult,
    FontSizeClassification,
    LineGroup,
    ModeResult,
    TextBlock,
    ThresholdResult,
    Valley,
)

__all__ = [
    "BreaksResult",
    "ClusterResult",
    "ColumnResult",
    "FontSizeClassification",
    "LineGroup",
    "ModeResult",
    "TextBlock",
    "ThresholdResult",
    "Valley",
    "classify_font_sizes",
    "cluster_1d",
    "detect_columns",
    "detect_headers_footers",
    "detect_paragraph_breaks",
    "detect_table_regions",
    "find_modes",
    "find_valleys",
    "find_widest_valley",
    "group_into_lines",
    "jenks_breaks",
    "otsu_threshold",
    "projection_profile",
    "xy_cut",
]
