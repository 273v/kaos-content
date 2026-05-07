"""Universal input types for layout analysis primitives.

TextBlock is the sole input type — a positioned text fragment with optional
metadata. It is deliberately minimal and source-agnostic: works with PDF,
Word, OCR, HTML, or any system that produces positioned text.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TextBlock:
    """A positioned text rectangle with optional metadata.

    Coordinate system: origin at top-left, Y increases downward.
    Units are arbitrary (points, pixels, mm) — primitives work with
    any consistent unit system.
    """

    left: float
    top: float
    right: float
    bottom: float
    text: str = ""
    font_size: float = 0.0
    page: int = 0

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def x_center(self) -> float:
        return (self.left + self.right) / 2.0

    @property
    def y_center(self) -> float:
        return (self.top + self.bottom) / 2.0

    @property
    def area(self) -> float:
        return self.width * self.height

    def overlaps_x(self, other: TextBlock) -> bool:
        """True if horizontal extents overlap."""
        return self.left < other.right and other.left < self.right

    def overlaps_y(self, other: TextBlock) -> bool:
        """True if vertical extents overlap."""
        return self.top < other.bottom and other.top < self.bottom

    def overlaps(self, other: TextBlock) -> bool:
        """True if rectangles overlap in both axes."""
        return self.overlaps_x(other) and self.overlaps_y(other)

    def contains(self, other: TextBlock) -> bool:
        """True if this block fully contains another."""
        return (
            self.left <= other.left
            and self.top <= other.top
            and self.right >= other.right
            and self.bottom >= other.bottom
        )

    def merge(self, other: TextBlock) -> TextBlock:
        """Merge two blocks into their bounding union."""
        return TextBlock(
            left=min(self.left, other.left),
            top=min(self.top, other.top),
            right=max(self.right, other.right),
            bottom=max(self.bottom, other.bottom),
            text=self.text + " " + other.text
            if self.text and other.text
            else self.text or other.text,
            font_size=max(self.font_size, other.font_size),
            page=self.page,
        )


@dataclass(frozen=True, slots=True)
class ClusterResult:
    """Result of a 1D clustering operation with diagnostics.

    groups: list of index-lists (each inner list contains indices into the
            original values array that belong to that cluster)
    centroids: representative value for each cluster (mean of members)
    sizes: number of members per cluster
    """

    groups: list[list[int]]
    centroids: list[float]
    sizes: list[int]

    @property
    def k(self) -> int:
        """Number of clusters."""
        return len(self.groups)


@dataclass(frozen=True, slots=True)
class ThresholdResult:
    """Result of a binary threshold computation with diagnostics."""

    threshold: float
    variance: float  # between-class variance at the threshold
    below_count: int
    above_count: int
    below_mean: float
    above_mean: float


@dataclass(frozen=True, slots=True)
class BreaksResult:
    """Result of a multi-class breaks computation (Jenks/Fisher)."""

    breaks: list[float]  # k-1 break points
    labels: list[int]  # class label (0..k-1) for each input value
    gvf: float  # Goodness of Variance Fit [0, 1]

    @property
    def k(self) -> int:
        """Number of classes."""
        return len(self.breaks) + 1


@dataclass(frozen=True, slots=True)
class ModeResult:
    """Result of mode detection with diagnostics."""

    modes: list[float]  # modal values (peak centers)
    counts: list[int]  # count per mode bin
    bin_width: float


@dataclass(frozen=True, slots=True)
class Valley:
    """A gap (zero or below-threshold region) in a projection profile."""

    start: int
    end: int
    width: int
    center: int


@dataclass(frozen=True, slots=True)
class ColumnResult:
    """Detected column boundaries."""

    columns: list[tuple[float, float]]  # (left, right) ranges
    gutters: list[Valley]  # the valleys that separate columns


@dataclass(frozen=True, slots=True)
class FontSizeClassification:
    """Result of font-size classification with diagnostics."""

    classes: dict[str, list[int]]  # {"heading": [indices], "body": [indices], ...}
    thresholds: list[float]  # break points used
    method: str  # "otsu" or "jenks"


@dataclass(frozen=True, slots=True)
class LineGroup:
    """A group of blocks on the same visual line."""

    blocks: list[TextBlock]
    indices: list[int]  # indices into original block list
    y_center: float
    top: float
    bottom: float
