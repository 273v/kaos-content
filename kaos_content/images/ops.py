"""Image preprocessing operations for KaosImage.

All operations return new KaosImage instances (immutable-style). Uses PIL
exclusively — no scikit-image or heavy dependencies. Advanced preprocessing
(Hough deskew, morphological segmentation) belongs in consumer packages.
"""

from __future__ import annotations

from PIL import Image as PILImage
from PIL import ImageEnhance, ImageFilter

from kaos_content.images.model import KaosImage


def to_grayscale(image: KaosImage) -> KaosImage:
    """Convert to grayscale (L mode)."""
    return image.to_grayscale()


def to_rgb(image: KaosImage) -> KaosImage:
    """Convert to RGB, compositing alpha onto white background."""
    return image.to_rgb()


def sharpen(image: KaosImage, factor: float = 1.5) -> KaosImage:
    """Sharpen the image. factor=1.0 is original, >1.0 sharpens, <1.0 blurs."""
    enhanced = ImageEnhance.Sharpness(image.pil).enhance(factor)
    return image._derive(enhanced)


def enhance_contrast(image: KaosImage, factor: float = 1.3) -> KaosImage:
    """Enhance contrast. factor=1.0 is original, >1.0 increases contrast."""
    enhanced = ImageEnhance.Contrast(image.pil).enhance(factor)
    return image._derive(enhanced)


def denoise(
    image: KaosImage,
    method: str = "median",
    strength: int = 3,
) -> KaosImage:
    """Denoise the image.

    Methods:
        median: Median filter (good for salt-and-pepper noise). strength = kernel size.
        gaussian: Gaussian blur. strength = radius.
        smooth: PIL SMOOTH filter (single pass).
    """
    if method == "median":
        # Kernel size must be odd
        size = strength if strength % 2 == 1 else strength + 1
        filtered = image.pil.filter(ImageFilter.MedianFilter(size=size))
    elif method == "gaussian":
        filtered = image.pil.filter(ImageFilter.GaussianBlur(radius=strength))
    elif method == "smooth":
        filtered = image.pil.filter(ImageFilter.SMOOTH)
    else:
        msg = f"Unknown denoise method: {method}. Use 'median', 'gaussian', or 'smooth'."
        raise ValueError(msg)
    return image._derive(filtered)


def threshold(
    image: KaosImage,
    value: int = 128,
    method: str = "simple",
) -> KaosImage:
    """Apply thresholding (binarization).

    Methods:
        simple: Fixed threshold value.
        otsu: Automatic threshold based on histogram analysis.
    """
    gray = image.to_grayscale()

    if method == "otsu":
        value = _otsu_threshold(gray.pil)

    binarized = gray.pil.point(lambda p: 255 if p > value else 0, mode="L")
    return gray._derive(binarized)


def rotate(image: KaosImage, degrees: int, *, expand: bool = True) -> KaosImage:
    """Rotate the image counter-clockwise by the given degrees."""
    return image.rotate(degrees, expand=expand)


def crop(image: KaosImage, left: int, top: int, right: int, bottom: int) -> KaosImage:
    """Crop to the given pixel coordinates."""
    return image.crop(left, top, right, bottom)


def remove_borders(image: KaosImage, margin_percent: float = 5.0) -> KaosImage:
    """Remove borders by cropping a percentage from each edge."""
    w, h = image.size
    margin_x = int(w * margin_percent / 100)
    margin_y = int(h * margin_percent / 100)
    return image.crop(margin_x, margin_y, w - margin_x, h - margin_y)


def auto_contrast(image: KaosImage) -> KaosImage:
    """Normalize contrast by stretching histogram to full range."""
    from PIL import ImageOps

    normalized = ImageOps.autocontrast(image.pil)
    return image._derive(normalized)


def enhance_brightness(image: KaosImage, factor: float = 1.2) -> KaosImage:
    """Adjust brightness. factor=1.0 is original, >1.0 brightens."""
    enhanced = ImageEnhance.Brightness(image.pil).enhance(factor)
    return image._derive(enhanced)


def invert(image: KaosImage) -> KaosImage:
    """Invert image colors."""
    from PIL import ImageOps

    inverted = ImageOps.invert(image.pil.convert("RGB"))
    return image._derive(inverted)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _otsu_threshold(gray_image: PILImage.Image) -> int:
    """Compute Otsu's threshold from a grayscale PIL Image histogram."""
    histogram = gray_image.histogram()
    total = sum(histogram)
    if total == 0:
        return 128

    sum_total = sum(i * h for i, h in enumerate(histogram))
    sum_bg = 0.0
    weight_bg = 0
    max_variance = 0.0
    best_threshold = 128

    for t in range(256):
        weight_bg += histogram[t]
        if weight_bg == 0:
            continue
        weight_fg = total - weight_bg
        if weight_fg == 0:
            break

        sum_bg += t * histogram[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_total - sum_bg) / weight_fg

        variance = weight_bg * weight_fg * (mean_bg - mean_fg) ** 2
        if variance > max_variance:
            max_variance = variance
            best_threshold = t

    return best_threshold
