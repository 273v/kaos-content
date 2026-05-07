"""Preprocessing profiles: named pipelines for common image preparation tasks.

Profiles match the patterns from kelvin_pdf and kelvin-ocr:
- OCR: grayscale + denoise + contrast (for Tesseract / OCR engines)
- VLM: RGB + denoise + contrast + sharpen (for vision language models)
- Thumbnail: resize + light enhancement (for previews)
"""

from __future__ import annotations

from enum import StrEnum

from kaos_content.images.model import KaosImage
from kaos_content.images.ops import (
    denoise,
    enhance_contrast,
    sharpen,
    to_grayscale,
    to_rgb,
)


class PreprocessingProfile(StrEnum):
    """Named preprocessing pipeline."""

    OCR = "ocr"
    VLM = "vlm"
    THUMBNAIL = "thumbnail"


def for_ocr(
    image: KaosImage,
    *,
    target_dpi: int = 300,
    denoise_method: str = "gaussian",
    denoise_strength: int = 1,
    contrast_factor: float = 1.2,
) -> KaosImage:
    """Prepare image for OCR: grayscale, denoise, contrast, DPI normalization.

    Matches kelvin_pdf's ocr_profile() and kelvin-ocr's preprocessing patterns.
    """
    result = to_grayscale(image)
    if denoise_strength > 0:
        result = denoise(result, method=denoise_method, strength=denoise_strength)
    result = enhance_contrast(result, factor=contrast_factor)
    result = result.with_dpi(target_dpi)
    return result


def for_vlm(
    image: KaosImage,
    *,
    max_size: int | None = None,
    target_dpi: int = 150,
    denoise_method: str = "median",
    denoise_strength: int = 3,
    contrast_factor: float = 1.4,
    sharpen_factor: float = 1.5,
) -> KaosImage:
    """Prepare image for vision language models: RGB, denoise, contrast, sharpen.

    Matches kelvin_pdf's vlm_profile(). VLMs typically want RGB, moderate
    resolution (150 DPI), and enhanced contrast/sharpness for readability.
    """
    result = to_rgb(image)
    if max_size is not None:
        result = result.resize(max_size=max_size)
    result = denoise(result, method=denoise_method, strength=denoise_strength)
    result = enhance_contrast(result, factor=contrast_factor)
    result = sharpen(result, factor=sharpen_factor)
    result = result.with_dpi(target_dpi)
    return result


def for_thumbnail(
    image: KaosImage,
    *,
    max_size: int = 512,
    contrast_factor: float = 1.3,
    sharpen_factor: float = 1.1,
) -> KaosImage:
    """Create a thumbnail with light enhancement.

    Matches kelvin_pdf's thumbnail_profile(). Resizes to max_size (longest
    edge), then applies light contrast and sharpness enhancement.
    """
    result = image.thumbnail(max_size=max_size)
    result = enhance_contrast(result, factor=contrast_factor)
    result = sharpen(result, factor=sharpen_factor)
    return result


def apply_profile(
    image: KaosImage,
    profile: PreprocessingProfile | str,
    **kwargs,
) -> KaosImage:
    """Apply a named preprocessing profile."""
    name = profile.value if isinstance(profile, PreprocessingProfile) else profile
    if name == "ocr":
        return for_ocr(image, **kwargs)
    elif name == "vlm":
        return for_vlm(image, **kwargs)
    elif name == "thumbnail":
        return for_thumbnail(image, **kwargs)
    else:
        msg = f"Unknown profile: {name}. Use 'ocr', 'vlm', or 'thumbnail'."
        raise ValueError(msg)
