"""Image handling for kaos-content: KaosImage wrapper, preprocessing, artifact integration.

Requires the ``images`` extra: ``pip install kaos-content[images]``
"""

from kaos_content.images.model import ColorMode, ImageFormat, KaosImage
from kaos_content.images.ops import (
    crop,
    denoise,
    enhance_contrast,
    rotate,
    sharpen,
    threshold,
    to_grayscale,
    to_rgb,
)
from kaos_content.images.profiles import PreprocessingProfile, for_ocr, for_thumbnail, for_vlm

__all__ = [
    "ColorMode",
    "ImageFormat",
    "KaosImage",
    "PreprocessingProfile",
    "crop",
    "denoise",
    "enhance_contrast",
    "for_ocr",
    "for_thumbnail",
    "for_vlm",
    "rotate",
    "sharpen",
    "threshold",
    "to_grayscale",
    "to_rgb",
]
