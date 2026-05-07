"""Image-decompression-bomb regression tests (audit M2).

Pins the resource-budget contract introduced in 0.1.0a1:

- ``KaosImage.from_bytes`` and ``KaosImage.from_path`` raise
  ``ImageDecompressionBombError`` when the decoded image exceeds
  ``MAX_IMAGE_PIXELS`` (100 megapixels by default).
- PIL's process-wide ``Image.MAX_IMAGE_PIXELS`` is set to our cap on
  module import so any code path that opens an image inherits the
  protection.
- ``load_image`` defaults to ``max_bytes=50_000_000`` (50 MB) and
  rejects oversize artifacts before they're decoded.

Audit findings addressed: M2 (PIL unbounded load, max_bytes=None).
"""

from __future__ import annotations

import io

import pytest
from PIL import Image as PILImage

from kaos_content.errors import ImageDecompressionBombError
from kaos_content.images import model as image_model
from kaos_content.images.artifacts import DEFAULT_LOAD_IMAGE_MAX_BYTES
from kaos_content.images.model import MAX_IMAGE_PIXELS, KaosImage

# ────────────────────────────────────────────────────────────────────
# MAX_IMAGE_PIXELS — the package cap is applied to PIL globally
# ────────────────────────────────────────────────────────────────────


def test_max_image_pixels_default_is_100m() -> None:
    """The package default cap is 100 megapixels — covers virtually all
    real-world raster images while blocking decompression bombs."""
    assert MAX_IMAGE_PIXELS == 100_000_000


def test_max_image_pixels_applied_to_pil_globally() -> None:
    """Importing kaos_content.images.model sets PIL's process-wide cap so
    any code path that opens an image inherits the protection — not just
    our explicit constructors."""
    assert PILImage.MAX_IMAGE_PIXELS == MAX_IMAGE_PIXELS


# ────────────────────────────────────────────────────────────────────
# from_bytes — pixel budget enforced
# ────────────────────────────────────────────────────────────────────


def _png_bytes(width: int, height: int) -> bytes:
    """Encode a small PNG of given dimensions."""
    img = PILImage.new("RGB", (width, height), (255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_from_bytes_under_cap_loads() -> None:
    """A normal image well under the cap loads cleanly."""
    data = _png_bytes(64, 64)
    img = KaosImage.from_bytes(data)
    assert img.width == 64 and img.height == 64


def test_from_bytes_at_cap_loads() -> None:
    """An image right at the cap is allowed (not strictly over)."""
    monkey_cap = 4096  # 64x64 image is 4096 pixels
    original = image_model.MAX_IMAGE_PIXELS
    image_model.MAX_IMAGE_PIXELS = monkey_cap
    try:
        data = _png_bytes(64, 64)
        img = KaosImage.from_bytes(data)
        assert img.width == 64
    finally:
        image_model.MAX_IMAGE_PIXELS = original


def test_from_bytes_over_cap_rejected() -> None:
    """An image whose pixel count exceeds MAX_IMAGE_PIXELS is rejected
    with a typed ImageDecompressionBombError that names the cap."""
    original = image_model.MAX_IMAGE_PIXELS
    image_model.MAX_IMAGE_PIXELS = 100  # absurdly low — easy to overflow
    try:
        data = _png_bytes(64, 64)  # 4096 pixels >> 100
        with pytest.raises(ImageDecompressionBombError, match="MAX_IMAGE_PIXELS=100"):
            KaosImage.from_bytes(data)
    finally:
        image_model.MAX_IMAGE_PIXELS = original


def test_from_bytes_disable_cap_with_none() -> None:
    """Setting MAX_IMAGE_PIXELS=None disables the kaos-level check
    (PIL's own warning still applies, but we don't gate on it)."""
    original = image_model.MAX_IMAGE_PIXELS
    image_model.MAX_IMAGE_PIXELS = None
    try:
        data = _png_bytes(64, 64)
        img = KaosImage.from_bytes(data)
        assert img.width == 64
    finally:
        image_model.MAX_IMAGE_PIXELS = original


# ────────────────────────────────────────────────────────────────────
# from_path — same protection
# ────────────────────────────────────────────────────────────────────


def test_from_path_over_cap_rejected(tmp_path) -> None:
    """The same pixel cap applies when loading from a file."""
    p = tmp_path / "small.png"
    p.write_bytes(_png_bytes(64, 64))

    original = image_model.MAX_IMAGE_PIXELS
    image_model.MAX_IMAGE_PIXELS = 100
    try:
        with pytest.raises(ImageDecompressionBombError):
            KaosImage.from_path(p)
    finally:
        image_model.MAX_IMAGE_PIXELS = original


def test_from_path_under_cap_loads(tmp_path) -> None:
    p = tmp_path / "small.png"
    p.write_bytes(_png_bytes(32, 32))
    img = KaosImage.from_path(p)
    assert img.size == (32, 32)


# ────────────────────────────────────────────────────────────────────
# load_image — byte budget on the artifact body
# ────────────────────────────────────────────────────────────────────


def test_default_load_image_max_bytes_is_50mb() -> None:
    """The default cap is 50 MB — well above any legitimate raster
    image and far below the multi-GB scale at which a load would OOM."""
    assert DEFAULT_LOAD_IMAGE_MAX_BYTES == 50_000_000
