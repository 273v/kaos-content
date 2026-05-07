"""KaosImage: standard image wrapper with metadata, format conversion, and artifact support."""

from __future__ import annotations

import io
import warnings
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from PIL import Image as PILImage

from kaos_content.errors import ImageDecompressionBombError

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from kaos_content.model.attr import Provenance


# Maximum allowed pixel count for any image loaded through ``KaosImage``.
# Defends against decompression-bomb attacks where a small compressed
# payload expands to gigabytes of pixels in memory. PIL's own default
# (~89M pixels) only emits a warning; we promote it to a hard error
# and tighten the cap to 100 megapixels.
#
# Override globally with ``kaos_content.images.model.MAX_IMAGE_PIXELS = N``
# before the first image load; or ``None`` to disable (NOT recommended).
MAX_IMAGE_PIXELS: int | None = 100_000_000

# Apply our cap as PIL's process-wide default so any code path that
# opens an image (even outside our explicit constructors) inherits the
# protection. Callers can still override ``PILImage.MAX_IMAGE_PIXELS``.
PILImage.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS


def _check_pixel_budget(pil: PILImage.Image) -> None:
    """Raise if the image exceeds the configured pixel budget."""
    cap = MAX_IMAGE_PIXELS
    if cap is None:
        return
    pixels = int(pil.width) * int(pil.height)
    if pixels > cap:
        msg = (
            f"Image has {pixels} pixels ({pil.width}x{pil.height}), "
            f"exceeds MAX_IMAGE_PIXELS={cap}. This is a defence against "
            f"decompression-bomb attacks. To allow, raise "
            f"kaos_content.images.model.MAX_IMAGE_PIXELS."
        )
        raise ImageDecompressionBombError(msg)


def _open_with_bomb_check(opener: Any) -> PILImage.Image:
    """Open a PIL image with decompression-bomb warnings promoted to errors.

    ``opener`` is a zero-argument callable that returns a PIL Image
    (typically ``lambda: PILImage.open(source)``). PIL emits a
    ``DecompressionBombWarning`` when an image's pixel count exceeds
    its internal cap; we treat that as fatal.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error", PILImage.DecompressionBombWarning)
        try:
            pil = opener()
        except PILImage.DecompressionBombWarning as exc:
            msg = (
                f"PIL flagged this image as a decompression bomb: {exc}. "
                f"Refusing to load. To allow, raise PIL's "
                f"Image.MAX_IMAGE_PIXELS."
            )
            raise ImageDecompressionBombError(msg) from exc
    _check_pixel_budget(pil)
    return pil


class ImageFormat(StrEnum):
    """Supported image output formats."""

    PNG = "png"
    JPEG = "jpeg"
    TIFF = "tiff"
    BMP = "bmp"
    WEBP = "webp"


class ColorMode(StrEnum):
    """Image color modes."""

    RGB = "RGB"
    RGBA = "RGBA"
    GRAYSCALE = "L"
    BINARY = "1"


# MIME type mapping
_FORMAT_TO_MIME: dict[str, str] = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "tiff": "image/tiff",
    "bmp": "image/bmp",
    "webp": "image/webp",
}

_MIME_TO_FORMAT: dict[str, str] = {v: k for k, v in _FORMAT_TO_MIME.items()}

# File signature detection
_SIGNATURES: list[tuple[bytes, str]] = [
    (b"\x89PNG", "png"),
    (b"\xff\xd8\xff", "jpeg"),
    (b"II\x2a\x00", "tiff"),
    (b"MM\x00\x2a", "tiff"),
    (b"BM", "bmp"),
    (b"RIFF", "webp"),
]


def _detect_format(data: bytes) -> str | None:
    """Detect image format from file signature."""
    for sig, fmt in _SIGNATURES:
        if data[: len(sig)] == sig:
            return fmt
    return None


class KaosImage:
    """Standard image wrapper with metadata, format conversion, and preprocessing.

    Wraps a PIL Image with provenance, DPI, and format tracking. All mutation
    methods return new KaosImage instances (immutable-style API matching
    kaos-content's frozen model convention).

    Usage::

        img = KaosImage.from_path("page.png")
        gray = img.to_grayscale()
        thumb = img.thumbnail(max_size=512)
        data = img.to_bytes(format="jpeg", quality=85)

        # Store as artifact
        from kaos_content.images.artifacts import store_image
        manifest = await store_image(img, runtime, context, name="page-1")
    """

    __slots__ = ("_dpi", "_metadata", "_pil", "_provenance", "_source_format")

    def __init__(
        self,
        pil_image: PILImage.Image,
        *,
        dpi: tuple[int, int] | None = None,
        source_format: str | None = None,
        provenance: Provenance | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._pil = pil_image
        self._dpi = dpi or _extract_dpi(pil_image)
        self._source_format = source_format
        self._provenance = provenance
        self._metadata = metadata or {}

    # ── Properties ──

    @property
    def pil(self) -> PILImage.Image:
        """Underlying PIL Image."""
        return self._pil

    @property
    def width(self) -> int:
        return self._pil.width

    @property
    def height(self) -> int:
        return self._pil.height

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) in pixels."""
        return self._pil.size

    @property
    def mode(self) -> str:
        """PIL color mode (RGB, L, RGBA, 1, etc.)."""
        return self._pil.mode

    @property
    def dpi(self) -> tuple[int, int]:
        """Image DPI (horizontal, vertical)."""
        return self._dpi

    @property
    def source_format(self) -> str | None:
        """Original format when loaded (png, jpeg, etc.)."""
        return self._source_format

    @property
    def provenance(self) -> Provenance | None:
        """Source provenance (page, bbox, extractor, etc.)."""
        return self._provenance

    @property
    def metadata(self) -> dict[str, Any]:
        """Arbitrary metadata dict."""
        return self._metadata

    @property
    def mime_type(self) -> str:
        """MIME type based on source format or default to PNG."""
        fmt = self._source_format or "png"
        return _FORMAT_TO_MIME.get(fmt, "image/png")

    @property
    def is_grayscale(self) -> bool:
        return self._pil.mode in ("L", "LA")

    @property
    def is_rgb(self) -> bool:
        return self._pil.mode in ("RGB", "RGBA")

    # ── Constructors ──

    @classmethod
    def from_pil(
        cls,
        image: PILImage.Image,
        *,
        dpi: tuple[int, int] | None = None,
        source_format: str | None = None,
        provenance: Provenance | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KaosImage:
        """Wrap an existing PIL Image."""
        return cls(
            image,
            dpi=dpi,
            source_format=source_format,
            provenance=provenance,
            metadata=metadata,
        )

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        dpi: tuple[int, int] | None = None,
        provenance: Provenance | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KaosImage:
        """Load image from bytes (auto-detects format).

        Raises ``ImageDecompressionBombError`` if the image exceeds
        ``MAX_IMAGE_PIXELS`` (defence against decompression bombs).
        """
        fmt = _detect_format(data)
        pil = _open_with_bomb_check(lambda: PILImage.open(io.BytesIO(data)))
        pil.load()  # Force load so buffer can be freed
        detected_dpi = dpi or _extract_dpi(pil)
        return cls(
            pil,
            dpi=detected_dpi,
            source_format=fmt or (pil.format or "").lower() or None,
            provenance=provenance,
            metadata=metadata,
        )

    @classmethod
    def from_path(
        cls,
        path: str | Path,
        *,
        dpi: tuple[int, int] | None = None,
        provenance: Provenance | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KaosImage:
        """Load image from file path.

        Raises ``ImageDecompressionBombError`` if the image exceeds
        ``MAX_IMAGE_PIXELS`` (defence against decompression bombs).
        """
        p = Path(path)
        pil = _open_with_bomb_check(lambda: PILImage.open(p))
        pil.load()
        fmt = (pil.format or "").lower() or p.suffix.lstrip(".").lower()
        detected_dpi = dpi or _extract_dpi(pil)
        return cls(
            pil,
            dpi=detected_dpi,
            source_format=fmt or None,
            provenance=provenance,
            metadata=metadata,
        )

    @classmethod
    def from_numpy(
        cls,
        array: npt.NDArray[np.uint8],
        *,
        mode: str | None = None,
        dpi: tuple[int, int] | None = None,
        provenance: Provenance | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> KaosImage:
        """Create from a numpy array (uint8).

        For grayscale: shape (H, W). For RGB: shape (H, W, 3).
        """
        if mode is None:
            mode = "L" if array.ndim == 2 else "RGB"
        pil = PILImage.fromarray(array, mode=mode)
        return cls(pil, dpi=dpi or (300, 300), provenance=provenance, metadata=metadata)

    # ── Output ──

    def to_bytes(
        self,
        *,
        format: str = "png",
        quality: int = 95,
        compression: int = 6,
        dpi: tuple[int, int] | None = None,
    ) -> bytes:
        """Serialize to bytes in the specified format.

        Args:
            format: Output format (png, jpeg, tiff, bmp, webp).
            quality: JPEG/WebP quality (1-100).
            compression: PNG compression level (0-9).
            dpi: Override DPI metadata in output.
        """
        buf = io.BytesIO()
        img = self._pil
        out_dpi = dpi or self._dpi
        fmt = format.upper()

        # JPEG does not support RGBA
        if fmt == "JPEG" and img.mode == "RGBA":
            background = PILImage.new("RGB", img.size, (255, 255, 255))
            background.paste(img, mask=img.split()[3])
            img = background
        elif fmt == "JPEG" and img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        save_kwargs: dict[str, Any] = {"dpi": out_dpi}
        if fmt == "JPEG":
            save_kwargs["quality"] = quality
            save_kwargs["optimize"] = True
        elif fmt == "PNG":
            save_kwargs["compress_level"] = compression
        elif fmt == "WEBP":
            save_kwargs["quality"] = quality

        img.save(buf, format=fmt, **save_kwargs)
        return buf.getvalue()

    def to_pil(self) -> PILImage.Image:
        """Return a copy of the underlying PIL Image."""
        return self._pil.copy()

    def to_numpy(self) -> npt.NDArray[np.uint8]:
        """Convert to numpy array (uint8)."""
        import numpy as np

        return np.array(self._pil, dtype=np.uint8)

    def to_base64(self, *, format: str = "png", **kwargs: Any) -> str:
        """Encode as base64 string (for wire format / data URIs)."""
        import base64

        data = self.to_bytes(format=format, **kwargs)
        return base64.b64encode(data).decode("ascii")

    def to_data_uri(self, *, format: str = "png", **kwargs: Any) -> str:
        """Encode as a data: URI (for HTML embedding)."""
        mime = _FORMAT_TO_MIME.get(format, f"image/{format}")
        b64 = self.to_base64(format=format, **kwargs)
        return f"data:{mime};base64,{b64}"

    # ── Transformations (return new KaosImage) ──

    def _derive(self, pil: PILImage.Image, **overrides: Any) -> KaosImage:
        """Create a derived image preserving metadata."""
        return KaosImage(
            pil,
            dpi=overrides.get("dpi", self._dpi),
            source_format=overrides.get("source_format", self._source_format),
            provenance=overrides.get("provenance", self._provenance),
            metadata=overrides.get("metadata", self._metadata),
        )

    def resize(
        self,
        width: int | None = None,
        height: int | None = None,
        *,
        max_size: int | None = None,
        maintain_aspect: bool = True,
        resample: PILImage.Resampling = PILImage.Resampling.LANCZOS,
    ) -> KaosImage:
        """Resize the image. Specify width/height or max_size (longest edge)."""
        if max_size is not None:
            # Scale longest edge to max_size
            w, h = self._pil.size
            scale = max_size / max(w, h)
            if scale >= 1.0:
                return self  # Already small enough
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
        elif width is not None and height is not None and not maintain_aspect:
            new_w, new_h = width, height
        elif width is not None and height is not None:
            # Sec-6 (security finding #7): ``PIL.Image.thumbnail()`` is
            # the only PIL op in this method that mutates IN PLACE.
            # Pre-fix the call ``self._pil.thumbnail(...)`` rewrote the
            # source KaosImage's underlying PIL image, then returned a
            # copy of the now-mutated original — violating the
            # immutable-style ``resize() returns a new KaosImage``
            # contract. Fix: copy FIRST, mutate the copy.
            #
            # All other branches use ``self._pil.resize(...)`` which is
            # already non-mutating — they don't need the copy dance.
            scaled = self._pil.copy()
            scaled.thumbnail((width, height), resample)
            return self._derive(scaled)
        elif width is not None:
            scale = width / self._pil.width
            new_w = width
            new_h = max(1, int(self._pil.height * scale))
        elif height is not None:
            scale = height / self._pil.height
            new_w = max(1, int(self._pil.width * scale))
            new_h = height
        else:
            return self

        return self._derive(self._pil.resize((new_w, new_h), resample))

    def thumbnail(self, max_size: int = 512) -> KaosImage:
        """Create a thumbnail (aspect-ratio-preserving resize)."""
        img = self._pil.copy()
        img.thumbnail((max_size, max_size), PILImage.Resampling.LANCZOS)
        return self._derive(img)

    def to_grayscale(self) -> KaosImage:
        """Convert to grayscale (L mode)."""
        if self._pil.mode == "L":
            return self
        return self._derive(self._pil.convert("L"))

    def to_rgb(self) -> KaosImage:
        """Convert to RGB, compositing alpha onto white background if needed."""
        if self._pil.mode == "RGB":
            return self
        if self._pil.mode == "RGBA":
            background = PILImage.new("RGB", self._pil.size, (255, 255, 255))
            background.paste(self._pil, mask=self._pil.split()[3])
            return self._derive(background)
        return self._derive(self._pil.convert("RGB"))

    def convert_mode(self, mode: str) -> KaosImage:
        """Convert to arbitrary PIL mode."""
        if self._pil.mode == mode:
            return self
        return self._derive(self._pil.convert(mode))

    def rotate(self, degrees: int, *, expand: bool = True, fill: int = 255) -> KaosImage:
        """Rotate by degrees (counter-clockwise). Expand canvas to fit."""
        return self._derive(
            self._pil.rotate(
                degrees, expand=expand, fillcolor=fill, resample=PILImage.Resampling.BICUBIC
            )
        )

    def crop(self, left: int, top: int, right: int, bottom: int) -> KaosImage:
        """Crop to the given pixel coordinates."""
        return self._derive(self._pil.crop((left, top, right, bottom)))

    def with_dpi(self, dpi: int | tuple[int, int]) -> KaosImage:
        """Return a copy with updated DPI metadata."""
        d = (dpi, dpi) if isinstance(dpi, int) else dpi
        return self._derive(self._pil, dpi=d)

    def with_provenance(self, provenance: Provenance) -> KaosImage:
        """Return a copy with updated provenance."""
        return self._derive(self._pil, provenance=provenance)

    def with_metadata(self, **kv: Any) -> KaosImage:
        """Return a copy with merged metadata."""
        merged = {**self._metadata, **kv}
        return self._derive(self._pil, metadata=merged)

    # ── Repr ──

    def __repr__(self) -> str:
        fmt = self._source_format or "unknown"
        return (
            f"KaosImage({self.width}x{self.height}, mode={self.mode}, "
            f"dpi={self._dpi}, format={fmt})"
        )


def _extract_dpi(pil: PILImage.Image) -> tuple[int, int]:
    """Extract DPI from PIL Image metadata, default (72, 72)."""
    info = pil.info
    dpi = info.get("dpi")
    if dpi and isinstance(dpi, tuple) and len(dpi) >= 2:
        return (int(dpi[0]), int(dpi[1]))
    return (72, 72)
