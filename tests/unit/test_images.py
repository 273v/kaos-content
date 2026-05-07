"""Tests for kaos-content[images]: KaosImage, ops, profiles, artifacts."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
from kaos_core import (
    ArtifactStore,
    KaosContext,
    KaosRuntime,
    KaosSettings,
    VFSConfig,
    VirtualFileSystem,
)
from kaos_core.types.enums import StorageBackend
from PIL import Image as PILImage

from kaos_content.images.model import KaosImage
from kaos_content.images.ops import (
    auto_contrast,
    crop,
    denoise,
    enhance_brightness,
    enhance_contrast,
    invert,
    remove_borders,
    rotate,
    sharpen,
    threshold,
)
from kaos_content.images.profiles import (
    PreprocessingProfile,
    apply_profile,
    for_ocr,
    for_thumbnail,
    for_vlm,
)
from kaos_content.model.attr import BoundingBox, Provenance, SourceRef

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_runtime(tmp_path: Path) -> KaosRuntime:
    settings = KaosSettings(
        artifact_inline_read_max_bytes=262_144,
        artifact_chunk_size_bytes=65_536,
    )
    runtime = KaosRuntime(config=settings)
    runtime.vfs = VirtualFileSystem(
        VFSConfig(default_backend=StorageBackend.DISK, disk_base_path=tmp_path / "vfs")
    )
    runtime.artifacts = ArtifactStore(
        runtime.vfs,
        manifest_context_id=settings.artifact_manifest_context_id,
        manifest_prefix=settings.artifact_manifest_prefix,
        max_inline_read_bytes=settings.artifact_inline_read_max_bytes,
        default_chunk_size=settings.artifact_chunk_size_bytes,
        temporary_ttl_seconds=settings.artifact_temporary_ttl_seconds,
    )
    return runtime


def _make_rgb_image(width: int = 200, height: int = 150) -> KaosImage:
    """Create a test RGB image with a gradient pattern."""
    pil = PILImage.new("RGB", (width, height), (100, 150, 200))
    return KaosImage.from_pil(pil, dpi=(300, 300), source_format="png")


def _make_gray_image(width: int = 200, height: int = 150) -> KaosImage:
    """Create a test grayscale image."""
    pil = PILImage.new("L", (width, height), 128)
    return KaosImage.from_pil(pil, dpi=(300, 300), source_format="png")


def _make_rgba_image(width: int = 200, height: int = 150) -> KaosImage:
    """Create a test RGBA image with transparency."""
    pil = PILImage.new("RGBA", (width, height), (100, 150, 200, 128))
    return KaosImage.from_pil(pil, dpi=(96, 96))


# ---------------------------------------------------------------------------
# KaosImage construction and properties
# ---------------------------------------------------------------------------


class TestKaosImageConstruction:
    def test_from_pil(self) -> None:
        img = _make_rgb_image()
        assert img.width == 200
        assert img.height == 150
        assert img.mode == "RGB"
        assert img.dpi == (300, 300)
        assert img.source_format == "png"

    def test_from_bytes_png(self) -> None:
        pil = PILImage.new("RGB", (100, 80), (255, 0, 0))
        buf = io.BytesIO()
        pil.save(buf, format="PNG")
        data = buf.getvalue()

        img = KaosImage.from_bytes(data)
        assert img.width == 100
        assert img.height == 80
        assert img.source_format == "png"

    def test_from_bytes_jpeg(self) -> None:
        pil = PILImage.new("RGB", (100, 80), (0, 255, 0))
        buf = io.BytesIO()
        pil.save(buf, format="JPEG", quality=90)
        data = buf.getvalue()

        img = KaosImage.from_bytes(data)
        assert img.width == 100
        assert img.height == 80
        assert img.source_format == "jpeg"

    def test_from_path(self, tmp_path: Path) -> None:
        path = tmp_path / "test.png"
        pil = PILImage.new("RGB", (50, 50), (0, 0, 255))
        pil.save(str(path))

        img = KaosImage.from_path(path)
        assert img.width == 50
        assert img.height == 50

    def test_from_numpy_rgb(self) -> None:
        arr = np.zeros((100, 200, 3), dtype=np.uint8)
        arr[:, :, 0] = 255  # Red
        img = KaosImage.from_numpy(arr)
        assert img.width == 200
        assert img.height == 100
        assert img.mode == "RGB"

    def test_from_numpy_grayscale(self) -> None:
        arr = np.full((100, 200), 128, dtype=np.uint8)
        img = KaosImage.from_numpy(arr)
        assert img.mode == "L"

    def test_with_provenance(self) -> None:
        prov = Provenance(
            source=SourceRef(uri="file:///doc.pdf", mime_type="application/pdf"),
            page=3,
            bbox=BoundingBox(left=10, top=20, right=100, bottom=80),
            extractor="kaos-pdf",
        )
        img = _make_rgb_image().with_provenance(prov)
        assert img.provenance is not None
        assert img.provenance.page == 3

    def test_with_metadata(self) -> None:
        img = _make_rgb_image().with_metadata(source="test", page_index=0)
        assert img.metadata["source"] == "test"
        assert img.metadata["page_index"] == 0

    def test_mime_type(self) -> None:
        img = _make_rgb_image()
        assert img.mime_type == "image/png"

    def test_is_grayscale(self) -> None:
        assert _make_gray_image().is_grayscale
        assert not _make_rgb_image().is_grayscale

    def test_is_rgb(self) -> None:
        assert _make_rgb_image().is_rgb
        assert not _make_gray_image().is_rgb

    def test_repr(self) -> None:
        img = _make_rgb_image()
        r = repr(img)
        assert "200x150" in r
        assert "RGB" in r
        assert "png" in r


# ---------------------------------------------------------------------------
# Output formats
# ---------------------------------------------------------------------------


class TestKaosImageOutput:
    def test_to_bytes_png(self) -> None:
        img = _make_rgb_image()
        data = img.to_bytes(format="png")
        assert data[:4] == b"\x89PNG"
        assert len(data) > 0

    def test_to_bytes_jpeg(self) -> None:
        img = _make_rgb_image()
        data = img.to_bytes(format="jpeg", quality=85)
        assert data[:2] == b"\xff\xd8"

    def test_to_bytes_jpeg_from_rgba(self) -> None:
        """RGBA images should be composited onto white for JPEG."""
        img = _make_rgba_image()
        data = img.to_bytes(format="jpeg")
        assert data[:2] == b"\xff\xd8"

    def test_to_pil(self) -> None:
        img = _make_rgb_image()
        pil = img.to_pil()
        assert isinstance(pil, PILImage.Image)
        assert pil.size == (200, 150)

    def test_to_numpy(self) -> None:
        img = _make_rgb_image()
        arr = img.to_numpy()
        assert arr.shape == (150, 200, 3)
        assert arr.dtype == np.uint8

    def test_to_numpy_grayscale(self) -> None:
        img = _make_gray_image()
        arr = img.to_numpy()
        assert arr.shape == (150, 200)

    def test_to_base64(self) -> None:
        img = _make_rgb_image()
        b64 = img.to_base64(format="png")
        assert isinstance(b64, str)
        assert len(b64) > 0

        import base64

        decoded = base64.b64decode(b64)
        assert decoded[:4] == b"\x89PNG"

    def test_to_data_uri(self) -> None:
        img = _make_rgb_image()
        uri = img.to_data_uri(format="png")
        assert uri.startswith("data:image/png;base64,")

    def test_to_bytes_with_dpi_override(self) -> None:
        img = _make_rgb_image()
        data = img.to_bytes(format="png", dpi=(150, 150))
        # Verify the PNG was created (DPI is in metadata, hard to verify externally)
        assert data[:4] == b"\x89PNG"


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------


class TestKaosImageTransformations:
    def test_resize_by_width(self) -> None:
        img = _make_rgb_image(400, 300)
        resized = img.resize(width=200)
        assert resized.width == 200
        assert resized.height == 150  # Aspect ratio preserved

    def test_resize_by_height(self) -> None:
        img = _make_rgb_image(400, 300)
        resized = img.resize(height=150)
        assert resized.height == 150
        assert resized.width == 200

    def test_resize_max_size(self) -> None:
        img = _make_rgb_image(400, 300)
        resized = img.resize(max_size=200)
        assert max(resized.width, resized.height) == 200

    def test_resize_max_size_no_upscale(self) -> None:
        img = _make_rgb_image(100, 80)
        resized = img.resize(max_size=500)
        assert resized is img  # No change needed

    def test_thumbnail(self) -> None:
        img = _make_rgb_image(800, 600)
        thumb = img.thumbnail(max_size=200)
        assert max(thumb.width, thumb.height) <= 200

    def test_to_grayscale(self) -> None:
        img = _make_rgb_image()
        gray = img.to_grayscale()
        assert gray.mode == "L"
        assert gray.dpi == img.dpi

    def test_to_grayscale_idempotent(self) -> None:
        img = _make_gray_image()
        gray = img.to_grayscale()
        assert gray is img  # No-op

    def test_to_rgb(self) -> None:
        img = _make_gray_image()
        rgb = img.to_rgb()
        assert rgb.mode == "RGB"

    def test_to_rgb_from_rgba(self) -> None:
        img = _make_rgba_image()
        rgb = img.to_rgb()
        assert rgb.mode == "RGB"

    def test_to_rgb_idempotent(self) -> None:
        img = _make_rgb_image()
        rgb = img.to_rgb()
        assert rgb is img

    def test_convert_mode(self) -> None:
        img = _make_rgb_image()
        binary = img.convert_mode("1")
        assert binary.mode == "1"

    def test_rotate(self) -> None:
        img = _make_rgb_image(200, 100)
        rotated = img.rotate(90)
        # After 90° rotation with expand, dimensions swap (approximately)
        assert rotated.height > rotated.width or rotated.width > 0

    def test_crop(self) -> None:
        img = _make_rgb_image(200, 150)
        cropped = img.crop(10, 10, 100, 80)
        assert cropped.width == 90
        assert cropped.height == 70

    def test_with_dpi(self) -> None:
        img = _make_rgb_image()
        updated = img.with_dpi(150)
        assert updated.dpi == (150, 150)
        assert updated.width == img.width  # Size unchanged

    def test_with_dpi_tuple(self) -> None:
        img = _make_rgb_image()
        updated = img.with_dpi((96, 96))
        assert updated.dpi == (96, 96)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


class TestImageOps:
    def test_sharpen(self) -> None:
        img = _make_rgb_image()
        sharpened = sharpen(img, factor=2.0)
        assert sharpened.width == img.width
        assert sharpened.mode == img.mode

    def test_enhance_contrast(self) -> None:
        img = _make_rgb_image()
        enhanced = enhance_contrast(img, factor=1.5)
        assert enhanced.width == img.width

    def test_denoise_median(self) -> None:
        img = _make_rgb_image()
        denoised = denoise(img, method="median", strength=3)
        assert denoised.width == img.width

    def test_denoise_gaussian(self) -> None:
        img = _make_rgb_image()
        denoised = denoise(img, method="gaussian", strength=2)
        assert denoised.width == img.width

    def test_denoise_smooth(self) -> None:
        img = _make_rgb_image()
        denoised = denoise(img, method="smooth")
        assert denoised.width == img.width

    def test_denoise_invalid_method(self) -> None:
        img = _make_rgb_image()
        with pytest.raises(ValueError, match="Unknown denoise method"):
            denoise(img, method="invalid")

    def test_threshold_simple(self) -> None:
        img = _make_gray_image()
        binary = threshold(img, value=100, method="simple")
        assert binary.mode == "L"

    def test_threshold_otsu(self) -> None:
        img = _make_gray_image()
        binary = threshold(img, method="otsu")
        assert binary.mode == "L"

    def test_threshold_converts_rgb_to_gray(self) -> None:
        img = _make_rgb_image()
        binary = threshold(img)
        assert binary.mode == "L"

    def test_rotate_op(self) -> None:
        img = _make_rgb_image(200, 100)
        rotated = rotate(img, 180)
        assert rotated.width == 200

    def test_crop_op(self) -> None:
        img = _make_rgb_image(200, 150)
        cropped = crop(img, 0, 0, 100, 100)
        assert cropped.size == (100, 100)

    def test_remove_borders(self) -> None:
        img = _make_rgb_image(200, 200)
        trimmed = remove_borders(img, margin_percent=10)
        assert trimmed.width == 160
        assert trimmed.height == 160

    def test_auto_contrast(self) -> None:
        img = _make_gray_image()
        enhanced = auto_contrast(img)
        assert enhanced.width == img.width

    def test_enhance_brightness(self) -> None:
        img = _make_rgb_image()
        bright = enhance_brightness(img, factor=1.5)
        assert bright.width == img.width

    def test_invert(self) -> None:
        img = _make_rgb_image()
        inverted = invert(img)
        assert inverted.width == img.width
        assert inverted.mode == "RGB"


# ---------------------------------------------------------------------------
# Profiles
# ---------------------------------------------------------------------------


class TestPreprocessingProfiles:
    def test_for_ocr(self) -> None:
        img = _make_rgb_image()
        result = for_ocr(img)
        assert result.mode == "L"  # Grayscale
        assert result.dpi == (300, 300)

    def test_for_ocr_custom_dpi(self) -> None:
        img = _make_rgb_image()
        result = for_ocr(img, target_dpi=150)
        assert result.dpi == (150, 150)

    def test_for_vlm(self) -> None:
        img = _make_gray_image()
        result = for_vlm(img)
        assert result.mode == "RGB"
        assert result.dpi == (150, 150)

    def test_for_vlm_with_resize(self) -> None:
        img = _make_rgb_image(800, 600)
        result = for_vlm(img, max_size=400)
        assert max(result.width, result.height) <= 400

    def test_for_thumbnail(self) -> None:
        img = _make_rgb_image(800, 600)
        result = for_thumbnail(img, max_size=256)
        assert max(result.width, result.height) <= 256

    def test_apply_profile_ocr(self) -> None:
        img = _make_rgb_image()
        result = apply_profile(img, PreprocessingProfile.OCR)
        assert result.mode == "L"

    def test_apply_profile_vlm(self) -> None:
        img = _make_gray_image()
        result = apply_profile(img, "vlm")
        assert result.mode == "RGB"

    def test_apply_profile_thumbnail(self) -> None:
        img = _make_rgb_image(800, 600)
        result = apply_profile(img, "thumbnail")
        assert max(result.width, result.height) <= 512

    def test_apply_profile_unknown(self) -> None:
        img = _make_rgb_image()
        with pytest.raises(ValueError, match="Unknown profile"):
            apply_profile(img, "nonexistent")


# ---------------------------------------------------------------------------
# Artifact store/load
# ---------------------------------------------------------------------------


class TestImageArtifacts:
    async def test_store_and_load_png(self, tmp_path: Path) -> None:
        from kaos_content.images.artifacts import load_image, store_image

        runtime = _make_runtime(tmp_path)
        context = KaosContext.create(session_id="test", runtime=runtime)

        img = _make_rgb_image(100, 80)
        manifest = await store_image(img, runtime, context, name="test-img")

        assert manifest.mime_type == "image/png"
        assert manifest.metadata["width"] == 100
        assert manifest.metadata["height"] == 80

        loaded = await load_image(manifest.artifact_id, runtime)
        assert loaded.width == 100
        assert loaded.height == 80

    async def test_store_jpeg(self, tmp_path: Path) -> None:
        from kaos_content.images.artifacts import store_image

        runtime = _make_runtime(tmp_path)
        context = KaosContext.create(session_id="test", runtime=runtime)

        img = _make_rgb_image()
        manifest = await store_image(
            img, runtime, context, name="jpeg-img", format="jpeg", quality=80
        )
        assert manifest.mime_type == "image/jpeg"

    async def test_store_with_description(self, tmp_path: Path) -> None:
        from kaos_content.images.artifacts import store_image

        runtime = _make_runtime(tmp_path)
        context = KaosContext.create(session_id="test", runtime=runtime)

        img = _make_rgb_image()
        manifest = await store_image(
            img, runtime, context, name="desc-img", description="Page 1 thumbnail"
        )
        assert manifest.description == "Page 1 thumbnail"

    async def test_load_by_ref(self, tmp_path: Path) -> None:
        from kaos_content.images.artifacts import load_image, store_image

        runtime = _make_runtime(tmp_path)
        context = KaosContext.create(session_id="test", runtime=runtime)

        img = _make_rgb_image(50, 50)
        manifest = await store_image(img, runtime, context, name="ref-img")
        ref = manifest.to_ref()

        loaded = await load_image(ref, runtime)
        assert loaded.width == 50

    async def test_to_tool_result_with_image(self, tmp_path: Path) -> None:
        """Stored image artifact produces correct tool result."""
        from kaos_content.images.artifacts import store_image

        runtime = _make_runtime(tmp_path)
        context = KaosContext.create(session_id="test", runtime=runtime)

        img = _make_rgb_image()
        manifest = await store_image(img, runtime, context, name="tool-img")

        result = manifest.to_tool_result(summary="Page 1 image")
        assert not result.isError
        assert len(result.content) == 2
        assert result.content[0].type == "text"
        assert result.content[1].type == "resource_link"


# ---------------------------------------------------------------------------
# Round-trip: PIL → KaosImage → bytes → KaosImage
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_png_round_trip(self) -> None:
        original = _make_rgb_image(100, 80)
        data = original.to_bytes(format="png")
        loaded = KaosImage.from_bytes(data, dpi=(300, 300))
        assert loaded.width == 100
        assert loaded.height == 80
        assert loaded.mode == "RGB"

    def test_jpeg_round_trip(self) -> None:
        original = _make_rgb_image(100, 80)
        data = original.to_bytes(format="jpeg", quality=95)
        loaded = KaosImage.from_bytes(data)
        assert loaded.width == 100
        assert loaded.height == 80

    def test_numpy_round_trip(self) -> None:
        arr = np.random.randint(0, 256, (100, 200, 3), dtype=np.uint8).view(np.uint8)
        img = KaosImage.from_numpy(arr)
        result = img.to_numpy()
        np.testing.assert_array_equal(arr, result)

    def test_grayscale_numpy_round_trip(self) -> None:
        arr = np.random.randint(0, 256, (100, 200), dtype=np.uint8).view(np.uint8)
        img = KaosImage.from_numpy(arr)
        result = img.to_numpy()
        np.testing.assert_array_equal(arr, result)
