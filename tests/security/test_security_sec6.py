"""Sec-6 regression tests: KaosImage.resize() mutated source (#7).

Pre-fix the width+height aspect-preserving branch of
``KaosImage.resize()`` called ``self._pil.thumbnail(...)`` — the
only PIL op in the method that mutates the underlying ``PIL.Image``
in place. The class API is immutable-style (resize returns a new
KaosImage), so this was a contract violation: the source image's
size was silently changed by the side effect.

Fix: copy the PIL image first, mutate the copy, return a new
KaosImage wrapping the copy. The ``self._pil.copy()`` call gives
us a fresh PIL.Image object that ``thumbnail()`` can safely mutate
without touching the original.
"""

from __future__ import annotations

import pytest

PIL = pytest.importorskip("PIL")
PILImage = pytest.importorskip("PIL.Image")

from kaos_content.images.model import KaosImage  # noqa: E402


def _make_kaos_image(width: int = 800, height: int = 600) -> KaosImage:
    """Build an in-memory KaosImage of the requested dimensions."""
    pil = PILImage.new("RGB", (width, height), color="red")
    return KaosImage(pil_image=pil)


class TestResizeWidthAndHeightDoesNotMutateSource:
    """The original Sec-6 PoC: width+height aspect-preserving branch."""

    def test_returns_fresh_pil_object(self) -> None:
        original = _make_kaos_image(800, 600)
        original_pil_id = id(original._pil)
        resized = original.resize(width=400, height=400)
        # Different KaosImage object — already true pre-fix.
        assert resized is not original
        # Different underlying PIL object — pre-fix the new KaosImage
        # wrapped a *copy* of the original PIL image, but the original's
        # PIL image had been mutated first.
        assert id(resized._pil) != original_pil_id

    def test_original_size_unchanged(self) -> None:
        # The smoking gun: pre-fix the original's .size shrank to fit
        # within the (width, height) box because ``thumbnail()`` ran
        # in place on the source.
        original = _make_kaos_image(800, 600)
        original_size = original._pil.size
        _ = original.resize(width=200, height=200)
        assert original._pil.size == original_size, (
            f"Original PIL was mutated: {original._pil.size} != {original_size}"
        )

    def test_resized_size_correct(self) -> None:
        # Sanity: the new image has the right dimensions.
        original = _make_kaos_image(800, 600)
        resized = original.resize(width=400, height=400)
        # Aspect preserved within the (400, 400) box: 800x600 → 400x300.
        assert resized._pil.size == (400, 300)


class TestOtherResizeBranchesAlsoSafe:
    """The other resize branches use ``PIL.resize()`` which is
    already non-mutating — these tests verify they stay that way."""

    def test_max_size_branch(self) -> None:
        original = _make_kaos_image(1000, 500)
        original_size = original._pil.size
        resized = original.resize(max_size=400)
        assert original._pil.size == original_size  # not mutated
        assert resized._pil.size == (400, 200)  # scaled correctly

    def test_width_only_branch(self) -> None:
        original = _make_kaos_image(800, 600)
        original_size = original._pil.size
        resized = original.resize(width=400)
        assert original._pil.size == original_size
        assert resized._pil.size == (400, 300)

    def test_height_only_branch(self) -> None:
        original = _make_kaos_image(800, 600)
        original_size = original._pil.size
        resized = original.resize(height=300)
        assert original._pil.size == original_size
        assert resized._pil.size == (400, 300)

    def test_no_aspect_branch(self) -> None:
        original = _make_kaos_image(800, 600)
        original_size = original._pil.size
        resized = original.resize(width=200, height=200, maintain_aspect=False)
        assert original._pil.size == original_size
        assert resized._pil.size == (200, 200)


class TestThumbnailMethodAlreadyCopies:
    """The ``thumbnail()`` method (separate from the ``resize()``
    width+height branch) was already correctly copying-before-mutation.
    Sanity test that this remains true."""

    def test_thumbnail_does_not_mutate_source(self) -> None:
        original = _make_kaos_image(800, 600)
        original_size = original._pil.size
        thumb = original.thumbnail(max_size=200)
        assert original._pil.size == original_size
        # Thumbnail fits within (200, 200): 800x600 → 200x150
        assert thumb._pil.size == (200, 150)
