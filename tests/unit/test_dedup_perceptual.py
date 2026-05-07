"""Unit tests for ``PerceptualHashLevel``.

Closes the coverage gap surfaced by the SemanticDedupLevel relocation
audit: the existing ``test_presets_run_without_error`` runs the level
against a doc with no ``page_images``, so it short-circuits at the
missing-input guard and ``imagehash`` is never actually exercised.

Two layers of coverage:

1. Synthetic images (numpy checkerboards) — exercise identity,
   one-pixel sensitivity, dissimilarity, missing-image skip, and the
   dhash-vs-phash control knob without any I/O.

2. Real PDF page renders (``tests/fixtures/perceptual/page_*.png``,
   first page of three distinct kaos-pdf fixture PDFs at 75 DPI
   grayscale) put through the realistic perturbations a re-saved or
   re-scanned document goes through: JPEG round-trip, light blur, and
   brightness/contrast shift. These are the operations
   PerceptualHashLevel exists to be robust to.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

PIL = pytest.importorskip("PIL")
pytest.importorskip("imagehash")

from PIL import Image, ImageEnhance, ImageFilter  # noqa: E402

from kaos_content.dedup.levels import PerceptualHashLevel  # noqa: E402
from kaos_content.dedup.types import DedupDocument  # noqa: E402
from kaos_content.images.model import KaosImage  # noqa: E402

_FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "perceptual"
_PDF_PAGE_FIXTURES = ("page_gpo.png", "page_staten.png", "page_plaster.png")


def _checkerboard(size: int = 256, tile: int = 32, seed: int = 0) -> Image.Image:
    """Synthesize a deterministic high-contrast checkerboard image.

    Perceptual hashes (dHash, pHash) need real image structure to work
    on — a flat or noise-only image collapses to a near-uniform hash.
    Checkerboards give stable gradients that produce well-separated
    hashes for visually-different patterns.
    """
    rng = np.random.default_rng(seed)
    arr = np.zeros((size, size, 3), dtype=np.uint8)
    for y in range(0, size, tile):
        for x in range(0, size, tile):
            shade = int(rng.integers(0, 256))
            arr[y : y + tile, x : x + tile] = shade
    return Image.fromarray(arr, mode="RGB")


def _kaos_doc(doc_id: str, images: list[Image.Image]) -> DedupDocument:
    return DedupDocument(
        doc_id=doc_id,
        page_images=tuple(KaosImage.from_pil(img) for img in images),
    )


class TestPerceptualHashLevel:
    def test_identical_images_clustered(self) -> None:
        img = _checkerboard(seed=1)
        d1 = _kaos_doc("d1", [img])
        d2 = _kaos_doc("d2", [img.copy()])

        level = PerceptualHashLevel(pages_to_hash=1, min_page_overlap=1)
        clusters = level.find_clusters([d1, d2])

        assert len(clusters) == 1
        assert clusters[0].size == 2
        assert set(clusters[0].member_doc_ids) == {"d1", "d2"}
        assert clusters[0].canonical_doc_id == "d1"
        assert clusters[0].level == "perceptual"

    def test_visually_similar_clustered(self) -> None:
        """A one-pixel color shift on a single tile must remain within Hamming budget."""
        base = _checkerboard(seed=2)
        # Bump exactly one pixel's red channel by 1 — invisible to the
        # eye, well below the perceptual-hash sensitivity threshold.
        arr = np.array(base)
        arr[0, 0, 0] = (int(arr[0, 0, 0]) + 1) % 256
        shifted = Image.fromarray(arr, mode="RGB")

        d1 = _kaos_doc("d1", [base])
        d2 = _kaos_doc("d2", [shifted])

        level = PerceptualHashLevel(pages_to_hash=1, max_hamming_distance=5)
        clusters = level.find_clusters([d1, d2])

        assert len(clusters) == 1, (
            "expected near-identical images to cluster under max_hamming_distance=5"
        )
        assert set(clusters[0].member_doc_ids) == {"d1", "d2"}

    def test_visually_different_not_clustered(self) -> None:
        d1 = _kaos_doc("d1", [_checkerboard(seed=10)])
        d2 = _kaos_doc("d2", [_checkerboard(seed=999)])

        level = PerceptualHashLevel(pages_to_hash=1, max_hamming_distance=2)
        clusters = level.find_clusters([d1, d2])

        assert clusters == [], (
            f"expected unrelated checkerboards (different RNG seeds) not to cluster "
            f"at max_hamming_distance=2; got {[c.member_doc_ids for c in clusters]}"
        )

    def test_missing_image_skipped(self) -> None:
        """Documents with no ``page_images`` are dropped from input, not errored."""
        img = _checkerboard(seed=3)
        d1 = _kaos_doc("d1", [img])
        d2 = _kaos_doc("d2", [img.copy()])
        d3 = DedupDocument(doc_id="d3")  # no page_images

        level = PerceptualHashLevel(pages_to_hash=1)
        clusters = level.find_clusters([d1, d2, d3])

        assert len(clusters) == 1
        assert "d3" not in clusters[0].member_doc_ids

    def test_dhash_vs_phash_algorithm(self) -> None:
        """Both ``dhash`` and ``phash`` produce coherent clusters; their
        hash digests differ, proving the ``algorithm`` knob isn't a no-op."""
        import imagehash

        img = _checkerboard(seed=42)
        kimg = KaosImage.from_pil(img)

        dhash_digest = imagehash.dhash(kimg.pil, hash_size=16)
        phash_digest = imagehash.phash(kimg.pil, hash_size=16)
        assert str(dhash_digest) != str(phash_digest), (
            "dhash and phash should produce different digests for the same image"
        )

        d1 = _kaos_doc("d1", [img])
        d2 = _kaos_doc("d2", [img.copy()])
        for algo in ("dhash", "phash"):
            level = PerceptualHashLevel(algorithm=algo, pages_to_hash=1, min_page_overlap=1)
            clusters = level.find_clusters([d1, d2])
            assert len(clusters) == 1, f"identical images must cluster under {algo}"


# ------------------------------------------------------------------
# Real PDF-page renders + realistic perturbations
# ------------------------------------------------------------------


def _load_fixture(name: str) -> Image.Image:
    path = _FIXTURE_DIR / name
    if not path.exists():
        pytest.skip(f"perceptual fixture missing: {path}")
    return Image.open(path).convert("L")


def _jpeg_roundtrip(img: Image.Image, quality: int = 60) -> Image.Image:
    """Re-encode through JPEG — the dominant artifact source for re-saved scans."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("L").copy()


def _light_blur(img: Image.Image, radius: float = 0.8) -> Image.Image:
    """Sub-pixel Gaussian blur — simulates scanner optical PSF or a small resample round-trip."""
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def _brightness_shift(img: Image.Image, factor: float = 1.08) -> Image.Image:
    """Modest brightness shift — common between scans of the same page on different devices."""
    return ImageEnhance.Brightness(img).enhance(factor)


@pytest.fixture(scope="module")
def pdf_page_fixtures() -> list[Image.Image]:
    """Three pre-rendered PDF first pages, grayscale at 75 DPI.

    Sources (all public domain):
    - page_gpo.png    : GPO report
    - page_staten.png : federal court opinion (Staten v. United States)
    - page_plaster.png: USPTO patent (ornamental plaster design)
    """
    return [_load_fixture(name) for name in _PDF_PAGE_FIXTURES]


class TestPerceptualOnRealPdfPages:
    def test_jpeg_roundtrip_clusters_with_original(
        self, pdf_page_fixtures: list[Image.Image]
    ) -> None:
        """A JPEG-recompressed page is the same document under perceptual hash."""
        docs: list[DedupDocument] = []
        for i, page in enumerate(pdf_page_fixtures):
            docs.append(_kaos_doc(f"orig_{i}", [page]))
            docs.append(_kaos_doc(f"jpeg_{i}", [_jpeg_roundtrip(page, quality=60)]))

        level = PerceptualHashLevel(
            algorithm="phash", hash_size=16, pages_to_hash=1, max_hamming_distance=12
        )
        clusters = level.find_clusters(docs)

        # Each (orig_i, jpeg_i) pair must end up in the same cluster.
        for i in range(len(pdf_page_fixtures)):
            cluster = next(
                (c for c in clusters if f"orig_{i}" in c.member_doc_ids),
                None,
            )
            assert cluster is not None, f"original {i} did not cluster"
            assert f"jpeg_{i}" in cluster.member_doc_ids, (
                f"JPEG-recompressed copy of page {i} did not cluster with its original; "
                f"cluster members: {cluster.member_doc_ids}"
            )

    def test_blur_clusters_with_original(self, pdf_page_fixtures: list[Image.Image]) -> None:
        """A lightly blurred page (scanner PSF) clusters with its original."""
        docs: list[DedupDocument] = []
        for i, page in enumerate(pdf_page_fixtures):
            docs.append(_kaos_doc(f"orig_{i}", [page]))
            docs.append(_kaos_doc(f"blur_{i}", [_light_blur(page, radius=0.8)]))

        level = PerceptualHashLevel(
            algorithm="phash", hash_size=16, pages_to_hash=1, max_hamming_distance=12
        )
        clusters = level.find_clusters(docs)

        for i in range(len(pdf_page_fixtures)):
            cluster = next(
                (c for c in clusters if f"orig_{i}" in c.member_doc_ids),
                None,
            )
            assert cluster is not None and f"blur_{i}" in cluster.member_doc_ids, (
                f"blurred copy of page {i} did not cluster with its original"
            )

    def test_brightness_shift_clusters_with_original(
        self, pdf_page_fixtures: list[Image.Image]
    ) -> None:
        """An 8% brightness shift does not move the perceptual hash off-cluster."""
        docs: list[DedupDocument] = []
        for i, page in enumerate(pdf_page_fixtures):
            docs.append(_kaos_doc(f"orig_{i}", [page]))
            docs.append(_kaos_doc(f"bright_{i}", [_brightness_shift(page, factor=1.08)]))

        level = PerceptualHashLevel(
            algorithm="phash", hash_size=16, pages_to_hash=1, max_hamming_distance=12
        )
        clusters = level.find_clusters(docs)

        for i in range(len(pdf_page_fixtures)):
            cluster = next(
                (c for c in clusters if f"orig_{i}" in c.member_doc_ids),
                None,
            )
            assert cluster is not None and f"bright_{i}" in cluster.member_doc_ids, (
                f"brightness-shifted copy of page {i} did not cluster with its original"
            )

    def test_distinct_pdf_pages_do_not_cluster(self, pdf_page_fixtures: list[Image.Image]) -> None:
        """Three visually distinct PDF pages each form their own cluster.

        Without this control, a perturbation test could pass spuriously
        if every page's hash collapsed into one bucket. Verifies that
        on real document layouts (multi-column government report, court
        opinion, patent diagram), the perceptual hash actually
        discriminates across documents.
        """
        docs = [_kaos_doc(f"d{i}", [page]) for i, page in enumerate(pdf_page_fixtures)]
        level = PerceptualHashLevel(
            algorithm="phash", hash_size=16, pages_to_hash=1, max_hamming_distance=12
        )
        clusters = level.find_clusters(docs)

        assert clusters == [], (
            f"three distinct PDF pages should not cluster at max_hamming_distance=12; "
            f"got {[c.member_doc_ids for c in clusters]}"
        )

    def test_combined_perturbations_still_cluster(
        self, pdf_page_fixtures: list[Image.Image]
    ) -> None:
        """JPEG + blur + brightness stacked together — the realistic
        scan-then-resave case. Original still clusters with the
        perturbed copy, and the three documents stay separable."""
        docs: list[DedupDocument] = []
        for i, page in enumerate(pdf_page_fixtures):
            perturbed = _brightness_shift(
                _light_blur(_jpeg_roundtrip(page, quality=60)), factor=1.06
            )
            docs.append(_kaos_doc(f"orig_{i}", [page]))
            docs.append(_kaos_doc(f"resave_{i}", [perturbed]))

        level = PerceptualHashLevel(
            algorithm="phash", hash_size=16, pages_to_hash=1, max_hamming_distance=12
        )
        clusters = level.find_clusters(docs)

        # Three pairs → three clusters.
        assert len(clusters) == len(pdf_page_fixtures), (
            f"expected exactly {len(pdf_page_fixtures)} clusters (one per page); "
            f"got {len(clusters)}: {[c.member_doc_ids for c in clusters]}"
        )
        for i in range(len(pdf_page_fixtures)):
            cluster = next(
                (c for c in clusters if f"orig_{i}" in c.member_doc_ids),
                None,
            )
            assert cluster is not None and f"resave_{i}" in cluster.member_doc_ids, (
                f"page {i} original did not cluster with its perturbed copy"
            )
