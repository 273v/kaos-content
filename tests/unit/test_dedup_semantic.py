"""Unit tests for ``SemanticDedupLevel``.

KNT-602 Option A migration tests — pin the contract of the level
that moved here from ``kaos_nlp_transformers.clustering`` in
kaos-content 0.1.0a3:

- Constructor validates ``distance_threshold`` against the cosine-
  distance domain [0, 2].
- ``find_clusters`` raises ``ImportError`` with a kaos-content-side
  install hint when scipy is missing — not the cryptic
  ModuleNotFoundError from ``import scipy``.
- ``find_clusters`` raises ``ImportError`` with a kaos-content-side
  install hint when ``kaos-nlp-transformers`` is missing.
- The constructor accepts a ``settings`` kwarg (KNT-004 contract).
- find_clusters honors a custom ``model_id``, threshold, batch_size
  and produces deterministic clusters via a fake ``EmbeddingModel``.

The fake-model tests run whether or not kaos-nlp-transformers is
installed; the live integration tests live in
``tests/integration/test_semantic_dedup.py``.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pytest

from kaos_content.dedup.levels.semantic import SemanticDedupLevel
from kaos_content.dedup.types import DedupDocument

_has_scipy = importlib.util.find_spec("scipy") is not None
_has_transformers = importlib.util.find_spec("kaos_nlp_transformers") is not None


# ─── Constructor validation ─────────────────────────────────────────────


class TestConstructor:
    def test_accepts_default_threshold(self) -> None:
        level = SemanticDedupLevel()
        assert level._distance_threshold == 0.10

    def test_accepts_explicit_threshold(self) -> None:
        level = SemanticDedupLevel(distance_threshold=0.25)
        assert level._distance_threshold == 0.25

    def test_rejects_negative_threshold(self) -> None:
        with pytest.raises(ValueError, match=r"\[0\.0, 2\.0\]"):
            SemanticDedupLevel(distance_threshold=-0.01)

    def test_rejects_above_two_threshold(self) -> None:
        with pytest.raises(ValueError, match=r"\[0\.0, 2\.0\]"):
            SemanticDedupLevel(distance_threshold=2.5)

    def test_accepts_settings_kwarg(self) -> None:
        """``settings`` kwarg must be plumbed through (KNT-004)."""
        import inspect

        params = inspect.signature(SemanticDedupLevel.__init__).parameters
        assert "settings" in params

    def test_constructor_does_not_load_kaos_nlp_transformers(self) -> None:
        """Constructor must not import kaos_nlp_transformers eagerly.

        The lazy-import discipline matters because users without the
        [transformers] extra need to be able to instantiate the level
        — only ``find_clusters`` requires the optional deps.
        """
        # If the module was already imported elsewhere (e.g. by the dev
        # group install), simply check the constructor doesn't TRIGGER
        # an import. We monkeypatch sys.modules to forbid a fresh import
        # for the duration of the test.
        if "kaos_nlp_transformers" in sys.modules:
            pytest.skip("kaos_nlp_transformers already imported by test runner")

        level = SemanticDedupLevel()
        assert level is not None
        assert "kaos_nlp_transformers" not in sys.modules


# ─── find_clusters install-hint paths ────────────────────────────────────


class TestInstallHints:
    _MISSING: object = object()

    @classmethod
    def _hide_modules(cls, *names: str) -> dict[str, object]:
        """Sentinel-hide each of ``names`` plus any already-loaded
        submodules thereof. Returns a snapshot mapping module name →
        prior value (or ``cls._MISSING`` for keys that didn't exist
        before — restoring those means deleting the key, not setting
        it to ``_MISSING``). Pair with :meth:`_restore_modules` in a
        finally block.
        """
        snapshot: dict[str, object] = {}
        # Capture both the explicit names AND any cached submodules.
        targets: set[str] = set(names)
        for mod in list(sys.modules):
            for prefix in names:
                if mod == prefix or mod.startswith(f"{prefix}."):
                    targets.add(mod)
        for mod in targets:
            snapshot[mod] = sys.modules.get(mod, cls._MISSING)
            sys.modules[mod] = None  # ty: ignore[invalid-assignment]
        return snapshot

    @classmethod
    def _restore_modules(cls, snapshot: dict[str, object]) -> None:
        for mod, val in snapshot.items():
            if val is cls._MISSING:
                # Original sys.modules had no entry — pop the sentinel
                # so a future fresh import can succeed.
                sys.modules.pop(mod, None)
            else:
                sys.modules[mod] = val  # ty: ignore[invalid-assignment]

    @pytest.mark.skipif(not _has_scipy, reason="needs scipy installed to monkey-hide it")
    def test_missing_scipy_raises_install_hint(self) -> None:
        """``find_clusters`` raises ImportError pointing at the
        ``[clustering]`` extra when scipy is unavailable.

        Uses raw sys.modules manipulation rather than
        ``monkeypatch.setitem`` because pytest's monkeypatch saves the
        prior value via ``dict.__getitem__`` semantics, which loses
        the distinction between "key was None" and "key was absent" —
        important here because we MUST fully restore the dict to its
        pre-test state so subsequent tests can re-import scipy.
        """
        snapshot = self._hide_modules(
            "scipy",
            "scipy.cluster",
            "scipy.cluster.hierarchy",
            "scipy.spatial",
            "scipy.spatial.distance",
        )
        try:
            docs = [DedupDocument(doc_id=str(i), text=f"document {i}") for i in range(3)]
            with pytest.raises(ImportError, match=r"\[clustering\]"):
                SemanticDedupLevel().find_clusters(docs)
        finally:
            self._restore_modules(snapshot)

    @pytest.mark.skipif(
        not _has_transformers,
        reason="needs kaos-nlp-transformers installed to monkey-hide it",
    )
    def test_missing_transformers_raises_install_hint(self) -> None:
        """``find_clusters`` raises ImportError pointing at the
        ``[transformers]`` extra when kaos-nlp-transformers is unavailable.
        """
        snapshot = self._hide_modules(
            "kaos_nlp_transformers",
            "kaos_nlp_transformers.settings",
        )
        try:
            docs = [DedupDocument(doc_id=str(i), text=f"document {i}") for i in range(3)]
            with pytest.raises(ImportError, match=r"\[transformers\]"):
                SemanticDedupLevel().find_clusters(docs)
        finally:
            self._restore_modules(snapshot)


# ─── find_clusters happy path with a fake EmbeddingModel ────────────────


@pytest.mark.skipif(not _has_scipy, reason="scipy not installed")
class TestFindClustersWithFakeModel:
    """Exercise the clustering pipeline with a deterministic fake model.

    Avoids the kaos-nlp-transformers / model-download dance — the
    real path is covered by the live integration suite.
    """

    @staticmethod
    def _patch_model(monkeypatch: pytest.MonkeyPatch) -> list[str]:
        """Replace EmbeddingModel.load with a fake whose embeddings encode
        the first 4 ASCII chars of each text. Returns a list of model_ids
        the fake `load` was called with (test plumbing assertion hook).
        """
        seen_model_ids: list[str] = []

        class _FakeModel:
            def embed(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
                _ = batch_size
                n = len(texts)
                mat = np.zeros((n, 4), dtype=np.float32)
                for i, t in enumerate(texts):
                    for j, ch in enumerate((t + "    ")[:4]):
                        mat[i, j] = (ord(ch) % 32) / 32.0
                norms = np.linalg.norm(mat, axis=1, keepdims=True)
                norms = np.where(norms == 0, 1.0, norms)
                return (mat / norms).astype(np.float32)

        class _FakeEmbeddingModel:
            @classmethod
            def load(cls, model_id: str, **kwargs: object) -> _FakeModel:
                seen_model_ids.append(model_id)
                return _FakeModel()

        # Patch kaos_nlp_transformers.EmbeddingModel — the level imports it
        # lazily inside find_clusters, so we patch the module-attribute that
        # the lazy import resolves to.
        import kaos_nlp_transformers

        monkeypatch.setattr(kaos_nlp_transformers, "EmbeddingModel", _FakeEmbeddingModel)
        return seen_model_ids

    def test_paraphrases_cluster_at_loose_threshold(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = self._patch_model(monkeypatch)
        docs = [
            DedupDocument(doc_id="a", text="alpha similar text alpha"),
            DedupDocument(doc_id="b", text="alpha similar text alpha"),  # dup of a
            DedupDocument(doc_id="c", text="zebra completely different"),
        ]
        # With identical first-4-chars, a and b have cosine sim 1.0
        # under the fake. distance_threshold=0.5 cuts above their
        # identical pair into one cluster; c stays separate.
        level = SemanticDedupLevel(distance_threshold=0.5)
        clusters = level.find_clusters(docs)
        assert len(clusters) == 1
        cluster = clusters[0]
        assert set(cluster.member_doc_ids) == {"a", "b"}
        assert cluster.level == "semantic"
        assert seen, "expected the fake EmbeddingModel.load to be called"

    def test_skips_empty_or_whitespace_only_documents(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        self._patch_model(monkeypatch)
        docs = [
            DedupDocument(doc_id="empty", text=""),
            DedupDocument(doc_id="whitespace", text="   \n\t  "),
            DedupDocument(doc_id="real", text="something real here"),
        ]
        # After filtering only "real" remains — single document means no
        # clustering possible.
        clusters = SemanticDedupLevel().find_clusters(docs)
        assert clusters == []

    def test_custom_model_id_propagates(self, monkeypatch: pytest.MonkeyPatch) -> None:
        seen = self._patch_model(monkeypatch)
        docs = [
            DedupDocument(doc_id="a", text="hello"),
            DedupDocument(doc_id="b", text="hello"),
        ]
        SemanticDedupLevel(model_id="custom/model", distance_threshold=0.5).find_clusters(docs)
        assert seen == ["custom/model"], seen

    def test_max_chars_truncates_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``max_chars=N`` slices each text to N before embedding."""
        seen_texts: list[str] = []

        class _ProbeModel:
            def embed(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
                _ = batch_size
                seen_texts.extend(texts)
                # Return enough variance so clustering is well-defined.
                return np.eye(len(texts), dtype=np.float32)[:, :4]

        class _ProbeEmbeddingModel:
            @classmethod
            def load(cls, model_id: str, **kwargs: object) -> _ProbeModel:
                _ = model_id
                return _ProbeModel()

        import kaos_nlp_transformers

        monkeypatch.setattr(kaos_nlp_transformers, "EmbeddingModel", _ProbeEmbeddingModel)

        docs = [
            DedupDocument(doc_id="a", text="x" * 5000),
            DedupDocument(doc_id="b", text="y" * 5000),
        ]
        SemanticDedupLevel(max_chars=10).find_clusters(docs)
        assert all(len(t) == 10 for t in seen_texts), seen_texts
