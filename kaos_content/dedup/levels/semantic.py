"""Semantic embedding clustering — implements ``DedupLevel`` via cosine
distance over dense embeddings.

Embeds documents with the kaos-nlp-transformers default model
(``BAAI/bge-small-en-v1.5``) and clusters them with scipy hierarchical
agglomerative clustering on cosine distance. Catches paraphrases,
template variants, and topic clusters that lexical levels miss.

KNT-602 Option A: this level lives in kaos-content (the consumer side
of the dependency arrow). It depends on the optional
``kaos-nlp-transformers`` and ``scipy`` packages — the imports are
lazy (inside ``find_clusters``) so the module is constructible
without them. ``find_clusters`` raises ``ImportError`` with an
actionable install hint when the dependencies are missing.

The same lazy-import pattern is used by :class:`SearchableDocument`
in ``kaos_content.indexing`` — kaos-content owns the AST-grounded
integration, kaos-nlp-transformers stays a clean inference primitive
with no reverse dependency on this package.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from kaos_core.logging import get_logger

from kaos_content.dedup.types import DedupCluster, DedupDocument, DedupLevel

if TYPE_CHECKING:
    from kaos_nlp_transformers.settings import KaosNLPTransformersSettings

logger = get_logger(__name__)


class SemanticDedupLevel(DedupLevel):
    """Embedding + cosine + agglomerative clustering."""

    name: ClassVar[str] = "semantic"

    def __init__(
        self,
        *,
        model_id: str | None = None,
        distance_threshold: float = 0.10,
        batch_size: int = 64,
        max_chars: int = 8000,
        device: str | None = None,
        backend: str | None = None,
        settings: KaosNLPTransformersSettings | None = None,
    ) -> None:
        """
        Args:
            model_id: Embedding model identifier. Must be registered in
                ``kaos_nlp_transformers.models.REGISTRY``. ``None``
                resolves to the package default at
                ``find_clusters`` time
                (``KaosNLPTransformersSettings.default_model``,
                currently ``BAAI/bge-small-en-v1.5``). Resolution is
                deferred so the constructor stays import-light — no
                kaos-nlp-transformers load happens until clustering is
                actually requested.
            distance_threshold: Cosine distance (1 - similarity)
                threshold for ``scipy.cluster.hierarchy.fcluster``.
                0.02 = near-exact semantic match (>0.98 cosine sim).
                0.10 = same template/form (~0.90 cosine sim).
                0.20 = same topic (~0.80 cosine sim).
            batch_size: Embedding batch size.
            max_chars: Truncate documents longer than this before
                embedding. The model context window is the hard limit;
                this avoids wasting memory on very long docs that
                won't fit anyway.
            device: Forwarded to ``EmbeddingModel.load(device=...)``.
                ``None`` defers to the package settings (default
                ``"auto"``). Pin to ``"cpu"`` to force CPU even on GPU
                hosts.
            backend: Forwarded to ``EmbeddingModel.load(backend=...)``.
            settings: Module settings forwarded to
                ``EmbeddingModel.load`` (cache/offline/device policy
                injection — kaos-nlp-transformers KNT-004).
        """
        # Audit-02 KNT-105 (carried over from kaos-nlp-transformers):
        # validate distance_threshold against the cosine-distance domain
        # [0, 2]. fcluster will accept any positive float, but values >2
        # silently flatten everything into one cluster (every cosine
        # distance fits) and values <0 raise inside scipy with a
        # confusing message. Catch it at construction.
        if not 0.0 <= distance_threshold <= 2.0:
            msg = (
                f"distance_threshold={distance_threshold!r} is outside the "
                "cosine distance domain [0.0, 2.0]. "
                "Fix: pick a value in (0.0, 1.0] for typical near-duplicate / "
                "topic clustering. 0.10 is the default for same-template "
                "matches; 0.20 for same-topic clusters."
            )
            raise ValueError(msg)

        self._model_id = model_id
        self._distance_threshold = distance_threshold
        self._batch_size = batch_size
        self._max_chars = max_chars
        self._device = device
        self._backend = backend
        self._settings = settings

    def find_clusters(
        self,
        documents: list[DedupDocument],
    ) -> list[DedupCluster]:
        # scipy is gated on the kaos-content `[clustering]` extra. Raise an
        # actionable install-hint error rather than letting the import fail
        # with a cryptic ModuleNotFoundError.
        try:
            from scipy.cluster.hierarchy import fcluster, linkage
            from scipy.spatial.distance import pdist
        except ImportError as exc:
            msg = (
                "SemanticDedupLevel requires scipy. "
                "Fix: pip install kaos-content[clustering] (or "
                "pip install scipy>=1.14.1 directly). "
                "Alternative: use kaos_content.dedup.levels.minhash for "
                "non-semantic near-duplicate detection without scipy."
            )
            raise ImportError(msg) from exc

        # kaos-nlp-transformers gates the [transformers] extra. Same hint
        # contract — actionable install path before the cryptic resolver
        # error surfaces.
        try:
            from kaos_nlp_transformers import EmbeddingModel
            from kaos_nlp_transformers.settings import (
                KaosNLPTransformersSettings,
            )
        except ImportError as exc:
            msg = (
                "SemanticDedupLevel requires kaos-nlp-transformers. "
                "Fix: pip install kaos-content[transformers] (or "
                "pip install kaos-nlp-transformers>=0.2.0a2 directly). "
                "Alternative: use kaos_content.dedup.levels.minhash for "
                "non-semantic near-duplicate detection."
            )
            raise ImportError(msg) from exc

        valid: list[tuple[int, DedupDocument]] = []
        texts: list[str] = []
        for i, doc in enumerate(documents):
            if doc.text and doc.text.strip():
                valid.append((i, doc))
                texts.append(doc.text[: self._max_chars])

        if len(valid) < 2:
            return []

        # Resolve model_id at run time so the constructor stays
        # import-light. Mirrors the settings-driven default contract
        # documented in kaos-nlp-transformers' KNT-004.
        model_id = (
            self._model_id or KaosNLPTransformersSettings.model_fields["default_model"].default
        )

        model = EmbeddingModel.load(
            model_id,
            device=self._device,
            backend=self._backend,
            settings=self._settings,
        )
        embeddings = model.embed(texts, batch_size=self._batch_size)

        dists = pdist(embeddings, metric="cosine")
        linkage_matrix = linkage(dists, method="average")
        labels = fcluster(linkage_matrix, t=self._distance_threshold, criterion="distance")

        groups: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(labels):
            groups[int(label)].append(idx)

        # EmbeddingModel.embed enforces L2 normalization (kaos-nlp-transformers
        # KNT-101), so dot products on these rows already give cosine
        # similarity. The defensive normalize-here-too step is cheap and
        # keeps this block correct even if a future code path skips the
        # model layer.
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        safe = np.where(norms == 0.0, 1.0, norms)
        unit = embeddings / safe

        clusters: list[DedupCluster] = []
        for label, members in groups.items():
            if len(members) < 2:
                continue
            member_docs = [valid[m][1] for m in members]

            # Audit-02 KNT-105: compute mean intra-cluster cosine similarity.
            # The DedupCluster default (similarity=1.0) was inherited unset
            # before that fix, so every semantic cluster reported 1.0
            # regardless of cluster tightness. With unit-norm rows, sim is
            # the upper-triangular mean of unit @ unit.T over `members`.
            block = unit[members]
            sim_matrix: Any = block @ block.T
            n_members = len(members)
            # Sum the strict upper triangle, count = n*(n-1)/2.
            triu_sum = float(np.triu(sim_matrix, k=1).sum())
            n_pairs = n_members * (n_members - 1) // 2
            mean_sim = triu_sum / n_pairs if n_pairs else 1.0
            # Clamp into [0.0, 1.0] for numeric jitter on near-1.0 values.
            mean_sim = float(min(max(mean_sim, 0.0), 1.0))

            clusters.append(
                DedupCluster(
                    cluster_id=f"semantic_{label}_{member_docs[0].doc_id}",
                    canonical_doc_id=member_docs[0].doc_id,
                    member_doc_ids=tuple(d.doc_id for d in member_docs),
                    level=self.name,
                    similarity=mean_sim,
                )
            )
        return clusters


__all__ = ["SemanticDedupLevel"]
