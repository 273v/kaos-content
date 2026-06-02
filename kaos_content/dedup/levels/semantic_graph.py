"""Semantic reachability dedup — embedding similarity graph + connected
components.

Catches items that are semantically equivalent but phrased differently, a
gap the lexical levels (binary/text hash, MinHash) cannot close. The
mechanism is *reachability*: embed every document, build a sparse cosine
similarity graph (kNN and/or a fixed threshold), then take the connected
components of that graph as duplicate groups. Because components are a
transitive closure over the over-threshold edges, ``A ~ B`` and ``B ~ C``
place ``A``, ``B`` and ``C`` in one group even when ``A`` and ``C`` are not
directly above the threshold.

This is a different mechanism from :class:`SemanticDedupLevel`, which runs
hierarchical agglomerative clustering (average linkage) at a cosine
*distance* threshold. Reachability over a kNN/threshold graph avoids the
``O(n^2)`` dense distance matrix and follows transitive chains; agglomerative
linkage instead bounds the average within-cluster distance. They are
complementary and live side by side.

The graph build and component labelling are the released Rust primitives —
this module only orchestrates:

- edges: ``kaos_nlp_core.similarity.knn_graph`` / ``near_duplicates`` (SIMD
  cosine sweeps, GIL released);
- components: ``kaos_graph.algorithms.connected_components_from_edges``
  (union-find over the integer edge list).

The optional deps (numpy, ``kaos-nlp-core``, ``kaos-graph``) are imported
lazily inside :meth:`find_clusters`, so the module is constructible without
them; the embedder is supplied by the caller (any object with
``.embed(list[str]) -> ndarray``), mirroring ``label_clusters``.

References: near-duplicate detection via embedding similarity graphs and
connected-components (transitive-closure) clustering; the kNN/threshold
graph keeps the comparison sparse instead of pairwise quadratic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

from kaos_core.logging import get_logger

from kaos_content.dedup.types import DedupCluster, DedupDocument, DedupLevel

if TYPE_CHECKING:
    import numpy as np  # noqa: F401  (used by `find_clusters` lazy import)

logger = get_logger(__name__)


class SemanticGraphDedupLevel(DedupLevel):
    """Embedding similarity graph + connected-components reachability.

    Groups documents whose embeddings are transitively reachable above a
    cosine threshold. The heavy compute (cosine edges, union-find
    components) runs in the kaos-nlp-core / kaos-graph Rust kernels; this
    level embeds the texts and maps the resulting component labels onto
    :class:`DedupCluster` groups.
    """

    name: ClassVar[str] = "semantic_graph"

    def __init__(
        self,
        embedder: Any,
        *,
        threshold: float = 0.85,
        k: int = 10,
        max_chars: int = 8000,
        batch_size: int = 64,
        assume_normalized: bool = True,
    ) -> None:
        """
        Args:
            embedder: an object with ``.embed(list[str]) -> ndarray`` (e.g.
                ``kaos_nlp_transformers.EmbeddingModel``). Required — this
                level exists to consume caller-provided embeddings; the
                ``dedup(embedder=...)`` convenience API constructs the
                level from this object.
            threshold: minimum cosine similarity for two documents to share
                an edge, in ``[0.0, 1.0]``. Pairs at or above it are joined;
                connected components of the resulting graph are the
                duplicate groups. ``0.85`` is a conservative
                near-duplicate / paraphrase default; raise it toward
                ``0.95`` to require near-exact semantic matches, lower it
                toward ``0.75`` for same-topic grouping. Because components
                follow chains, a lower threshold merges more transitively —
                tune against a labelled sample.
            k: neighbours per row for the kNN graph. The threshold still
                filters every kNN edge, so ``k`` only bounds how many
                candidate neighbours each row contributes; it does not by
                itself force a row into a group. Keep it small for large
                corpora. Capped internally at the number available.
            max_chars: truncate documents longer than this before embedding
                (the model context window is the hard limit anyway).
            batch_size: embedding batch size passed to ``embedder.embed``
                when the embedder accepts it.
            assume_normalized: when ``True`` (default), take the Rust
                unit-norm fast path — the caller guarantees L2-unit-norm
                rows, which ``EmbeddingModel.embed`` output already
                satisfies. When ``False``, the embeddings are L2-normalised
                here before the similarity sweep.
        """
        if embedder is None:
            msg = (
                "SemanticGraphDedupLevel requires an embedder with "
                ".embed(list[str]) -> ndarray. "
                "Fix: pass embedder=EmbeddingModel.load(...) (or any object "
                "exposing .embed). To dedup without embeddings, use a "
                "lexical level (binary/text hash, MinHash)."
            )
            raise ValueError(msg)
        if not 0.0 <= threshold <= 1.0:
            msg = (
                f"threshold={threshold!r} is outside the cosine-similarity "
                "domain [0.0, 1.0]. Fix: pick a value in (0.0, 1.0]; 0.85 is "
                "the near-duplicate default, ~0.95 for near-exact, ~0.75 for "
                "same-topic grouping."
            )
            raise ValueError(msg)
        if k < 1:
            msg = f"k={k!r} must be >= 1 (neighbours per row)."
            raise ValueError(msg)

        self._embedder = embedder
        self._threshold = threshold
        self._k = k
        self._max_chars = max_chars
        self._batch_size = batch_size
        self._assume_normalized = assume_normalized
        self.last_embeddings: dict[str, Any] = {}
        """Unit-norm embedding row per ``doc_id`` from the most recent
        :meth:`find_clusters` call (empty before the first run). Exposed so
        the ``dedup(embedder=...)`` convenience API can reuse the vectors
        this level already computed to drive ``canonical='medoid'`` survivor
        selection without re-embedding. Reset at the start of each call."""

    def _embed(self, texts: list[str]) -> Any:
        """Embed ``texts`` via the caller's embedder.

        Passes ``batch_size`` when the embedder accepts it (matches the
        ``EmbeddingModel.embed`` signature) and falls back to the bare call
        otherwise — ``label_clusters`` treats the embedder the same duck-typed
        way.
        """
        try:
            return self._embedder.embed(texts, batch_size=self._batch_size)
        except TypeError:
            return self._embedder.embed(texts)

    def find_clusters(
        self,
        documents: list[DedupDocument],
    ) -> list[DedupCluster]:
        try:
            import numpy as np
            from kaos_nlp_core.similarity import (
                as_contiguous_f32,
                knn_graph,
                near_duplicates,
            )
        except ImportError as exc:
            msg = (
                "SemanticGraphDedupLevel requires numpy + kaos-nlp-core (the "
                "knn_graph / near_duplicates similarity kernels). "
                "Fix: pip install kaos-content[nlp] (or pip install "
                "kaos-nlp-core>=0.1.6 and numpy>=2.1.0). Alternative: use "
                "kaos_content.dedup.levels.minhash for non-semantic "
                "near-duplicate detection."
            )
            raise ImportError(msg) from exc

        try:
            from kaos_graph.algorithms import connected_components_from_edges
        except ImportError as exc:
            msg = (
                "SemanticGraphDedupLevel requires kaos-graph (the "
                "connected_components_from_edges union-find kernel) to take "
                "connected components of the similarity graph. "
                "Fix: pip install kaos-content[graph] (or pip install "
                "kaos-graph>=0.1.4 directly)."
            )
            raise ImportError(msg) from exc

        self.last_embeddings = {}

        valid: list[tuple[int, DedupDocument]] = []
        texts: list[str] = []
        for i, doc in enumerate(documents):
            if doc.text and doc.text.strip():
                valid.append((i, doc))
                texts.append(doc.text[: self._max_chars])

        if len(valid) < 2:
            return []

        # Coerce to the C-contiguous float32 layout the Rust similarity
        # boundary requires (reuses kaos-nlp-core's own coercion helper).
        matrix = as_contiguous_f32(self._embed(texts))
        if matrix.ndim != 2 or matrix.shape[0] != len(valid):
            msg = (
                "embedder.embed must return a 2-D (n_texts, dim) array; got "
                f"shape {getattr(matrix, 'shape', None)!r} for {len(valid)} texts."
            )
            raise ValueError(msg)

        if not self._assume_normalized:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = as_contiguous_f32(matrix / np.where(norms == 0.0, 1.0, norms))

        # Stash the rows we just computed so the convenience API can reuse
        # them for canonical='medoid' instead of re-embedding (one source of
        # truth for the vectors). Keyed by doc_id; rows are unit-norm.
        self.last_embeddings = {doc.doc_id: matrix[row] for row, (_, doc) in enumerate(valid)}

        n = matrix.shape[0]

        # Two over-threshold edge sources, both Rust SIMD sweeps:
        #   - near_duplicates: every pair >= threshold (the dense edge set
        #     reachability wants, capped only by max_pairs);
        #   - knn_graph: each row's top-k neighbours, threshold-filtered.
        # Their union is fed to the union-find. near_duplicates alone is
        # complete for reachability; knn adds nothing past it but is cheap
        # insurance when a future max_pairs cap clips the pair list.
        nd = near_duplicates(
            matrix,
            self._threshold,
            assume_normalized=True,
        )
        edge_rows: list[Any] = [nd.pairs]

        knn = knn_graph(matrix, min(self._k, n - 1), assume_normalized=True)
        knn_edges = knn.edges()
        if knn_edges.shape[0]:
            # `edges()` drops NO_NEIGHBOR padding in row-major order; apply
            # the same padding mask to the flat score/index arrays so the
            # threshold filter stays aligned with the surviving edges.
            from kaos_nlp_core.similarity import NO_NEIGHBOR

            flat_scores = knn.scores.reshape(-1)
            pad_mask = knn.indices.reshape(-1) != NO_NEIGHBOR
            keep = flat_scores[pad_mask] >= self._threshold
            kept = knn_edges[keep]
            if kept.shape[0]:
                edge_rows.append(kept)

        edge_matrix = (
            np.concatenate(edge_rows, axis=0) if edge_rows else np.empty((0, 2), np.uint32)
        )
        # connected_components_from_edges declares Sequence[tuple[int, int]];
        # hand it integer tuples (its own internal coercion does the same)
        # rather than the numpy (m, 2) array, so the typed boundary is exact.
        edges: list[tuple[int, int]] = [(int(a), int(b)) for a, b in edge_matrix]

        # Union-find transitive closure in Rust. Each node's label is the
        # smallest node id in its component (deterministic, edge-order
        # independent).
        labels = connected_components_from_edges(n, edges)

        groups: dict[int, list[int]] = {}
        for idx, label in enumerate(labels):
            groups.setdefault(int(label), []).append(idx)

        # Per-pair cosine for the within-group mean similarity. Rows are
        # unit-norm (assume_normalized contract / explicit normalise above),
        # so the dot product is the cosine.
        clusters: list[DedupCluster] = []
        for label, members in groups.items():
            if len(members) < 2:
                continue
            member_docs = [valid[m][1] for m in members]

            block = matrix[members]
            sim_matrix: Any = block @ block.T
            n_members = len(members)
            triu_sum = float(np.triu(sim_matrix, k=1).sum())
            n_pairs = n_members * (n_members - 1) // 2
            mean_sim = triu_sum / n_pairs if n_pairs else 1.0
            mean_sim = float(min(max(mean_sim, 0.0), 1.0))

            clusters.append(
                DedupCluster(
                    cluster_id=f"semantic_graph_{label}_{member_docs[0].doc_id}",
                    canonical_doc_id=member_docs[0].doc_id,
                    member_doc_ids=tuple(d.doc_id for d in member_docs),
                    level=self.name,
                    similarity=mean_sim,
                )
            )
        return clusters


__all__ = ["SemanticGraphDedupLevel"]
