"""Automatic cluster labelling via class-based TF-IDF (c-TF-IDF).

Turns a set of texts plus a cluster assignment (e.g. the component labels
from ``kaos_graph.algorithms.connected_components_from_edges`` over
``kaos_nlp_core.similarity.near_duplicates`` edges) into a keyword label
per cluster — the BERTopic c-TF-IDF approach: treat each cluster as one
document and run TF-IDF over the set of clusters, so a term scores high
when frequent *within* a cluster and distinctive *across* clusters.

The c-TF-IDF compute (tokenisation, n-grams, per-class counts, weighting)
runs in the Rust kernel ``kaos_nlp_core.ctfidf.class_tfidf`` — same word
tokenizer as the rest of the stack, GIL released, deterministic. This
module is the orchestration layer: it adds keyword **diversification**
(semantic MMR via ``kaos_nlp_core.similarity.mmr_select`` when an embedder
is supplied), the per-cluster **exemplar** (medoid when embeddings are
given, else the longest text), and the :class:`ClusterLabel` result type.

Requires the ``[nlp]`` extra (kaos-nlp-core). The MMR diversification and
medoid exemplar additionally need numpy.

References: BERTopic c-TF-IDF
(https://maartengr.github.io/BERTopic/getting_started/ctfidf/ctfidf.html),
KeyBERT/Carbonell-Goldstein MMR.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class ClusterLabel:
    """The label of one cluster.

    Attributes:
        cluster_id: the cluster's id (as passed in ``cluster_ids``).
        keywords: top c-TF-IDF terms, most distinctive first — the label.
        scores: c-TF-IDF weights aligned element-wise with ``keywords``.
        exemplar: index (into the original ``texts``) of a representative
            member — the medoid when ``embeddings`` were given, else the
            longest text.
        size: number of texts in the cluster.
    """

    cluster_id: Any
    keywords: tuple[str, ...]
    scores: tuple[float, ...]
    exemplar: int
    size: int


def _mmr_diversify(
    candidates: list[tuple[str, float]],
    top_k: int,
    diversity: float,
    embedder: Any,
) -> list[tuple[str, float]]:
    """Semantic MMR keyword diversification via ``mmr_select``.

    Maps keyword-MMR onto ``kaos_nlp_core.similarity.mmr_select``: matrix
    rows are candidate-keyword embeddings (the primitive's internal
    pairwise cosine is the keyword-similarity term), relevance is the
    c-TF-IDF scores scaled to ``[0, 1]``, and ``lambda_ = 1 - diversity``
    (the primitive weights relevance, so higher ``diversity`` lowers
    ``lambda_``).
    """
    try:
        import numpy as np
        from kaos_nlp_core.similarity import as_contiguous_f32, mmr_select
    except ImportError as exc:
        msg = (
            "label_clusters(embedder=...) needs numpy + kaos-nlp-core for MMR "
            "keyword diversification. Fix: pip install kaos-content[nlp] and numpy. "
            "Alternative: drop embedder= (no diversification) or set token_prefix= "
            "to conflate morphological variants in the Rust tokenizer."
        )
        raise ImportError(msg) from exc

    if len(candidates) <= top_k:
        return candidates

    terms = [t for t, _ in candidates]
    matrix = as_contiguous_f32(embedder.embed(terms))

    scores_arr = np.asarray([s for _, s in candidates], dtype=np.float32)
    lo, hi = float(scores_arr.min()), float(scores_arr.max())
    relevance = (scores_arr - lo) / (hi - lo) if hi > lo else np.ones_like(scores_arr)

    result = mmr_select(matrix, as_contiguous_f32(relevance), top_k, lambda_=1.0 - diversity)
    return [candidates[i] for i in result.indices.tolist()]


def _exemplar_index(
    member_indices: list[int],
    texts: Sequence[str],
    embeddings: Any | None,
) -> int:
    """Pick a representative member: medoid if embeddings given, else longest."""
    if embeddings is not None:
        import numpy as np

        block = np.asarray(embeddings, dtype=np.float64)[member_indices]
        norms = np.linalg.norm(block, axis=1, keepdims=True)
        unit = block / np.where(norms == 0.0, 1.0, norms)
        centroid = unit.mean(axis=0)
        sims = unit @ centroid
        return member_indices[int(np.argmax(sims))]
    return max(member_indices, key=lambda i: len(texts[i]))


def label_clusters(
    texts: Sequence[str],
    cluster_ids: Sequence[Any],
    *,
    top_k: int = 10,
    ngram_range: tuple[int, int] = (1, 2),
    stopwords: Iterable[str] | None = None,
    min_df: int = 1,
    reduce_frequent_words: bool = False,
    bm25_weighting: bool = False,
    lowercase: bool = True,
    token_prefix: int = 0,
    diversify: bool = True,
    diversity: float = 0.3,
    embedder: Any | None = None,
    embeddings: Any | None = None,
) -> dict[Any, ClusterLabel]:
    """Label each cluster with its most distinctive keywords (c-TF-IDF).

    The c-TF-IDF ranking comes from the Rust kernel
    ``kaos_nlp_core.ctfidf.class_tfidf``; this function adds optional
    semantic keyword diversification and a per-cluster exemplar.

    Args:
        texts: the documents, one per item.
        cluster_ids: the cluster assignment per text (any hashable). Same
            length as ``texts``.
        top_k: keywords per label (BERTopic default 10; 5-10 reads best).
        ngram_range: inclusive ``(min_n, max_n)`` for n-gram terms.
        stopwords: words to drop before counting. ``None`` (default)
            removes none — c-TF-IDF already down-weights terms common
            *across* clusters via the cross-class IDF, so generic words
            rarely surface as distinctive labels. For extra cleanup pass a
            set; the recommended source is the derived
            ``kaos_nlp_core.stopwords.stopwords()`` (no ad-hoc list is
            baked in here).
        min_df: drop terms whose total count across clusters is below this.
        reduce_frequent_words: ``sqrt`` the term frequency (suppresses
            residual high-frequency words).
        bm25_weighting: smoothed BM25-style IDF (steadier on small corpora).
        lowercase: lowercase tokens before counting.
        token_prefix: when ``> 0``, truncate tokens to this many characters
            in the Rust tokenizer — a dependency-free conflation of
            morphological/derivational variants (``4`` merges
            ``automobile``/``automotive``/``autos`` → ``auto``). Yields
            truncated surface forms, so prefer it for grouping.
        diversify: when ``True`` *and* an ``embedder`` is supplied, apply
            semantic MMR to drop redundant keywords. No-op without an
            embedder (use ``token_prefix`` for surface-variant conflation).
        diversity: MMR diversity in ``[0, 1]`` (``0`` = pure relevance,
            ``1`` = max diversity). ``0.3`` is a sane keyword default.
        embedder: optional object with ``.embed(list[str]) -> ndarray``
            (e.g. ``kaos_nlp_transformers.EmbeddingModel``). Enables MMR.
        embeddings: optional ``(n_texts, dim)`` per-text embedding array;
            when given, each ``exemplar`` is the cluster medoid.

    Returns:
        ``{cluster_id: ClusterLabel}`` for every distinct cluster id, in
        first-seen order.

    Raises:
        ValueError: ``texts``/``cluster_ids`` length mismatch, or invalid
            ``top_k``/``ngram_range``.
        ImportError: kaos-nlp-core (the ``[nlp]`` extra) is not installed.
    """
    if len(texts) != len(cluster_ids):
        msg = f"texts and cluster_ids must match in length ({len(texts)} != {len(cluster_ids)})."
        raise ValueError(msg)
    if top_k < 1:
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    lo, hi = ngram_range
    if lo < 1 or hi < lo:
        raise ValueError(f"ngram_range must be (min>=1, max>=min), got {ngram_range}")

    try:
        from kaos_nlp_core.ctfidf import class_tfidf
    except ImportError as exc:
        msg = (
            "label_clusters requires kaos-nlp-core (the class_tfidf kernel). "
            "Fix: pip install kaos-content[nlp] (or pip install "
            "kaos-nlp-core>=0.1.6 directly)."
        )
        raise ImportError(msg) from exc

    stops = None if stopwords is None else set(stopwords)
    use_mmr = diversify and embedder is not None
    # Pull a wider candidate pool only when MMR will prune it back to top_k.
    pool = max(top_k * 3, 30) if use_mmr else top_k

    candidates = class_tfidf(
        texts,
        cluster_ids,
        top_k=pool,
        ngram_range=ngram_range,
        stopwords=stops,
        min_df=min_df,
        reduce_frequent_words=reduce_frequent_words,
        bm25_weighting=bm25_weighting,
        lowercase=lowercase,
        token_prefix=token_prefix,
    )

    # Member indices per cluster (for exemplar + size), first-seen order to
    # match class_tfidf's key ordering.
    members: dict[Any, list[int]] = {}
    for idx, cid in enumerate(cluster_ids):
        members.setdefault(cid, []).append(idx)

    labels: dict[Any, ClusterLabel] = {}
    for cid, cand in candidates.items():
        chosen = _mmr_diversify(cand, top_k, diversity, embedder) if use_mmr else cand[:top_k]
        labels[cid] = ClusterLabel(
            cluster_id=cid,
            keywords=tuple(t for t, _ in chosen),
            scores=tuple(round(s, 6) for _, s in chosen),
            exemplar=_exemplar_index(members[cid], texts, embeddings),
            size=len(members[cid]),
        )
    return labels


__all__ = ["ClusterLabel", "label_clusters"]
