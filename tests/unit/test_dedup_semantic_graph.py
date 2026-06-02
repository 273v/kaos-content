"""Unit tests for the semantic *reachability* dedup path through the
public ``dedup(embedder=...)`` entry point.

The level (``SemanticGraphDedupLevel``) embeds documents, builds a cosine
similarity graph via the kaos-nlp-core Rust kernels, and groups the
connected components (kaos-graph union-find). These tests use a tiny,
deterministic, hand-crafted fake embedder so they run offline with no
model download — the assertions exercise the *graph reachability* contract,
not embedding quality:

- a transitive chain ``A ~ B ~ C`` (with ``A`` orthogonal to ``C``) must
  merge all three into one group;
- unrelated items stay separate;
- omitting ``embedder`` leaves behavior unchanged.
"""

from __future__ import annotations

import pytest

from kaos_content.dedup import DedupDocument, dedup

# The level needs the Rust kernels: kaos-nlp-core (knn_graph / near_duplicates)
# and kaos-graph (connected_components_from_edges), plus numpy.
np = pytest.importorskip("numpy", reason="semantic graph dedup needs numpy")
pytest.importorskip(
    "kaos_nlp_core.similarity",
    reason="needs kaos-nlp-core>=0.1.6 (knn_graph / near_duplicates)",
)
pytest.importorskip(
    "kaos_graph.algorithms",
    reason="needs kaos-graph>=0.1.4 (connected_components_from_edges)",
)


class _VectorEmbedder:
    """Deterministic embedder mapping each text to a fixed unit vector.

    ``embed`` looks the text up in a ``{text: vector}`` table and returns
    an L2-normalised float32 matrix in input order. This lets a test pin
    the exact cosine geometry (chains, orthogonality) without a model.
    """

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table

    def embed(self, texts: list[str]) -> np.ndarray:
        rows = np.asarray([self._table[t] for t in texts], dtype=np.float32)
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        return rows / np.where(norms == 0.0, 1.0, norms)


def test_reachability_merges_transitive_chain() -> None:
    """A ~ B ~ C merge into ONE group even though A is orthogonal to C.

    Vectors are chosen so cos(A, B) and cos(B, C) clear the threshold but
    cos(A, C) = 0 (orthogonal). Connected components must still place all
    three in a single duplicate group — the transitive-closure behavior.
    """
    # 2-D unit vectors at angles 0deg, 30deg, 60deg.
    # cos(A,B)=cos30~0.866, cos(B,C)=cos30~0.866 (both >= 0.8 threshold),
    # cos(A,C)=cos60=0.5 (< threshold). A and C are NOT directly similar.
    deg = np.pi / 180.0
    table = {
        "A": [float(np.cos(0 * deg)), float(np.sin(0 * deg))],
        "B": [float(np.cos(30 * deg)), float(np.sin(30 * deg))],
        "C": [float(np.cos(60 * deg)), float(np.sin(60 * deg))],
        "Z": [0.0, -1.0],  # orthogonal/opposite to all of A,B,C
    }
    embedder = _VectorEmbedder(table)

    report = dedup(
        ["A", "B", "C", "Z"],
        levels=[],  # no lexical levels — isolate the semantic path
        embedder=embedder,
        semantic_threshold=0.8,
        semantic_k=3,
    )

    semantic = [c for c in report.clusters if c.level == "semantic_graph"]
    assert len(semantic) == 1, "the A~B~C chain must form exactly one group"
    members = set(semantic[0].member_doc_ids)
    assert members == {"0", "1", "2"}, "reachability must merge A, B and C"
    # Z is unrelated -> singleton, never clustered.
    assert "3" in report.singletons


def test_unrelated_items_stay_separate() -> None:
    """Mutually orthogonal items produce no semantic clusters."""
    table = {
        "x": [1.0, 0.0, 0.0],
        "y": [0.0, 1.0, 0.0],
        "z": [0.0, 0.0, 1.0],
    }
    embedder = _VectorEmbedder(table)

    report = dedup(
        ["x", "y", "z"],
        levels=[],
        embedder=embedder,
        semantic_threshold=0.8,
        semantic_k=2,
    )

    assert [c for c in report.clusters if c.level == "semantic_graph"] == []
    assert set(report.singletons) == {"0", "1", "2"}
    assert report.total_unique == 3


def test_high_threshold_blocks_weak_chain() -> None:
    """Raising the threshold above the chain edges leaves items separate.

    With the 30deg-step chain (cos ~ 0.866 per edge), a 0.95 threshold
    admits no edge, so no group forms — guards against a level that ignores
    the threshold.
    """
    deg = np.pi / 180.0
    table = {
        "A": [float(np.cos(0 * deg)), float(np.sin(0 * deg))],
        "B": [float(np.cos(30 * deg)), float(np.sin(30 * deg))],
        "C": [float(np.cos(60 * deg)), float(np.sin(60 * deg))],
    }
    report = dedup(
        ["A", "B", "C"],
        levels=[],
        embedder=_VectorEmbedder(table),
        semantic_threshold=0.95,
        semantic_k=3,
    )
    assert [c for c in report.clusters if c.level == "semantic_graph"] == []
    assert set(report.singletons) == {"0", "1", "2"}


def test_canonical_strategy_applies_to_semantic_group() -> None:
    """``canonical='longest'`` re-picks the survivor of a semantic group."""
    deg = np.pi / 180.0
    table = {
        "short": [float(np.cos(0 * deg)), float(np.sin(0 * deg))],
        "the longest member text here": [
            float(np.cos(20 * deg)),
            float(np.sin(20 * deg)),
        ],
    }
    report = dedup(
        ["short", "the longest member text here"],
        levels=[],
        embedder=_VectorEmbedder(table),
        semantic_threshold=0.8,
        semantic_k=1,
        canonical="longest",
    )
    semantic = [c for c in report.clusters if c.level == "semantic_graph"]
    assert len(semantic) == 1
    # doc_id "1" is the longer text -> survivor under 'longest'.
    assert semantic[0].canonical_doc_id == "1"


def test_provenance_records_semantic_level() -> None:
    """Semantic clusters are tagged with the ``semantic_graph`` level and a
    within-group mean cosine similarity in (0, 1]."""
    table = {
        "p": [1.0, 0.0],
        "q": [float(np.cos(10 * np.pi / 180)), float(np.sin(10 * np.pi / 180))],
    }
    report = dedup(
        ["p", "q"],
        levels=[],
        embedder=_VectorEmbedder(table),
        semantic_threshold=0.8,
    )
    semantic = [c for c in report.clusters if c.level == "semantic_graph"]
    assert len(semantic) == 1
    assert semantic[0].level == "semantic_graph"
    assert 0.0 < semantic[0].similarity <= 1.0
    assert report.per_level_stats["semantic_graph"]["clusters"] == 1


def test_no_embedder_behavior_unchanged() -> None:
    """Without an embedder, the semantic level never runs and the result
    matches the plain lexical pipeline."""
    items = ["alpha alpha alpha", "beta beta beta", "alpha alpha alpha"]
    baseline = dedup(items)  # default preset, no embedder
    with_param = dedup(items, embedder=None)  # explicit None is identical

    assert "semantic_graph" not in baseline.per_level_stats
    assert "semantic_graph" not in with_param.per_level_stats
    assert baseline.total_unique == with_param.total_unique
    assert {c.level for c in baseline.clusters} == {c.level for c in with_param.clusters}
    # The two identical strings still collapse via the lexical text-hash level.
    assert baseline.total_unique == 2


def test_embedder_appended_after_lexical_levels() -> None:
    """With a real preset + embedder, the semantic level runs after the
    lexical levels. Under the default short-circuit, lexically-clustered
    docs are NOT re-fed to the semantic level (so a lone semantic neighbour
    stays a singleton); with ``short_circuit=False`` the semantic level
    sees every doc and merges the near-paraphrase."""
    deg = np.pi / 180.0
    table = {
        "the quick brown fox": [float(np.cos(0 * deg)), float(np.sin(0 * deg))],
        "a fast auburn fox": [float(np.cos(15 * deg)), float(np.sin(15 * deg))],
    }
    docs = [
        DedupDocument(doc_id="d1", text="the quick brown fox"),
        DedupDocument(doc_id="d2", text="the quick brown fox"),
        DedupDocument(doc_id="d3", text="a fast auburn fox"),
    ]

    # Short-circuit ON (default): d1/d2 collapse lexically; only d3 reaches
    # the semantic level, alone -> no semantic group, d3 is a singleton.
    sc = dedup(docs, preset="standard", embedder=_VectorEmbedder(table), semantic_threshold=0.8)
    assert "semantic_graph" in sc.per_level_stats  # the level participated
    assert {c.level for c in sc.clusters} == {"text_hash"}
    assert "d3" in sc.singletons
    assert sc.total_unique == 2

    # Short-circuit OFF: the semantic level sees all three and groups the
    # near-paraphrase d3 with the d1/d2 originals.
    nsc = dedup(
        docs,
        preset="standard",
        embedder=_VectorEmbedder(table),
        semantic_threshold=0.8,
        short_circuit=False,
    )
    assert "semantic_graph" in {c.level for c in nsc.clusters}


def test_invalid_threshold_rejected() -> None:
    """An out-of-range semantic threshold is a ValueError at level
    construction (surfaced through dedup)."""
    with pytest.raises(ValueError, match="threshold"):
        dedup(
            ["a", "b"],
            levels=[],
            embedder=_VectorEmbedder({"a": [1.0], "b": [1.0]}),
            semantic_threshold=1.5,
        )
