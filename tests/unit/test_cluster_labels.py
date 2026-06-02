"""Unit tests for label_clusters() — c-TF-IDF cluster labelling.

Requires kaos-nlp-core >= 0.1.6 (the class_tfidf kernel); skipped cleanly
when that isn't importable so the gate stays green pre-release.
"""

from __future__ import annotations

import pytest

pytest.importorskip("kaos_nlp_core.ctfidf", reason="needs kaos-nlp-core>=0.1.6 (class_tfidf)")

from kaos_content.cluster import ClusterLabel, label_clusters

_LITIGATION = [
    "The court granted the motion for summary judgment.",
    "Plaintiff filed a motion for summary judgment yesterday.",
    "Summary judgment motions were dismissed by the judge.",
]
_BAKING = [
    "The recipe calls for two cups of flour and three eggs.",
    "Mix the flour, sugar, and eggs into a smooth batter.",
    "Bake the batter with flour at 350 degrees for thirty minutes.",
]


def _two_topic_corpus() -> tuple[list[str], list[int]]:
    texts = [*_LITIGATION, *_BAKING]
    ids = [0, 0, 0, 1, 1, 1]
    return texts, ids


def test_separates_topics() -> None:
    texts, ids = _two_topic_corpus()
    labels = label_clusters(texts, ids, top_k=4)
    assert set(labels) == {0, 1}
    assert all(isinstance(v, ClusterLabel) for v in labels.values())
    kw0 = " ".join(labels[0].keywords)
    kw1 = " ".join(labels[1].keywords)
    assert any(w in kw0 for w in ("judgment", "motion", "summary", "court"))
    assert any(w in kw1 for w in ("flour", "batter", "eggs", "bake"))
    assert "flour" not in kw0 and "judgment" not in kw1
    assert labels[0].size == 3 and labels[1].size == 3


def test_scores_descending_and_aligned() -> None:
    texts, ids = _two_topic_corpus()
    labels = label_clusters(texts, ids, top_k=5)
    for lbl in labels.values():
        assert len(lbl.keywords) == len(lbl.scores)
        assert list(lbl.scores) == sorted(lbl.scores, reverse=True)


def test_exemplar_longest_without_embeddings() -> None:
    texts, ids = _two_topic_corpus()
    labels = label_clusters(texts, ids, top_k=3)
    # Cluster 0's longest text is index 2 (".. dismissed by the judge.").
    assert labels[0].exemplar == max(range(3), key=lambda i: len(texts[i]))


def test_top_k_respected() -> None:
    texts, ids = _two_topic_corpus()
    labels = label_clusters(texts, ids, top_k=2)
    assert all(len(lbl.keywords) <= 2 for lbl in labels.values())


def test_token_prefix_conflates_variants() -> None:
    texts = ["automobile automotive autos vehicles", "kitchen cooking baking recipe"]
    plain = label_clusters(texts, [0, 1], top_k=10, stopwords=set())
    prefixed = label_clusters(texts, [0, 1], top_k=10, stopwords=set(), token_prefix=4)
    assert "auto" in prefixed[0].keywords
    assert "automobile" not in prefixed[0].keywords
    assert len(prefixed[0].keywords) <= len(plain[0].keywords)


def test_bm25_and_reduce_frequent_words_run() -> None:
    texts, ids = _two_topic_corpus()
    labels = label_clusters(texts, ids, top_k=4, bm25_weighting=True, reduce_frequent_words=True)
    assert all(all(s >= 0.0 for s in lbl.scores) for lbl in labels.values())


def test_min_df_filters() -> None:
    texts = ["common common rare", "common common common"]
    labels = label_clusters(texts, [0, 1], min_df=2, stopwords=set())
    assert "rare" not in labels[0].keywords


def test_exemplar_medoid_with_embeddings() -> None:
    np = pytest.importorskip("numpy")
    # Cluster 0 = rows 0,1,2; row 1 is the centroid-nearest (medoid).
    texts = ["a", "b", "c", "d"]
    ids = [0, 0, 0, 1]
    emb = np.array(
        [[1.0, 0.0], [0.92, 0.39], [0.7, 0.71], [0.0, 1.0]],
        dtype=np.float32,
    )
    labels = label_clusters(texts, ids, top_k=1, embeddings=emb, stopwords=set())
    assert labels[0].exemplar == 1


def test_mmr_diversify_with_fake_embedder() -> None:
    np = pytest.importorskip("numpy")

    class _FakeEmbedder:
        # Deterministic per-term embedding from a hash, so MMR has a real
        # similarity structure to diversify over.
        def embed(self, terms: list[str]) -> np.ndarray:
            rows = []
            for t in terms:
                h = abs(hash(t))
                rows.append([(h % 97) / 97.0, ((h // 97) % 89) / 89.0, ((h // 9000) % 83) / 83.0])
            return np.asarray(rows, dtype=np.float32)

    texts = ["alpha beta gamma delta epsilon zeta eta theta", "x y z"]
    labels = label_clusters(
        texts, [0, 1], top_k=3, stopwords=set(), embedder=_FakeEmbedder(), diversity=0.5
    )
    assert len(labels[0].keywords) <= 3


def test_diversify_noop_without_embedder() -> None:
    texts, ids = _two_topic_corpus()
    a = label_clusters(texts, ids, top_k=4, diversify=True)
    b = label_clusters(texts, ids, top_k=4, diversify=False)
    assert {k: v.keywords for k, v in a.items()} == {k: v.keywords for k, v in b.items()}


def test_length_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="match in length"):
        label_clusters(["a", "b"], [0])


def test_invalid_top_k_and_ngram_raise() -> None:
    with pytest.raises(ValueError, match="top_k"):
        label_clusters(["a"], [0], top_k=0)
    with pytest.raises(ValueError, match="ngram_range"):
        label_clusters(["a"], [0], ngram_range=(2, 1))


def test_empty() -> None:
    assert label_clusters([], []) == {}


def test_deterministic() -> None:
    texts, ids = _two_topic_corpus()
    a = label_clusters(texts, ids, top_k=5)
    b = label_clusters(texts, ids, top_k=5)
    assert {k: v.keywords for k, v in a.items()} == {k: v.keywords for k, v in b.items()}
