"""Live integration tests for ``SemanticDedupLevel``.

Loads ``BAAI/bge-small-en-v1.5`` (~30-90s first time, cached after) and
runs hierarchical agglomerative clustering on toy paraphrase corpora.
Marked ``integration`` so the unit-only tier skips it.

KNT-602 Option A: this test was ported from
``kaos-nlp-transformers/tests/integration/test_semantic_dedup.py`` in
kaos-content 0.1.0a3 alongside the SemanticDedupLevel implementation
move. Assertions verify content understanding (paraphrase clustering,
threshold sensitivity, error ergonomics), not just shape or non-empty
output.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from kaos_content.dedup.types import DedupDocument

pytestmark = pytest.mark.integration

# Reuse the kaos-nlp-core USC fixture (~69k sections, decoded from
# alea-institute/kl3m-data-usc via the kl3m-004-128k tokenizer). The
# fixture is staged by kaos-nlp-core/tests/fixtures/download_hf_fixtures.py;
# the corpus-scale test below skips cleanly when it's missing so this
# package's test suite stays self-contained.
_USC_FIXTURE = (
    Path(__file__).resolve().parents[3] / "kaos-nlp-core" / "tests" / "fixtures" / "usc.jsonl"
)


def _skip_if_offline() -> None:
    pytest.importorskip("scipy")
    pytest.importorskip("kaos_nlp_transformers")
    if os.environ.get("KAOS_NLP_TRANSFORMERS_OFFLINE", "").lower() in ("1", "true", "yes"):
        pytest.skip("offline mode set")


def test_paraphrases_clustered() -> None:
    """Two paraphrases of the same sentence cluster; an unrelated sentence stays separate."""
    _skip_if_offline()
    from kaos_content.dedup.levels.semantic import SemanticDedupLevel

    docs = [
        DedupDocument(
            doc_id="d1",
            text="The quick brown fox jumps over the lazy dog.",
        ),
        DedupDocument(
            doc_id="d2",
            text="A fast brown fox leaps over a sleepy hound.",
        ),
        DedupDocument(
            doc_id="d3",
            text="Quarterly revenue grew twelve percent year over year.",
        ),
    ]
    level = SemanticDedupLevel(distance_threshold=0.30, device="cpu")
    clusters = level.find_clusters(docs)

    paraphrase_clusters = [c for c in clusters if {"d1", "d2"}.issubset(set(c.member_doc_ids))]
    assert len(paraphrase_clusters) == 1, (
        f"expected d1+d2 to cluster as paraphrases under distance_threshold=0.30; "
        f"got clusters={[c.member_doc_ids for c in clusters]}"
    )
    assert all("d3" not in c.member_doc_ids for c in paraphrase_clusters), (
        "d3 (revenue) should not cluster with d1/d2 (fox paraphrases)"
    )
    assert paraphrase_clusters[0].level == "semantic"


def test_distance_threshold_sensitivity() -> None:
    """Tightening distance_threshold strictly shrinks cluster membership."""
    _skip_if_offline()
    from kaos_content.dedup.levels.semantic import SemanticDedupLevel

    docs = [
        DedupDocument(doc_id="d1", text="The quick brown fox jumps over the lazy dog."),
        DedupDocument(doc_id="d2", text="A fast brown fox leaps over a sleepy hound."),
        DedupDocument(doc_id="d3", text="Quarterly revenue grew twelve percent year over year."),
    ]

    loose = SemanticDedupLevel(distance_threshold=0.50, device="cpu").find_clusters(docs)
    tight = SemanticDedupLevel(distance_threshold=0.001, device="cpu").find_clusters(docs)

    loose_members = {m for c in loose for m in c.member_doc_ids}
    tight_members = {m for c in tight for m in c.member_doc_ids}
    assert len(tight_members) <= len(loose_members), (
        f"tightening distance_threshold should not grow cluster membership; "
        f"loose={loose_members}, tight={tight_members}"
    )
    assert len(tight_members) == 0, (
        f"distance_threshold=0.001 should cluster nothing; got {tight_members}"
    )


def test_corpus_scale_clustering_on_usc_sections() -> None:
    """SemanticDedupLevel produces non-degenerate clusters on real legal text.

    Loads ~500 USC sections from the kaos-nlp-core fixture and asserts:

    1. Embedding + clustering completes on the full sample.
    2. The corpus produces *some* clustering (US Code is template-heavy
       — cross-reference language, definitional repetition — so a sane
       threshold will surface real duplicates).
    3. Clustering does not collapse: the largest cluster covers less
       than half the sample at ``distance_threshold=0.15``.
    4. Hand-crafted paraphrases of two anchor sections cluster with
       their anchors.

    Sample size overridable via ``KAOS_DEDUP_USC_SAMPLE`` env var
    (default 500). Higher values give more confidence at higher cost.
    """
    _skip_if_offline()
    if not _USC_FIXTURE.exists():
        pytest.skip(
            f"USC fixture not staged at {_USC_FIXTURE}. "
            "Run kaos-nlp-core/tests/fixtures/download_hf_fixtures.py to populate."
        )
    from kaos_content.dedup.levels.semantic import SemanticDedupLevel

    sample_size = int(os.environ.get("KAOS_DEDUP_USC_SAMPLE", "500"))

    docs: list[DedupDocument] = []
    with _USC_FIXTURE.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            text = row.get("text", "")
            if len(text) < 800:
                continue
            docs.append(DedupDocument(doc_id=str(row["id"]), text=text))
            if len(docs) >= sample_size:
                break

    assert len(docs) == sample_size, (
        f"expected {sample_size} USC sections >= 800 chars; got {len(docs)}"
    )

    anchor_a, anchor_b = docs[10], docs[42]
    anchor_a_text = anchor_a.text or ""
    anchor_b_text = anchor_b.text or ""
    para_a = DedupDocument(
        doc_id=f"para_{anchor_a.doc_id}",
        text=(
            "Plain-English restatement of the following statutory text, "
            "paraphrasing key terms while preserving meaning:\n\n" + anchor_a_text[:600]
        ),
    )
    para_b = DedupDocument(
        doc_id=f"para_{anchor_b.doc_id}",
        text=(
            "A summary in different words of this section of federal law, "
            "rewriting the language while keeping the substance:\n\n" + anchor_b_text[:600]
        ),
    )
    docs_with_paraphrases = [*docs, para_a, para_b]

    level = SemanticDedupLevel(distance_threshold=0.15, batch_size=64, device="cpu")
    clusters = level.find_clusters(docs_with_paraphrases)

    assert clusters, (
        f"expected at least one cluster on {len(docs_with_paraphrases)} USC docs "
        "at distance_threshold=0.15"
    )

    largest = max(c.size for c in clusters)
    assert largest < len(docs_with_paraphrases) // 2, (
        f"largest cluster has {largest}/{len(docs_with_paraphrases)} docs — "
        f"threshold too loose or embedding signal degraded"
    )

    for anchor, para in ((anchor_a, para_a), (anchor_b, para_b)):
        para_cluster = next(
            (c for c in clusters if para.doc_id in c.member_doc_ids),
            None,
        )
        assert para_cluster is not None, f"paraphrase {para.doc_id} did not cluster with anything"
        assert anchor.doc_id in para_cluster.member_doc_ids, (
            f"paraphrase {para.doc_id} clustered into "
            f"{para_cluster.member_doc_ids} but anchor {anchor.doc_id} is missing"
        )
        assert para_cluster.level == "semantic"


def test_unregistered_model_error_message() -> None:
    """Unknown model id raises ModelNotRegisteredError naming the model."""
    _skip_if_offline()
    if importlib.util.find_spec("kaos_nlp_transformers") is None:
        pytest.skip("kaos-nlp-transformers not installed")
    from kaos_nlp_transformers.errors import (
        ModelNotRegisteredError,
    )

    from kaos_content.dedup.levels.semantic import SemanticDedupLevel

    docs = [
        DedupDocument(doc_id="d1", text="alpha beta gamma delta"),
        DedupDocument(doc_id="d2", text="alpha beta gamma delta"),
    ]
    bogus = "definitely-not-a-real-model/does-not-exist"
    level = SemanticDedupLevel(model_id=bogus, device="cpu")
    with pytest.raises(ModelNotRegisteredError) as exc_info:
        level.find_clusters(docs)
    msg = str(exc_info.value)
    assert bogus in msg
    assert "registry" in msg.lower()
