"""Unit tests for canonical-record selection + the dedup() convenience."""

from __future__ import annotations

import pytest

from kaos_content.dedup import (
    DedupDocument,
    dedup,
    recanonicalize,
    select_canonical,
)
from kaos_content.dedup.types import DedupCluster, DedupReport


def _docs(*pairs: tuple[str, str]) -> list[DedupDocument]:
    return [DedupDocument(doc_id=i, text=t) for i, t in pairs]


# ---------------------------------------------------------------------------
# select_canonical
# ---------------------------------------------------------------------------


def test_select_first_is_input_order() -> None:
    members = _docs(("a", "short"), ("b", "much longer text here"))
    assert select_canonical(members, "first") == "a"


def test_select_longest_and_shortest() -> None:
    # lengths: a=2, b=10, c=4 → longest b, shortest a.
    members = _docs(("a", "aa"), ("b", "bbbbbbbbbb"), ("c", "cccc"))
    assert select_canonical(members, "longest") == "b"
    assert select_canonical(members, "shortest") == "a"


def test_select_longest_tiebreak_is_earliest() -> None:
    members = _docs(("a", "same"), ("b", "same"))
    assert select_canonical(members, "longest") == "a"
    assert select_canonical(members, "shortest") == "a"


def test_select_callable() -> None:
    members = _docs(("a", "x"), ("b", "y"))
    assert select_canonical(members, lambda ms: ms[-1].doc_id) == "b"


def test_select_callable_non_member_raises() -> None:
    members = _docs(("a", "x"))
    with pytest.raises(ValueError, match="not a member"):
        select_canonical(members, lambda ms: "ghost")


def test_select_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="unknown canonical strategy"):
        select_canonical(_docs(("a", "x")), "biggest")  # ty: ignore[invalid-argument-type]


def test_select_empty_raises() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        select_canonical([], "first")


def test_select_medoid() -> None:
    np = pytest.importorskip("numpy")
    # Three vectors; doc 'b' sits between 'a' and 'c' → it's the medoid.
    members = [
        DedupDocument(doc_id="a", text="x", embedding=np.array([1.0, 0.0], dtype=np.float32)),
        DedupDocument(doc_id="b", text="x", embedding=np.array([0.92, 0.39], dtype=np.float32)),
        DedupDocument(doc_id="c", text="x", embedding=np.array([0.7, 0.71], dtype=np.float32)),
    ]
    assert select_canonical(members, "medoid") == "b"


def test_select_medoid_without_embedding_raises() -> None:
    pytest.importorskip("numpy")
    members = _docs(("a", "x"), ("b", "y"))
    with pytest.raises(ValueError, match="embedding=None"):
        select_canonical(members, "medoid")


# ---------------------------------------------------------------------------
# recanonicalize
# ---------------------------------------------------------------------------


def test_recanonicalize_preserves_membership_changes_only_canonical() -> None:
    docs = _docs(("a", "tiny"), ("b", "the longest most complete text"), ("c", "mid"))
    report = DedupReport(
        clusters=(
            DedupCluster(
                cluster_id="k1",
                canonical_doc_id="a",
                member_doc_ids=("a", "b", "c"),
                level="text_hash",
                similarity=1.0,
            ),
        ),
        singletons=(),
        per_level_stats={},
        total_input=3,
        total_unique=1,
    )
    out = recanonicalize(report, docs, "longest")
    assert out.clusters[0].canonical_doc_id == "b"
    assert out.clusters[0].member_doc_ids == ("a", "b", "c")  # unchanged
    assert out.total_unique == report.total_unique  # counts unchanged
    # Original is not mutated.
    assert report.clusters[0].canonical_doc_id == "a"


def test_recanonicalize_first_is_noop() -> None:
    report = DedupReport(
        clusters=(), singletons=(), per_level_stats={}, total_input=0, total_unique=0
    )
    assert recanonicalize(report, [], "first") is report


# ---------------------------------------------------------------------------
# dedup() convenience
# ---------------------------------------------------------------------------


def test_dedup_strings_exact_and_near() -> None:
    items = [
        "The quick brown fox jumps over the lazy dog.",
        "The quick brown fox jumps over the lazy dog.",  # exact dup of 0
        "Entirely unrelated content about quarterly tax filings.",
    ]
    report = dedup(items, preset="standard")
    assert report.total_input == 3
    assert report.total_unique == 2
    # doc_ids are positional indices as strings.
    cluster = next(c for c in report.clusters)
    assert set(cluster.member_doc_ids) == {"0", "1"}
    assert cluster.canonical_doc_id == "0"  # default 'first'


def test_dedup_longest_canonical() -> None:
    from kaos_content.dedup.levels.minhash import MinHashLevel

    # Two near-duplicates with heavy shingle overlap so MinHash clusters
    # them despite the length difference; item 1 is the longer (more
    # complete) record, so canonical='longest' must pick it over the
    # input-order default (item 0).
    items = [
        "the annual report covers fiscal performance strategy and outlook",
        "the annual report covers fiscal performance strategy and outlook "
        "with detailed appendices and footnotes",
    ]
    report = dedup(
        items,
        levels=[MinHashLevel(shingle_size=3, threshold=0.4)],
        canonical="longest",
    )
    assert len(report.clusters) == 1, "near-duplicates should cluster"
    assert report.clusters[0].canonical_doc_id == "1"  # the longer member


def test_dedup_passthrough_dedupdocument() -> None:
    docs = [
        DedupDocument(doc_id="x", text="repeated text here"),
        DedupDocument(doc_id="y", text="repeated text here"),
    ]
    report = dedup(docs, preset="fast")
    assert report.total_unique == 1
    assert {report.clusters[0].canonical_doc_id} <= {"x", "y"}


def test_dedup_levels_override() -> None:
    from kaos_content.dedup.levels.text_hash import TextHashLevel

    items = ["same", "same", "different"]
    report = dedup(items, levels=[TextHashLevel(lowercase=True)])
    assert report.total_unique == 2
    assert report.per_level_stats.keys() == {"text_hash"}


def test_dedup_unknown_preset_raises() -> None:
    with pytest.raises(ValueError, match="unknown preset"):
        dedup(["a", "b"], preset="turbo")


def test_dedup_bad_item_type_raises() -> None:
    with pytest.raises(TypeError, match="str or DedupDocument"):
        dedup([123, 456])  # ty: ignore[invalid-argument-type]


def test_dedup_empty() -> None:
    report = dedup([])
    assert report.total_input == 0
    assert report.total_unique == 0
    assert report.dedup_rate == 0.0
