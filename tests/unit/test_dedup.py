"""Unit tests for the FUND-11 document dedup pipeline.

Covers:
1. BinaryHashLevel — file-level exact dedup
2. TextHashLevel — normalized text dedup
3. DedupPipeline — multi-level orchestration + short-circuit
4. DedupReport — metrics (dedup_rate, total_unique, singletons)
5. Presets — FAST / STANDARD configurations
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest

from kaos_content.dedup import (
    DedupCluster,
    DedupDocument,
    DedupLevel,
    DedupPipeline,
    DedupReport,
)
from kaos_content.dedup.levels import BinaryHashLevel, FuzzyBinaryLevel, MinHashLevel, TextHashLevel
from kaos_content.dedup.pipeline import DedupPipelineConfig
from kaos_content.dedup.presets import COMPREHENSIVE, FAST, LEGAL, STANDARD

# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------


def _write_file(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.write_text(content)
    return p


# ------------------------------------------------------------------
# BinaryHashLevel
# ------------------------------------------------------------------


class TestBinaryHashLevel:
    def test_identical_files_clustered(self, tmp_path: Path) -> None:
        content = "This is the exact same content."
        f1 = _write_file(tmp_path, "a.txt", content)
        f2 = _write_file(tmp_path, "b.txt", content)
        f3 = _write_file(tmp_path, "c.txt", content)

        docs = [
            DedupDocument(doc_id="d1", file_path=f1),
            DedupDocument(doc_id="d2", file_path=f2),
            DedupDocument(doc_id="d3", file_path=f3),
        ]
        level = BinaryHashLevel()
        clusters = level.find_clusters(docs)
        assert len(clusters) == 1
        assert clusters[0].size == 3
        assert clusters[0].canonical_doc_id == "d1"
        assert clusters[0].similarity == 1.0
        assert clusters[0].level == "binary_hash"

    def test_different_files_no_cluster(self, tmp_path: Path) -> None:
        f1 = _write_file(tmp_path, "a.txt", "content A")
        f2 = _write_file(tmp_path, "b.txt", "content B")

        docs = [
            DedupDocument(doc_id="d1", file_path=f1),
            DedupDocument(doc_id="d2", file_path=f2),
        ]
        level = BinaryHashLevel()
        assert level.find_clusters(docs) == []

    def test_missing_file_path_skipped(self) -> None:
        docs = [DedupDocument(doc_id="d1")]
        level = BinaryHashLevel()
        assert level.find_clusters(docs) == []

    def test_nonexistent_file_skipped(self) -> None:
        docs = [DedupDocument(doc_id="d1", file_path=Path("/nonexistent/x.pdf"))]
        level = BinaryHashLevel()
        assert level.find_clusters(docs) == []

    def test_blake2b_algorithm(self, tmp_path: Path) -> None:
        content = "same"
        f1 = _write_file(tmp_path, "a.txt", content)
        f2 = _write_file(tmp_path, "b.txt", content)
        docs = [
            DedupDocument(doc_id="d1", file_path=f1),
            DedupDocument(doc_id="d2", file_path=f2),
        ]
        level = BinaryHashLevel(algorithm="blake2b")
        clusters = level.find_clusters(docs)
        assert len(clusters) == 1


# ------------------------------------------------------------------
# TextHashLevel
# ------------------------------------------------------------------


class TestTextHashLevel:
    def test_identical_text_clustered(self) -> None:
        docs = [
            DedupDocument(doc_id="d1", text="Hello world."),
            DedupDocument(doc_id="d2", text="Hello world."),
        ]
        level = TextHashLevel()
        clusters = level.find_clusters(docs)
        assert len(clusters) == 1
        assert clusters[0].level == "text_hash"

    def test_whitespace_variation_clustered(self) -> None:
        docs = [
            DedupDocument(doc_id="d1", text="Hello   world."),
            DedupDocument(doc_id="d2", text="Hello\tworld.\n"),
        ]
        level = TextHashLevel()
        clusters = level.find_clusters(docs)
        assert len(clusters) == 1

    def test_case_variation_clustered_when_lowercase(self) -> None:
        docs = [
            DedupDocument(doc_id="d1", text="HELLO WORLD"),
            DedupDocument(doc_id="d2", text="hello world"),
        ]
        level = TextHashLevel(lowercase=True)
        clusters = level.find_clusters(docs)
        assert len(clusters) == 1

    def test_case_variation_not_clustered_when_case_sensitive(self) -> None:
        docs = [
            DedupDocument(doc_id="d1", text="HELLO WORLD"),
            DedupDocument(doc_id="d2", text="hello world"),
        ]
        level = TextHashLevel(lowercase=False)
        clusters = level.find_clusters(docs)
        assert clusters == []

    def test_unicode_normalization(self) -> None:
        docs = [
            DedupDocument(doc_id="d1", text="caf\u00e9"),  # é precomposed
            DedupDocument(doc_id="d2", text="cafe\u0301"),  # e + combining accent
        ]
        level = TextHashLevel(unicode_normalize=True)
        clusters = level.find_clusters(docs)
        assert len(clusters) == 1

    def test_punctuation_stripping(self) -> None:
        docs = [
            DedupDocument(doc_id="d1", text="Hello, world!"),
            DedupDocument(doc_id="d2", text="Hello world"),
        ]
        level = TextHashLevel(strip_punctuation=True)
        clusters = level.find_clusters(docs)
        assert len(clusters) == 1

    def test_different_text_no_cluster(self) -> None:
        docs = [
            DedupDocument(doc_id="d1", text="Document about cats"),
            DedupDocument(doc_id="d2", text="Document about dogs"),
        ]
        level = TextHashLevel()
        assert level.find_clusters(docs) == []

    def test_missing_text_skipped(self) -> None:
        docs = [DedupDocument(doc_id="d1"), DedupDocument(doc_id="d2")]
        level = TextHashLevel()
        assert level.find_clusters(docs) == []

    def test_empty_text_skipped(self) -> None:
        docs = [
            DedupDocument(doc_id="d1", text=""),
            DedupDocument(doc_id="d2", text="   "),
        ]
        level = TextHashLevel()
        assert level.find_clusters(docs) == []


# ------------------------------------------------------------------
# Pipeline
# ------------------------------------------------------------------


class TestDedupPipeline:
    def test_empty_input(self) -> None:
        config = DedupPipelineConfig(levels=(TextHashLevel(),))
        pipeline = DedupPipeline(config)
        report = pipeline.run([])
        assert report.total_input == 0
        assert report.total_unique == 0
        assert report.clusters == ()
        assert report.singletons == ()

    def test_no_duplicates(self) -> None:
        docs = [
            DedupDocument(doc_id="d1", text="aaa"),
            DedupDocument(doc_id="d2", text="bbb"),
            DedupDocument(doc_id="d3", text="ccc"),
        ]
        config = DedupPipelineConfig(levels=(TextHashLevel(),))
        report = DedupPipeline(config).run(docs)
        assert report.total_input == 3
        assert report.total_unique == 3
        assert report.total_duplicates == 0
        assert report.dedup_rate == pytest.approx(0.0)
        assert len(report.singletons) == 3

    def test_all_duplicates(self) -> None:
        docs = [
            DedupDocument(doc_id="d1", text="same"),
            DedupDocument(doc_id="d2", text="same"),
            DedupDocument(doc_id="d3", text="same"),
        ]
        config = DedupPipelineConfig(levels=(TextHashLevel(),))
        report = DedupPipeline(config).run(docs)
        assert report.total_input == 3
        assert report.total_unique == 1
        assert report.total_duplicates == 2
        assert report.dedup_rate == pytest.approx(2 / 3)

    def test_multi_level_short_circuit(self, tmp_path: Path) -> None:
        """Binary hash catches the file dup; text hash doesn't re-process."""
        content = "same content"
        f1 = _write_file(tmp_path, "a.txt", content)
        f2 = _write_file(tmp_path, "b.txt", content)

        docs = [
            DedupDocument(doc_id="d1", file_path=f1, text=content),
            DedupDocument(doc_id="d2", file_path=f2, text=content),
            DedupDocument(doc_id="d3", text="unique"),
        ]
        config = DedupPipelineConfig(
            levels=(BinaryHashLevel(), TextHashLevel()),
            short_circuit=True,
        )
        report = DedupPipeline(config).run(docs)
        assert report.total_unique == 2
        assert len(report.clusters) == 1
        assert report.clusters[0].level == "binary_hash"
        assert "text_hash" in report.per_level_stats
        assert report.per_level_stats["text_hash"]["clusters"] == 0

    def test_multi_level_no_short_circuit(self, tmp_path: Path) -> None:
        """Without short-circuit, both levels see all docs."""
        content = "same content"
        f1 = _write_file(tmp_path, "a.txt", content)
        f2 = _write_file(tmp_path, "b.txt", content)

        docs = [
            DedupDocument(doc_id="d1", file_path=f1, text=content),
            DedupDocument(doc_id="d2", file_path=f2, text=content),
        ]
        config = DedupPipelineConfig(
            levels=(BinaryHashLevel(), TextHashLevel()),
            short_circuit=False,
        )
        report = DedupPipeline(config).run(docs)
        assert len(report.clusters) == 2
        assert report.clusters[0].level == "binary_hash"
        assert report.clusters[1].level == "text_hash"

    def test_mixed_dup_types(self, tmp_path: Path) -> None:
        """Binary dups + text dups detected in one run."""
        f1 = _write_file(tmp_path, "a.txt", "binary dup content")
        f2 = _write_file(tmp_path, "b.txt", "binary dup content")

        docs = [
            DedupDocument(doc_id="d1", file_path=f1, text="binary dup content"),
            DedupDocument(doc_id="d2", file_path=f2, text="binary dup content"),
            DedupDocument(doc_id="d3", text="text  dup  content"),
            DedupDocument(doc_id="d4", text="text dup content"),
            DedupDocument(doc_id="d5", text="unique text"),
        ]
        config = DedupPipelineConfig(
            levels=(BinaryHashLevel(), TextHashLevel()),
            short_circuit=True,
        )
        report = DedupPipeline(config).run(docs)
        assert report.total_input == 5
        assert report.total_unique == 3
        assert len(report.clusters) == 2

    def test_per_level_stats(self) -> None:
        docs = [
            DedupDocument(doc_id="d1", text="aaa"),
            DedupDocument(doc_id="d2", text="aaa"),
        ]
        config = DedupPipelineConfig(levels=(TextHashLevel(),))
        report = DedupPipeline(config).run(docs)
        assert "text_hash" in report.per_level_stats
        stats = report.per_level_stats["text_hash"]
        assert stats["clusters"] == 1
        assert stats["docs_deduped"] == 1


# ------------------------------------------------------------------
# DedupCluster + DedupReport
# ------------------------------------------------------------------


class TestDedupCluster:
    def test_duplicate_doc_ids_excludes_canonical(self) -> None:
        cluster = DedupCluster(
            cluster_id="c1",
            canonical_doc_id="d1",
            member_doc_ids=("d1", "d2", "d3"),
            level="test",
        )
        assert cluster.duplicate_doc_ids == ("d2", "d3")
        assert cluster.size == 3

    def test_frozen(self) -> None:
        cluster = DedupCluster(
            cluster_id="c1",
            canonical_doc_id="d1",
            member_doc_ids=("d1",),
            level="test",
        )
        with pytest.raises(AttributeError):
            cluster.__setattr__("level", "other")


class TestDedupReport:
    def test_dedup_rate_empty(self) -> None:
        report = DedupReport(
            clusters=(),
            singletons=(),
            per_level_stats={},
            total_input=0,
            total_unique=0,
        )
        assert report.dedup_rate == 0.0

    def test_dedup_rate_nonzero(self) -> None:
        report = DedupReport(
            clusters=(),
            singletons=(),
            per_level_stats={},
            total_input=10,
            total_unique=7,
        )
        assert report.dedup_rate == pytest.approx(0.3)


# ------------------------------------------------------------------
# Presets
# ------------------------------------------------------------------


class TestFuzzyBinaryLevel:
    def test_similar_files_clustered(self, tmp_path: Path) -> None:
        content = "This is a long enough document for CTPH. " * 20
        f1 = _write_file(tmp_path, "a.txt", content)
        f2 = _write_file(tmp_path, "b.txt", content + " tiny edit")
        docs = [
            DedupDocument(doc_id="d1", file_path=f1),
            DedupDocument(doc_id="d2", file_path=f2),
        ]
        level = FuzzyBinaryLevel(threshold=0.3)
        clusters = level.find_clusters(docs)
        assert len(clusters) == 1
        assert clusters[0].level == "fuzzy_binary"

    def test_very_different_files_no_cluster(self, tmp_path: Path) -> None:
        f1 = _write_file(tmp_path, "a.txt", "aaaa " * 100)
        f2 = _write_file(tmp_path, "b.txt", "zzzz " * 100)
        docs = [
            DedupDocument(doc_id="d1", file_path=f1),
            DedupDocument(doc_id="d2", file_path=f2),
        ]
        level = FuzzyBinaryLevel(threshold=0.7)
        assert level.find_clusters(docs) == []

    def test_missing_file_skipped(self) -> None:
        docs = [DedupDocument(doc_id="d1")]
        level = FuzzyBinaryLevel()
        assert level.find_clusters(docs) == []


class TestMinHashLevel:
    def test_identical_text_clustered(self) -> None:
        text = "The quick brown fox jumps over the lazy dog. " * 5
        docs = [
            DedupDocument(doc_id="d1", text=text),
            DedupDocument(doc_id="d2", text=text),
        ]
        level = MinHashLevel(shingle_size=3, threshold=0.5)
        clusters = level.find_clusters(docs)
        assert len(clusters) == 1
        assert clusters[0].level == "minhash"

    def test_near_dup_text_clustered(self) -> None:
        base = "The quick brown fox jumps over the lazy dog. This is a test. " * 5
        docs = [
            DedupDocument(doc_id="d1", text=base),
            DedupDocument(doc_id="d2", text=base + " And one more sentence."),
        ]
        level = MinHashLevel(shingle_size=3, threshold=0.5)
        clusters = level.find_clusters(docs)
        assert len(clusters) == 1

    def test_different_text_no_cluster(self) -> None:
        docs = [
            DedupDocument(doc_id="d1", text="cats dogs birds fish snakes lizards frogs. " * 5),
            DedupDocument(doc_id="d2", text="math science physics chemistry biology ecology. " * 5),
        ]
        level = MinHashLevel(shingle_size=3, threshold=0.8)
        assert level.find_clusters(docs) == []

    def test_short_text_skipped(self) -> None:
        docs = [
            DedupDocument(doc_id="d1", text="hi"),
            DedupDocument(doc_id="d2", text="hi"),
        ]
        level = MinHashLevel(shingle_size=5, threshold=0.5)
        assert level.find_clusters(docs) == []

    def test_missing_text_skipped(self) -> None:
        docs = [DedupDocument(doc_id="d1"), DedupDocument(doc_id="d2")]
        level = MinHashLevel()
        assert level.find_clusters(docs) == []

    def test_char_shingles(self) -> None:
        text = "abcdefghijklmnopqrstuvwxyz " * 10
        docs = [
            DedupDocument(doc_id="d1", text=text),
            DedupDocument(doc_id="d2", text=text),
        ]
        level = MinHashLevel(shingle_size=3, use_tokens=False, threshold=0.5)
        clusters = level.find_clusters(docs)
        assert len(clusters) == 1


class TestPresets:
    def test_fast_has_two_levels(self) -> None:
        assert len(FAST.levels) == 2
        assert FAST.levels[0].name == "binary_hash"
        assert FAST.levels[1].name == "text_hash"

    def test_standard_has_four_levels(self) -> None:
        assert len(STANDARD.levels) == 4
        level_names = [lvl.name for lvl in STANDARD.levels]
        assert "binary_hash" in level_names
        assert "fuzzy_binary" in level_names
        assert "text_hash" in level_names
        assert "minhash" in level_names

    def test_comprehensive_uses_13_word_shingles(self) -> None:
        from kaos_content.dedup.levels.minhash import MinHashLevel

        mh_levels = [lvl for lvl in COMPREHENSIVE.levels if lvl.name == "minhash"]
        assert mh_levels
        assert isinstance(mh_levels[0], MinHashLevel)
        assert mh_levels[0]._shingle_size == 13

    def test_legal_preset_no_fuzzy_binary(self) -> None:
        level_names = [lvl.name for lvl in LEGAL.levels]
        assert "fuzzy_binary" not in level_names
        assert "minhash" in level_names

    def test_comprehensive_includes_semantic_when_plugin_installed(self) -> None:
        """COMPREHENSIVE picks up the kaos-nlp-transformers semantic plugin.

        When kaos-nlp-transformers is installed (always the case in
        this dev environment, see uv.sources), the preset includes a
        level named ``"semantic"``. When it isn't, the preset stays at
        four lexical levels. This guards against typos in the
        try-import path silently dropping the plugin.
        """
        try:
            import kaos_nlp_transformers.clustering  # noqa: F401
        except ImportError:
            pytest.skip("kaos-nlp-transformers not installed")
        level_names = [lvl.name for lvl in COMPREHENSIVE.levels]
        assert "semantic" in level_names, (
            f"expected COMPREHENSIVE to include the semantic plugin level "
            f"when kaos-nlp-transformers is installed; got {level_names}"
        )

    def test_presets_run_without_error(self) -> None:
        docs = [DedupDocument(doc_id="d1", text="hello world this is enough tokens for shingles")]
        for preset in (FAST, STANDARD, COMPREHENSIVE, LEGAL):
            report = DedupPipeline(preset).run(docs)
            assert report.total_input == 1


# ------------------------------------------------------------------
# Custom level integration
# ------------------------------------------------------------------


class TestCustomLevel:
    def test_custom_level_plugs_into_pipeline(self) -> None:
        """Any class implementing DedupLevel works in the pipeline."""

        class AlwaysDupLevel(DedupLevel):
            name: ClassVar[str] = "always_dup"

            def find_clusters(self, documents: list[Any]) -> list[DedupCluster]:
                if len(documents) < 2:
                    return []
                return [
                    DedupCluster(
                        cluster_id="all",
                        canonical_doc_id=documents[0].doc_id,
                        member_doc_ids=tuple(d.doc_id for d in documents),
                        level=self.name,
                        similarity=0.99,
                    )
                ]

        docs = [
            DedupDocument(doc_id="d1", text="a"),
            DedupDocument(doc_id="d2", text="b"),
            DedupDocument(doc_id="d3", text="c"),
        ]
        config = DedupPipelineConfig(levels=(AlwaysDupLevel(),))
        report = DedupPipeline(config).run(docs)
        assert report.total_unique == 1
        assert report.clusters[0].similarity == pytest.approx(0.99)
