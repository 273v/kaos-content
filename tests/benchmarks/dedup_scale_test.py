"""FUND-11 scale benchmark: 100K records with controlled duplication + mutation.

Loads kl3m legal pairs, takes the `positive` (passage) field from up to 100K
records, injects exact duplicates and near-duplicates (1% char-level mutation),
then runs the full dedup pipeline and reports precision/recall/throughput.

Test matrix:
- 50K unique records (base)
- 25K exact duplicates (randomly sampled from the base)
- 25K near-duplicates (1% character mutation of randomly sampled base records)
Total: 100K records, 50% duplication rate

Expected results:
- text_hash should catch all 25K exact dups (recall=1.0)
- minhash should catch most 25K near-dups (recall>0.9 at Jaccard=0.8)
- Overall: 50K unique → dedup_rate ~0.50

Run::

    uv run python tests/benchmarks/dedup_scale_test.py
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

KL3M_TRAIN = Path(__file__).parent.parent.parent.parent / (
    "scripts/embedding-finetune/output/kl3m-legal-pairs-v1/train.jsonl"
)

N_UNIQUE = 50_000
N_EXACT_DUP = 25_000
N_NEAR_DUP = 25_000
MUTATION_RATE = 0.01  # 1% char-level mutation


def _load_passages(path: Path, limit: int) -> list[str]:
    """Load `positive` field from kl3m training pairs."""
    passages: list[str] = []
    with path.open() as f:
        for line in f:
            if len(passages) >= limit:
                break
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = rec.get("positive", "")
            if len(text) > 50:
                passages.append(text)
    return passages


def _mutate(text: str, rate: float, rng: random.Random) -> str:
    """Randomly replace `rate` fraction of characters."""
    chars = list(text)
    n_mutations = max(1, int(len(chars) * rate))
    positions = rng.sample(range(len(chars)), min(n_mutations, len(chars)))
    alphabet = "abcdefghijklmnopqrstuvwxyz "
    for pos in positions:
        chars[pos] = rng.choice(alphabet)
    return "".join(chars)


def main() -> None:
    if not KL3M_TRAIN.exists():
        print(f"ERROR: kl3m training data not found at {KL3M_TRAIN}")
        print("This benchmark requires the kl3m-legal-pairs dataset.")
        return

    print(f"Loading {N_UNIQUE} unique passages from {KL3M_TRAIN.name}...")
    t0 = time.perf_counter()
    base_passages = _load_passages(KL3M_TRAIN, N_UNIQUE)
    if len(base_passages) < N_UNIQUE:
        print(f"WARNING: only {len(base_passages)} passages available, need {N_UNIQUE}")
        return
    print(f"  Loaded {len(base_passages)} passages in {time.perf_counter() - t0:.1f}s")

    rng = random.Random(42)

    from kaos_content.dedup import DedupDocument, DedupPipeline
    from kaos_content.dedup.levels import MinHashLevel, TextHashLevel
    from kaos_content.dedup.pipeline import DedupPipelineConfig

    # Build the 100K dataset
    print("Building 100K document set...")
    documents: list[DedupDocument] = []

    # 50K unique
    for i, text in enumerate(base_passages):
        documents.append(DedupDocument(doc_id=f"base_{i}", text=text))

    # 25K exact duplicates
    exact_dup_sources: list[int] = []
    for i in range(N_EXACT_DUP):
        src_idx = rng.randint(0, N_UNIQUE - 1)
        exact_dup_sources.append(src_idx)
        documents.append(
            DedupDocument(
                doc_id=f"exact_dup_{i}",
                text=base_passages[src_idx],
            )
        )

    # 25K near-duplicates (1% mutation)
    near_dup_sources: list[int] = []
    for i in range(N_NEAR_DUP):
        src_idx = rng.randint(0, N_UNIQUE - 1)
        near_dup_sources.append(src_idx)
        mutated = _mutate(base_passages[src_idx], MUTATION_RATE, rng)
        documents.append(
            DedupDocument(
                doc_id=f"near_dup_{i}",
                text=mutated,
            )
        )

    print(f"  Total documents: {len(documents)}")
    print(f"  Unique base: {N_UNIQUE}")
    print(f"  Exact duplicates: {N_EXACT_DUP}")
    print(f"  Near-duplicates (1% mutation): {N_NEAR_DUP}")
    print()

    # Pipeline: text_hash (catches exact) + minhash (catches near-dup)
    config = DedupPipelineConfig(
        levels=(
            TextHashLevel(lowercase=True),
            MinHashLevel(shingle_size=5, num_perms=128, threshold=0.8),
        ),
        short_circuit=True,
    )
    pipeline = DedupPipeline(config)

    print("Running dedup pipeline...")
    t0 = time.perf_counter()
    report = pipeline.run(documents)
    elapsed = time.perf_counter() - t0

    print(f"\n{'=' * 60}")
    print(f"RESULTS ({elapsed:.1f}s)")
    print(f"{'=' * 60}")
    print(f"Total input:      {report.total_input:,}")
    print(f"Total unique:     {report.total_unique:,}")
    print(f"Total duplicates: {report.total_duplicates:,}")
    print(f"Dedup rate:       {report.dedup_rate:.1%}")
    print(f"Throughput:       {report.total_input / elapsed:,.0f} docs/sec")
    print()

    for level_name, stats in report.per_level_stats.items():
        print(f"  {level_name}: {stats['clusters']} clusters, {stats['docs_deduped']} docs removed")

    # Precision/recall analysis
    # Ground truth: exact_dup_{i} is a duplicate of base_{exact_dup_sources[i]}
    #               near_dup_{i} is a near-dup of base_{near_dup_sources[i]}
    clustered_ids = set()
    for cluster in report.clusters:
        for doc_id in cluster.duplicate_doc_ids:
            clustered_ids.add(doc_id)

    exact_tp = sum(1 for i in range(N_EXACT_DUP) if f"exact_dup_{i}" in clustered_ids)
    near_tp = sum(1 for i in range(N_NEAR_DUP) if f"near_dup_{i}" in clustered_ids)
    base_fp = sum(1 for i in range(N_UNIQUE) if f"base_{i}" in clustered_ids)

    exact_recall = exact_tp / N_EXACT_DUP if N_EXACT_DUP else 0
    near_recall = near_tp / N_NEAR_DUP if N_NEAR_DUP else 0

    # FP: base docs that got clustered as duplicates of OTHER base docs
    # (Some base docs might legitimately be similar — real FP rate)
    print(f"\nExact-dup recall:  {exact_tp}/{N_EXACT_DUP} = {exact_recall:.1%}")
    print(f"Near-dup recall:   {near_tp}/{N_NEAR_DUP} = {near_recall:.1%}")
    print(f"Base docs clustered (includes legitimate near-dups): {base_fp}")
    print(f"\nThroughput: {report.total_input / elapsed:,.0f} docs/sec ({elapsed:.1f}s total)")


if __name__ == "__main__":
    main()
