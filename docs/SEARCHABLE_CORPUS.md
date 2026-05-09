# SearchableCorpus

Design doc for `kaos_content.indexing.SearchableCorpus` — the
corpus-level analog of `SearchableDocument`. Authored alongside the
KNT-601 0.2.0 cycle of `kaos-nlp-transformers`; targets a
`kaos-content` 0.2.x landing.

## 1. TL;DR

`SearchableCorpus(documents=[ContentDocument, ...])` is the
corpus-level analog of `SearchableDocument`: one shared BM25 index
with corpus-wide IDF, one shared embedding matrix, and AST-grounded
results that carry `doc_index` / `doc_uri` in addition to the existing
`block_ref` / `page` / `section_ref`. It lives in
`kaos_content.indexing` next to `SearchableDocument`, builds the BM25
index eagerly and the embedding matrix lazily, and threads
`(doc_idx, row)` through the `kaos-nlp-core` `Searcher` via per-record
metadata (keeping `block_ref` intact as `external_id`). Replaces the
per-doc fanout pattern that loses corpus-wide IDF.

## 2. Goal & Non-Goals

**Goal.** A pre-built, multi-document index that:

- shares one BM25 inverted index (corpus-wide IDF, one tokenizer pass);
- shares one embedding matrix and one query-embed call per query;
- returns `SearchResult`s that name the source document AND the AST
  node;
- mirrors `SearchableDocument`'s three retrieval modes (`bm25`,
  `embeddings`, `hybrid`) and propagates `model_id` /
  `reranker_model_id`;
- handles the realistic legal workload (1000+ ContentDocuments, 100K–
  1M paragraph rows) without surprising memory or rebuild costs.

**Non-goals.**

- Distributed / on-disk index. In-memory only; serve from a worker.
- Mutable corpus. Add/remove a document = rebuild. ContentDocument is
  frozen, so this is consistent with `SearchableDocument`.
- Cross-corpus federation. Out of scope; do at the application layer.
- Reranker model loading or device management. Stays in
  `kaos-nlp-transformers`. We propagate `reranker_model_id` only.
- AST-aware logic moving down into `kaos-nlp-*`. The package boundary
  forbids it (kaos-content depends on kaos-nlp-*, never the inverse).

## 3. API Sketch

```python
class SearchableCorpus:
    """N ContentDocuments with one shared search index.

    Like SearchableDocument but corpus-wide. Builds a single BM25
    inverted index (so IDF is computed across the whole corpus, not
    per-document) and shares one embedding matrix across all dense
    queries.
    """

    __slots__ = (
        "_documents", "_views", "_level", "_retrieval",
        "_model_id", "_reranker_model_id",
        "_units",                # list[ParagraphUnit | SentenceUnit] flattened
        "_doc_offsets",          # list[int], len = N+1, cumulative row count
        "_doc_uris",             # list[str], len = N
        "_records",              # parallel to _units, dict shape for Searcher
        "_searcher",             # kaos_nlp_core.search.Searcher | None
        "_doc_embeddings",       # np.ndarray | None  (lazy)
        "_section_titles",       # dict[(doc_idx, section_ref) -> title]
        "_heading_paths",        # dict[(doc_idx, section_ref) -> tuple[str,...]]
        "_max_embed_rows",       # int — guardrail
    )

    def __init__(
        self,
        documents: Sequence[ContentDocument],
        *,
        level: Literal["paragraph", "sentence"] = "paragraph",
        retrieval: RetrievalMode = "bm25",
        doc_uris: Sequence[str] | None = None,
        model_id: str | None = None,
        reranker_model_id: str | None = None,
        max_embed_rows: int = 200_000,
    ) -> None: ...

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        preview_length: int = 200,
        rerank_top_k: int | None = None,
        rerank_candidate_k: int = HYBRID_DEFAULT_CANDIDATE_K,
    ) -> SearchResults: ...

    @property
    def documents(self) -> tuple[ContentDocument, ...]: ...
    @property
    def size(self) -> int: ...                # total rows
    @property
    def num_documents(self) -> int: ...
    @property
    def doc_uris(self) -> tuple[str, ...]: ...
    @property
    def retrieval(self) -> str: ...
    @property
    def level(self) -> str: ...

    def doc_for_row(self, row: int) -> tuple[int, ContentDocument]:
        """Map global row index back to (doc_index, document)."""
```

`SearchResult` gains two optional, default-`None` fields (additive,
backwards-compatible — existing single-doc callers see unchanged
behavior):

```python
@dataclass(frozen=True, slots=True)
class SearchResult:
    text: str
    score: float
    block_ref: str
    page: int | None
    section_ref: str | None
    section_title: str | None
    char_start: int | None = None
    char_end: int | None = None
    heading_path: tuple[str, ...] = ()
    # ---- new for SearchableCorpus -------------------------------------
    doc_index: int | None = None
    doc_uri: str | None = None
```

## 4. Design Decisions

### 4.1 Inverted index strategy: ONE corpus-wide index

Build one `kaos_nlp_core.search.Searcher` over the concatenated row
stream from all documents. This is the only choice that gives
corpus-wide IDF — the whole point of the class. Per-document indexes
plus score-merging would be cheaper to rebuild incrementally, but it
breaks BM25 score comparability across documents (a rare term in doc
A gets a different IDF than the same term in doc B). The cost (one
extra list concatenation at build time) is dominated by tokenization,
which runs once either way. **Recommendation: one shared index.**

### 4.2 Row-index threading

Global row index `i` ∈ `[0, N_total)` is the index into the flat
`_units` list. The mapping `i → (doc_idx, doc_row)` is recovered by
binary search into `_doc_offsets` (a length-`N+1` cumulative count).
Through `kaos-nlp-core` we use `external_id_field="block_ref"`
exactly as `SearchableDocument` does, but we extend the per-record
metadata with `doc_index` so the round-trip survives without parsing
strings:

```python
records.append({
    "id": global_row,                                 # dense
    "text": unit.text,
    "block_ref": unit.block_ref,                      # external_id
    "page": unit.page,
    "section_ref": unit.section_ref,
    "doc_index": doc_idx,                             # NEW
    # sentence-level only:
    "char_start": ..., "char_end": ...,
})
searcher = Searcher.from_documents(
    records,
    external_id_field="block_ref",
    metadata_fields=["page", "section_ref", "doc_index", ...],
)
```

`SearchHit.metadata["doc_index"]` plus `SearchHit.doc_id` (== global
row) tells us everything needed to fan out into a `SearchResult`. We
do NOT pack `(doc_idx, row)` into the `external_id` — keep
`block_ref` intact as the AST address, which downstream consumers
(e.g. `search_corpus`'s passage URI builder) already rely on.

### 4.3 Embedding cache scope

Three layers, each correct for its scope:

1. **Process-wide LRU** in `kaos-nlp-transformers`
   (`embedding_cache_size`, keyed on
   `(model_id, revision, blake2b(text))`). Survives across
   `SearchableCorpus` instances. Free; no per-class work needed.
2. **Per-instance dense matrix** `_doc_embeddings` (shape
   `(N_total, dim)`). Computed once on first dense query, reused
   thereafter. Same pattern `SearchableDocument` uses.
3. **No corpus-wide pre-warm cache.** Don't replicate the
   process-wide LRU. The matrix already covers it.

**Recommendation: matrix + process-wide LRU is sufficient. Do not add
a third tier.**

### 4.4 Memory model: BM25 eager, embeddings lazy

- BM25 index: built in `__init__`. Cost is one tokenization pass + one
  Rust-side index build; for 1000 documents at ~200 paragraphs each,
  this is a few seconds and well under 1 GB. Eager is right because
  BM25 is the default `retrieval`.
- Embedding matrix: built lazily on first dense query. For
  N_total = 200K rows × dim=384 × 4 bytes ≈ 300 MB. Acceptable on a
  workstation; must not be paid by callers who only use BM25.
- Guardrail (see 4.7): refuse to build the matrix if
  `N_total > max_embed_rows`.

### 4.5 Result shape: add both `doc_index` and `doc_uri`

`doc_index` is the cheap, unambiguous handle (O(1) into
`self.documents`). `doc_uri` is the human-readable / URL-shaped value
that downstream tooling (e.g. `_searchable_passage_uri`) already keys
on. Both are `Optional` with `None` defaults so existing callers of
`search_document` / `SearchableDocument.search` are untouched.

### 4.6 Rerank candidate pool: corpus-wide BM25 → embedding rerank

In hybrid mode, BM25 picks the top `rerank_candidate_k` from the
corpus-wide index, then the embedding matrix reranks just those rows.
Matches `SearchableDocument._search_hybrid` exactly with a larger
`N`. Per-doc-then-merge fanout is wrong: it forces an arbitrary cap
on how many results can come from any single document, and it loses
the shared-IDF benefit that motivates the shared index.

Default `rerank_candidate_k` stays at
`HYBRID_DEFAULT_CANDIDATE_K = 50`; operators can raise it (e.g.
200) for large corpora where the relevant document is buried.

### 4.7 Scaling guardrails

- BM25-only: bounded by tokenizer + Rust index. Soft-tested up to
  ~1M rows in `kaos-nlp-core`'s benches; no hard cap.
- Dense / hybrid: hard cap via `max_embed_rows` (default 200_000 ≈
  ~300 MB at dim=384). Construction succeeds; the FIRST dense query
  raises a clear `ValueError`:
  > "Corpus has 312_847 rows; refusing to build a (312_847, 384)
  > embedding matrix (~458 MB). Fix: raise max_embed_rows
  > explicitly, or use retrieval='bm25', or chunk the corpus and
  > merge results upstream."
- Empty corpus / all-empty docs: legal; returns empty
  `SearchResults`.

### 4.8 Pickle / serialization: NO

`SearchableCorpus` is **not picklable in v1**. The `Searcher` /
`InvertedIndex` from `kaos-nlp-core` is a Rust extension type whose
pickle support is not guaranteed across releases. Embedding matrices
ARE numpy-picklable but reproducing them from a snapshot is strictly
worse than rebuilding from the (frozen, pickle-clean) ContentDocument
list. Document-level pickle works:
`pickle.dumps((corpus.documents, corpus.level, corpus.retrieval, ...))`
and rebuild on the other side.

If users need persistence later, the right primitive is a
`SearchableCorpus.save(path)` / `.load(path)` pair that snapshots
the documents + the embedding matrix only, and reconstructs the BM25
index on load. Defer to a follow-up.

## 5. Test Plan

All under
`/home/mjbommar/projects/273v/kaos-content/tests/unit/test_searchable_corpus.py`
unless noted. Mark dense / hybrid suites with the existing skipif
guards (`_has_nlp`, `_has_transformers`).

| File | Test | Asserts |
|---|---|---|
| `tests/unit/test_searchable_corpus.py::TestConstruction` | `test_doc_offsets_dense_and_correct` | `_doc_offsets[0] == 0`, `_doc_offsets[-1] == size`, monotonic |
| | `test_size_matches_sum_of_units` | `corpus.size == sum(len(iter_paragraph_units(d)) for d in docs)` |
| | `test_doc_uris_default_to_metadata_source` | uri pulled from `metadata.source.uri` when present |
| | `test_doc_uris_fallback_to_anon` | `"doc:anon-2"` when neither override nor source.uri |
| | `test_invalid_doc_uris_length_raises` | length mismatch raises ValueError |
| `::TestBm25CorpusWideIDF` | `test_corpus_wide_idf_differs_from_per_doc` | a term unique to one doc gets a higher score in the corpus index than in a fanned-out per-doc query (proves shared IDF) |
| | `test_block_ref_round_trips` | every result's `block_ref` lives in the named document |
| | `test_doc_index_present_on_every_result` | `r.doc_index in range(num_documents)` |
| | `test_doc_uri_matches_doc_index` | `r.doc_uri == corpus.doc_uris[r.doc_index]` |
| | `test_results_can_come_from_multiple_docs` | a query hitting both docs returns results from both |
| | `test_doc_for_row_inverse` | `corpus.doc_for_row(global_row)[0] == doc_index` for every result |
| `::TestEmbeddings` (skipif transformers) | `test_lazy_matrix_build` | matrix is `None` before first dense query, populated after |
| | `test_matrix_reuse_across_queries` | second query does not re-embed (assert `_doc_embeddings is same object`) |
| | `test_max_embed_rows_guardrail` | constructing with low cap and querying dense raises ValueError with hint |
| | `test_model_id_propagated` | `model_id="..."` is honored (mock `EmbeddingModel.load`) |
| `::TestHybrid` (skipif transformers) | `test_hybrid_uses_corpus_wide_candidates` | candidate pool is drawn corpus-wide, not per-doc |
| | `test_rerank_top_k_default` | when `rerank_top_k=None`, falls back to `top_k` |
| | `test_reranker_model_id_propagated` | parameter threads through |
| `::TestEmpty` | `test_no_documents` | empty list yields empty `SearchResults` |
| | `test_all_empty_documents` | returns empty results, no crash |
| | `test_mixed_empty_and_nonempty` | only the non-empty doc's rows appear |
| `::TestSentenceLevel` (skipif nlp) | `test_sentence_char_offsets_threaded` | sentence hits carry `char_start` / `char_end` and correct `block_ref` |
| `::TestBackcompat` | `test_searchable_document_unchanged` | `SearchResult(...)` without `doc_index` / `doc_uri` still constructs (defaults `None`) |
| | `test_search_document_results_no_doc_fields` | existing `search_document` API still produces `doc_index=None` |
| `tests/integration/test_searchable_corpus_scale.py` (marked `slow`) | `test_1000_doc_corpus_bm25` | constructs a 1000-doc synthetic corpus, queries, asserts elapsed < 5 s after build |
| | `test_1000_doc_corpus_dense_guardrail` | with default cap, raises; with raised cap and `dim=64` mock, succeeds |

## 6. Open Risks / Limitations

- **Memory cliff at the embedding-matrix boundary.** A 1M-row corpus
  at dim=384 is ~1.5 GB. The `max_embed_rows` cap is the user's seat
  belt; picking a sane default is judgment, not science. 200K is
  conservative.
- **No incremental update.** Adding one document = full rebuild.
  Fine for the legal-research workload (corpora are loaded once and
  queried many times); not fine for streaming pipelines.
- **Reranker integration is propagation-only.** This class accepts
  and exposes `reranker_model_id` but does NOT load a
  `CrossEncoderReranker` itself. The actual reranker step lives in
  upstream consumers (e.g. `kaos-rag` retrieval pipelines) that
  already own reranker lifecycles. Documented as such; no surprise
  behavior change.
- **Tokenizer invariance.** The default `Tokenizer(lowercase=True)`
  from kaos-nlp-core is shared across all docs. A multilingual
  corpus that needs per-document tokenizers is out of scope.
- **`doc_uri` collisions.** If two `ContentDocument`s share a
  `metadata.source.uri`, results disambiguate by `doc_index` only.
  We log a single info-level warning at construction.

## 7. Appendix: Example

```python
from kaos_content.indexing import SearchableCorpus
from kaos_content.parsers import extract_pdf
import glob

docs = [extract_pdf(p) for p in glob.glob("contracts/*.pdf")]

corpus = SearchableCorpus(
    documents=docs,
    level="sentence",
    retrieval="hybrid",
    model_id="BAAI/bge-small-en-v1.5",
)

results = corpus.search("non-compete enforceability in California", top_k=10)
for r in results.results:
    print(f"{r.doc_uri}  p.{r.page}  {r.section_title}: {r.text[:80]}")
    print(f"  -> doc_index={r.doc_index}, block_ref={r.block_ref}")
```
