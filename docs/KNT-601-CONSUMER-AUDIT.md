# KNT-601 Consumer Audit — kaos-content

> Audit run on 2026-05-09 to identify integration gaps between kaos-content
> and kaos-nlp-transformers 0.2.0 (KNT-601: ONNX-via-Rust migration).
> The audit precedes P6.2 / P6.3 / P6.4 work in kaos-content. Findings
> here drive the patches in those phases.

## 1. Summary

kaos-content's `kaos_nlp_transformers` integration is small and well-isolated:
the public surface still works under KNT-601 because `EmbeddingModel.load()`
/ `.embed()` semantics are preserved. There are **two real correctness bugs**
worth fixing in the upcoming P6.2 pass — model-default propagation and
re-loading the model on every call — and several lower-severity ergonomics
/ docs items. **No legacy package names (`fastembed`, `onnxruntime`,
`sentence-transformers`, `EmbeddingRetriever`) appear anywhere in this
repo**, so KNT-601's surface is mostly invisible from this side. The
biggest correctness gap is that `_embed_texts` / `_embed_query` call
`EmbeddingModel.load()` with **no arguments** on every call — there is
no way for callers (including `SearchableDocument`) to inject a non-default
model, and the load runs four times for one hybrid query.

Recommendation: fix model-id propagation + cache the loaded model, then
thread `count_tokens` and `max_seq_len` into `SectionChunker` so chunk
caps line up with the embedding model's window.

## 2. Findings table (sorted by severity)

| File:line | Severity | What | Recommended fix |
|---|---|---|---|
| `kaos_content/search.py:589-591` | **High** | `_embed_texts` hard-codes `EmbeddingModel.load()` with no model_id arg; no way for caller to specify a model. The P6 design plumbs `retrieval=` through but not the model_id. | Add `model_id: str \| None = None` (and matching backend / device hints if useful) parameter to `_embed_texts`, `_embed_query`, `_search_embeddings`, `_search_hybrid`, `_build_search_records`, `search_document`, `SearchableDocument.__init__`. Plumb to `EmbeddingModel.load(model_id=model_id)`. |
| `kaos_content/search.py:603-605` | **High** | Same as above for the query embedder. | Same fix; share one helper. |
| `kaos_content/search.py:581-606` and `kaos_content/indexing.py:193-208` | **High** | Each `_embed_texts` / `_embed_query` call runs `EmbeddingModel.load()` again. A single `_search_hybrid` call loads the model **4×** (texts via `_embed_texts`, query via `_embed_query`, candidates via `_embed_texts` again, query again). Even with the new `ort` backend's faster init this is wasteful, and worse, each `load()` re-allocates the cdylib session. | Module-level `lru_cache(maxsize=4)` keyed on model_id (None ⇒ default) returning a loaded `EmbeddingModel`; reuse across `_embed_texts` / `_embed_query`. `SearchableDocument` should also keep a per-instance handle so the cache key isn't strictly necessary on the hot path. |
| `kaos_content/search.py:909` | **Medium** | `search_corpus` is declared `async def` but contains zero `await` and zero thread-pool offload — the entire BM25 + embedding workload runs on the event loop thread. With the new ort backend the heavy embedding call now genuinely releases the GIL (KNT-601 #5), so this is the moment to wrap each per-document `sdoc.search(...)` in `asyncio.to_thread` (or run them concurrently via `asyncio.gather`). | Wrap the per-doc `sdoc.search(...)` body in `asyncio.to_thread(...)` and gather. Optionally reuse a single `EmbeddingModel` across docs to amortize load. |
| `kaos_content/chunking/section_chunker.py:43-51, 213-290` | **Medium** | `SectionChunker.max_chars` is a character cap, but the only meaningful budget for an embedding model is **tokens**. `SearchableDocument.chunks()` (`indexing.py:421-445`) passes the same `max_chars` straight through, so a chunker fed into the dense path can produce chunks longer than `EmbeddingModel.max_seq_len` (KNT-601 #2 added that property and `count_tokens(...)`). The model will silently truncate on the Rust side. | Add `max_tokens: int \| None = None` to `SectionChunker` (and from `SearchableDocument.chunks`) using `EmbeddingModel.count_tokens(...)` to enforce the budget when an embedding model is in scope. Keep `max_chars` as the cheap default. |
| `kaos_content/search.py:909-1013` (`search_corpus` dict-mode) | **Medium** | When passed `dict[str, str]` it builds a fresh `SearchableDocument` per `(uri, text)`. With `retrieval="embeddings"` or `"hybrid"`, this re-loads the embedding model **per document** AND embeds each tiny one-paragraph corpus separately rather than batching. For a 100-doc dict corpus this is 100 model loads + 100 1-row embedding calls. Provenance loss: the temporary documents drop `block_ref` since each builder produces only `#/body/0` — `_searchable_passage_uri` papers over this with the special-case at line 1027 (`block_ref != "#/body/0"`), so the `block_ref` field on the returned `RetrievalResult.metadata` is uniformly `#/body/0` for dict-mode hits. | (a) Share one `EmbeddingModel` across the loop, (b) batch all dict-mode texts into one `embed()` call before scoring, (c) document the dict-mode `block_ref` shape explicitly. |
| `kaos_content/search.py:14-17, 61-62, 584` | **Low** | Comments / docstrings still pin the embedding model name (`bge-small-en-v1.5`) and the `~1ms` per-text latency claim. These were calibrated against fastembed and are no longer authoritative under ort; the hard-coded model name will mislead callers when KNT-601's `EmbeddingModel.load(model_id=...)` exposes other models. | Drop the specific model name from the docstrings, or wrap it as "default model (currently bge-small-en-v1.5; see kaos-nlp-transformers for the active default)". Update the `~1ms` figure or remove it. |
| `kaos_content/search.py:233-240` | **Low** | `_ensure_transformers_available` install hint mentions `[transformers]` extra of kaos-content; per `pyproject.toml:54-65` that extra was **stripped** at v0.1.0a1 and not yet re-added. The hint is correct that the package is needed, but the `pip install kaos-content[transformers]` alternative does not work on the current alpha. | Either re-add the `[transformers]` extra in pyproject.toml or remove that bullet from the hint until it does. |
| `kaos_content/search.py:581` | **Low** | `_embed_texts(texts: list[str])` annotation forces a list. KNT-601 #2 says `embed()` now accepts `Iterable[str]`. | Widen the parameter type. Consider passing a generator from `_search_embeddings` and `_search_hybrid`. |
| `kaos_content/search.py:592-593` | **Low** | `_embed_texts` accesses `model.dim` to size the empty-input fallback. If the property name ever shifts (`dimension`, `embed_dim`), this breaks silently. | Defensively pull from `np.asarray(model.embed([""])).shape[-1]` in a one-time fallback, or just return `np.zeros((0, 0))` when there are no texts. |
| `kaos_content/search.py:1025-1031` | **Low** | `_searchable_passage_uri` builds `passage_uri` for SearchableDocument hits — the special case `block_ref != "#/body/0"` exists because dict-mode `search_corpus` always produces a single-block document. This is a provenance hazard: a real one-paragraph document also yields `#/body/0`, and its passage URI silently degrades to a `#h<text_hash>` fallback rather than the actual block_ref. | Mark synthetic docs with a metadata sentinel so `_searchable_passage_uri` can distinguish "synthetic single-block" from "legitimate first-block hit". |
| `kaos_content/dedup/presets.py:33-39` | **Low** | The plugin-import path for `SemanticDedupLevel` is fine under KNT-601 (the API is unchanged), but the comment at line 29 ("Wave 3 sibling — not on PyPI at v0.1.0a1") is stale once kaos-nlp-transformers 0.2.0 ships. | Update the comment when the dependency is re-pinned in pyproject.toml. |
| `kaos_content/search.py:599-601` | **Low** | `_embed_query` materializes the query through `model.embed([query])[0]` and then `np.linalg.norm`. Identical pattern repeated in three call sites. | Refactor into one normalized embed helper used by both. |
| `README.md:48, 187` | **Low** | README mentions `kaos-nlp-transformers` for "Dense embeddings + retrieval"; once KNT-601 lands, also worth noting that the package is now pure-Rust under the hood (no Python `onnxruntime` dep) so `pip install kaos-nlp-transformers` no longer drags 100MB of ORT wheels. | Optional one-sentence update in the optional-extras table when shipping kaos-content 0.1.0a2. |
| `kaos_content/dedup/levels/minhash.py:97-101` | **Low** (false positive) | `tokens = text.lower().split()` is a hand-rolled tokenizer — checked whether KNT-601's `EmbeddingModel.count_tokens` should replace it. **Verdict: no.** MinHash is a lexical hashing pipeline, not embedding-aware; whitespace shingling is correct here. | None. Flagged only so future audits don't re-walk. |

## 3. Per-finding detail (high & medium)

### H-1, H-2: Hard-coded model defaults in `_embed_texts` / `_embed_query`

`kaos_content/search.py:589` and `:603` both call `EmbeddingModel.load()` with
**no arguments**. The `retrieval=` argument plumbed all the way from
`search_document(..., retrieval="embeddings")` into `_search_embeddings` is
the *mode*, not the *model*. There is no public surface to say "use
`intfloat/e5-large-v2` instead of the default" without monkeypatching
kaos-nlp-transformers' default. Worse, when the default flips on a
kaos-nlp-transformers minor release, every kaos-content user picks it up
silently — there's no pin in the consumer.

**Fix sketch.** Add `model_id: str | None = None` to the public APIs
(`search_document`, `SearchableDocument.__init__`, `search_corpus`) and
thread it through `_embed_texts(model_id)` / `_embed_query(model_id)`.
Pass to `EmbeddingModel.load(model_id=model_id)` (None ⇒ kaos-nlp-transformers
default). Add a one-paragraph note to the `SearchableDocument` docstring:
"If you change `model_id` between calls, the cached embedding matrix
becomes stale — construct a new `SearchableDocument`."

### H-3: Repeated `EmbeddingModel.load()` per call

`_embed_texts` and `_embed_query` are called once per query, and
`_search_hybrid` calls each of them twice. That's **four
`EmbeddingModel.load()` calls for one hybrid query**. Even though
`SearchableDocument` caches the corpus embedding matrix
(`_doc_embeddings`), it still calls `_embed_query` on every search,
which re-loads the model. KNT-601's Rust backend has faster init than
fastembed but model load is still non-trivial — it has to mmap the
ONNX file and bind the cdylib session.

**Fix sketch.** Module-level `functools.lru_cache(maxsize=4)` on a
`_get_model(model_id)` helper. The cache key is `(model_id,)` (None
for default). All three call sites go through it. `SearchableDocument`
can additionally keep a `self._model` slot to avoid the lookup on
every query.

### M-1: `search_corpus` is async-shaped but synchronous

`search_corpus` (`search.py:909`) is `async def` but the body has no
`await` and no thread offload. With KNT-601 #5 (free-threaded Python
now works, GIL released during embed), this is the moment to *actually*
parallelize per-document search:

```python
search_tasks = [
    asyncio.to_thread(sdoc.search, query, top_k=top_k, preview_length=0, rerank_top_k=rerank_top_k)
    for _uri, sdoc in indexed_docs
]
results_per_doc = await asyncio.gather(*search_tasks)
```

This gives a real concurrency win on the embedding path: 10 docs × 50ms
embed each becomes ~50ms wall instead of 500ms.

### M-2: Token-aware chunking via `count_tokens` / `max_seq_len`

`SectionChunker` (`chunking/section_chunker.py:43-51`) takes `max_chars`
only. `SearchableDocument.chunks()` (`indexing.py:421-445`) passes that
through. When the chunked output is fed back into the dense retrieval
path (chunk → embed → index), chunks longer than the model's
`max_seq_len` get silently truncated by the embedding model. Pre-KNT-601
there was no Python API to ask the model "how many tokens is this?";
KNT-601 #2 added `EmbeddingModel.count_tokens(texts) -> list[int]` and
the `max_seq_len` property. Now the chunker can enforce the right budget.

**Fix sketch.** Add `max_tokens: int | None = None` to
`SectionChunker.__init__`. When set, the `_enforce_max_chars` discipline
gains a sibling pass: after building each chunk, call
`model.count_tokens([chunk_text])[0]`; if over budget, split at sentence
boundaries (the existing `_split_paragraph_at_sentences` machinery
handles this). Default `max_tokens=None` keeps the cheap char-only path.

### M-3: dict-mode `search_corpus` re-loads the embedding model per document

When called as `search_corpus({"uri1": text1, ...}, retrieval="embeddings")`,
the loop at `search.py:959-972` builds one temporary `SearchableDocument`
per dict entry. Each constructor with `retrieval="embeddings"` triggers
`_ensure_transformers_available`, then `.search()` triggers `_embed_query`
and `_ensure_doc_embeddings()` — every one of those re-loads the model
under H-3.

Compounding this, the synthetic documents always have `block_ref="#/body/0"`
(single-paragraph builder). `_searchable_passage_uri` (`search.py:1027`)
special-cases this with `if block_ref and block_ref != "#/body/0"`, but a
*legitimate* corpus with one-paragraph documents falls through the same
special case — provenance is lost identically for both.

**Fix sketch.** (a) Once H-3 is fixed, the dict-mode loop benefits
automatically. (b) Mark synthetic docs with a
`metadata.extra["_kaos_synthetic_corpus"] = True` flag and check that
flag in `_searchable_passage_uri`. (c) Better still: short-circuit
dict-mode entirely — skip the `SearchableDocument` round-trip and embed
all texts in one batched call.

## 4. NOT findings (verified clean)

- `kaos_content/dedup/presets.py:33-39` — `SemanticDedupLevel` plugin
  import path is unaffected by KNT-601. Keep as-is.
- `kaos_content/dedup/levels/__init__.py:3-7` — docstring stable surface,
  no update needed.
- `kaos_content/dedup/levels/minhash.py:97-101` — `text.lower().split()`
  is a whitespace tokenizer for MinHash shingling; lexical by design,
  do **not** replace with `count_tokens`.
- `kaos_content/dedup/levels/fuzzy_binary.py:61` — uses
  `kaos_nlp_core.hashing.CTPH`, not `kaos_nlp_transformers`. Out of scope.
- `kaos_content/structure.py:95` — `from kaos_nlp_core.structure import
  label_lines`. Not transformers; no change.
- `kaos_content/chunking/section_chunker.py:302` — `from
  kaos_nlp_core.segmentation import segment_sentences`. Not transformers.
- `kaos_content/units.py` — pure AST enumeration, no NLP-package imports.
- `kaos_content/corpus.py` — Protocol + thin adapter; no NLP-package
  imports beyond a docstring reference.
- `kaos_content/indexing.py:489` — `from kaos_nlp_core.structures import
  SpanIndex` (in `AnnotationIndex._ensure_built`). Not transformers.
- **No imports of `onnxruntime` anywhere** (verified via grep).
- **No imports of `fastembed` / `sentence_transformers` /
  `EmbeddingRetriever`** (verified via grep). No code path will break on
  KNT-601's deprecation.
- **No reads of `KAOS_NLP_TRANSFORMERS_OFFLINE` or `HF_HUB_OFFLINE`**
  (verified via grep). kaos-content does not set or check offline-mode
  env vars.
- **No string-comparison branches on `backend_name`** (verified). kaos-content
  never inspects which backend is active.
- **No reads of `SystemDevices.onnx_providers`** (verified). The
  capabilities-introspection path was only ever in kaos-nlp-transformers.
- **No references to `_check_gil_enabled`** (verified).
- **`SearchResult` dataclass** (`search.py:73-107`) preserves `block_ref`,
  `page`, `section_ref`, `section_title`, `char_start`, `char_end`,
  `heading_path`. `_records_to_search_results` (`search.py:534-578`) and
  `_results_to_search_results` (`search.py:308-348`) both populate every
  field. Provenance is **not lost** in the dense / hybrid path within a
  single document. The only provenance hazard is the dict-mode
  `search_corpus` synthetic-document case (M-3 above).
- **`SearchableDocument._search_hybrid`** correctly slices the cached
  embedding matrix (`indexing.py:368: cand_vecs = doc_vecs[candidate_indices]`)
  rather than re-embedding candidates. Already efficient.
- **`SearchableDocument._search_embeddings`** correctly reuses the cached
  doc embedding matrix; only the query is re-embedded per call.

## 5. Top-5 250-word summary

1. **Hard-coded `EmbeddingModel.load()` with no model_id argument**
   (`search.py:589`, `:603`) — the only way to use a non-default
   embedding model is monkeypatching kaos-nlp-transformers. The
   `retrieval=` argument plumbed through `search_document` and
   `SearchableDocument` is the *mode*, not the *model*. P6.2 needs to
   thread a `model_id: str | None` parameter from the public API down
   through `_embed_texts` / `_embed_query`. (High)

2. **`EmbeddingModel.load()` is called four times per hybrid query** —
   `_embed_texts` and `_embed_query` each load on every call, and
   `_search_hybrid` calls them twice apiece. Even with the new ort
   backend's faster init, this re-mmaps the ONNX session and re-binds
   the cdylib each time. Add an `lru_cache` on a `_get_model(model_id)`
   helper. (High)

3. **`async def search_corpus` performs zero `await` and no thread
   offload** (`search.py:909`) — with KNT-601's free-threaded Python
   support and GIL-releasing Rust embed, per-document searches can
   finally run concurrently. Wrap each `sdoc.search(...)` in
   `asyncio.to_thread` and `gather`. (Medium)

4. **`SectionChunker` is char-budgeted but feeds an embedding pipeline**
   — chunks longer than `EmbeddingModel.max_seq_len` get silently
   truncated. KNT-601 #2 added `count_tokens` / `max_seq_len`; add a
   `max_tokens` knob to `SectionChunker` and `SearchableDocument.chunks`.
   (Medium)

5. **dict-mode `search_corpus` re-loads the embedding model per document
   and loses block_ref provenance** — N temporary `SearchableDocument`s
   = N model loads; all synthetic docs share `block_ref="#/body/0"`,
   indistinguishable from real first-paragraph hits in
   `_searchable_passage_uri`. Mark synthetic docs with a metadata flag
   and batch the embed call. (Medium)
