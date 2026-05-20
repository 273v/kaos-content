# Changelog

All notable changes to `kaos-content` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-05-20

### Changed — WU-L of 0.1.0 GA plan

- 0.1.0 GA — WU-L of the 0.1.0 GA plan. First stable release of
  `kaos-content`. The public API is frozen for the 0.1.x line: no
  breaking changes will land until 0.2.0. Runtime `kaos-core` pin
  raised from `>=0.1.0rc1,<0.2` to `>=0.1.0,<0.2`. `[nlp]` extra +
  dev-group `kaos-nlp-core` raised from `>=0.1.0a1,<0.2` to
  `>=0.1.0,<0.2`. `[transformers]` extra + dev-group
  `kaos-nlp-transformers` raised from `>=0.1.0a7,<0.2` to
  `>=0.1.0rc1,<0.2` (rc1 is on PyPI; 0.1.0 follows in this same WU-L
  wave and will resolve correctly). No source changes vs 0.1.0rc1.


## [0.1.0rc1] — 2026-05-20

WU-J of the 0.1.0 GA plan
(`kaos-modules/docs/plans/2026-05-20-0.1.0-ga-plan.md`): release
candidate; pin floor raised to `>=0.1.0rc1,<0.2` across kaos-* deps;
freezes the public API surface ahead of GA. No source changes vs
0.1.0a12.

### Changed

- Runtime `kaos-core` pin raised to `>=0.1.0rc1,<0.2` (was
  `>=0.1.0a10,<0.2`). The `<0.2` ceiling protects against legacy
  `0.2.0a*` lines leaking into resolution.
- `[transformers]` extra pin raised to
  `kaos-nlp-transformers>=0.1.0a7,<0.2` (was `>=0.2.0a2,<0.3`).
  The 0.2.0a* line is retired per the kaos-nlp-transformers
  0.1.0a7 release notes (WU-F.8 / GA decision #1 — stay on 0.1.x);
  the `<0.2` ceiling matches.

### TODO

- The `[transformers]` extra pin floor will tighten to
  `kaos-nlp-transformers>=0.1.0rc1,<0.2` in a follow-up patch once
  kaos-nlp-transformers ships its own 0.1.0rc1 later in WU-J.

## [0.1.0a12] — 2026-05-18

### Added

- **`numbering_label: str | None` on `Paragraph`, `Heading`, and
  `ListItem`.** Carries the rendered visible numeral from the source
  document (e.g. `"11."`, `"(a)"`, `"Section 11."`, `"11(a)(i)"`) so
  serializers and downstream consumers can emit / cite the exact label
  an attorney sees on the page. Word's auto-numbering stores the
  visible numeral as `numbering.xml` + `numPr` + a running counter
  rather than inline in the run text; the kaos-office DOCX reader will
  resolve the counter and populate this field in a follow-up release.
  `DocumentBuilder.heading()`, `.paragraph()`, and `.begin_list_item()`
  accept a keyword-only `numbering_label` argument. Default `None`
  preserves existing behavior for AST-constructed documents and for
  legacy JSON payloads serialized before this field existed.
  See `kaos-modules/docs/plans/docx-numbering-resolution.md` for the
  full design.

### Changed

- **All three core serializers (`serialize_text`,
  `serialize_markdown`, `serialize_html`) honor
  `numbering_label`.** When set on `Paragraph`, `Heading`, or
  `ListItem`, the rendered label is prepended verbatim to the
  block's body:
  - text / markdown: `"11. GOVERNING LAW. ..."`,
    `"## Section 11. GOVERNING LAW"`,
    `"(a) Subject to..."`.
  - markdown ordered / bullet lists: the per-item label replaces the
    position-based marker. Items without a label keep position-based
    numbering (back-compatible for AST-constructed documents).
  - HTML: the label is emitted both as a `data-numbering-label="..."`
    attribute and as inline visible text on the `<p>` / `<h*>` /
    `<li>`. Labels are HTML-escaped so they cannot be an injection
    vector. The inline text is required because default CSS does not
    surface the data-attribute — without it an attorney's
    "Section 11(a)(i)" citation diverges from the rendered page.

Stage 4 of
`kaos-modules/docs/plans/docx-numbering-resolution.md`. New test
suite `tests/unit/test_serializer_numbering_label.py` locks the
emission shape across all three formats and asserts the XSS-safe
escape on HTML.

## [0.1.0a11] — 2026-05-18

### Added

- **`DocumentView.block_path(block_ref) -> tuple[str, ...]`** — public
  method returning the chain of enclosing heading texts for a block,
  root-first and INCLUDING the nearest containing section's heading
  text. Empty tuple is the explicit "no enclosing heading" contract
  (preamble, no-headings document, or unknown ref) and downstream
  agents must NOT invent section identifiers for blocks with empty
  path. Solves the "agent fabricates section numbers" failure mode
  observed in the kaos-ui SPA 2026-05-18 persona matrix where every
  numeric "Section N" claim was confidently wrong despite verbatim
  clause text being correct. See
  `kaos-modules/docs/plans/persona-matrix-followups.md` §4.
- **`SearchResult.path: tuple[str, ...]`** — new field on every search
  hit carrying the full structural breadcrumb (root-first, INCLUDING
  the immediate section). Equivalent to
  `heading_path + (section_title,)` when `section_title` is set,
  `heading_path` otherwise. Hits in document preamble have empty
  `path`. The `path` value is identical to
  `DocumentView.block_path(hit.block_ref)` — search and view share one
  structural truth. Existing `heading_path` (ancestors only) and
  `section_title` (immediate section) fields are unchanged and remain
  the documented surface for callers that want them separately.
- **`kaos-content-search-document` MCP tool result dicts** now include
  `path: list[str]` per hit alongside the existing `block_ref` / `page`
  / `section_ref` / `section_title` fields. Empty list signals no
  enclosing heading.
- **`kaos-content-context-window` MCP tool result** now includes a
  top-level `path: list[str]` describing the section the target block
  lives in. Empty list signals no enclosing heading.

### Changed

- **sdist now includes `AGENTS.md`, `CODE_OF_CONDUCT.md`,
  `CONTRIBUTING.md`, and `docs/standards/*.md`.** These are the
  canonical contributor-facing docs referenced from the project root.
  Previously they were tracked in git but excluded from the published
  sdist, which broke distro packagers and source-built installs that
  needed them for context. `.pre-commit-config.yaml` and `uv.lock`
  remain excluded by design (dev infra / application lockfile).
  `check-manifest` now passes; the hatch sdist include-list remains the
  source of truth.
- **Private NDA integration tests now look at `$KAOS_CONTENT_NDA_DIR`
  instead of a maintainer-local default path.** `tests/integration/
  test_summary_real_ndas.py`, `test_corpus_tools_real_ndas.py`, and
  `test_entity_filters_real_ndas.py` previously hard-coded
  `~/projects/273v/kelvin-app/samples/docx/`. Without the corpus the
  suites already skipped, but the path-as-default leaked the
  maintainer's layout into the public sdist. The env var lets external
  contributors point at any directory of `MNDA*.docx` files; absent
  the env var the tests skip with a self-explanatory reason. Also
  removed an absolute local path from `docs/SEARCHABLE_CORPUS.md`.

### Fixed

- **`SearchResult.path` is now populated by `search_tabular()` and
  forwarded by `search_corpus()`.** The 0.1.0a11 breadcrumb work
  introduced the canonical `path` field but `search_tabular()` set
  `section_title` while leaving `path=()`, and `search_corpus()` only
  forwarded `heading_path` / `section_title` to its
  `RetrievalResult.metadata`. Empty `path` is the documented contract
  for "no structural identifier available", so downstream agents
  refused to cite the column / section even though the data was
  present. `search_tabular()` now sets `path=(section_title,)`,
  `search_corpus()` includes `path` in its metadata dict, and the
  `kaos-content-search-tabular` MCP tool's JSON output gains
  `path: list[str]` to match `kaos-content-search-document`.
- **`kaos_content.artifacts.load_tabular()` now applies the same
  mime-type guard as `load_document()`.** Passing an HTML / XML
  artifact previously produced an opaque `JSONDecodeError` after the
  body had already been read. The guard checks `manifest.mime_type`
  before reading bytes (rejecting anything outside
  `{"application/json", "application/x-ndjson"}`) and, when the
  manifest carries no mime hint, sniffs the first non-whitespace byte
  for `<` to catch untyped HTML/XML. The new error is
  `ArtifactMimeTypeError` with an agent-friendly hint pointing at
  `store_tabular()`, matching the load_document parity from 0.1.0a9.

## [0.1.0a10] — 2026-05-17

### Changed

- **kaos-core floor raised to `>=0.1.0a10`** to pick up the URI
  contract redesign (bare names route through
  `context.default_vfs_namespace`; `file://` and `vfs://` schemes).
  See `kaos-modules/docs/plans/uri-contract-redesign.md`. No
  internal callers in `kaos-content` pass synthetic bare names to
  `resolve_input_path`, so the contract change is a pass-through.

## [0.1.0a9] — 2026-05-17

### Fixed

- **`kaos_content.artifacts.load_document()` now rejects non-JSON
  artifacts up front with an agent-friendly hint instead of leaking a
  pydantic JSON-decode error.** Previously, passing an HTML artifact
  (e.g. an unparsed web page) produced `Invalid JSON: expected value
  at line 2 column 1 [type=jso...` — useless to the LLM and a real
  source of agent self-loop failures observed in
  single-user-chat. The library now checks `manifest.mime_type` before
  validation and rejects anything outside
  `{"application/json", "application/x-ndjson"}`; when the manifest
  has no mime hint, a first-byte sniff for `<` catches HTML / XML
  payloads after the bytes are read. A new
  `kaos_content.errors.ArtifactMimeTypeError` carries the offending
  artifact id + mime and the message points at the right preparatory
  tool (`kaos-content-parse-html` for HTML,
  `kaos-content-parse-markdown` for Markdown / text) so the agent can
  self-correct on its next turn. Stage 5.2 of the
  vfs-blind-tools-audit-and-fix plan. kaos-mcp's resource adapter
  already guarded at the MCP boundary; this hardens the library
  itself so in-process callers (kaos-agents tools that call
  `load_document` directly) get the same protection.


## [0.1.0a8] — 2026-05-16

### Changed

- **Lockfile refreshed to track the rest of the kaos-* org.** Pinned
  development versions of sibling packages updated to today's
  releases — `kaos-core` 0.1.0a4 → 0.1.0a7, `kaos-nlp-core`
  0.1.0a1 → 0.1.0a8, `kaos-nlp-transformers` 0.2.0a3 → 0.2.0a8,
  `kaos-office` 0.1.0a2 → 0.1.0a4. The base-install `kaos-core`
  constraint (`>=0.1.0a4,<0.2`) is unchanged; published metadata is
  identical to 0.1.0a7. No public API, parser, serializer, MCP
  schema, or JSON output changes.


## [0.1.0a7] — 2026-05-15

### Fixed

- **Three array parameters across the MCP tool catalog now declare
  their element types.** Each was previously `type=array` with no
  `items`, which OpenAI's strict JSON Schema validator rejected
  with HTTP 400 `invalid_function_parameters` — the whole tool
  catalog for the turn was lost.
  - `kaos-content-corpus-cluster.documents` now declares
    `items: {type: "object", properties: {doc_id, text},
    required: ["doc_id", "text"]}` so the LLM populates each
    record with the correct shape on the first try.
  - `kaos-content-summarize-corpus.artifact_ids` and
    `kaos-content-corpus-narrow.artifact_ids` now declare
    `items: {type: "string"}`.
  kaos-core 0.1.0a7's defensive `items: {}` floor is belt +
  suspenders.

## [0.1.0a6] — 2026-05-11

### Added

- **``DocumentSummary`` value type + deterministic builder.** Cheap,
  no-LLM summary attached to ``ContentDocument`` for corpus-scale
  triage. Fields: ``head_tokens`` (first ~500 tokens verbatim),
  ``top_ngrams`` (50 most-frequent 1–3-grams after stopword removal),
  ``bottom_ngrams`` (50 rare-but-recurring n-grams — the distinctive
  fingerprint), ``char_length``, ``sentence_count``,
  ``paragraph_count``, ``entity_counts`` (per-type sentence counts),
  ``schema_version``. Built via ``kaos_content.summarize.build_document_summary``;
  uses kaos-nlp-core's Rust-backed ``Tokenizer`` and
  ``FrequencyVocabulary`` for tokenisation + counting (single FFI hop).
  ``ContentDocument`` gains a ``summary: DocumentSummary | None`` field,
  default ``None`` for backward compat. (K1; files:
  ``kaos_content/model/document.py``, ``kaos_content/model/summary.py``,
  ``kaos_content/summarize/__init__.py``, ``kaos_content/summarize/stopwords.py``.)
- **Typed-entity sentence/paragraph filter primitives.** Free
  functions over ``DocumentView`` that yield each sentence/paragraph
  containing a typed entity match: ``sentences_with_dates`` /
  ``money`` / ``percents`` / ``durations`` / ``numbers`` (plus
  paragraph-level counterparts). ``EntityMatch`` carries the typed
  value (datetime / Decimal / MoneyMatch / DurationMatch) so callers
  can sort/threshold without re-parsing. ``DocumentView`` gains
  ``.sentences_with_entity()`` / ``.paragraphs_with_entity()``
  convenience methods. (K2; files:
  ``kaos_content/views/entity_filters.py``,
  ``kaos_content/views/document_view.py``.)
- **Five entity-filter MCP tools.** ``kaos-content-sentences-with-{dates,
  money, percents, durations, numbers}`` — read-only, deterministic,
  zero-LLM. Each takes an artifact id and returns matches with
  serialised typed values + AST anchors + page/section context.
  (K3; files: ``kaos_content/tools.py``.)
- **``kaos-content-corpus-summarize`` MCP tool.** Builds
  ``DocumentSummary`` for each artifact in a corpus. Skips artifacts
  whose summary is already populated unless ``force_rebuild=True``.
  Failed loads are recorded in a ``failed[]`` list rather than
  aborting the call. (K4; files: ``kaos_content/tools.py``.)
- **``kaos-content-corpus-narrow`` MCP tool.** BM25-ranks corpus
  artifacts against a query using their summaries (head + top + bottom
  n-grams concatenated). Returns top-K with scores, head snippets,
  and distinguishing n-grams for triage. (K4; files: ``kaos_content/tools.py``.)
- **``kaos_content.tokens`` token-frequency primitives.**
  ``document_token_frequency``, ``section_token_frequency``,
  ``paragraph_token_frequency`` — free functions backed by
  kaos-nlp-core's Rust ``FrequencyVocabulary`` (single FFI hop;
  counting in Rust, not Python). Files: ``kaos_content/tokens.py``.

### Changed

- **Frequency counting now lives in Rust, not Python.** Both the K1
  summary builder's n-gram pass and the K9 token-frequency primitives
  accumulate counts via ``kaos_nlp_core.structures.FrequencyVocabulary``
  rather than ``collections.Counter``. Matches the
  ``feedback_rust_first.md`` convention; no functional change to
  outputs, only one FFI hop per call instead of per-token. Files:
  ``kaos_content/summarize/__init__.py``, ``kaos_content/tokens.py``.

## [0.1.0a5] — 2026-05-11

### Added

- **``kaos-content-stats`` MCP tool.** Returns per-document statistical
  summary: ``char_count``, ``word_count``, ``paragraph_count``,
  ``heading_count``, ``table_count``, ``image_count``,
  ``code_block_count``, ``footnote_count``, ``annotation_count``, and
  ``page_count`` (when provenance carries pages). Closes the
  aggregation-question gap (longest doc, most tables, etc.) at the
  kaos-content boundary — callers no longer need to retrieve passages
  and count manually. Pairs with kaos-agents' corpus-manifest tool
  which aggregates these stats across a session corpus. Files:
  ``kaos_content/tools.py``.

### Changed

- **Downgraded annotation-validation WARNING to DEBUG.** The
  ``NodeIndex._build`` log line ``Document has N annotation target(s)
  referencing non-existent nodes`` was emitted at ``WARNING`` level
  on every DOCX load (one entry per body block per doc — ~10 lines
  per document) and was not actionable for users. These references
  reflect parser lifecycle quirks where annotations target valid
  blocks indexed under a different ref shape (e.g. a footnote
  promoted to a body block) — not real data errors. WARNING noise
  conditioned users to ignore real warnings. Callers that want a
  hard error can still call ``validate_annotations()`` directly.
  Files: ``kaos_content/traversal/index.py``.

### Fixed

- **CI: Windows ``Install dependencies`` PowerShell parse error.** The
  Windows-x64 matrix leg was failing immediately at the install step
  with ``ParserError: Missing expression after unary operator '--'``
  because the multi-line ``uv sync`` command used ``\`` line
  continuations but the step lacked ``shell: bash``. The "Run unit
  tests" step already had ``shell: bash`` for the same reason; aligned
  the install step. Files: ``.github/workflows/ci.yml``.
- **CI: Python 3.15 ``scipy`` source-build failed with ``Dependency
  "OpenBLAS" not found``.** SciPy lands in the dep tree transitively
  on the 3.15 lane (no cached wheel yet) and its Meson build refuses
  to proceed without OpenBLAS + LAPACK headers. Added
  ``libopenblas-dev liblapack-dev gfortran`` to the Linux apt-get
  install step. The 3.15 lane stays ``experimental: true`` (rpds-py /
  PyO3 upstream still gates it for other repos) but the build path
  now reaches further. Files: ``.github/workflows/ci.yml``.
- **Tests: ``TestIsSafeUrlMalformed`` parametrize ID overflows
  Windows env-var cap.** The malformed-URL fuzz fixture includes
  ``"http://" + "a" * 100000 + ":99999999999999999999"`` (100 000+
  characters). Pytest writes the full parameter value into
  ``PYTEST_CURRENT_TEST`` for each parametrized test, and Windows
  limits a single environment variable to 32 767 characters; on the
  new Windows-x64 CI leg every URL test in the class collected with
  ``ValueError: the environment variable is longer than 32767
  characters``. Switched to ``pytest.param(url, id="...")`` so
  pytest uses a short ID for the env var while the test body still
  receives the full giant URL. No behavior change; the same five
  fixtures still run. Files:
  ``tests/security/test_security_sec1.py``.
### Changed

- **uv.lock is now tracked in git.** Previously gitignored at v0.1.0a1
  because the ``[mcp]`` optional extra (and the ``kaos-mcp`` dev
  dependency) referenced a sibling not yet on PyPI; ``uv lock``
  couldn't resolve them. ``kaos-mcp`` shipped (0.1.0a2), so the
  original gating reason no longer applies. Tracking the lockfile
  gives reproducible local dev environments, lets Dependabot surface
  sibling-version bumps as PRs, and makes the supply-chain pin set
  publicly auditable. Mirrors the org-wide convention being adopted
  across all 16 kaos-* repos.
### Security

- **bandit + vulture now run in both pre-commit and CI.** The
  ``.pre-commit-config.yaml`` gains two new hooks (bandit static
  security scan + vulture dead-code scan), mirrored by jobs in
  ``security.yml`` so the scan is publicly visible on every PR.
  Bandit skip list is justified inline per audit
  (``B101,B404,B603,B607``); vulture runs at ``--min-confidence
  100`` with a shared ``--ignore-names`` list for framework
  callbacks / signal handlers / OAuth field names that vulture
  can't infer from the import graph alone. Both hooks currently
  pass clean. Mirrors the rollout pattern from kaos-core.

## [0.1.0a4] — 2026-05-10

### Fixed

- **Passage-URI provenance for ``search_corpus(dict[uri, text])``
  results.** Synthetic single-paragraph documents constructed by
  dict-mode ``search_corpus`` now carry a ``_kaos_synthetic_corpus``
  sentinel on ``DocumentMetadata.extra``; the
  ``_searchable_passage_uri`` builder reads the flag and falls back
  to char_start / hash form for synthetic hits while keeping the
  block_ref-derived URI for legitimate one-paragraph documents.
  Pre-fix, both paths produced ``block_ref="#/body/0"`` and the
  heuristic at ``_searchable_passage_uri`` (``block_ref != "#/body/0"``)
  silently dropped block_ref provenance from real one-paragraph hits.
  Resolves KNT-601 consumer-audit finding M-3.

### Changed

- ``kaos_content.tools.py``'s ``_VERSION`` constant now reads from
  ``kaos_content._version.__version__`` rather than being hardcoded.
  Eliminates the recurring drift where MCP tool metadata ``version``
  fields lagged behind the package version. Test
  ``test_tool_name_matches_pattern`` tightened from
  ``startswith("0.1.0a")`` back to exact equality.

### Documentation

- README extras table refreshed for KNT-602: ``[transformers]`` row
  now mentions ``SemanticDedupLevel`` as a level that benefits from
  the embedding step; new ``[clustering]`` row (scipy) covers the
  clustering step. ``[dedup-perceptual]`` row's parenthetical updated
  to reflect that ``SemanticDedupLevel`` lives natively in this
  package now (not registered by an external plugin).

## [0.1.0a3] — 2026-05-10

### Breaking changes

- **MCP tool renamed**: ``kaos-nlp-transformers-dedup-semantic`` →
  ``kaos-content-dedup-semantic``. **No deprecation cycle.** The
  pre-1.0 alpha series accepts breaking changes per the cross-monorepo
  standards (``kaos-modules-auth/docs/oss/20-python-packaging/public-api-discipline.md``);
  this rename ships under that allowance because (a) it is the natural
  consequence of the SemanticDedupLevel move (KNT-602 Option A) and
  fixing the rename in two phases would temporarily expose two tools
  with identical behavior; (b) downstream consumers of the previous
  tool name are explicitly enumerated and can be migrated in
  lockstep with this release. Treat callers of the old name as
  broken on kaos-nlp-transformers 0.2.0a3+ — the old tool is removed
  there, not left behind as a deprecation shim. If you depended on
  the tool name through MCP discovery, update your tool list to the
  new name when bumping kaos-nlp-transformers to 0.2.0a3.

### Added

- **`kaos_content.dedup.levels.semantic.SemanticDedupLevel`** — moved
  here from ``kaos_nlp_transformers.clustering`` (KNT-602 Option A).
  kaos-content owns the AST-grounded integration; kaos-nlp-transformers
  goes back to being a clean inference primitive with no reverse
  dependency on this package. Imports of ``EmbeddingModel`` and
  ``KaosNLPTransformersSettings`` are lazy (inside ``find_clusters``)
  so the level is constructible without the optional deps. ``find_clusters``
  raises ``ImportError`` with an actionable install hint pointing at
  ``kaos-content[transformers]`` / ``kaos-content[clustering]`` when
  either dep is missing.
- **MCP tool ``kaos-content-dedup-semantic``** — registered via
  ``register_content_tools``. Replaces the breaking-renamed
  ``kaos-nlp-transformers-dedup-semantic`` (see Breaking changes
  above).
- **`[clustering]` extra (`scipy>=1.14.1`)** — orthogonal to
  ``[transformers]``. Pair both extras to actually run
  ``SemanticDedupLevel``; without them the ``COMPREHENSIVE`` and
  ``OCR_AWARE`` presets gracefully degrade to lexical-only.
- **`SearchableDocument(model_id=...)`, `search_document(..., model_id=...)`,
  `search_corpus(..., model_id=...)`.** The HF Hub embedding model is now
  selectable per call and per index. ``None`` selects the
  ``kaos-nlp-transformers`` default
  (``KaosNLPTransformersSettings.default_model``). The cached embedding
  matrix on a ``SearchableDocument`` is implicitly keyed on this value
  — construct a new instance to switch models. Addresses KNT-601
  consumer-audit findings H-1 / H-2: the embedding model is no longer
  hard-coded behind ``EmbeddingModel.load()`` with no arguments.
- **Process-level ``EmbeddingModel`` cache.** ``kaos_content.search``
  now memoizes ``EmbeddingModel.load(model_id=...)`` via an
  ``lru_cache(maxsize=4)``; previously a single hybrid query loaded
  the model four times (audit finding H-3).
- **Token-aware chunking via ``SectionChunker(max_tokens=...,
  model_id=...)``.** When set, a follow-up pass calls
  ``EmbeddingModel.count_tokens`` and re-splits any chunk whose
  token count exceeds the budget — at block boundaries when possible,
  at sentence boundaries within an oversized paragraph as a fallback.
  The default ``max_tokens=None`` keeps the cheap char-only path
  unchanged. ``SectionChunker.from_outline`` and
  ``SearchableDocument.chunks(...)`` accept the same parameters; the
  latter defaults ``model_id`` to the index's model so chunker output
  and dense retrieval stay aligned by default. Addresses KNT-601
  consumer-audit finding M-2 (oversized chunks getting silently
  truncated by the embedding model's tokenizer at ``max_seq_len``).
- **`SearchableCorpus(documents=[ContentDocument, ...])` — the
  corpus-level analog of ``SearchableDocument``.** Builds one shared
  BM25 inverted index (so IDF is computed across the whole corpus,
  not per-document) and shares one embedding matrix across all dense
  queries. ``SearchResult`` gains two additive optional fields —
  ``doc_index`` (cheap O(1) handle into ``corpus.documents``) and
  ``doc_uri`` (human-readable URL) — so multi-document hits are
  fully addressable. Three retrieval modes mirror
  ``SearchableDocument`` (``"bm25"`` / ``"embeddings"`` / ``"hybrid"``).
  BM25 index is eager, embedding matrix lazy; ``max_embed_rows``
  guardrail (default 200K) prevents accidental multi-GB allocations.
  Backwards-compatible: existing single-document ``SearchResult``
  callers see ``doc_index = doc_uri = None``. See
  ``docs/SEARCHABLE_CORPUS.md`` for the full design rationale.

### Changed

- **`search_corpus` runs per-document searches concurrently.** Each
  ``SearchableDocument.search`` now executes inside ``asyncio.to_thread``
  and the futures are awaited via ``asyncio.gather``. Previously the
  function was ``async def`` but contained no ``await``s — the entire
  BM25 + embedding workload ran on the event loop thread. Combined with
  KNT-601's GIL-releasing Rust embed path, this makes per-document
  embed truly parallel. (Audit finding M-1.)
- **`_embed_texts` now accepts ``Iterable[str]``** instead of strictly
  ``list[str]``; matches the upstream ``EmbeddingModel.embed`` signature
  introduced in kaos-nlp-transformers 0.2.0.

## [0.1.0a2] — 2026-05-07

### Security

- **KCONT-01 — XXE / entity-expansion guard on the HTML parser.** `parse_html`
  previously called :func:`lxml.html.document_fromstring` with default parser
  settings. lxml's HTML parser tolerates DOCTYPE entity declarations, so a
  payload with billion-laughs-shaped entity nesting could either expand
  exponentially in memory or attempt to fetch a `SYSTEM` reference over the
  network. Fix: a module-level `lxml.html.HTMLParser(no_network=True,
  huge_tree=False, recover=True, remove_blank_text=False)` is now passed
  explicitly to every `document_fromstring` call. `no_network=True` blocks
  external entity fetches; `huge_tree=False` keeps the libxml2 input-size
  and entity-expansion caps in place. New regression tests
  (`tests/security/test_security_sec7.py`) confirm a billion-laughs payload
  parses in well under one second and that SYSTEM-entity references are
  not fetched.

### Changed

- **`kaos_content._security` is now a re-export shim.** The canonical
  implementation of `is_safe_url` and `UNSAFE_SCHEMES` has moved to
  :mod:`kaos_core.security.url` (introduced in `kaos-core` 0.1.0a4),
  where it is joined by `validate_outbound_url` (full SSRF guard),
  response size-cap helpers, and the `KaosSecuritySettings` knob set.
  In-tree callers in `serializers/html.py`, `serializers/markdown.py`,
  and `parsers/html.py` now import directly from `kaos_core.security`.
  The `_security` module remains for back-compat and re-exports the
  prior names byte-for-byte; behavior of the predicate is unchanged
  (same canonicalization algorithm, same blocklist).
- **`kaos-core` minimum bumped to `>=0.1.0a4`** in `[project.dependencies]`
  (was `>=0.1.0a1,<0.2`). Required because `kaos_core.security` debuts
  in 0.1.0a4. The ceiling stays `<0.2`.

## [0.1.0a1] — 2026-05-04

First public alpha release.

### Added

- `ContentDocument` typed Block/Inline AST: 17 Block types and 17 Inline
  types as frozen Pydantic v2 models, with content-model enforcement
  at construction (blocks contain blocks or inlines, never mixed).
- Provenance on every node: source file, page, bounding box, char span,
  extraction confidence (`Provenance` model in `model/provenance.py`).
- 15 `AnnotationType` values (3 generic + 7 legal + 3 NLP + 1 extraction
  + 1 tracked-change) with typed `body` schemas via
  `model/annotation_bodies.py`.
- `Attr` triple — `(id, classes, key-value)` on every node — for
  domain-specific extension via `Div` / `Span` containers.
- 17 `ColumnType` values (8 universal + 3 analytical + 2 structured
  + 4 KAOS extraction-specific) for `TabularDocument`.
- `DocumentView` and `TabularDocument`: dynamic hierarchical views
  (pages, sections, paragraphs, sentences) and the universal tabular
  AST.
- Three serializers: `serialize_markdown()`, `serialize_html()`,
  `serialize_text()`, each supporting `view` parameter for tracked-
  changes rendering (`final` / `original` / `markup`).
- Five tabular serializers: `serialize_csv()`, `serialize_tsv()`,
  `serialize_markdown_table()`, `serialize_json_records()`,
  `serialize_tabular_summary()`.
- `SectionChunker` — canonical heading-aware chunker that preserves
  AST validity, footnote partitioning, and annotation propagation.
- 9 optional extras: `[markdown]`, `[html]`, `[images]`, `[layout]`,
  `[nlp]`, `[polars]`, `[duckdb]`, `[mcp]`, `[dedup-perceptual]`.
- `kaos_content.shortcuts` — terse AST construction helpers
  (`text`, `bold`, `paragraph`, `heading`, `bullet_list`, etc.).
- `kaos_content.traversal` — depth-first walk + typed queries
  (`find_by_type`, `find_links`, `find_tables`, `find_images`, etc.).
- `kaos_content.transforms` — `DocumentTransform` Protocol +
  `compose()` + `apply()`.
- `kaos_content.revision` — tracked-change API: `accept_all`,
  `reject_all`, `at_time` transforms.
- Polars and DuckDB bridges in `bridges/` for tabular round-trip.
- `search_document()` BM25 search (optional `[nlp]` extra) with
  term-frequency fallback.
- `WS-TR` extraction primitives: `ExtractionCell`,
  `ExtractionCitation`, `ExtractionError`, `CellStatus` with
  reviewer-overlay invariants (immutable AI / immutable reviewer).
- 7 MCP tools registered via `register_content_tools()`.
- Python 3.13 and 3.14 support.

### Security

- HTML and Markdown serializers no longer pass through raw HTML
  blocks by default; `allow_raw_html=False` is the default. Raw HTML
  blocks emit `<!-- raw HTML stripped -->` (HTML) or
  `<!-- raw {format} stripped -->` (Markdown) unless the caller
  explicitly opts in. Audit C1+C2.
- Both serializers neuter unsafe URI schemes (`javascript:`, `data:`,
  `vbscript:`, `file:`) in `<a href>` and `<img src>`. The URL is
  replaced with `#` and the original captured as `data-unsafe-url`
  for forensics. Audit C3.
- New shared module `kaos_content._security` consolidates
  `is_safe_url()` + `UNSAFE_SCHEMES`. Canonicalises through HTML-
  entity decoding, URL percent-decoding, and whitespace removal
  before scheme matching — closes the `jav\nascript:`,
  `&#x6A;avascript:`, and `javascript%3A` bypasses (audit C4).
  Used by HTML parser, HTML serializer, and Markdown serializer so
  one fix applies everywhere.
- DuckDB bridge gains layered defence (audit C5+C6+C7):
  - `execute_query(..., untrusted_sql=True)` is the default; it
    rejects SQL containing dangerous functions (`read_csv`,
    `read_csv_auto`, `read_parquet`, `read_json`, `read_json_auto`,
    `read_blob`, `read_text`, `read_xml`, `glob`, `scan_jsonl`) and
    keywords (`attach`, `detach`, `copy`, `install`, `load`,
    `pragma`, `export`, `import`). SQL line and block comments are
    stripped before the deny-list runs so comment-evasion bypasses
    fail.
  - New `create_safe_connection()` returns an in-memory DuckDB
    connection with `enable_external_access=false` and unsigned-
    extension loads disabled — the engine-level half of the
    defence.
  - `_register_table_fallback` quotes column identifiers via
    `_quote_ident`; embedded `"` characters can no longer break
    out of the identifier.
  - `list_tables` uses parameterised queries instead of f-string
    interpolation.
- Hypothesis-based fuzz suite under `tests/fuzz/` (dev-only, gated
  on the `hypothesis` dev dependency): 42 property tests covering
  the URL filter, serializer XSS contracts, DuckDB deny-list
  evasion, JSON deserialization robustness, and Markdown
  serializer/parser no-crash properties. Surfaced and fixed an
  adjacent-emphasis serializer escape-rule bug (see Fixed).

### Fixed

- Model field constraints reject structurally-invalid values at
  construction time (audit M1):
  - `BoundingBox` rejects right<left and (top_left) bottom<top.
  - `Provenance.page` is `>=1`; `0` and negative values rejected.
  - `Provenance.confidence` is bounded to `[0.0, 1.0]`.
  - `Provenance.char_span` is non-negative and `end >= start`.
  - `Cell.row_span` and `Cell.col_span` are `>= 1`.
  - `Image.width` and `Image.height` are `> 0` when set.
- Image loading is bounded against decompression-bomb attacks
  (audit M2): `kaos_content.images.model.MAX_IMAGE_PIXELS =
  100_000_000` is enforced, and PIL's process-wide
  `Image.MAX_IMAGE_PIXELS` is set to the same cap on import.
  Decompression-bomb warnings are promoted to a typed
  `ImageDecompressionBombError`. `images/artifacts.load_image()`
  defaults `max_bytes=50_000_000` (was unbounded).
- Markdown serializer alternates `*` / `_` (and `**` / `__`)
  delimiters when an adjacent inline sibling already used the same
  delimiter. Without this, `Emphasis(0), Emphasis(0), Emphasis(0)`
  rendered as `*0**0**0*` and re-tokenised under CommonMark
  flanker rules into `Emphasis[Text(0), Strong(Text(0)), Text(0)]`
  on round-trip. Surfaced by the new fuzz tier.
- MCP tool annotations split between `_QUERY_ANNOTATIONS`
  (read-only/idempotent — search tools) and
  `_ARTIFACT_WRITER_ANNOTATIONS` (parse / chunk / extract /
  serialize-to-large — they materialise new VFS artifacts).
  Previously every tool advertised `readOnlyHint=True` even when
  it wrote artifacts, which auto-approving agents
  (Claude Code) would skip confirmation for. Audit M3.
- `kaos_content.parsers.parse_markdown` is now lazily re-exported
  from the package so importing `kaos_content` no longer pulls in
  `markdown_it` for users who don't have the `[markdown]` extra
  installed.
- Tool `_VERSION` synced to `0.1.0a1` (was stale at `0.1.0`).

### Removed

- `kaos-content-serve` script entry point and `kaos_content.serve`
  module — exposing tools over the Model Context Protocol is the
  responsibility of the companion package
  [`kaos-mcp`](https://github.com/273v/kaos-mcp), which ships
  separately. Bundling a stub server in `kaos-content` whose only
  resolution path went through `kaos-mcp` was a misleading
  dependency contract.

### Packaging

- `kaos-core` dependency spec is now `>=0.1.0a1,<0.2` (was
  `>=0.1,<0.2`). PEP 440 excludes pre-releases from a bare `>=0.1`
  spec, so plain `pip install kaos-content` could not satisfy the
  dependency. Same fix applied to the `[nlp]` and `[mcp]` extras'
  sibling-package pins.
- `SECURITY.md` is now included in the PyPI sdist (was missing).
- `[tool.check-manifest]` declares the intentional sdist exclusions
  (`CLAUDE.md`, internal `docs/*.md`, `uv.lock`) so the parity
  check passes cleanly.

### Dependencies

- `cryptography>=46.0.7` (was `>=44.0.2`) — closes
  CVE-2026-34073 and CVE-2026-39892.
- `lxml>=6.1.0` (was `>=5.0.0`, `[html]` extra) — closes
  CVE-2026-41066.
- `Pillow>=12.2.0` (`[images]` extra) — closes CVE-2026-40192.

### License

This release is the first to ship under the Apache License 2.0.
Earlier internal versions were proprietary.

[Unreleased]: https://github.com/273v/kaos-content/compare/v0.1.0a10...HEAD
[0.1.0a10]: https://github.com/273v/kaos-content/compare/v0.1.0a9...v0.1.0a10
[0.1.0a9]: https://github.com/273v/kaos-content/compare/v0.1.0a8...v0.1.0a9
[0.1.0a8]: https://github.com/273v/kaos-content/compare/v0.1.0a7...v0.1.0a8
[0.1.0a7]: https://github.com/273v/kaos-content/compare/v0.1.0a6...v0.1.0a7
[0.1.0a6]: https://github.com/273v/kaos-content/compare/v0.1.0a5...v0.1.0a6
[0.1.0a5]: https://github.com/273v/kaos-content/compare/v0.1.0a4...v0.1.0a5
[0.1.0a1]: https://github.com/273v/kaos-content/releases/tag/v0.1.0a1
