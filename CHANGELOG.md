# Changelog

All notable changes to `kaos-content` are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/273v/kaos-content/compare/v0.1.0a1...HEAD
[0.1.0a1]: https://github.com/273v/kaos-content/releases/tag/v0.1.0a1
