# kaos-content

> **Part of [Kelvin Agentic OS](https://kelvin.legal) (KAOS)** — open agentic
> infrastructure for legal work, built by
> [273 Ventures](https://273ventures.com).
> See the [full KAOS package map](https://github.com/273v) for the rest of the stack.

[![PyPI - Version](https://img.shields.io/pypi/v/kaos-content)](https://pypi.org/project/kaos-content/)
[![Python](https://img.shields.io/pypi/pyversions/kaos-content)](https://pypi.org/project/kaos-content/)
[![License](https://img.shields.io/pypi/l/kaos-content)](https://github.com/273v/kaos-content/blob/main/LICENSE)
[![quality](https://github.com/273v/kaos-content/actions/workflows/quality.yml/badge.svg)](https://github.com/273v/kaos-content/actions/workflows/quality.yml)

`kaos-content` is the canonical document model for KAOS — a typed
Block/Inline AST with provenance, annotations, views, and round-trip
serializers.

Every KAOS document processor (`kaos-pdf`, `kaos-office`, `kaos-web`)
produces a `ContentDocument`; every downstream consumer (search,
chunking, LLM programs, MCP resources) reads one. The shape is inspired
by Pandoc's Block/Inline discipline and Docling's provenance model.
`kaos-content` does **not** parse PDFs, fetch URLs, or call LLMs —
companion packages do. This package is what makes them interoperable.

To expose `kaos-content` parsers and serializers over MCP, add the
companion package [`kaos-mcp`](https://github.com/273v/kaos-mcp).

## Install

```bash
uv add kaos-content
# or
pip install kaos-content
```

`kaos-content` requires Python **3.13** or newer. Optional extras unlock
specific capabilities:

| Extra | Pulls in | Unlocks |
|---|---|---|
| `[markdown]` | `markdown-it-py[plugins]` | `parse_markdown` round-trip |
| `[html]` | `lxml>=6.1` | HTML parser to AST |
| `[images]` | `Pillow>=12.2`, `numpy` | `KaosImage` wrapper (PIL + DPI + provenance) |
| `[layout]` | `numpy` only | Layout primitives (X-Y cut, projection profiles, clustering, valley detection). Operates on numeric coordinate arrays — no PIL, no raster IO. Install `[images]` separately if you also need to load source images. |
| `[polars]` | `polars>=1.5.0` | `TabularDocument` ↔ Polars DataFrame |
| `[duckdb]` | `duckdb>=1.1.1` | DuckDB SQL bridge for tabular |
| `[nlp]` | `kaos-nlp-core` | BM25 search + sentence-level units + `fuzzy_binary` / `minhash` dedup levels + the cosine similarity-graph kernels (`knn_graph` / `near_duplicates`) behind `SemanticGraphDedupLevel` |
| `[graph]` | `kaos-graph>=0.1.4` | The `connected_components_from_edges` union-find kernel that turns the cosine similarity graph into duplicate groups for `SemanticGraphDedupLevel` / `dedup(embedder=...)`. Pair with `[nlp]` and a caller-supplied embedder. |
| `[transformers]` | `kaos-nlp-transformers>=0.2.0a2` | Dense embedding + cross-encoder retrieval (`SearchableDocument(retrieval='embeddings'|'hybrid')`, `SearchableCorpus`, token-aware `SectionChunker(max_tokens=...)`) AND the embedding step of `SemanticDedupLevel`. Pure-Rust backend under the hood — `libonnxruntime` is statically linked into the cdylib, so no Python `onnxruntime` install. |
| `[clustering]` | `scipy>=1.14.1` | The clustering step of `SemanticDedupLevel` (hierarchical agglomerative on cosine distance). Pair with `[transformers]` to actually run the level — both extras are required at `find_clusters` time. Without them the `COMPREHENSIVE` and `OCR_AWARE` presets gracefully degrade to lexical-only. |
| `[dedup-perceptual]` | `imagehash` | Perceptual page-image dedup (`PerceptualHashLevel`). For embedding-based semantic dedup, install `kaos-content[transformers,clustering]` to enable `SemanticDedupLevel` (KNT-602 0.1.0a3 — the level lives natively in this package now, no longer registered by an external plugin). |

## Quick start

Build a small document, serialize it, search it, and walk its sections:

```python
from kaos_content.model.document import ContentDocument
from kaos_content.model.metadata import DocumentMetadata
from kaos_content.shortcuts import bold, heading, link, paragraph, text
from kaos_content.search import search_document
from kaos_content.serializers.markdown import serialize_markdown
from kaos_content.serializers.html import serialize_html
from kaos_content.views.document_view import DocumentView

doc = ContentDocument(
    metadata=DocumentMetadata(title="Hello"),
    body=(
        heading(1, "Hello, KAOS"),
        paragraph(
            text("Built on a "),
            bold("typed AST"),
            text(" with "),
            link("https://kelvin.legal", "provenance"),
            text("."),
        ),
    ),
)

print(serialize_markdown(doc))
# # Hello, KAOS
#
# Built on a **typed AST** with [provenance](https://kelvin.legal).

print(serialize_html(doc, allow_raw_html=False))
# <h1>Hello, KAOS</h1><p>Built on a <strong>typed AST</strong> with
# <a href="https://kelvin.legal">provenance</a>.</p>

results = search_document(doc, "typed AST", top_k=5)
for r in results.results:
    print(r.score, r.block_ref, r.text[:80])

for section in DocumentView(doc).flat_sections:
    print(section.heading_text, section.heading_ref)
```

The AST is constructed entirely from frozen Pydantic models — every
field is type-validated at construction time, content-model
constraints (blocks contain blocks or inlines, never mixed) are
enforced before a node ever reaches a serializer, and the same
`ContentDocument` round-trips losslessly through JSON via
`model_dump_json()` / `model_validate_json()`.

## Concepts

The package is built around nine composable primitives.

| Concept | What it is |
|---|---|
| `ContentDocument` | Frozen Pydantic AST: metadata + body of `Block` nodes |
| `Block` / `Inline` | Strictly separated — blocks contain blocks or inlines, never mixed; enforced at construction |
| `Provenance` | Source file, page, bounding box, char span, confidence — on any node |
| `Attr` | Pandoc-style `(id, classes, key-value)` triple — universal extension mechanism |
| `Annotation` | Standoff layer for overlapping marks (redactions, defined terms, citations, NLP entities) |
| `node_ref` | JSON pointer addressing (e.g. `#/body/5`) — stable target for MCP resources |
| `DocumentView` | Dynamic hierarchical views (pages, sections, paragraphs, sentences) computed from the flat AST |
| `TabularDocument` | Universal tabular AST — peer to `ContentDocument`, 17-type column system, Polars/DuckDB bridges |
| `KaosImage` | PIL wrapper carrying DPI + provenance, with bomb-resistant load (100 MP cap) |

## Compatibility & status

| Aspect | |
|---|---|
| **Python** | 3.13, 3.14, 3.15 pre-release (CI runs all three on linux-x64; 3.13 on macOS-arm64 and Windows-x64). 3.14t free-threaded is intentionally skipped until the dependency ecosystem ships wheels for it. |
| **OS** | Linux, macOS, Windows (pure-Python wheel; no native code) |
| **Maturity** | Alpha (`0.1.0a1`). The Block/Inline grammar is near-stable; serializer flags, traversal helpers, and the `[dedup-perceptual]` / `[layout]` APIs are subject to refinement during the alpha cycle. |
| **Stability policy** | Pre-1.0: minor bumps (`0.x → 0.(x+1)`) may break behaviour; patch bumps are additive only. Every change is documented in [`CHANGELOG.md`](CHANGELOG.md). The MCP tool surface and the safe-by-default serializer/parser/SQL contracts are public API. |
| **Test coverage** | 2,068 unit tests + 42 Hypothesis property tests pass on Python 3.13. |
| **Type checker** | Validated with [`ty`](https://docs.astral.sh/ty/), Astral's Python type checker. |

## Security model

`kaos-content` ships **safe-by-default** serializers and bridges. The
contract:

| Surface | Default | Override |
|---|---|---|
| `serialize_html(allow_raw_html=False)` | strips raw HTML blocks; neuters `javascript:` / `data:` / `vbscript:` / `file:` URLs to `#` with a `data-unsafe-url` forensic attribute | `allow_raw_html=True` for trusted content |
| `serialize_markdown(allow_raw_html=False)` | same URL-neutering; `<!-- raw {format} stripped -->` placeholder for raw blocks | `allow_raw_html=True` |
| `parsers.html` | URLs canonicalised through HTML-entity decode → percent-decode → whitespace removal before scheme checks; defeats `jav&#x09;ascript:` and `javascript%3A` style bypasses | — |
| `bridges.duckdb.execute_query(untrusted_sql=True)` | application-level deny-list rejects `read_csv`, `read_parquet`, `attach`, `copy`, `install`, `load`, `pragma` (and SQL-comment evasions); strips line/block comments before matching | `untrusted_sql=False` for fully trusted SQL |
| `bridges.duckdb.create_safe_connection()` | engine-level sandbox: `enable_external_access=false`, unsigned-extension loads disabled | use a raw `duckdb.connect()` if you need filesystem access |
| `KaosImage.from_bytes` / `from_path` | rejects images > 100 MP via `ImageDecompressionBombError` (PIL's warning promoted to a hard error) | `kaos_content.images.model.MAX_IMAGE_PIXELS = N` |
| `images.artifacts.load_image(max_bytes=50_000_000)` | rejects artifact bodies > 50 MB before decoding | `max_bytes=None` for trusted artifacts |
| `BoundingBox` / `Provenance` / `Cell` / `Image` | Pydantic `Field` constraints reject inverted boxes, page=0, confidence>1, zero/negative spans, zero/negative dimensions at construction time | — |

See [SECURITY.md](SECURITY.md) for the disclosure policy and threat
model.

## MCP tools

`kaos-content` registers 17 MCP tools through
`kaos_content.tools.register_content_tools(runtime)` (count enforced by
[`tests/unit/test_tools.py::TestRegistration`](tests/unit/test_tools.py)
to prevent README drift):

| Tool | What it does |
|---|---|
| `kaos-content-parse-markdown` | Parse markdown text into a `ContentDocument` artifact |
| `kaos-content-serialize` | Load an artifact and serialize to markdown / HTML / text |
| `kaos-content-chunk-document` | Split a document at heading boundaries, store chunks as artifacts |
| `kaos-content-search-document` | BM25 / term-frequency search with AST `block_ref` results |
| `kaos-content-search-table` | Case-insensitive substring search inside a `TabularDocument` |
| `kaos-content-extract-section` | Pull a section by heading ref into a standalone document |
| `kaos-content-extract-page` | Pull a single page (requires page provenance) |
| `kaos-content-context-window` | Expand a node ref into a structural context window |
| `kaos-content-dedup-semantic` | Lexical / semantic dedup over a corpus of artifacts |
| `kaos-content-stats` | Per-document statistical summary (counts, structure, pages) |
| `kaos-content-sentences-with-dates` | Yield sentences containing typed date entities |
| `kaos-content-sentences-with-money` | Yield sentences containing typed money entities |
| `kaos-content-sentences-with-percents` | Yield sentences containing typed percent entities |
| `kaos-content-sentences-with-durations` | Yield sentences containing typed duration entities |
| `kaos-content-sentences-with-numbers` | Yield sentences containing typed numeric entities |
| `kaos-content-corpus-summarize` | Build / refresh `DocumentSummary` for each artifact in a corpus |
| `kaos-content-corpus-narrow` | BM25-rank a corpus by query against per-document summaries |

## Configuration

`kaos-content` has no module-level environment variables — its public
APIs are all in-process. Settings that affect behaviour are documented
inline at the call site (`MAX_IMAGE_PIXELS`, `DEFAULT_LOAD_IMAGE_MAX_BYTES`,
`allow_raw_html`, `untrusted_sql`).

## Companion packages

`kaos-content` is one of the packages in the
[Kelvin Agentic OS](https://kelvin.legal). The broader stack:

| Package | Layer | What it does |
|---|---|---|
| [`kaos-core`](https://github.com/273v/kaos-core) | Core | Foundational runtime, MCP-native types, registries, execution engine, VFS |
| [`kaos-content`](https://github.com/273v/kaos-content) | Core | Typed document AST: Block/Inline, provenance, views |
| [`kaos-mcp`](https://github.com/273v/kaos-mcp) | Bridge | FastMCP server, `kaos` management CLI, MCP resource templates |
| [`kaos-pdf`](https://github.com/273v/kaos-pdf) | Extraction | PDF → AST with provenance |
| [`kaos-web`](https://github.com/273v/kaos-web) | Extraction | Web extraction, browser automation, search, domain intelligence |
| [`kaos-office`](https://github.com/273v/kaos-office) | Extraction | DOCX / PPTX / XLSX readers + writers to AST |
| [`kaos-tabular`](https://github.com/273v/kaos-tabular) | Extraction | DuckDB-powered SQL analytics |
| [`kaos-source`](https://github.com/273v/kaos-source) | Data | Government + financial data connectors (Federal Register, eCFR, EDGAR, GovInfo, PACER, GLEIF) |
| [`kaos-llm-client`](https://github.com/273v/kaos-llm-client) | LLM | Multi-provider LLM transport |
| [`kaos-llm-core`](https://github.com/273v/kaos-llm-core) | LLM | Typed LLM programming (Signatures, Programs, Optimizers) |
| [`kaos-nlp-core`](https://github.com/273v/kaos-nlp-core) | Primitives (Rust) | High-performance NLP primitives |
| [`kaos-nlp-transformers`](https://github.com/273v/kaos-nlp-transformers) | ML (Rust) | Dense embeddings + cross-encoder reranking via Rust `ort` cdylib (libonnxruntime statically linked) |
| [`kaos-graph`](https://github.com/273v/kaos-graph) | Primitives (Rust) | Graph algorithms + RDF/SPARQL |
| [`kaos-ml-core`](https://github.com/273v/kaos-ml-core) | Primitives (Rust) | Classical ML on the document AST |
| [`kaos-citations`](https://github.com/273v/kaos-citations) | Legal | Legal citation extraction, resolution, verification |
| [`kaos-agents`](https://github.com/273v/kaos-agents) | Agentic | Agent runtime, memory, recipes |
| [`kaos-reference`](https://github.com/273v/kaos-reference) | Sample | Reference module for module authors |

Packages depend on `kaos-core`; everything else is opt-in. Mix and match
the ones you need.

## Development

```bash
git clone https://github.com/273v/kaos-content
cd kaos-content
uv sync --group dev
```

Install pre-commit hooks (recommended — they run the same checks as CI on
every commit, scoped to staged files):

```bash
uvx pre-commit install
uvx pre-commit run --all-files     # one-time full sweep
```

Manual QA commands (the same set CI runs):

```bash
uv run ruff format --check kaos_content tests
uv run ruff check kaos_content tests
uv run ty check kaos_content tests
uv run pytest -m "not live and not network and not slow"
```

## Build from source

```bash
uv build
uv pip install dist/*.whl
```

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md)
for setup, quality gates, pull request expectations, and engineering
standards. By contributing you agree to follow the
[project conduct expectations](CODE_OF_CONDUCT.md) and certify the
[Developer Certificate of Origin v1.1](https://developercertificate.org/) —
sign every commit with `git commit -s`. Please open an issue before starting
on a non-trivial change so we can align on scope.

## Security

For security issues, **please do not file a public issue**. Report privately
via [GitHub Private Vulnerability Reporting](https://github.com/273v/kaos-content/security/advisories/new)
or email **security@273ventures.com**. See [SECURITY.md](SECURITY.md) for the
full disclosure policy.

## License

Apache License 2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE).

Copyright 2026 [273 Ventures LLC](https://273ventures.com).
Built for [kelvin.legal](https://kelvin.legal).
