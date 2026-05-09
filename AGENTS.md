# Repository Agent Guidance

## Scope

This file is the canonical instruction file for coding agents working in
this repository. It applies to the whole repository unless a more specific
`AGENTS.md` is added in a subdirectory.

Keep agent-driven changes focused on the requested task. Preserve existing
user changes, avoid unrelated cleanup, and do not edit generated files unless
the requested change explicitly requires regenerating them.

## Project Identity

- Distribution: `kaos-content`
- Import package: `kaos_content`
- Runtime: Python 3.13+
- Package type: pure Python typed library with `py.typed`
- Primary model: `ContentDocument`, a frozen Pydantic Block/Inline AST with
  provenance, annotations, views, traversal, parsers, serializers, and
  tabular peers.

The repository does not currently publish a CLI entry point. Treat documented
modules, exported names, Pydantic models, JSON output, schemas, MCP tool
contracts, parser behavior, and serializer behavior as public API.

## Setup

Use `uv` for environments, dependency resolution, builds, and tool execution.

```bash
uv sync --group dev
uvx pre-commit install
```

For contributor details, read [CONTRIBUTING.md](CONTRIBUTING.md). For the
durable engineering standards, link to these files instead of duplicating
their contents:

- [Python design and architecture](docs/standards/python-design-and-architecture.md)
- [Code quality standards](docs/standards/code-quality-standards.md)
- [Engineering process](docs/standards/engineering-process.md)
- [Tests, fixtures, and CI](docs/standards/tests-fixtures-ci.md)

## Local Checks

Run the narrowest useful checks while developing, then run the relevant gate
before handing off:

```bash
uv run ruff format --check kaos_content tests
uv run ruff check kaos_content tests
uv run ty check kaos_content tests
uv run pytest -m "not live and not network and not slow" --no-cov
```

Use `ty`, not mypy. Inline suppressions use `# ty: ignore[...]` with the
narrowest applicable rule and a reason when the reason is not obvious.

When packaging, metadata, README rendering, or release behavior changes, also
run:

```bash
uv build
uvx --from twine twine check --strict dist/*
```

## Architecture Rules

- Keep the Block/Inline AST discipline intact: block containers hold blocks,
  inline containers hold inlines, and serializers/parsers must not blur that
  boundary.
- Preserve provenance on nodes when transforming, parsing, chunking,
  serializing, or bridging content. Page, bounding box, character span,
  confidence, and source references are part of the model contract.
- Use stable JSON-pointer node refs such as `#/body/5` for cross-references,
  annotations, views, search results, and MCP-facing outputs.
- Keep annotations as standoff metadata that can target nodes or offsets
  without forcing overlapping spans into the tree.
- Keep `DocumentView` and `TabularView` as computed, lazy, non-mutating views
  over immutable document models.
- Prefer AST-grounded operations over raw text heuristics. Search, chunking,
  extraction, and transformations should retain node refs and structure.
- Keep parser and serializer contracts stable for JSON, markdown, HTML, text,
  tabular output, and schemas. Changes to escaping, raw HTML handling, URL
  safety, output ordering, or schema shape are public behavior changes.
- Keep optional integrations behind extras and lazy imports. Do not import
  optional dependencies at module import time.
- Keep base dependencies small, declared, and compatible with the repository's
  license policy.

## Testing

New behavior needs tests through the real public entry point. Bug fixes need
regression tests. Parser, serializer, URL, SQL, file, image, and untrusted
input changes need both accepted and rejected cases where practical.

Use the existing test tiers and markers from `pyproject.toml`. The fast local
gate must not require network access, credentials, live services, or large
downloads:

```bash
uv run pytest -m "not live and not network and not slow" --no-cov
```

Fixtures must be small, redistributable, documented, and free of secrets,
customer data, privileged content, and PII.

## Security

Maintain safe-by-default behavior for serializers, parsers, image loading,
artifact loading, and DuckDB bridges. Preserve existing limits and canonical
validation around URLs, raw HTML, SQL deny-lists, image sizes, bounding boxes,
page numbers, confidence scores, and character spans.

Never commit secrets, credentials, `.env` files, private keys, build caches,
or virtual environments. Do not expose credentials, internal paths, provider
payloads, or stack traces in user-facing errors.

Report suspected vulnerabilities through [SECURITY.md](SECURITY.md), not
public issues.

## Commits, PRs, And Releases

Use focused conventional commits and sign off commits with `git commit -s`.
Do not force-push unless a maintainer explicitly asks for it.

Before opening a PR, confirm the change is scoped, tested, rebased on `main`,
and documented where public behavior changes. PR descriptions should state
what changed, why it changed, how it was tested, and whether public API,
schemas, parser/serializer output, package metadata, fixtures, or release
artifacts are affected.

Update `CHANGELOG.md` for released user-visible behavior changes, public API
changes, schema changes, package metadata changes, security changes, and
deprecations. Do not edit release metadata for docs-only agent guidance.
