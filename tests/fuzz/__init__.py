"""Property-based and pathological-edge-case tests for kaos-content.

These tests use Hypothesis to generate adversarial inputs against the
package's security contracts:

- ``test_fuzz_url_filter`` — ``_security.is_safe_url`` canonicalisation
- ``test_fuzz_serializer_xss`` — ``serialize_html`` / ``serialize_markdown``
  safe-by-default contract for arbitrary URLs and raw-HTML strings
- ``test_fuzz_sql_safety`` — ``bridges.duckdb._assert_sql_safe`` deny-list
  cannot be evaded by comment / casing / whitespace tricks
- ``test_fuzz_json_roundtrip`` — ``ContentDocument.model_validate_json``
  never crashes on arbitrary bytes; valid documents round-trip
- ``test_fuzz_markdown_roundtrip`` — generated documents survive
  ``serialize_markdown`` → ``parse_markdown`` (idempotent on the
  second pass)

Hypothesis profile is tuned in ``conftest.py`` so the suite stays
fast in normal CI but can be cranked up for nightly tiers.
"""
