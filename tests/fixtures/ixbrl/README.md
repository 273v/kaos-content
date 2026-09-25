# ixbrl — provenance

`synthetic_annual_report.htm` is a synthetic Inline XBRL (iXBRL 1.1)
annual report used by `tests/unit/test_html_inline_xbrl.py`.

Every name, number and sentence in it is fictional. Only the markup
shape follows the layout that filing tools emit for EDGAR iXBRL
documents:

- an XML declaration and an `<html>` root that declares the `ix`,
  `xbrli`, `link`, `dei`, `us-gaap` and `iso4217` namespaces;
- a `<div style="display:none">` holding `ix:header` (`ix:hidden`
  facts, `ix:references`, `ix:resources` contexts and units) — none of
  which may appear in parsed text;
- a flat `<body>` of sibling paragraph `<div>`s separated by
  `<hr style="page-break-after:always"/>` page breaks, with no
  article-level container;
- visible facts: `ix:nonNumeric` on cover-page values and on a text
  block that spans a page break via `ix:continuation`, and
  `ix:nonFraction` values in running text and in a table.

Hidden sentinels asserted absent from parsed text: `0009999999`,
`HIDDENFACTFY`, `2025-01-01`, `iso4217:USD`.

| File | Source | License | Created | SHA-256 |
|---|---|---|---|---|
| `synthetic_annual_report.htm` | Hand-written synthetic document | Apache-2.0 (this repository) | 2026-09-25 | `9bc5742d7326b613fb3e4c1d7a10efdecabe1726bbadfa0583cb96077199a53f` |
