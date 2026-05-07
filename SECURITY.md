# Security policy

## Reporting a vulnerability

We take security seriously. If you believe you have found a security
vulnerability in `kaos-content`, please report it privately so we can
address it before public disclosure.

**Please do not file a public GitHub issue for security reports.**

### How to report

Use [GitHub Private Vulnerability Reporting](https://github.com/273v/kaos-content/security/advisories/new)
to send a report. Alternatively, email **security@273ventures.com**.

Include as much of the following as you can:

- A description of the vulnerability and its impact
- Steps to reproduce, including affected versions
- Any proof-of-concept code, if available
- Suggested mitigations, if you have any

### What to expect

- **Acknowledgement** — within 3 business days of your report.
- **Initial triage** — within 7 business days, including a severity
  assessment.
- **Fix and disclosure** — coordinated with you. Our target window is
  90 days from acknowledgement to public disclosure, faster for
  high-severity issues.
- **Credit** — we credit reporters in the release notes and security
  advisory unless you prefer to remain anonymous.

## Supported versions

`kaos-content` follows Semantic Versioning. While the project is
pre-1.0, only the latest minor release receives security fixes. After
1.0, the latest two minor releases will be supported.

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |
| < 0.1   | No        |

## Scope

In-scope:

- The `kaos-content` Python package as published on PyPI
- The `273v/kaos-content` GitHub repository (CI, release, supply chain)
- The HTML and Markdown serializers' XSS contract
- The HTML parser's URL-scheme filter
- The DuckDB bridge's SQL safety boundary
- The image-loading decompression-bomb mitigations

Out of scope:

- Third-party dependencies (report to the upstream project)
- Issues caused by user-supplied configuration that explicitly
  disables safety features (e.g., `allow_raw_html=True`,
  `untrusted_sql=False`, `enable_external_access=true`)
