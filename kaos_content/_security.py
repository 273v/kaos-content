"""Back-compat shim. The canonical implementation now lives in kaos_core.

As of ``kaos-content`` 0.1.0a2 the ``is_safe_url`` predicate and the
``UNSAFE_SCHEMES`` constant have moved to :mod:`kaos_core.security.url`,
where they are joined by a full SSRF guard (``validate_outbound_url``)
plus response-size cap helpers. This module remains as a re-export shim
so any in-tree caller still importing from ``kaos_content._security``
continues to work without change. New code should import directly from
:mod:`kaos_core.security`.

This module is package-private (``_security``); external callers were
asked in the prior docstring to file an issue before depending on it,
so the re-export is intentionally narrow (only the two prior names).
"""

from __future__ import annotations

from kaos_core.security.url import UNSAFE_SCHEMES, is_safe_url

__all__ = ["UNSAFE_SCHEMES", "is_safe_url"]
