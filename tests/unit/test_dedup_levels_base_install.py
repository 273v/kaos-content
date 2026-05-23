"""Pin the base-install honesty contract for ``kaos_content.dedup.levels``.

audit-04 F-001: ``kaos_content/dedup/levels/__init__.py:3-7`` and the
``semantic.py:9-14`` docstring both promise that the module imports
without ``numpy`` / ``scipy`` / ``kaos-nlp-transformers`` installed,
and that ``find_clusters`` raises an actionable ``ImportError`` when
they are actually needed. The promise was false before this test
landed because ``semantic.py`` imported ``numpy`` at module scope,
so any ``from kaos_content.dedup.levels import ...`` blew up on a
base install (numpy is only in the ``[images]`` / ``[layout]`` /
``[clustering]`` extras).

This test simulates the missing-numpy install by routing ``numpy``
through a meta-path finder that refuses imports, then exercises the
public import surface that the audit reproducer used. ``find_clusters``
is intentionally NOT called — the test only pins the import-time
contract; ``find_clusters`` correctness with numpy installed is
covered elsewhere.
"""

from __future__ import annotations

import importlib
import importlib.abc
import sys
from types import ModuleType


class _BlockNumpy(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "numpy" or fullname.startswith("numpy."):
            raise ImportError(f"blocked {fullname}")
        return None


def test_dedup_levels_import_without_numpy() -> None:
    """``import kaos_content.dedup.levels`` must succeed without numpy.

    Reproduces the audit-04 F-001 evidence block. Closes the
    documented-but-broken lazy-import promise in
    ``kaos_content/dedup/levels/__init__.py:3-7``.
    """
    # Snapshot every numpy/scipy/kaos_content.dedup.levels module so we can
    # restore the prior state after the block — a half-cleared numpy in
    # sys.modules leaves `numpy.typing` (re-)imports recursing forever for
    # any later test that pulls scipy.
    blocker = _BlockNumpy()
    snapshot: dict[str, ModuleType] = {
        name: mod
        for name, mod in sys.modules.items()
        if name == "numpy"
        or name.startswith("numpy.")
        or name == "scipy"
        or name.startswith("scipy.")
        or name.startswith("kaos_content.dedup.levels")
    }
    for name in snapshot:
        del sys.modules[name]
    sys.meta_path.insert(0, blocker)

    try:
        levels = importlib.import_module("kaos_content.dedup.levels")
        # The promised public exports must all be importable.
        assert hasattr(levels, "MinHashLevel")
        assert hasattr(levels, "SemanticDedupLevel")
        # The constructor must not touch numpy either.
        instance = levels.SemanticDedupLevel()
        assert instance is not None
    finally:
        if blocker in sys.meta_path:
            sys.meta_path.remove(blocker)
        # Drop any modules imported under the block, then restore the
        # snapshot so subsequent tests see numpy/scipy/kaos_content.dedup
        # exactly as they were before this test ran.
        for name in [
            n
            for n in sys.modules
            if n.startswith("kaos_content.dedup.levels")
            or n == "numpy"
            or n.startswith("numpy.")
            or n == "scipy"
            or n.startswith("scipy.")
        ]:
            del sys.modules[name]
        sys.modules.update(snapshot)
