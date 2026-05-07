"""Hypothesis configuration for the fuzz tier.

Two profiles are registered:

- ``ci`` (default) — small example budget, hard deadline. Designed to
  run alongside the regular unit suite without inflating wall time.
- ``nightly`` — large example budget, no deadline. Use when running
  ``pytest tests/fuzz --hypothesis-profile=nightly`` on a longer tier.
"""

from __future__ import annotations

from hypothesis import HealthCheck, settings

settings.register_profile(
    "ci",
    max_examples=100,
    deadline=1000,  # ms
    suppress_health_check=[HealthCheck.too_slow],
)
settings.register_profile(
    "nightly",
    max_examples=2000,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile("ci")
