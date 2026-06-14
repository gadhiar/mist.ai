"""General compound-id canonicalization. Used by BOTH the normalizer/resolver and
the F2 scorer so a metric's id never depends on the LLM's word order.
"""

from __future__ import annotations

import re


def _slug(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\s_/]+", "-", s)
    s = re.sub(r"[^a-z0-9\-]", "", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-")


def canonical_metric_id(value: str | float | int, unit: str) -> str:
    """Deterministic Metric id as <value>-<unit>, both slugified. Value always
    leads, so '12000-requests-per-second' and 'requests-per-second-12000' collapse.
    """
    return f"{_slug(str(value))}-{_slug(unit)}".strip("-")
