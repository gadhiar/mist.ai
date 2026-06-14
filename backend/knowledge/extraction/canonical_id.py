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


def canonical_metric_id_from_id(id_str: str) -> str:
    """Fallback Metric-id canonicalization when structured value+unit props are
    absent: move the single numeric token to the front so `requests-per-second-12000`
    and `12000-requests-per-second` collapse to the same id. Preserves the order of
    the remaining (unit) tokens; does NOT attempt to reorder unit words. Returns the
    slug unchanged when there is not exactly one numeric token, or it is already first.
    """
    s = _slug(id_str)
    tokens = s.split("-")
    nums = [i for i, t in enumerate(tokens) if re.fullmatch(r"\d+(?:\.\d+)?", t)]
    if len(nums) != 1 or nums[0] == 0:
        return s
    i = nums[0]
    return "-".join([tokens[i]] + tokens[:i] + tokens[i + 1 :])
