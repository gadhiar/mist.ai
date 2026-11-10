"""Id canonicalization shared by the normalizer and the eval scorer.

The graph and the scorer import these functions rather than keeping two copies, so a
stored id and a scored id cannot drift apart, and a metric's id never depends on the
LLM's word order.
"""

from __future__ import annotations

import re


def canonical_id(name: str) -> str:
    """Canonical entity id: lowercase, hyphenated, alnum+hyphen only.

    Version collapsing is not part of the slug; it is a separate rail that only some
    entity types take.
    """
    s = name.lower().strip()
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"[^a-z0-9\-]", "", s)
    s = re.sub(r"-{2,}", "-", s)
    s = s.strip("-")
    return s or name.lower().replace(" ", "-")


def _slug(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"[\s_/]+", "-", s)
    s = re.sub(r"[^a-z0-9\-]", "", s)
    s = re.sub(r"-{2,}", "-", s)
    return s.strip("-")


def canonical_metric_id(
    value: str | float | int, unit: str | None, quantity: str | None = None
) -> str:
    """Deterministic Metric id as `<value>-<unit>[-<quantity>]`, the form
    `data/gold/extraction/LABELLING.md` writes.

    Value always leads, so `12000-requests-per-second` and `requests-per-second-12000`
    collapse. The quantity is part of the id so that a p95 and a p99 latency of `120 ms`
    get `120-ms-p95-latency` and `120-ms-p99-latency` rather than one shared `120-ms`.

    The id joins the non-empty slugs of value, unit and quantity, in that order, so a
    count with no unit still gets `180-max-queue-depth`. A quantity is dropped when the
    unit's slug already ends with it on a token boundary (`percent-coverage` with
    `coverage`), which keeps an id from repeating its tail when the model folds the
    quantity into the unit. With no quantity the id is the `<value>-<unit>` this builder
    returned before it took one.

    Args:
        value: The metric's value; slugified, so `12,000` becomes `12000`.
        unit: The unit, or None or empty for a unitless count.
        quantity: What is measured (`p95 latency`), or None or empty.

    Returns:
        The slug parts joined with `-`.
    """
    value_slug = _slug(str(value))
    unit_slug = _slug(str(unit)) if unit else ""
    quantity_slug = _slug(str(quantity)) if quantity else ""
    if quantity_slug and (unit_slug == quantity_slug or unit_slug.endswith(f"-{quantity_slug}")):
        quantity_slug = ""
    return "-".join(part for part in (value_slug, unit_slug, quantity_slug) if part)


def canonical_metric_id_from_id(id_str: str) -> str:
    """Fallback Metric-id canonicalization when structured props cannot rebuild the
    id (no value, or neither a unit nor a quantity): move the single numeric token to
    the front so `requests-per-second-12000` and `12000-requests-per-second` collapse
    to the same id. Preserves the order of the remaining (unit) tokens; does NOT
    attempt to reorder unit words. Returns the slug unchanged when there is not
    exactly one numeric token, or it is already first.
    """
    s = _slug(id_str)
    tokens = s.split("-")
    nums = [i for i, t in enumerate(tokens) if re.fullmatch(r"\d+(?:\.\d+)?", t)]
    if len(nums) != 1 or nums[0] == 0:
        return s
    i = nums[0]
    return "-".join([tokens[i]] + tokens[:i] + tokens[i + 1 :])
