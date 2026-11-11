"""The v1.4.0 pack's plug-ins: what its id rules need that data cannot state.

The pack loader reads `PLUGINS`. Each name is the one a rule in `conventions.py` cites.

The pack's rails hash covers this module's source, so an edit here moves that hash and
the release line with it. Each plug-in calls pure helpers only. None runs the rail
runner or resolves a pack: the runner checks a pack's pinned cases, which call these
functions, before the pack's first row, and a plug-in that ran the rails would start
that check again.
"""

from __future__ import annotations

import re
from collections.abc import Mapping

from backend.knowledge.extraction.canonical_id import (
    canonical_metric_id,
    canonical_metric_id_from_id,
)
from backend.knowledge.extraction.id_rails import ISO_DATE_RE
from backend.knowledge.ontologies.pack import ID, Plugin

from .types import TECHNOLOGY

TECHNOLOGY_VERSION = "technology-version"
METRIC_ID = "metric-id"
EVENT_DATE = "event-date"

# A version suffix on a display name: "Python 3.11", "Node 18.x", "React v18". Requires
# whitespace before the number or an explicit "v" prefix, so the trailing digits of
# "web3", "oauth2" or "base64" are not a version.
VERSION_PATTERN: re.Pattern = re.compile(
    r"(?:\s+v?|\bv)(\d+\.?\d*\.?\d*\.?\d*)([-.]?\w+)*$", re.IGNORECASE
)

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]+")
_TRAILING_ISO_DATE_RE = re.compile(r"-\d{4}-\d{2}-\d{2}$")


def version_rail(entity_id: str, name: str | None, entity_type: str) -> str:
    """Remove a Technology's version, read from its name, from the end of its id.

    Rule 3 collapses versions ("Python 3.11" is `python`). The rail fires only for a
    `Technology` whose name `VERSION_PATTERN` matches: the matched suffix is split on
    non-alphanumerics (` 3.11` gives `3`, `11`; ` v18` gives `v18`), and when the id ends
    with those tokens hyphen-joined (`-3-11`) or concatenated (`-311`), they are removed.
    Every other id keeps its digits, so `web3`, `covid-19` and an Event's date stay.
    """
    if entity_type != TECHNOLOGY.type_name or not name:
        return entity_id
    match = VERSION_PATTERN.search(name)
    if match is None:
        return entity_id
    tokens = [t for t in _NON_ALNUM_RE.split(match.group(0).lower()) if t]
    if not tokens:
        return entity_id
    for suffix in ("-".join(tokens), "".join(tokens)):
        tail = f"-{suffix}"
        if entity_id.endswith(tail) and len(entity_id) > len(tail):
            return entity_id[: -len(tail)]
    return entity_id


def technology_version_id(entity: Mapping, entity_id: str, target: Mapping | None) -> str:
    """The id plug-in of the Technology `typed_id` rule: `version_rail` on the entity."""
    name = str(entity.get("name") or entity.get("id") or "")
    return version_rail(entity_id, name, TECHNOLOGY.type_name)


def metric_id(entity: Mapping, entity_id: str, target: Mapping | None) -> str:
    """The id plug-in of the Metric `typed_id` rule.

    The id is built with `canonical_metric_id` when the properties carry a value and a
    unit or a quantity, and otherwise from the raw `id` field with
    `canonical_metric_id_from_id`, so a dotted decimal in the id survives.
    """
    props = entity.get("properties")
    props = props if isinstance(props, Mapping) else {}
    value, unit, quantity = props.get("value"), props.get("unit"), props.get("quantity")
    if value is not None and (unit or quantity):
        return canonical_metric_id(
            value, str(unit) if unit else None, str(quantity) if quantity else None
        )
    return canonical_metric_id_from_id(str(entity.get("id") or entity.get("name") or ""))


def event_date_id(entity: Mapping, entity_id: str, target: Mapping | None) -> str | None:
    """The id plug-in of the Event-date `derived_edge` rule (ADR-054 Decision 4).

    A target whose id is not an ISO day does not date the Event. Otherwise the id loses
    a trailing `-YYYY-MM-DD` and gains `-<the target's id>`.
    """
    date = str((target or {}).get("id", ""))
    if not ISO_DATE_RE.match(date):
        return None
    return f"{_TRAILING_ISO_DATE_RE.sub('', entity_id)}-{date}"


PLUGINS: tuple[Plugin, ...] = (
    # A Technology's version, read from its name, leaves the end of its id.
    Plugin(TECHNOLOGY_VERSION, ID, technology_version_id),
    # A Metric's id puts the value first, spelled as the metric rail spells it.
    Plugin(METRIC_ID, ID, metric_id),
    # An Event dated to one ISO day gains that day at the end of its id.
    Plugin(EVENT_DATE, ID, event_date_id),
)
