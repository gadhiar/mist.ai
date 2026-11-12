"""The v1.4.0 pack's plug-ins: what its rules need that data cannot state.

The pack loader reads `PLUGINS`. Each name is the one a rule in `conventions.py` cites.

The pack's rails hash covers this module's source, so an edit here moves that hash and
the release line with it. Each plug-in calls pure helpers only. None runs the rail
runner or resolves a pack: the runner checks a pack's pinned cases, which call these
functions, before the pack's first row, and a plug-in that ran the rails would start
that check again. For that reason too this module does not import the runner: the
team plug-in reads the runner's group and row by the attributes `_Group` and `_Row`
name.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Protocol

from backend.knowledge.extraction.canonical_id import (
    canonical_id,
    canonical_metric_id,
    canonical_metric_id_from_id,
)
from backend.knowledge.extraction.id_rails import ISO_DATE_RE
from backend.knowledge.ontologies.pack import ID, RELATION, Plugin

from .types import ORGANIZATION, TECHNOLOGY, WORKS_AT

TECHNOLOGY_VERSION = "technology-version"
METRIC_ID = "metric-id"
EVENT_DATE = "event-date"
TEAM_EMPLOYER = "team-employer"

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


class _Group(Protocol):
    """The runner's `FanOutGroup`, as `team_employer` reads it."""

    anchor: object
    members: tuple[str, ...]
    entities: tuple[Mapping, ...]
    relationships: tuple[Mapping, ...]


class _Row(Protocol):
    """The runner's `RailRow`, as `team_employer` reads it."""

    batch_record: tuple[Mapping, ...]
    turn_plan: Mapping | None


def _names(line: str, entity: Mapping) -> bool:
    """True when a slugged line holds the entity's id or display name as whole words."""
    padded = f"-{line}-"
    for spelling in (entity.get("id"), entity.get("name")):
        slug = canonical_id(str(spelling or ""))
        if slug and f"-{slug}-" in padded:
            return True
    return False


def _holds(batch: Mapping, entity_id: object) -> bool:
    """True when a batch's own payload holds the entity, as an entity or as an edge's end."""
    payload = batch.get("payload") or {}
    if any(entity.get("id") == entity_id for entity in payload.get("entities") or ()):
        return True
    return any(
        entity_id in (relationship.get("source"), relationship.get("target"))
        for relationship in payload.get("relationships") or ()
    )


def _team_lines(team: Mapping, row: _Row) -> Iterator[str]:
    """The team's own headings and lead-ins, each slugged; never a unit's own text.

    - From a batch whose own payload holds the team, a unit's context lines, the
      headings and lead-in it was read under, are the team's when one of them names
      the team. A unit's own text is never read, prose included: a member's own item
      names that member's employer, not the team's, and a club whose member works
      somewhere must not give every member that employer (ADR-054's v14 amendment).
      The rail only infers, and a wrong `WORKS_AT` would overwrite a stored employer,
      so a sentence that names a team and an employer together gives none; the
      model's own `WORKS_AT` still covers it. A batch's own `context`, the earlier
      units it was shown for reference, is not read.
    - A template of the turn plan is the team's when its fixed endpoint is the team or
      its lead-in names the team, and then its lead-in is the team's.
    """
    team_id = team.get("id")
    for batch in row.batch_record:
        if not _holds(batch, team_id):
            continue
        for unit in batch.get("units") or ():
            context = [canonical_id(str(line)) for line in unit.get("context") or ()]
            if any(_names(line, team) for line in context):
                yield from context
    team_slug = canonical_id(str(team_id or ""))
    for template in (row.turn_plan or {}).get("templates") or ():
        lead_in = canonical_id(str(template.get("lead_in") or ""))
        fixed = canonical_id(str(template.get("fixed_endpoint") or ""))
        if (fixed and fixed == team_slug) or _names(lead_in, team):
            yield lead_in


def team_employer(group: _Group, row: _Row) -> str | None:
    """The `team-fan-out` rule's `WORKS_AT` endpoint: the employer of a member's team.

    ADR-054's v14 amendment, point 2 and register row 1. The employer is the one
    `Organization` E, other than the team, that both:

    - one of the team's own headings or lead-ins names, read from the batch record or
      from the turn plan's template (`_team_lines`), by E's id or display name as whole
      words; a unit's own text, a member's item or a line of prose, is never read; and
    - a member of the team holds a `WORKS_AT` to. The runner lists only the members
      whose every `MEMBER_OF` the team is a current assertion, and shows only current
      edges, so both input edges are current.

    A team's `PART_OF` its employer is not read: Rule 34 forbids that edge, and a rail
    adds no edge from an output the prompt forbids. A row with neither a batch record
    nor a plan names no heading, so no employer resolves. When two organizations
    qualify, the team has no one employer and none resolves: a member gains no
    `WORKS_AT` rather than a wrong one.

    Returns:
        E's id as the payload writes it, or None.
    """
    entities: dict = {}
    for entity in group.entities:
        entities.setdefault(entity.get("id"), entity)
    team = entities.get(group.anchor)
    if team is None:
        return None
    lines = list(_team_lines(team, row))
    members = set(group.members)
    found: list[str] = []
    for relationship in group.relationships:
        employer = relationship.get("target")
        entity = entities.get(employer)
        if (
            relationship.get("type") != WORKS_AT.type_name
            or relationship.get("source") not in members
            or not isinstance(employer, str)
            or employer == group.anchor
            or employer in found
            or entity is None
            or entity.get("type") != ORGANIZATION.type_name
        ):
            continue
        if any(_names(line, entity) for line in lines):
            found.append(employer)
    return found[0] if len(found) == 1 else None


PLUGINS: tuple[Plugin, ...] = (
    # A Technology's version, read from its name, leaves the end of its id.
    Plugin(TECHNOLOGY_VERSION, ID, technology_version_id),
    # A Metric's id puts the value first, spelled as the metric rail spells it.
    Plugin(METRIC_ID, ID, metric_id),
    # An Event dated to one ISO day gains that day at the end of its id.
    Plugin(EVENT_DATE, ID, event_date_id),
    # A named team's employer, which each member of the team gains.
    Plugin(TEAM_EMPLOYER, RELATION, team_employer),
)
