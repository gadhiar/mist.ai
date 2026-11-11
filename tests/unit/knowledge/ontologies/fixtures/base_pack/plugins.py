"""The base fixture pack's plug-ins: what its rules cannot state as data."""

import re

from backend.knowledge.ontologies.pack import ID, RELATION, Plugin

_NUMBER_RE = re.compile(r"(\d+)")


def mooring_number(entity, entity_id, target):
    """A mooring is stored under its number: `Mooring No. 7` is `mooring-7`."""
    match = _NUMBER_RE.search(str(entity.get("name") or entity.get("id") or ""))
    return f"mooring-{match.group(1)}" if match else None


def moored_id(entity, entity_id, target):
    """A deckhand moored at one numbered mooring carries it in the id."""
    target_id = str(target.get("id", ""))
    if not target_id.startswith("mooring-"):
        return None
    return f"{entity_id.removesuffix('-at-' + target_id)}-at-{target_id}"


def skipper_lodging(group, row):
    """The mooring the crew's skipper lodges at; the engine shows live edges only."""
    for relationship in group.relationships:
        if relationship.get("type") == "LODGES_AT" and relationship.get("source") == group.anchor:
            return relationship.get("target")
    return None


def no_solo_crew(entities, relationships, row):
    """Nobody crews with themself."""
    kept = [
        relationship
        for relationship in relationships
        if not (
            relationship.get("type") == "CREWS_WITH"
            and relationship.get("source") == relationship.get("target")
        )
    ]
    return entities, kept


PLUGINS = (
    Plugin("mooring-number", ID, mooring_number),
    Plugin("moored-id", ID, moored_id),
    Plugin("skipper-lodging", RELATION, skipper_lodging),
    Plugin("no-solo-crew", RELATION, no_solo_crew),
)
