"""The v1.4.0 pack's manifest: its version and the data the engine applies without a rule.

The pack loader reads `MANIFEST`. The version, its parent and its migration are read from
the ontology definition in `types.py`, so the version is stated once. The four counts
below are stated here and nowhere else: the pack's test reads them from this module and
holds the pack's lists to them.

This module imports the pack loader, so nothing `types.py` imports may import it.
"""

from __future__ import annotations

from backend.knowledge.ontologies.pack import Manifest, RetiredType

from .types import ABSTRACTION, ONTOLOGY_V1_0_0, RELATED_TO, RETIRED_TYPES, SUPERTYPES, USER

# What an entity of a retired type gains, unless it has it, when its type is replaced: a
# Milestone becomes an Event whose `event_type` says what it was.
_RETIRED_GAINS: dict[str, tuple[tuple[str, str], ...]] = {
    "Milestone": (("event_type", "milestone"),),
}

# How many types and edges the version has, and how many of each the model may produce.
NODE_TYPE_COUNT = 30
EXTRACTABLE_NODE_TYPE_COUNT = 21
EDGE_TYPE_COUNT = 51
EXTRACTABLE_EDGE_TYPE_COUNT = 39

MANIFEST = Manifest(
    version=ONTOLOGY_V1_0_0.version,
    # An entity whose type the version does not have becomes an Abstraction, the
    # supertype of the abstract types.
    fallback_type=ABSTRACTION.type_name,
    user_id="user",
    user_type=USER.type_name,
    parent_version=ONTOLOGY_V1_0_0.parent_version,
    # The retired types and the supertypes are the tuples `types.py` states, which the
    # hierarchy module reads too: an edge that accepts a Concept accepts an Abstraction.
    retired_types=tuple(
        RetiredType(name, replacement, _RETIRED_GAINS.get(name, ()))
        for name, replacement in RETIRED_TYPES
    ),
    supertypes=SUPERTYPES,
    escape_edge=RELATED_TO.type_name,
    migration_script_path=ONTOLOGY_V1_0_0.migration_script_path,
    deprecated=ONTOLOGY_V1_0_0.deprecated,
)
