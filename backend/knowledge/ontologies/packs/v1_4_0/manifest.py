"""The v1.4.0 pack's manifest: its version and the data the engine applies without a rule.

The pack loader reads `MANIFEST`. The version, its parent and its migration are read from
the ontology definition in `types.py`, so the version is stated once. The four counts
below are stated here and nowhere else: the pack's test reads them from this module and
holds the pack's lists to them.

This module imports the pack loader, so nothing `types.py` imports may import it.
"""

from __future__ import annotations

from backend.knowledge.ontologies.pack import Manifest, RetiredType

from .types import ABSTRACTION, CONCEPT, EVENT, ONTOLOGY_V1_0_0, RELATED_TO, USER

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
    # v1.4.0 retired both. A Milestone becomes an Event whose `event_type` says what it
    # was, unless the entity already states one.
    retired_types=(
        RetiredType("Topic", CONCEPT.type_name),
        RetiredType("Milestone", EVENT.type_name, (("event_type", "milestone"),)),
    ),
    # An edge that accepts a Concept accepts an Abstraction.
    supertypes=((ABSTRACTION.type_name, CONCEPT.type_name),),
    escape_edge=RELATED_TO.type_name,
    migration_script_path=ONTOLOGY_V1_0_0.migration_script_path,
    deprecated=ONTOLOGY_V1_0_0.deprecated,
)
