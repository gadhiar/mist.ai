"""The base fixture pack's two node types and its edges."""

from backend.knowledge.ontologies.base import (
    EdgeTypeDefinition,
    KnowledgeDomain,
    NodeTypeDefinition,
)

DECKHAND = NodeTypeDefinition(
    type_name="Deckhand",
    description="A person aboard, the user included.",
    knowledge_domain=KnowledgeDomain.EXTERNAL,
)
MOORING = NodeTypeDefinition(
    type_name="Mooring",
    description="A numbered place a deckhand ties up at.",
    knowledge_domain=KnowledgeDomain.EXTERNAL,
)

CREWS_WITH = EdgeTypeDefinition(
    type_name="CREWS_WITH",
    description="A deckhand sails in a skipper's crew.",
    allowed_source_types=("Deckhand", "Stowaway"),
    allowed_target_types=("Deckhand",),
)
SERVES_UNDER = EdgeTypeDefinition(
    type_name="SERVES_UNDER",
    description="A deckhand takes orders from another.",
    allowed_source_types=("Deckhand",),
    allowed_target_types=("Deckhand",),
)
KNOWS_HAND = EdgeTypeDefinition(
    type_name="KNOWS_HAND",
    description="A deckhand knows another.",
    allowed_source_types=("Deckhand",),
    allowed_target_types=("Deckhand",),
    directional=False,
)
OUTRANKS = EdgeTypeDefinition(
    type_name="OUTRANKS",
    description="A deckhand is senior to another.",
    allowed_source_types=("Deckhand",),
    allowed_target_types=("Deckhand",),
)
MOORED_AT = EdgeTypeDefinition(
    type_name="MOORED_AT",
    description="A deckhand's boat is tied up at a mooring.",
    allowed_source_types=("Deckhand",),
    allowed_target_types=("Mooring",),
)
LODGES_AT = EdgeTypeDefinition(
    type_name="LODGES_AT",
    description="A deckhand sleeps at a mooring.",
    allowed_source_types=("Deckhand",),
    allowed_target_types=("Mooring",),
)
ALONGSIDE = EdgeTypeDefinition(
    type_name="ALONGSIDE",
    description="Anything beside anything: the pack's any-to-any edge.",
    allowed_source_types=("Deckhand", "Mooring"),
    allowed_target_types=("Deckhand", "Mooring"),
    directional=False,
)

NODE_TYPES = (DECKHAND, MOORING)
EDGE_TYPES = (CREWS_WITH, SERVES_UNDER, KNOWS_HAND, OUTRANKS, MOORED_AT, LODGES_AT, ALONGSIDE)
EXTRACTABLE_NODE_TYPES = ("Deckhand", "Mooring")
EXTRACTABLE_EDGE_TYPES = tuple(edge.type_name for edge in EDGE_TYPES)
