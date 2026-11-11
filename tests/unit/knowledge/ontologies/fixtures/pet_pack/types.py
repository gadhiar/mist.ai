"""The pet fixture pack's types: the base pack's, plus `Pet` and `HAS_PET`."""

from backend.knowledge.ontologies.base import (
    EdgeTypeDefinition,
    KnowledgeDomain,
    NodeTypeDefinition,
)
from tests.unit.knowledge.ontologies.fixtures.base_pack import types as base

PET = NodeTypeDefinition(
    type_name="Pet",
    description="An animal somebody keeps.",
    knowledge_domain=KnowledgeDomain.EXTERNAL,
)
HAS_PET = EdgeTypeDefinition(
    type_name="HAS_PET",
    description="A deckhand keeps a pet.",
    allowed_source_types=("Deckhand",),
    allowed_target_types=("Pet",),
)

NODE_TYPES = (*base.NODE_TYPES, PET)
EDGE_TYPES = (*base.EDGE_TYPES, HAS_PET)
EXTRACTABLE_NODE_TYPES = (*base.EXTRACTABLE_NODE_TYPES, "Pet")
EXTRACTABLE_EDGE_TYPES = (*base.EXTRACTABLE_EDGE_TYPES, "HAS_PET")
