"""Confidence arithmetic for the curation pipeline.

Pure computation -- no I/O, no constructor dependencies. Reads confidence
policies from the v1.0.0 ontology to derive boost values and defaults.
"""

from backend.knowledge.ontologies.base import ConfidencePolicy, KnowledgeDomain
from backend.knowledge.ontologies.v1_0_0 import ONTOLOGY_V1_0_0


class ConfidenceManager:
    """Manages confidence scoring for curation operations.

    Reads correction_boost and default_confidence values from the ontology's
    ConfidencePolicy definitions. No hardcoded constants.
    """

    def __init__(self) -> None:
        self._policies: dict[KnowledgeDomain, ConfidencePolicy] = {
            p.domain: p for p in ONTOLOGY_V1_0_0.confidence_policies
        }
        self._type_to_domain: dict[str, KnowledgeDomain] = {
            nt.type_name: nt.knowledge_domain for nt in ONTOLOGY_V1_0_0.node_types
        }

    def reinforced_confidence(self, current: float, domain: KnowledgeDomain) -> float:
        """Apply diminishing-returns confidence boost.

        Formula: min(0.99, current + boost * (1.0 - current))

        Args:
            current: Current confidence value (0.0 to 1.0).
            domain: Knowledge domain determining the boost magnitude.

        Returns:
            Reinforced confidence value, capped at 0.99.
        """
        policy = self._policies.get(domain)
        boost = policy.correction_boost if policy else 0.10
        return min(0.99, current + boost * (1.0 - current))

    def initial_confidence(self, domain: KnowledgeDomain) -> float:
        """Return the default confidence for a knowledge domain.

        Args:
            domain: The knowledge domain.

        Returns:
            Default confidence from the ontology policy.
        """
        policy = self._policies.get(domain)
        return policy.default_confidence if policy else 0.8

    def determine_domain(self, entity_type: str) -> KnowledgeDomain:
        """Map an entity type name to its KnowledgeDomain.

        Reads from the ontology's NodeTypeDefinition.knowledge_domain field.
        Falls back to EXTERNAL if entity_type is not found.

        Args:
            entity_type: The entity type name (e.g. "User", "MistIdentity").

        Returns:
            The KnowledgeDomain for the entity type.
        """
        return self._type_to_domain.get(entity_type, KnowledgeDomain.EXTERNAL)
