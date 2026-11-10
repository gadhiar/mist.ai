"""Entity normalization stage.

Stage 5: stores each entity's `id` field through the id rails
(`backend/knowledge/extraction/id_rails.py`), the same functions the eval scorer uses.
Target <50ms. Graph-identity resolution (R1.1d) was removed; the curation deduper in
Stage 7a is the sole authority for graph-identity decisions.
"""

import logging

from backend.interfaces import EmbeddingProvider
from backend.knowledge.extraction import id_rails
from backend.knowledge.extraction.ontology_extractor import ExtractionResult
from backend.knowledge.storage.graph_executor import GraphExecutor

logger = logging.getLogger(__name__)


class EntityNormalizer:
    """Async wrapper over the id rails, the pure steps that build a stored id.

    Graph-identity resolution (R1.1d) was removed; the curation deduper in Stage 7a is
    the sole authority for graph-identity decisions. The steps, their order and the
    payload rail are in `id_rails.normalize_entity` and `id_rails.normalize_payload`.

    The class constants are the `id_rails` objects themselves, so a test that patches one
    in place (`monkeypatch.setitem(EntityNormalizer.CANONICAL_REGISTRY, ...)`) reaches the
    steps.
    """

    STATIC_ALIASES = id_rails.STATIC_ALIASES
    PRE_CANON_ALIASES = id_rails.PRE_CANON_ALIASES
    VERSION_PATTERN = id_rails.VERSION_PATTERN
    CANONICAL_REGISTRY = id_rails.CANONICAL_REGISTRY
    RESERVED_NAMES = id_rails.RESERVED_NAMES

    def __init__(
        self,
        embedding_generator: EmbeddingProvider,
        executor: GraphExecutor | None = None,
    ) -> None:
        """Initialize the normalizer.

        Args:
            embedding_generator: Retained for API compatibility; not used after R1.1d strip.
            executor: Retained for API compatibility; not used after R1.1d strip.
                Graph-identity resolution moved to the curation deduper (Stage 7a).
        """
        # retained: constructor params kept for call-site compatibility (factories.py +
        # ~16 test call sites). Graph-identity path (_find_in_graph) was removed in R1.1d.
        self._embedding_generator = embedding_generator
        self._executor = executor
        self._graph_available = executor is not None

    async def normalize(self, extraction: ExtractionResult) -> ExtractionResult:
        """Store every entity's `id` field through the id rails.

        Pure canonicalization: this method issues no graph queries; matching against
        existing graph nodes is the curation deduper's sole responsibility as of R1.1.

        Modifies the ExtractionResult in place and returns it.

        Args:
            extraction: The ExtractionResult with entities and relationships.

        Returns:
            The same ExtractionResult with normalized entity IDs.
        """
        if not extraction.entities:
            return extraction

        for entity in extraction.entities:
            reserved = id_rails.reserved_identity(entity)
            if reserved is not None:
                logger.warning(
                    "Reserved name '%s' remapped to canonical id='%s' type='%s'",
                    entity.get("name") or entity.get("id"),
                    reserved[0],
                    reserved[1],
                )

        extraction.entities, extraction.relationships = id_rails.normalize_payload(
            extraction.entities, extraction.relationships
        )
        return extraction
