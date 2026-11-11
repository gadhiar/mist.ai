"""Entity normalization stage.

Stage 5: runs the rail runner (`backend/knowledge/extraction/rails.py`) over the
extraction with an ontology pack, the same runner and pack the eval scorer's id rails
use. Target <50ms. Graph-identity resolution (R1.1d) was removed; the curation deduper in
Stage 7a is the sole authority for graph-identity decisions.
"""

import logging
from collections.abc import Mapping, Sequence

from backend.interfaces import EmbeddingProvider
from backend.knowledge.extraction import id_rails, rails
from backend.knowledge.extraction.ontology_extractor import ExtractionResult
from backend.knowledge.ontologies import registry
from backend.knowledge.ontologies.pack import Pack
from backend.knowledge.storage.graph_executor import GraphExecutor

logger = logging.getLogger(__name__)

# What a replay hands over from a cache row, besides the payload. The row's scope is a
# rail input too and `normalize` takes it, but the replay paths do not hand it over yet.
_RAIL_INPUTS: tuple[str, ...] = ("batch_record", "turn_plan")


def rail_inputs(cached: Mapping) -> dict:
    """The batch record and plan a cache row carries, as keyword arguments for `normalize`.

    A row written before a column existed, or by a turn that had no plan, holds None
    there. Those are left out, so `normalize` is called exactly as it was before the
    row could carry them.

    Args:
        cached: The row `ExtractionCache.get` returned.
    """
    return {name: cached[name] for name in _RAIL_INPUTS if cached.get(name) is not None}


class EntityNormalizer:
    """Async wrapper over the rail runner, the pure steps that build a stored payload.

    Graph-identity resolution (R1.1d) was removed; the curation deduper in Stage 7a is
    the sole authority for graph-identity decisions. The steps and their order are in
    `rails.run_rails`; what they act on is the pack's.

    The class constants are the `id_rails` objects themselves. `CANONICAL_REGISTRY` is
    read by the runner at each call, so a test that patches it in place
    (`monkeypatch.setitem(EntityNormalizer.CANONICAL_REGISTRY, ...)`) reaches the steps.
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
        pack: Pack | None = None,
    ) -> None:
        """Initialize the normalizer.

        Args:
            embedding_generator: Retained for API compatibility; not used after R1.1d strip.
            executor: Retained for API compatibility; not used after R1.1d strip.
                Graph-identity resolution moved to the curation deduper (Stage 7a).
            pack: The ontology pack whose rails run; None takes the registry's active
                pack.

        Raises:
            PackLoadError: One of the pack's pinned cases does not hold. The check runs
                here, when the normalizer is built, and not at its first `normalize`.
        """
        # retained: constructor params kept for call-site compatibility (factories.py +
        # ~16 test call sites). Graph-identity path (_find_in_graph) was removed in R1.1d.
        self._embedding_generator = embedding_generator
        self._executor = executor
        self._graph_available = executor is not None
        self._pack = pack if pack is not None else registry.active_pack()
        rails.require_pinned_cases(self._pack)

    async def normalize(
        self,
        extraction: ExtractionResult,
        *,
        scope: str | None = None,
        batch_record: Sequence[Mapping] | None = None,
        turn_plan: Mapping | None = None,
    ) -> ExtractionResult:
        """Run the pack's rails over the extraction.

        Pure: this method issues no graph queries; matching against existing graph nodes
        is the curation deduper's sole responsibility as of R1.1. The extraction and the
        three keyword arguments are the whole rail row, so a replay that passes what the
        cache row holds gets what the live apply got.

        Modifies the ExtractionResult in place and returns it.

        Args:
            extraction: The ExtractionResult with entities and relationships.
            scope: The cache row's scope label; None for a turn that has none.
            batch_record: The cache row's batch record; None for a turn extracted as
                one batch.
            turn_plan: The cache row's turn plan; None for a turn without one.

        Returns:
            The same ExtractionResult with the stored entities and relationships.
        """
        if not extraction.entities:
            return extraction

        for entity in extraction.entities:
            reserved = rails.reserved_identity(entity, self._pack)
            if reserved is not None:
                logger.warning(
                    "Reserved name '%s' remapped to canonical id='%s' type='%s'",
                    entity.get("name") or entity.get("id"),
                    reserved[0],
                    reserved[1],
                )

        row = rails.rail_row(
            extraction.entities,
            extraction.relationships,
            scope=scope,
            batch_record=batch_record,
            turn_plan=turn_plan,
        )
        extraction.entities, extraction.relationships = rails.run_rails(row, self._pack)
        return extraction
