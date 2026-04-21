"""Entity normalization and deduplication stage.

Stage 5: Normalizes entity IDs to canonical forms and deduplicates
against existing graph entities using static aliases, exact matches,
alias matches, and embedding similarity. Target <50ms.
"""

import logging
import re

from backend.errors import EmbeddingError, Neo4jQueryError
from backend.interfaces import EmbeddingProvider
from backend.knowledge.extraction.ontology_extractor import ExtractionResult
from backend.knowledge.storage.graph_executor import GraphExecutor

logger = logging.getLogger(__name__)


class EntityNormalizer:
    """Normalizes entity IDs and deduplicates against the graph.

    Algorithm per entity:
    1. Canonicalize: lowercase, strip version numbers, replace spaces with hyphens.
    2. Check static alias map (e.g. "js" -> "javascript").
    3. Check exact ID match in the graph.
    4. Check alias match in the graph (entity.aliases array).
    5. Check embedding similarity (>= threshold AND same entity_type).
    6. If match found: rewrite entity ID to the existing one.
       If no match: entity is new, keep the canonical ID.
    """

    # Static alias map: short name / alternate spelling -> canonical ID.
    # IMPORTANT: Keys must be unambiguous. If a short form could refer to
    # multiple technologies (e.g. "tf" -> terraform OR tensorflow), do NOT
    # include it here. Let the LLM resolve ambiguity via the full name.
    STATIC_ALIASES: dict[str, str] = {
        "js": "javascript",
        "ts": "typescript",
        "py": "python",
        "rb": "ruby",
        "node.js": "nodejs",
        "node": "nodejs",
        "react.js": "react",
        "reactjs": "react",
        "react-js": "react",
        "vue.js": "vue",
        "vuejs": "vue",
        "vue-js": "vue",
        "angular.js": "angular",
        "angularjs": "angular",
        "next.js": "nextjs",
        "nuxt.js": "nuxtjs",
        "mongo": "mongodb",
        "postgres": "postgresql",
        "pg": "postgresql",
        "k8s": "kubernetes",
        "kube": "kubernetes",
        "vscode": "visual-studio-code",
        "vs-code": "visual-studio-code",
        "gcp": "google-cloud-platform",
        "aws": "amazon-web-services",
        "gh": "github",
        "cpp": "c-plus-plus",
        "csharp": "c-sharp",
        "golang": "go",
        "objc": "objective-c",
        "sklearn": "scikit-learn",
        "torch": "pytorch",
    }

    # Pre-canonicalization aliases: checked against the raw lowered name
    # BEFORE stripping special characters. Handles names like "c++", "c#"
    # whose special chars would be lost during canonicalization.
    _PRE_CANON_ALIASES: dict[str, str] = {
        "c++": "c-plus-plus",
        "c#": "c-sharp",
    }

    # Strips version numbers: "Python 3.11" -> "Python", "Node 18.x" -> "Node"
    # Requires whitespace before the number or an explicit "v" prefix to avoid
    # stripping trailing digits from names like "web3", "oauth2", "base64".
    VERSION_PATTERN: re.Pattern = re.compile(
        r"(?:\s+v?|\bv)(\d+\.?\d*\.?\d*\.?\d*)([-.]?\w+)*$", re.IGNORECASE
    )

    # Bug G guard: reserved names for the MIST system itself always resolve
    # to the canonical mist-identity node (seeded in scripts/seed_data.yaml).
    # Extraction that introduces new "mist" or "the-ai" entities pollutes
    # the graph with duplicates of the system's own identity node.
    #
    # Cluster 1 extension: each entry maps to a (canonical_id, canonical_type)
    # tuple. The type override is required because Cluster 1 validator
    # constraints check MistIdentity as the source of IMPLEMENTED_WITH /
    # MIST_HAS_CAPABILITY / MIST_HAS_TRAIT / MIST_HAS_PREFERENCE edges; an
    # LLM-guessed type of "Organization" on the mist-identity node breaks
    # those constraints. Forcing type=MistIdentity on reserved-name matches
    # keeps node schema aligned with edge-contract validation.
    RESERVED_NAMES: dict[str, tuple[str, str]] = {
        "mist": ("mist-identity", "MistIdentity"),
        "mist.ai": ("mist-identity", "MistIdentity"),
        "mist ai": ("mist-identity", "MistIdentity"),
        "mist-ai": ("mist-identity", "MistIdentity"),
        "the ai": ("mist-identity", "MistIdentity"),
        "the-ai": ("mist-identity", "MistIdentity"),
        "the assistant": ("mist-identity", "MistIdentity"),
        "the-assistant": ("mist-identity", "MistIdentity"),
    }

    SIMILARITY_THRESHOLD: float = 0.92

    def __init__(
        self,
        embedding_generator: EmbeddingProvider,
        executor: GraphExecutor | None = None,
    ) -> None:
        """Initialize the normalizer.

        Args:
            embedding_generator: For computing embedding similarity.
            executor: Async graph executor for querying existing entities.
                If None, graph-based deduplication is skipped (local-only mode).
        """
        self._embedding_generator = embedding_generator
        self._executor = executor
        self._graph_available = executor is not None

    async def normalize(self, extraction: ExtractionResult) -> ExtractionResult:
        """Normalize all entity IDs and deduplicate against the graph.

        Modifies the ExtractionResult in place and returns it.

        Args:
            extraction: The ExtractionResult with entities and relationships.

        Returns:
            The same ExtractionResult with normalized entity IDs.
        """
        if not extraction.entities:
            return extraction

        # Build a mapping from old ID -> new canonical ID
        id_map: dict[str, str] = {}

        for entity in extraction.entities:
            old_id = entity.get("id", "")
            entity_name = entity.get("name", old_id)
            entity_type = entity.get("type", "")

            # Skip the "user" entity -- always canonical
            if old_id.lower() == "user":
                id_map[old_id] = "user"
                entity["id"] = "user"
                continue

            # Bug G / Cluster 1 guard: reserved-name remap takes precedence
            # over every other alias path and also overrides entity_type so
            # the mist-identity node carries the MistIdentity label required
            # by validator edge-contract constraints.
            raw_lower = entity_name.lower().strip()
            reserved = self.RESERVED_NAMES.get(raw_lower)
            if reserved is not None:
                canonical_id, canonical_type = reserved
                logger.warning(
                    "Reserved name '%s' remapped to canonical id='%s' type='%s'",
                    entity_name,
                    canonical_id,
                    canonical_type,
                )
                entity["type"] = canonical_type
                id_map[old_id] = canonical_id
                entity["id"] = canonical_id
                continue

            # Check pre-canonicalization aliases (e.g. "C++" before "+" is stripped).
            # Note: _PRE_CANON_ALIASES matches BEFORE _canonicalize runs. Keys in
            # RESERVED_NAMES must NOT appear in _PRE_CANON_ALIASES (the guard above
            # would already have handled them, but keeping invariants explicit).
            pre_canon = self._PRE_CANON_ALIASES.get(raw_lower)
            if pre_canon is not None:
                canonical_id = pre_canon
            else:
                canonical_id = self._canonicalize(entity_name)

                # Check static aliases (post-canonicalization)
                canonical_id = self.STATIC_ALIASES.get(canonical_id, canonical_id)

            # Graph-based deduplication (if available)
            if self._graph_available and self._executor is not None:
                graph_id = await self._find_in_graph(canonical_id, entity_type)
                if graph_id is not None:
                    canonical_id = graph_id

            id_map[old_id] = canonical_id
            entity["id"] = canonical_id

        # Update relationship source/target IDs
        for rel in extraction.relationships:
            source = rel.get("source", "")
            target = rel.get("target", "")
            if source in id_map:
                rel["source"] = id_map[source]
            if target in id_map:
                rel["target"] = id_map[target]

        # Deduplicate entities (multiple old IDs may map to the same canonical)
        seen: set[str] = set()
        deduped_entities: list[dict] = []
        for entity in extraction.entities:
            eid = entity["id"]
            if eid not in seen:
                seen.add(eid)
                deduped_entities.append(entity)
        extraction.entities = deduped_entities

        # Deduplicate relationships (same source-target-type triple)
        seen_rels: set[tuple[str, str, str]] = set()
        deduped_rels: list[dict] = []
        for rel in extraction.relationships:
            key = (rel.get("source", ""), rel.get("target", ""), rel.get("type", ""))
            if key not in seen_rels:
                seen_rels.add(key)
                deduped_rels.append(rel)
        extraction.relationships = deduped_rels

        return extraction

    def _canonicalize(self, name: str) -> str:
        """Convert a display name to a canonical entity ID.

        Lowercases, strips version numbers, replaces spaces/underscores
        with hyphens, and removes non-alphanumeric characters (except hyphens).

        Reserved-name remapping (Bug G / Cluster 1) is handled upstream in
        `normalize()` so that the caller can also override `entity["type"]`
        to MistIdentity; this method is pure id-canonicalization for
        non-reserved names.

        Args:
            name: The entity display name.

        Returns:
            Canonical entity ID string.
        """
        # Strip version numbers
        canonical = self.VERSION_PATTERN.sub("", name)

        # Lowercase
        canonical = canonical.lower().strip()

        # Replace spaces and underscores with hyphens
        canonical = re.sub(r"[\s_]+", "-", canonical)

        # Remove anything that's not alphanumeric or hyphen
        canonical = re.sub(r"[^a-z0-9\-]", "", canonical)

        # Collapse multiple hyphens
        canonical = re.sub(r"-{2,}", "-", canonical)

        # Strip leading/trailing hyphens
        canonical = canonical.strip("-")

        return canonical or name.lower().replace(" ", "-")

    async def _find_in_graph(self, canonical_id: str, entity_type: str) -> str | None:
        """Search the graph for an existing entity matching this canonical ID.

        Tries three tiers:
        1. Exact ID match.
        2. Alias match (entity.aliases array).
        3. Embedding similarity (>= threshold AND same entity_type).

        Args:
            canonical_id: The canonical entity ID to search for.
            entity_type: The entity type (for type-constrained similarity).

        Returns:
            The existing entity's ID if found, None otherwise.
        """
        if self._executor is None:
            return None

        # Tier 1: Exact ID match
        try:
            results = await self._executor.execute_query(
                "MATCH (e:__Entity__) WHERE toLower(e.id) = $canonical_id "
                "RETURN e.id AS id, e.entity_type AS entity_type, e.aliases AS aliases LIMIT 1",
                {"canonical_id": canonical_id},
            )
            if results:
                logger.debug("Exact match for '%s' -> '%s'", canonical_id, results[0]["id"])
                return results[0]["id"]
        except (Neo4jQueryError, Exception) as e:
            logger.warning("Exact ID query failed (%s): %s", type(e).__name__, e)
            return None

        # Tier 2: Alias match
        try:
            results = await self._executor.execute_query(
                "MATCH (e:__Entity__) "
                "WHERE $canonical_id IN [a IN e.aliases | toLower(a)] "
                "RETURN e.id AS id, e.entity_type AS entity_type, e.aliases AS aliases LIMIT 1",
                {"canonical_id": canonical_id},
            )
            if results:
                logger.debug("Alias match for '%s' -> '%s'", canonical_id, results[0]["id"])
                return results[0]["id"]
        except (Neo4jQueryError, Exception) as e:
            logger.warning("Alias query failed (%s): %s", type(e).__name__, e)
            return None

        # Tier 3: Embedding similarity
        try:
            candidate_embedding = self._embedding_generator.generate_embedding(canonical_id)
            results = await self._executor.execute_query(
                "CALL db.index.vector.queryNodes('entity_embeddings', 5, $candidate_embedding) "
                "YIELD node, score "
                "WHERE score >= $threshold AND node.entity_type = $entity_type "
                "RETURN node.id AS id, score "
                "ORDER BY score DESC LIMIT 3",
                {
                    "candidate_embedding": candidate_embedding,
                    "threshold": self.SIMILARITY_THRESHOLD,
                    "entity_type": entity_type,
                },
            )
            if results:
                best = results[0]
                logger.debug(
                    "Embedding match for '%s' -> '%s' (score=%.3f)",
                    canonical_id,
                    best["id"],
                    best["score"],
                )
                return best["id"]
        except (Neo4jQueryError, EmbeddingError, Exception) as e:
            # Vector index may not exist yet
            logger.debug(
                "Embedding similarity query failed (%s, expected if no vector index): %s",
                type(e).__name__,
                e,
            )

        return None
