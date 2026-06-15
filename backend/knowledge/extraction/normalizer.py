"""Entity normalization stage.

Stage 5: Normalizes entity IDs to canonical forms via string canonicalization,
static aliases, registry lookups, and resolver passes. Target <50ms.
Graph-identity resolution (R1.1d) was removed; the curation deduper in Stage 7a
is the sole authority for graph-identity decisions.
"""

import logging
import re

from backend.interfaces import EmbeddingProvider
from backend.knowledge.extraction.canonical_id import (
    canonical_metric_id,
    canonical_metric_id_from_id,
)
from backend.knowledge.extraction.ontology_extractor import ExtractionResult
from backend.knowledge.ontologies.hierarchy import RETIRED_TYPE_MAP
from backend.knowledge.ontologies.v1_0_0 import ALL_NODE_TYPE_NAMES
from backend.knowledge.storage.graph_executor import GraphExecutor

logger = logging.getLogger(__name__)


class EntityNormalizer:
    """Normalizes entity IDs via string canonicalization, static aliases, registry, and
    resolver passes (retired-type coercion, Metric compound-id, parent fallback).

    Graph-identity resolution (R1.1d) was removed; the curation deduper in Stage 7a is
    the sole authority for graph-identity decisions. This class is pure string/registry/
    resolver canonicalization.

    Algorithm per entity:
    1. Canonicalize: lowercase, strip version numbers, replace spaces with hyphens.
    2. Check static alias map (e.g. "js" -> "javascript").
    3. Check canonical-entity registry (curated overrides).
    4. Resolver passes: retired-type coercion, Metric compound-id, parent fallback.
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

    # Bounded curated (id, type) overrides for recurring high-value entities,
    # keyed by CANONICAL id (post _canonicalize / static-alias). Seeded later
    # from curated vocab (seed_data.yaml), NEVER from F2 probes. Authoritative:
    # a registry hit overrides both id and type and short-circuits graph dedup
    # and the resolver passes. Empty for now; populated in a later task.
    CANONICAL_REGISTRY: dict[str, tuple[str, str]] = {}

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
        """Normalize all entity IDs via string canonicalization + resolver passes.

        Pure canonicalization: reserved-name/registry resolution + the ontology
        resolver passes (retired-type coercion, Metric compound-id, parent
        fallback). Graph-identity resolution (matching against existing graph
        nodes) is the curation deduper's sole responsibility as of R1.1 -- this
        method issues no graph queries.

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

            # Pass 2: canonical-entity registry (authoritative curated overrides).
            # Keyed on the canonical id (post _canonicalize / static-alias) so one
            # entry catches every surface variant that canonicalizes to the same id.
            # A registry hit short-circuits graph dedup and the resolver passes.
            registry = self.CANONICAL_REGISTRY.get(canonical_id)
            if registry is not None:
                reg_id, reg_type = registry
                entity["type"] = reg_type
                entity["id"] = reg_id
                id_map[old_id] = reg_id
                continue

            # Set the pre-resolver canonical id first, then run resolver passes
            # 3-5 (retired-type coercion, Metric compound-id, parent fallback).
            # id_map captures entity["id"] AFTER the resolver so that the
            # relationship source/target remap below points to the final id
            # (e.g. the compound Metric id, not the pre-resolver canonical_id).
            entity["id"] = canonical_id
            self._resolve_type_and_id(entity)
            id_map[old_id] = entity["id"]

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

    def _resolve_type_and_id(self, entity: dict) -> None:
        """Resolver passes 3-5: retired-type coercion, Metric compound-id, parent
        fallback. (Pass 1 reserved-name and pass 2 registry are handled earlier in
        normalize() because they short-circuit the loop.) Mutates entity in place;
        the caller captures entity['id'] into id_map AFTER this runs.

        Pass 3: Retired types (Topic -> Concept, Milestone -> Event). For Milestone,
        also sets properties.event_type='milestone' via setdefault so an existing
        value is never overwritten.

        Pass 4: Metric compound-id from value + unit properties. Only fires when both
        value and unit are present; skips silently otherwise.

        Pass 5: Unknown type -> Abstraction fallback. Fires when the post-pass-3 type
        is not in ALL_NODE_TYPE_NAMES.
        """
        etype = entity.get("type", "")

        # Pass 3: retired-type coercion (Topic -> Concept, Milestone -> Event).
        if etype in RETIRED_TYPE_MAP:
            coerced = RETIRED_TYPE_MAP[etype]
            if etype == "Milestone":
                props = entity.setdefault("properties", {})
                props.setdefault("event_type", "milestone")
            entity["type"] = coerced
            etype = coerced

        # Pass 4: compound-id canonicalization for Metric (id from value + unit).
        # When structured props are present, rebuild as <value>-<unit> (canonical_metric_id).
        # When props are absent, fall back to string-based value-position normalization
        # (canonical_metric_id_from_id) so bare ids like `requests-per-second-12000`
        # still collapse to the canonical `12000-requests-per-second`.
        if etype == "Metric":
            props = entity.get("properties") or {}
            value, unit = props.get("value"), props.get("unit")
            if value is not None and unit:
                entity["id"] = canonical_metric_id(value, unit)
            else:
                entity["id"] = canonical_metric_id_from_id(entity["id"])

        # Pass 5: parent fallback for unknown types not otherwise handled.
        if entity.get("type", "") not in ALL_NODE_TYPE_NAMES:
            entity["type"] = "Abstraction"

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
