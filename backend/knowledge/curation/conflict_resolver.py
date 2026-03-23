"""Relationship conflict detection and resolution.

Stage 7b: Detects functional supersessions, contradictions, and
progressions. Newer always wins. Produces supersession actions
for the graph writer.
"""

import logging
from dataclasses import dataclass, field

from backend.knowledge.storage.graph_executor import GraphExecutor

logger = logging.getLogger(__name__)

# Functional relationships: single-valued (new replaces old)
FUNCTIONAL_RELATIONSHIPS: set[str] = {"WORKS_AT"}

# Contradiction pairs: bidirectional conflicts
CONTRADICTION_PAIRS: list[tuple[str, str]] = [
    ("USES", "DISLIKES"),
    ("PREFERS", "DISLIKES"),
    ("EXPERT_IN", "STRUGGLES_WITH"),
    ("INTERESTED_IN", "DISLIKES"),
]

# Progression pairs: (from_type, to_type) -- from is superseded by to
PROGRESSION_PAIRS: list[tuple[str, str]] = [
    ("LEARNING", "EXPERT_IN"),
    ("STRUGGLES_WITH", "EXPERT_IN"),
    ("STRUGGLES_WITH", "USES"),
]


def _get_contradiction_type(rel_type: str) -> set[str]:
    """Return the set of relationship types that contradict the given type."""
    contradictions: set[str] = set()
    for a, b in CONTRADICTION_PAIRS:
        if rel_type == a:
            contradictions.add(b)
        elif rel_type == b:
            contradictions.add(a)
    return contradictions


def _get_superseded_by(rel_type: str) -> set[str]:
    """Return types that are superseded when rel_type appears."""
    superseded: set[str] = set()
    for from_type, to_type in PROGRESSION_PAIRS:
        if rel_type == to_type:
            superseded.add(from_type)
    return superseded


@dataclass(frozen=True, slots=True)
class SupersessionAction:
    """Instruction to supersede an existing relationship."""

    old_rel_type: str
    old_target_id: str
    new_target_id: str
    reason: str  # "functional_supersession" | "contradiction" | "progression"


@dataclass(frozen=True, slots=True)
class ConflictResolutionResult:
    """Result of conflict resolution."""

    relationships: list[dict]
    supersession_actions: list[SupersessionAction] = field(default_factory=list)
    conflicts_detected: int = 0
    conflicts_resolved: int = 0


class ConflictResolver:
    """Detects and resolves relationship conflicts.

    Checks functional supersessions (WORKS_AT), contradiction pairs
    (USES vs DISLIKES), and progressions (LEARNING -> EXPERT_IN).
    Newer always wins.
    """

    def __init__(self, executor: GraphExecutor) -> None:
        self._executor = executor

    async def resolve(
        self, entities: list[dict], relationships: list[dict]
    ) -> ConflictResolutionResult:
        """Detect and resolve conflicts in relationships.

        Args:
            entities: Entity list (used for source/target context).
            relationships: Relationship list from extraction/dedup.

        Returns:
            ConflictResolutionResult with filtered relationships and actions.
        """
        if not relationships:
            return ConflictResolutionResult(relationships=[])

        # Phase 1: Intra-batch contradiction check
        relationships, intra_detected = self._resolve_intra_batch(relationships)

        supersession_actions: list[SupersessionAction] = []
        detected = intra_detected
        resolved = intra_detected

        # Phase 2: Check each relationship against graph state
        for rel in relationships:
            rel_type = rel.get("type", "")
            source = rel.get("source", "")
            target = rel.get("target", "")

            # Functional supersession (e.g. WORKS_AT)
            if rel_type in FUNCTIONAL_RELATIONSHIPS:
                actions = await self._check_functional(source, rel_type, target)
                if actions:
                    detected += len(actions)
                    resolved += len(actions)
                    supersession_actions.extend(actions)

            # Contradiction pairs
            contradiction_types = _get_contradiction_type(rel_type)
            for contra_type in contradiction_types:
                actions = await self._check_contradiction(source, target, rel_type, contra_type)
                if actions:
                    detected += len(actions)
                    resolved += len(actions)
                    supersession_actions.extend(actions)

            # Progression pairs
            superseded_types = _get_superseded_by(rel_type)
            for old_type in superseded_types:
                actions = await self._check_progression(source, target, rel_type, old_type)
                if actions:
                    detected += len(actions)
                    resolved += len(actions)
                    supersession_actions.extend(actions)

        return ConflictResolutionResult(
            relationships=relationships,
            supersession_actions=supersession_actions,
            conflicts_detected=detected,
            conflicts_resolved=resolved,
        )

    def _resolve_intra_batch(self, relationships: list[dict]) -> tuple[list[dict], int]:
        """Remove intra-batch contradictions. Last occurrence wins."""
        # Build lookup: (source, target) -> list of rels
        pair_rels: dict[tuple[str, str], list[dict]] = {}
        for rel in relationships:
            key = (rel.get("source", ""), rel.get("target", ""))
            pair_rels.setdefault(key, []).append(rel)

        detected = 0
        keep: list[dict] = []

        for (_source, _target), rels in pair_rels.items():
            if len(rels) <= 1:
                keep.extend(rels)
                continue

            # Check for contradiction pairs within this source-target group
            types = [r.get("type", "") for r in rels]
            contradicted: set[int] = set()

            for i, type_a in enumerate(types):
                for j, type_b in enumerate(types):
                    if i >= j:
                        continue
                    contras = _get_contradiction_type(type_a)
                    if type_b in contras:
                        # Later index wins (newer)
                        contradicted.add(i)
                        detected += 1

            for i, rel in enumerate(rels):
                if i not in contradicted:
                    keep.append(rel)

        return keep, detected

    async def _check_functional(
        self, source: str, rel_type: str, new_target: str
    ) -> list[SupersessionAction]:
        """Check for existing functional relationship to supersede."""
        results = await self._executor.execute_query(
            "MATCH (s:__Entity__ {id: $source})-[r]->(t:__Entity__) "
            "WHERE type(r) = $rel_type AND r.status = 'active' "
            "AND t.id <> $new_target "
            "RETURN s.id AS source_id, t.id AS target_id, type(r) AS type",
            {"source": source, "rel_type": rel_type, "new_target": new_target},
        )
        return [
            SupersessionAction(
                old_rel_type=r["type"],
                old_target_id=r["target_id"],
                new_target_id=new_target,
                reason="functional_supersession",
            )
            for r in results
        ]

    async def _check_contradiction(
        self, source: str, target: str, new_type: str, contra_type: str
    ) -> list[SupersessionAction]:
        """Check for existing contradictory relationship."""
        results = await self._executor.execute_query(
            "MATCH (s:__Entity__ {id: $source})-[r]->(t:__Entity__ {id: $target}) "
            "WHERE type(r) = $contra_type AND r.status = 'active' "
            "RETURN s.id AS source_id, t.id AS target_id, type(r) AS type",
            {"source": source, "target": target, "contra_type": contra_type},
        )
        return [
            SupersessionAction(
                old_rel_type=r["type"],
                old_target_id=r["target_id"],
                new_target_id=target,
                reason="contradiction",
            )
            for r in results
        ]

    async def _check_progression(
        self, source: str, target: str, new_type: str, old_type: str
    ) -> list[SupersessionAction]:
        """Check for existing relationship that is superseded by progression."""
        results = await self._executor.execute_query(
            "MATCH (s:__Entity__ {id: $source})-[r]->(t:__Entity__ {id: $target}) "
            "WHERE type(r) = $old_type AND r.status = 'active' "
            "RETURN s.id AS source_id, t.id AS target_id, type(r) AS type",
            {"source": source, "target": target, "old_type": old_type},
        )
        return [
            SupersessionAction(
                old_rel_type=r["type"],
                old_target_id=r["target_id"],
                new_target_id=target,
                reason="progression",
            )
            for r in results
        ]
