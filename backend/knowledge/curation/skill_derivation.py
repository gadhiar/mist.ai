"""Skill derivation from tool usage patterns.

Analyzes detected ToolUsagePatterns and creates/updates Skill and
MistCapability entities in the knowledge graph. Writes directly via
GraphExecutor -- does NOT use InternalKnowledgeDeriver or VALID_OPS.
"""

import logging
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime

from backend.knowledge.config import SkillDerivationConfig
from backend.knowledge.extraction.tool_usage_tracker import ToolUsageTracker
from backend.knowledge.storage.graph_executor import GraphExecutor

logger = logging.getLogger(__name__)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(text: str) -> str:
    """Convert text to a URL-safe slug identifier."""
    return _SLUG_RE.sub("-", text.lower()).strip("-")


@dataclass(frozen=True, slots=True)
class SkillDerivationResult:
    """Result of a skill derivation run."""

    patterns_detected: int
    skills_created: int
    skills_updated: int
    capabilities_created: int
    duration_ms: float


class SkillDerivationJob:
    """Curation job that derives Skill entities from tool usage patterns.

    Mirrors the SelfReflectionJob pattern: registered with CurationScheduler,
    exposes an async ``run()`` method, writes directly to the graph via
    GraphExecutor.

    Args:
        tracker: ToolUsageTracker with recorded tool calls.
        executor: GraphExecutor for async graph writes.
        config: SkillDerivationConfig controlling thresholds.
    """

    def __init__(
        self,
        tracker: ToolUsageTracker,
        executor: GraphExecutor,
        config: SkillDerivationConfig,
    ) -> None:
        self._tracker = tracker
        self._executor = executor
        self._config = config

    async def run(self) -> SkillDerivationResult:
        """Detect patterns and create/update Skill and MistCapability entities.

        For each detected pattern:
            1. Query graph for existing Skill with matching id (dedup).
            2. If exists: UPDATE proficiency via Cypher MERGE.
            3. If not exists and occurrence >= skill_threshold: CREATE Skill
               + KNOWS edge from User entity.
            4. If success_count >= capability_threshold: CREATE/UPDATE
               MistCapability + HAS_CAPABILITY edge from MistIdentity.

        Returns:
            SkillDerivationResult with counts.
        """
        if not self._config.enabled:
            return SkillDerivationResult(
                patterns_detected=0,
                skills_created=0,
                skills_updated=0,
                capabilities_created=0,
                duration_ms=0.0,
            )

        start = time.perf_counter()
        patterns = self._tracker.detect_patterns()

        skills_created = 0
        skills_updated = 0
        capabilities_created = 0

        for pattern in patterns:
            display_name = f"{pattern.context_summary} ({pattern.tool_type})"
            skill_id = f"skill-{_slugify(display_name)}"
            proficiency = min(1.0, pattern.success_count / (self._config.capability_threshold * 2))
            now = datetime.now(UTC).isoformat()

            try:
                existing = await self._find_existing_skill(skill_id)

                if existing:
                    await self._update_skill(skill_id, proficiency, now)
                    skills_updated += 1
                else:
                    await self._create_skill(
                        skill_id, display_name, proficiency, pattern.tool_type, now
                    )
                    skills_created += 1

                if pattern.success_count >= self._config.capability_threshold:
                    created = await self._ensure_capability(
                        skill_id, display_name, pattern.tool_type, proficiency, now
                    )
                    if created:
                        capabilities_created += 1

            except Exception as e:
                logger.error("Skill derivation failed for pattern '%s': %s", display_name, e)

        elapsed = (time.perf_counter() - start) * 1000

        logger.info(
            "Skill derivation: %d patterns, %d created, %d updated, %d capabilities in %.1fms",
            len(patterns),
            skills_created,
            skills_updated,
            capabilities_created,
            elapsed,
        )

        return SkillDerivationResult(
            patterns_detected=len(patterns),
            skills_created=skills_created,
            skills_updated=skills_updated,
            capabilities_created=capabilities_created,
            duration_ms=elapsed,
        )

    async def _find_existing_skill(self, skill_id: str) -> bool:
        """Check whether a Skill entity already exists in the graph."""
        results = await self._executor.execute_query(
            "MATCH (e:__Entity__ {id: $skill_id, entity_type: 'Skill'}) "
            "RETURN e.id AS id LIMIT 1",
            {"skill_id": skill_id},
        )
        return len(results) > 0

    async def _create_skill(
        self,
        skill_id: str,
        display_name: str,
        proficiency: float,
        tool_type: str,
        now: str,
    ) -> None:
        """Create a new Skill entity and KNOWS edge from User."""
        await self._executor.execute_write(
            "MERGE (e:__Entity__ {id: $skill_id}) "
            "ON CREATE SET "
            "  e.entity_type = 'Skill', "
            "  e.display_name = $display_name, "
            "  e.knowledge_domain = 'internal', "
            "  e.source_type = 'tool_usage_derived', "
            "  e.proficiency = $proficiency, "
            "  e.tool_type = $tool_type, "
            "  e.status = 'active', "
            "  e.created_at = $now, "
            "  e.updated_at = $now, "
            "  e.ontology_version = '1.0.0' "
            "WITH e "
            "MERGE (u:__Entity__ {id: 'user'}) "
            "MERGE (u)-[:KNOWS]->(e)",
            {
                "skill_id": skill_id,
                "display_name": display_name,
                "proficiency": proficiency,
                "tool_type": tool_type,
                "now": now,
            },
        )

    async def _update_skill(self, skill_id: str, proficiency: float, now: str) -> None:
        """Update proficiency on an existing Skill entity."""
        await self._executor.execute_write(
            "MATCH (e:__Entity__ {id: $skill_id, entity_type: 'Skill'}) "
            "SET e.proficiency = $proficiency, e.updated_at = $now",
            {"skill_id": skill_id, "proficiency": proficiency, "now": now},
        )

    async def _ensure_capability(
        self,
        skill_id: str,
        display_name: str,
        tool_type: str,
        proficiency: float,
        now: str,
    ) -> bool:
        """Create or update a MistCapability entity and HAS_CAPABILITY edge.

        Returns:
            True if the capability was newly created, False if updated.
        """
        capability_id = f"cap-{_slugify(display_name)}"

        existing = await self._executor.execute_query(
            "MATCH (e:__Entity__ {id: $cap_id, entity_type: 'MistCapability'}) "
            "RETURN e.id AS id LIMIT 1",
            {"cap_id": capability_id},
        )

        if existing:
            await self._executor.execute_write(
                "MATCH (e:__Entity__ {id: $cap_id, entity_type: 'MistCapability'}) "
                "SET e.proficiency = $proficiency, e.updated_at = $now",
                {"cap_id": capability_id, "proficiency": proficiency, "now": now},
            )
            return False

        await self._executor.execute_write(
            "MERGE (e:__Entity__ {id: $cap_id}) "
            "ON CREATE SET "
            "  e.entity_type = 'MistCapability', "
            "  e.display_name = $display_name, "
            "  e.knowledge_domain = 'internal', "
            "  e.source_type = 'tool_usage_derived', "
            "  e.capability_type = $tool_type, "
            "  e.proficiency = $proficiency, "
            "  e.status = 'active', "
            "  e.created_at = $now, "
            "  e.updated_at = $now, "
            "  e.ontology_version = '1.0.0' "
            "WITH e "
            "MATCH (m:MistIdentity {id: 'mist-identity'}) "
            "MERGE (m)-[:HAS_CAPABILITY]->(e)",
            {
                "cap_id": capability_id,
                "display_name": display_name,
                "tool_type": tool_type,
                "proficiency": proficiency,
                "now": now,
            },
        )
        return True
