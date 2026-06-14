"""Unit tests for ontology v1.4.0 migration.

Tests the idempotent retype of Topic -> Concept and Milestone -> Event.

Approach: captured-queries + simulated state. A stateful fake executor
interprets the two specific SET queries against an in-memory entity list,
which lets us assert both:
  1. The correct Cypher strings are issued (contract).
  2. The node state transitions match the expected retype (behavior).
  3. A second run is a no-op (idempotency).

The fake also exposes `entity_types()` for concise state assertions.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from scripts.migrations.ontology_v1_4_0 import CYPHER, migrate

# ---------------------------------------------------------------------------
# Stateful fake executor
# ---------------------------------------------------------------------------


class _StatefulMigrationExecutor:
    """In-memory fake that applies the two v1.4.0 migration SET queries.

    Maintains a list of entity dicts with at least `entity_type`. Interprets
    only the two specific migration queries; any other write is captured but
    not applied (this migration issues no other writes).
    """

    def __init__(self, entities: list[dict[str, Any]]) -> None:
        # Deep-copy so tests are independent of the fixture's mutable list.
        self._entities: list[dict[str, Any]] = copy.deepcopy(entities)
        self.writes: list[tuple[str, dict | None]] = []

    async def execute_write(self, query: str, params: dict | None = None) -> list[dict]:
        self.writes.append((query, params))
        self._apply(query)
        return []

    # execute_query is not called by the migration; included for interface completeness.
    async def execute_query(self, query: str, params: dict | None = None) -> list[dict]:
        return []

    def _apply(self, query: str) -> None:
        """Apply the mutation described by the migration Cypher to in-memory entities."""
        if "entity_type = 'Topic'" in query:
            for e in self._entities:
                if e.get("entity_type") == "Topic":
                    e["entity_type"] = "Concept"
                    e["ontology_version"] = "1.4.0"
        elif "entity_type = 'Milestone'" in query:
            for e in self._entities:
                if e.get("entity_type") == "Milestone":
                    e["entity_type"] = "Event"
                    if "event_type" not in e:
                        e["event_type"] = "milestone"
                    e["ontology_version"] = "1.4.0"

    def entity_types(self) -> set[str]:
        """Return the set of entity_type values currently in the fake graph."""
        return {e["entity_type"] for e in self._entities}

    def get_entity(self, entity_id: str) -> dict[str, Any] | None:
        """Return the entity dict for the given id, or None."""
        for e in self._entities:
            if e.get("id") == entity_id:
                return e
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_executor(entities: list[dict[str, Any]]) -> _StatefulMigrationExecutor:
    return _StatefulMigrationExecutor(entities)


@pytest.fixture
def executor_topic_milestone() -> _StatefulMigrationExecutor:
    """Fake executor seeded with one Topic, one Milestone, and one unrelated Skill."""
    return _make_executor(
        [
            {"id": "python-basics", "entity_type": "Topic"},
            {"id": "launch-v1", "entity_type": "Milestone"},
            {"id": "python", "entity_type": "Skill"},
        ]
    )


@pytest.fixture
def executor_already_migrated() -> _StatefulMigrationExecutor:
    """Fake executor whose nodes are already on v1.4.0 types (post-migration state)."""
    return _make_executor(
        [
            {"id": "python-basics", "entity_type": "Concept", "ontology_version": "1.4.0"},
            {
                "id": "launch-v1",
                "entity_type": "Event",
                "event_type": "milestone",
                "ontology_version": "1.4.0",
            },
            {"id": "python", "entity_type": "Skill"},
        ]
    )


@pytest.fixture
def executor_no_legacy_types() -> _StatefulMigrationExecutor:
    """Fake executor with no Topic/Milestone nodes -- near-noop scenario."""
    return _make_executor(
        [
            {"id": "mist-identity", "entity_type": "MistIdentity"},
            {"id": "user", "entity_type": "User"},
            {"id": "python", "entity_type": "Skill"},
        ]
    )


@pytest.fixture
def executor_milestone_with_event_type() -> _StatefulMigrationExecutor:
    """Milestone node that already has a custom event_type -- coalesce must preserve it."""
    return _make_executor(
        [
            {"id": "deadline-q3", "entity_type": "Milestone", "event_type": "deadline"},
        ]
    )


# ---------------------------------------------------------------------------
# Tests: contract (CYPHER constants)
# ---------------------------------------------------------------------------


class TestMigrationContract:
    """The CYPHER list exposes the exact queries the migration will issue."""

    def test_cypher_has_two_entries(self):
        assert len(CYPHER) == 2

    def test_first_query_targets_topic(self):
        assert "entity_type = 'Topic'" in CYPHER[0]

    def test_first_query_sets_concept(self):
        assert "entity_type = 'Concept'" in CYPHER[0] or "'Concept'" in CYPHER[0]

    def test_second_query_targets_milestone(self):
        assert "entity_type = 'Milestone'" in CYPHER[1]

    def test_second_query_sets_event(self):
        assert "entity_type = 'Event'" in CYPHER[1] or "'Event'" in CYPHER[1]

    def test_second_query_stamps_event_type(self):
        assert "event_type" in CYPHER[1]

    def test_queries_stamp_ontology_version(self):
        for q in CYPHER:
            assert "ontology_version" in q and "1.4.0" in q


# ---------------------------------------------------------------------------
# Tests: behavior (state transitions)
# ---------------------------------------------------------------------------


class TestMigrationRelabels:
    """After migrate(), no Topic/Milestone entities remain."""

    @pytest.mark.asyncio
    async def test_topic_becomes_concept(self, executor_topic_milestone):
        await migrate(executor_topic_milestone)

        assert "Topic" not in executor_topic_milestone.entity_types()
        entity = executor_topic_milestone.get_entity("python-basics")
        assert entity is not None
        assert entity["entity_type"] == "Concept"

    @pytest.mark.asyncio
    async def test_milestone_becomes_event(self, executor_topic_milestone):
        await migrate(executor_topic_milestone)

        assert "Milestone" not in executor_topic_milestone.entity_types()
        entity = executor_topic_milestone.get_entity("launch-v1")
        assert entity is not None
        assert entity["entity_type"] == "Event"

    @pytest.mark.asyncio
    async def test_milestone_gains_event_type_milestone(self, executor_topic_milestone):
        await migrate(executor_topic_milestone)

        entity = executor_topic_milestone.get_entity("launch-v1")
        assert entity is not None
        assert entity.get("event_type") == "milestone"

    @pytest.mark.asyncio
    async def test_unrelated_entity_type_unchanged(self, executor_topic_milestone):
        await migrate(executor_topic_milestone)

        entity = executor_topic_milestone.get_entity("python")
        assert entity is not None
        assert entity["entity_type"] == "Skill"

    @pytest.mark.asyncio
    async def test_ontology_version_stamped_on_retypes(self, executor_topic_milestone):
        await migrate(executor_topic_milestone)

        concept = executor_topic_milestone.get_entity("python-basics")
        event = executor_topic_milestone.get_entity("launch-v1")
        assert concept is not None and concept.get("ontology_version") == "1.4.0"
        assert event is not None and event.get("ontology_version") == "1.4.0"

    @pytest.mark.asyncio
    async def test_existing_event_type_preserved_by_coalesce(
        self, executor_milestone_with_event_type
    ):
        # event_type='deadline' is already set; coalesce must NOT overwrite it.
        # Our fake mirrors coalesce: only sets event_type when absent.
        await migrate(executor_milestone_with_event_type)

        entity = executor_milestone_with_event_type.get_entity("deadline-q3")
        assert entity is not None
        assert entity["entity_type"] == "Event"
        assert entity["event_type"] == "deadline"


# ---------------------------------------------------------------------------
# Tests: idempotency
# ---------------------------------------------------------------------------


class TestMigrationIdempotency:
    """A second migrate() call must be a no-op on the entity set."""

    @pytest.mark.asyncio
    async def test_double_run_preserves_concept(self, executor_topic_milestone):
        await migrate(executor_topic_milestone)
        await migrate(executor_topic_milestone)

        assert "Topic" not in executor_topic_milestone.entity_types()
        assert "Concept" in executor_topic_milestone.entity_types()

    @pytest.mark.asyncio
    async def test_double_run_preserves_event(self, executor_topic_milestone):
        await migrate(executor_topic_milestone)
        await migrate(executor_topic_milestone)

        assert "Milestone" not in executor_topic_milestone.entity_types()
        assert "Event" in executor_topic_milestone.entity_types()

    @pytest.mark.asyncio
    async def test_already_migrated_graph_unchanged(self, executor_already_migrated):
        types_before = executor_already_migrated.entity_types()
        await migrate(executor_already_migrated)
        types_after = executor_already_migrated.entity_types()

        assert types_before == types_after

    @pytest.mark.asyncio
    async def test_no_legacy_types_graph_unchanged(self, executor_no_legacy_types):
        types_before = executor_no_legacy_types.entity_types()
        await migrate(executor_no_legacy_types)
        types_after = executor_no_legacy_types.entity_types()

        assert types_before == types_after


# ---------------------------------------------------------------------------
# Tests: query issuance (contract verification at the executor boundary)
# ---------------------------------------------------------------------------


class TestMigrationQueryIssuance:
    """migrate() issues exactly CYPHER[0] and CYPHER[1] in order."""

    @pytest.mark.asyncio
    async def test_issues_exactly_two_writes(self, executor_topic_milestone):
        await migrate(executor_topic_milestone)

        assert len(executor_topic_milestone.writes) == 2

    @pytest.mark.asyncio
    async def test_first_write_is_topic_query(self, executor_topic_milestone):
        await migrate(executor_topic_milestone)

        query, _ = executor_topic_milestone.writes[0]
        assert query == CYPHER[0]

    @pytest.mark.asyncio
    async def test_second_write_is_milestone_query(self, executor_topic_milestone):
        await migrate(executor_topic_milestone)

        query, _ = executor_topic_milestone.writes[1]
        assert query == CYPHER[1]

    @pytest.mark.asyncio
    async def test_each_write_passes_empty_params(self, executor_topic_milestone):
        await migrate(executor_topic_milestone)

        for _, params in executor_topic_milestone.writes:
            assert params == {}
