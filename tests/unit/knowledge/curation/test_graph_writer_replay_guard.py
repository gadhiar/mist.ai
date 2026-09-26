"""Entity confidence is reinforced once per event, so a re-applied turn converges.

The extraction backlog re-applies a turn whose curation a crash interrupted
(`backend/extraction_backlog/dispatcher.py`, CRASH SAFETY). Before the guard, the
second run took the upsert's `ON MATCH` branch and raised the confidence of an
entity the first run had just created. The guard skips that reinforce when THIS
event's EXTRACTED_FROM edge to the entity already exists.

Two kinds of test:
- Cypher shape, over the recording `FakeNeo4jConnection`: the guard is in the
  upsert statement, ahead of the MERGE, bound to this event and session.
- Crash-then-re-apply through the real `CurationGraphWriter`, over
  `ReplayGraphConnection`, a stateful fake that evaluates MERGE, the guard and
  the reinforce for the `confidence` property only. It does not evaluate Cypher;
  the eval-Neo4j test in `tests/integration/extraction_backlog/` does.
"""

import pytest

from backend.knowledge.curation.confidence import ConfidenceManager
from backend.knowledge.curation.deduplication import MergeAction
from tests.unit.knowledge.curation._graph_writer_fakes import (
    PROVENANCE_EDGE_MERGE,
    ReplayGraphConnection,
    SimulatedCrashError,
    make_replay_writer,
    make_writer,
    writes_matching,
)

SESSION = "sess-1"
EXTRACTED = 0.8
# Derived the way the writer derives `$reinforced`, not hardcoded, so a change
# to the ontology's boost policy does not break these tests for the wrong reason.
_MANAGER = ConfidenceManager()
REINFORCED = _MANAGER.reinforced_confidence(EXTRACTED, _MANAGER.determine_domain("Technology"))


def _entity(entity_id: str) -> dict:
    return {
        "id": entity_id,
        "type": "Technology",
        "name": entity_id.title(),
        "confidence": EXTRACTED,
    }


async def _apply(conn: ReplayGraphConnection, event_id: str, ids: list[str]) -> None:
    """One curation write of a turn, with the merge actions dedup would produce.

    Dedup rewrites an incoming entity onto an existing node and emits a
    MergeAction for it (`deduplication.py`, `_find_existing` tier 1: exact id),
    so a re-applied turn arrives with merge actions for every entity the first
    run created.
    """
    entities = [_entity(i) for i in ids]
    merge_actions = [
        MergeAction(existing_entity_id=e["id"], incoming_entity=e, merge_instructions={})
        for e in entities
        if e["id"] in conn.node_confidence
    ]
    await make_replay_writer(conn).write(
        entities=entities, merge_actions=merge_actions, event_id=event_id, session_id=SESSION
    )


def test_fixture_reinforce_is_observable() -> None:
    """Every equality below would be vacuous if reinforcing changed nothing."""
    assert REINFORCED > EXTRACTED


class TestUpsertCarriesReplayGuard:
    @pytest.mark.asyncio
    async def test_guard_is_in_the_upsert_statement_ahead_of_the_merge(self) -> None:
        writer, conn = make_writer()

        await writer.write(
            entities=[_entity("rust")], merge_actions=[], event_id="evt-7", session_id=SESSION
        )

        (query, _params), *rest = writes_matching(conn, "MERGE (e:__Entity__ {id: $entity_id})")
        assert rest == []
        guard = query.index("OPTIONAL MATCH (:__Entity__ {id: $entity_id})-[seen:EXTRACTED_FROM]->")
        assert guard < query.index("MERGE (e:__Entity__ {id: $entity_id})")
        assert "(:ConversationContext {conversation_id: $session_id}) " in query
        assert "WHERE seen.source_utterance_id = $event_id " in query
        assert "WITH count(seen) > 0 AS event_already_applied " in query

    @pytest.mark.asyncio
    async def test_guard_parameters_carry_this_event_and_session(self) -> None:
        writer, conn = make_writer()

        await writer.write(
            entities=[_entity("rust")], merge_actions=[], event_id="evt-7", session_id=SESSION
        )

        [(_query, params)] = writes_matching(conn, "MERGE (e:__Entity__ {id: $entity_id})")
        assert params["event_id"] == "evt-7"
        assert params["session_id"] == SESSION

    @pytest.mark.asyncio
    async def test_guard_reads_the_property_the_provenance_edge_writes(self) -> None:
        """The guard's key and the edge's key must be the same event id."""
        writer, conn = make_writer()

        await writer.write(
            entities=[_entity("rust")], merge_actions=[], event_id="evt-7", session_id=SESSION
        )

        [(_upsert, upsert_params)] = writes_matching(conn, "MERGE (e:__Entity__ {id: $entity_id})")
        [(edge, edge_params)] = writes_matching(conn, PROVENANCE_EDGE_MERGE)
        assert "r.source_utterance_id = $event_id" in edge
        assert "MATCH (ctx:ConversationContext {conversation_id: $session_id})" in edge
        assert edge_params["event_id"] == upsert_params["event_id"]
        assert edge_params["session_id"] == upsert_params["session_id"]

    @pytest.mark.asyncio
    async def test_guard_gates_only_the_confidence_assignment(self) -> None:
        """display_name, description and updated_at keep their ON MATCH semantics."""
        writer, conn = make_writer()

        await writer.write(
            entities=[_entity("rust")], merge_actions=[], event_id="evt-7", session_id=SESSION
        )

        [(query, _params)] = writes_matching(conn, "MERGE (e:__Entity__ {id: $entity_id})")
        on_match = query.split("ON MATCH SET", 1)[1]
        assert on_match.count("event_already_applied") == 1
        assert on_match.startswith(
            " e.confidence = CASE WHEN event_already_applied THEN e.confidence "
        )
        assert "e.updated_at = $now" in on_match


class TestCrashThenReapply:
    @pytest.mark.asyncio
    async def test_reapplying_the_same_event_leaves_confidence_as_one_apply(self) -> None:
        once = ReplayGraphConnection()
        await _apply(once, "evt-0", ["rust"])
        replayed = ReplayGraphConnection()
        await _apply(replayed, "evt-0", ["rust"])

        # A crash after the graph write and before the `curated` marker re-runs it.
        await _apply(replayed, "evt-0", ["rust"])

        assert once.node_confidence == {"rust": EXTRACTED}
        assert replayed.node_confidence == once.node_confidence

    @pytest.mark.asyncio
    async def test_a_new_event_on_the_same_entity_still_reinforces(self) -> None:
        conn = ReplayGraphConnection()
        await _apply(conn, "evt-0", ["rust"])

        await _apply(conn, "evt-1", ["rust"])

        assert conn.node_confidence == {"rust": REINFORCED}

    @pytest.mark.asyncio
    async def test_a_new_event_after_a_replayed_one_still_reinforces(self) -> None:
        conn = ReplayGraphConnection()
        await _apply(conn, "evt-0", ["rust"])
        await _apply(conn, "evt-0", ["rust"])

        await _apply(conn, "evt-1", ["rust"])

        assert conn.node_confidence == {"rust": REINFORCED}

    @pytest.mark.asyncio
    async def test_two_turn_crash_replay_matches_the_uninterrupted_run(self) -> None:
        """The T2a scenario: turn 0 (dev, rust) crashes after its write, then turn 1."""
        uninterrupted = ReplayGraphConnection()
        await _apply(uninterrupted, "evt-0", ["dev", "rust"])
        await _apply(uninterrupted, "evt-1", ["dev", "zig"])
        crashed = ReplayGraphConnection()
        await _apply(crashed, "evt-0", ["dev", "rust"])

        await _apply(crashed, "evt-0", ["dev", "rust"])
        await _apply(crashed, "evt-1", ["dev", "zig"])

        assert uninterrupted.node_confidence == {
            "dev": REINFORCED,
            "rust": EXTRACTED,
            "zig": EXTRACTED,
        }
        assert crashed.node_confidence == uninterrupted.node_confidence

    @pytest.mark.asyncio
    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Known residual window, documented in `_upsert_entity`: a kill between "
            "the entity upsert and its EXTRACTED_FROM MERGE leaves no edge for the "
            "guard to find, so the replay reinforces. Closing it means writing the "
            "edge in the upsert statement; when that lands this XPASSes -- drop the "
            "xfail then."
        ),
    )
    async def test_kill_between_upsert_and_edge_write_still_converges(self) -> None:
        once = ReplayGraphConnection()
        await _apply(once, "evt-0", ["rust"])
        # Writes of one single-entity turn: 1 context, 2 upsert, 3 new_fact
        # LearningEvent, 4 EXTRACTED_FROM. Kill before the 4th takes effect.
        crashed = ReplayGraphConnection(crash_on_write=4)
        with pytest.raises(SimulatedCrashError):
            await _apply(crashed, "evt-0", ["rust"])
        assert PROVENANCE_EDGE_MERGE in crashed.writes[-1][0]

        await _apply(crashed, "evt-0", ["rust"])

        assert crashed.node_confidence == once.node_confidence
