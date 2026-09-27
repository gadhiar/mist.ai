"""Entity confidence is reinforced once per event, so a re-applied turn converges.

The extraction backlog re-applies a turn whose curation a crash interrupted
(`backend/extraction_backlog/dispatcher.py`, CRASH SAFETY). Before the guard, the
second run took the upsert's `ON MATCH` branch and raised the confidence of an
entity the first run had just created. The guard skips that reinforce when THIS
event's EXTRACTED_FROM edge to the entity already exists, and that edge (with the
entity's `new_fact` LearningEvent) is written by the same statement as the
entity, so a kill cannot leave the entity without it.

Two kinds of test:
- Cypher shape, over the recording `FakeNeo4jConnection`: the guard is in the
  upsert statement, ahead of the MERGE, bound to this event and session; the
  edge and LearningEvent MERGEs are in that statement too.
- Crash-then-re-apply through the real `CurationGraphWriter`, over
  `ReplayGraphConnection`, a stateful fake that evaluates MERGE, the guard, the
  reinforce for the `confidence` property, the edge's `source_utterance_id` and
  LearningEvent ids only. It does not evaluate Cypher; the eval-Neo4j test in
  `tests/integration/extraction_backlog/` does.
"""

import pytest

from backend.knowledge.curation.confidence import ConfidenceManager
from backend.knowledge.curation.deduplication import MergeAction
from tests.unit.knowledge.curation._graph_writer_fakes import (
    ENTITY_MERGE,
    LEARNING_EVENT_MERGE,
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


class TestProvenanceFoldedIntoUpsert:
    """The EXTRACTED_FROM edge and the `new_fact` LearningEvent are written by
    the entity's own statement, so no kill can separate them from the entity.
    """

    @pytest.mark.asyncio
    async def test_edge_merge_is_in_the_entity_statement_after_the_guard(self) -> None:
        writer, conn = make_writer()

        await writer.write(
            entities=[_entity("rust")], merge_actions=[], event_id="evt-7", session_id=SESSION
        )

        [(query, _params)] = writes_matching(conn, ENTITY_MERGE)
        assert writes_matching(conn, PROVENANCE_EDGE_MERGE) == writes_matching(conn, ENTITY_MERGE)
        guard = query.index("OPTIONAL MATCH (:__Entity__ {id: $entity_id})-[seen:EXTRACTED_FROM]->")
        entity_merge = query.index(ENTITY_MERGE)
        ctx_match = query.index(" MATCH (ctx:ConversationContext {conversation_id: $session_id}) ")
        edge_merge = query.index(PROVENANCE_EDGE_MERGE)
        assert guard < entity_merge < ctx_match < edge_merge

    @pytest.mark.asyncio
    async def test_new_entity_learning_event_is_in_the_entity_statement(self) -> None:
        writer, conn = make_writer()

        result = await writer.write(
            entities=[_entity("rust")], merge_actions=[], event_id="evt-7", session_id=SESSION
        )

        [(query, params)] = writes_matching(conn, "LearningEvent")
        assert ENTITY_MERGE in query
        assert query.index(ENTITY_MERGE) < query.index(LEARNING_EVENT_MERGE)
        assert "le.learning_type = 'new_fact'" in query
        assert "MERGE (le)-[:ABOUT]->(e)" in query
        assert query.endswith(" MERGE (le)-[:LEARNED_FROM]->(ctx)")
        assert params["learning_id"] == "learning-evt-7-new_fact-rust"
        assert params["learning_display_name"] == "new_fact: rust"
        assert result.learning_events_created == 1

    @pytest.mark.asyncio
    async def test_updated_entity_writes_no_learning_event(self) -> None:
        writer, conn = make_writer()
        rust = _entity("rust")
        merge = MergeAction(existing_entity_id="rust", incoming_entity=rust, merge_instructions={})

        result = await writer.write(
            entities=[rust], merge_actions=[merge], event_id="evt-7", session_id=SESSION
        )

        assert writes_matching(conn, "LearningEvent") == []
        [(query, _params)] = writes_matching(conn, ENTITY_MERGE)
        assert PROVENANCE_EDGE_MERGE in query
        assert result.learning_events_created == 0

    @pytest.mark.asyncio
    async def test_one_statement_per_entity_after_the_context(self) -> None:
        writer, conn = make_writer()

        await writer.write(
            entities=[_entity("dev"), _entity("rust")],
            merge_actions=[],
            event_id="evt-7",
            session_id=SESSION,
        )

        [context, dev, rust] = [q for q, _p in conn.writes]
        assert "MERGE (ctx:__Provenance__:ConversationContext" in context
        assert ENTITY_MERGE in dev and PROVENANCE_EDGE_MERGE in dev
        assert ENTITY_MERGE in rust and PROVENANCE_EDGE_MERGE in rust


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


# (history, crashed turn): `history` is applied uninterrupted first, then the
# crashed turn is killed and replayed. Covers a turn of one new entity, a turn
# of two (a kill between entities), and a turn that reinforces an entity an
# earlier turn created next to one it creates.
_KILL_SCENARIOS = [
    pytest.param([], ("evt-0", ["rust"]), id="one-new-entity"),
    pytest.param([], ("evt-0", ["dev", "rust"]), id="two-new-entities"),
    pytest.param(
        [("evt-0", ["dev", "rust"])], ("evt-1", ["dev", "zig"]), id="reinforce-and-create"
    ),
]


Turn = tuple[str, list[str]]


async def _graph_after(
    history: list[Turn], turn: Turn, *, kill_at: int | None
) -> tuple[ReplayGraphConnection, int]:
    """The graph after `history`, then `turn` killed before its `kill_at`-th
    write and re-applied. `kill_at=None` is the single uninterrupted apply.

    Also returns how many writes the first (killed or uninterrupted) apply of
    `turn` issued, the kill included.
    """
    conn = ReplayGraphConnection()
    for event_id, ids in history:
        await _apply(conn, event_id, ids)
    before = len(conn.writes)
    if kill_at is None:
        await _apply(conn, *turn)
        return conn, len(conn.writes) - before
    conn.crash_on_write = before + kill_at
    with pytest.raises(SimulatedCrashError):
        await _apply(conn, *turn)
    turn_writes = len(conn.writes) - before
    await _apply(conn, *turn)
    return conn, turn_writes


async def _replays_after_every_kill(history: list[Turn], turn: Turn):
    """(single apply, [(kill point, replayed graph)]) for a kill before every
    write of the turn and one after its last write.

    The kill points are counted from an uninterrupted apply, not hardcoded, so
    they follow the writer if its statement count changes. "after-last" is a
    kill after the whole curation write and before the `curated` marker: the
    replay re-runs every statement of the turn.
    """
    once, turn_writes = await _graph_after(history, turn, kill_at=None)
    replays: list[tuple[int | str, ReplayGraphConnection]] = []
    for kill_at in range(1, turn_writes + 1):
        crashed, killed_at = await _graph_after(history, turn, kill_at=kill_at)
        assert killed_at == kill_at, "the kill must land inside the turn"
        replays.append((kill_at, crashed))
    after_last, _ = await _graph_after(history, turn, kill_at=None)
    await _apply(after_last, *turn)
    replays.append(("after-last", after_last))
    return once, replays


class TestKillAtAnyStatementThenReplay:
    """A kill before ANY statement of a turn's curation, or after its last one,
    then a replay, leaves the graph a single apply leaves.

    Before the EXTRACTED_FROM and `new_fact` LearningEvent MERGEs moved into
    the entity statement, a kill before the separate edge write left the entity
    with no edge for the replay's guard to find, so it reinforced twice; and a
    kill before the separate LearningEvent write lost the LearningEvent, since
    the replay's dedup sees the entity and treats it as an update.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("history, turn", _KILL_SCENARIOS)
    async def test_node_confidence_equals_a_single_apply(self, history, turn) -> None:
        once, replays = await _replays_after_every_kill(history, turn)

        diverged = {
            k: g.node_confidence for k, g in replays if g.node_confidence != once.node_confidence
        }

        assert diverged == {}, f"single apply: {once.node_confidence}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("history, turn", _KILL_SCENARIOS)
    async def test_learning_events_equal_a_single_apply(self, history, turn) -> None:
        once, replays = await _replays_after_every_kill(history, turn)

        diverged = {
            k: g.learning_events for k, g in replays if g.learning_events != once.learning_events
        }

        assert once.learning_events, "the turn must create at least one new_fact event"
        assert diverged == {}, f"single apply: {sorted(once.learning_events)}"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("history, turn", _KILL_SCENARIOS)
    async def test_extracted_from_edges_equal_a_single_apply(self, history, turn) -> None:
        once, replays = await _replays_after_every_kill(history, turn)

        diverged = {
            k: g.extracted_from for k, g in replays if g.extracted_from != once.extracted_from
        }

        assert diverged == {}, f"single apply: {once.extracted_from}"
