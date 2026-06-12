"""C2 ReconciliationEngine -- graph adapter over the pure planner."""

import pytest

from backend.knowledge.curation.graph_writer import RebuildStamps
from backend.knowledge.curation.reconciliation import ReconciliationEngine
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection

STAMPS = RebuildStamps(ontology_version="1.2.0", extraction_version="v-test", model_hash="m-test")
RECORDED_AT = "2026-06-10T12:00:00+00:00"


def _engine(conn: FakeNeo4jConnection) -> ReconciliationEngine:
    return ReconciliationEngine(executor=FakeGraphExecutor(conn), rebuild_stamps=STAMPS)


def _rel(predicate="USES", source="user", target="rust", props=None):
    return {
        "source": source,
        "target": target,
        "type": predicate,
        "confidence": 0.9,
        "properties": props or {},
    }


def _belief_row(**overrides):
    row = {
        "edge_ref": "ref-1",
        "target": "rust",
        "valid_from": "2024-01-01T00:00:00+00:00",
        "valid_to": None,
        "recorded_at": "2024-01-01T00:00:00+00:00",
        "confidence": 0.8,
        "source_type": "extracted",
        "context": "",
        "temporal_status": "current",
        "evidence": ["e0"],
        "source_utterance_id": "e0",
    }
    row.update(overrides)
    return row


class TestReconcileTurn:
    @pytest.mark.asyncio
    async def test_fresh_assert_appends_bitemporal_version(self):
        conn = FakeNeo4jConnection()  # all reads return [] -> no priors
        result = await _engine(conn).reconcile_turn(
            [_rel()], recorded_at=RECORDED_AT, event_id="e1", session_id="s1"
        )
        assert result.appended == 1 and result.closed == 0
        query, params = conn.writes[0]
        assert "MERGE (s)-[r:USES {version_key: $vk}]->(t)" in query
        assert params["vk"] == f"e1|{RECORDED_AT}|open"
        assert params["eid"] == "e1"
        assert params["recorded_at"] == RECORDED_AT
        assert params["valid_from"] == RECORDED_AT
        assert params["valid_to"] is None
        assert params["ontology_version"] == "1.2.0"
        # Bug A regression guard (rel side): extraction-time edges must carry
        # provenance='extraction' (moved here from the old writer tests).
        assert "r.provenance = 'extraction'" in query

    @pytest.mark.asyncio
    async def test_stated_start_date_lands_in_valid_from(self):
        conn = FakeNeo4jConnection()
        await _engine(conn).reconcile_turn(
            [_rel(props={"start_date": "2026-05"})],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        _, params = conn.writes[0]
        assert params["valid_from"] == "2026-05-01T00:00:00+00:00"

    @pytest.mark.asyncio
    async def test_single_supersession_round_trip(self):
        # Prior WORKS_AT acme exists; new WORKS_AT initech must append the new
        # version, append a clamped copy of the prior, and transaction-close it.
        prior_row = _belief_row(edge_ref="ref-acme", target="acme")
        conn = FakeNeo4jConnection(
            query_responses={
                "t.id <> $target": [prior_row],  # single-conflict fetch
            }
        )
        result = await _engine(conn).reconcile_turn(
            [_rel(predicate="WORKS_AT", target="initech")],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        assert result.appended == 2  # new version + clamped copy
        assert result.closed == 1
        close_writes = [q for q, _ in conn.writes if "recorded_until" in q and "SET" in q]
        assert close_writes, "transaction close must be written"

    @pytest.mark.asyncio
    async def test_turn_replay_is_noop(self):
        conn = FakeNeo4jConnection(
            query_responses={"source_utterance_id: $eid": [{"n": 1}]}  # idempotency probe hit
        )
        result = await _engine(conn).reconcile_turn(
            [_rel()], recorded_at=RECORDED_AT, event_id="e1", session_id="s1"
        )
        assert result.appended == 0 and result.closed == 0
        conn.assert_no_writes()

    @pytest.mark.asyncio
    async def test_durable_structural_upsert(self):
        conn = FakeNeo4jConnection()
        result = await _engine(conn).reconcile_turn(
            [_rel(predicate="IS_A", source="python", target="programming-language")],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        assert result.structural == 1
        query, _ = conn.writes[0]
        assert "MERGE (s)-[r:IS_A]->(t)" in query  # keyed by pair, not eid

    @pytest.mark.asyncio
    async def test_symmetric_predicate_canonicalizes_order(self):
        conn = FakeNeo4jConnection()
        await _engine(conn).reconcile_turn(
            [_rel(predicate="WORKS_WITH", source="zustand", target="react")],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        _, params = conn.writes[0]
        assert params["source"] == "react" and params["target"] == "zustand"

    @pytest.mark.asyncio
    async def test_unknown_predicate_is_skipped_with_flag(self):
        conn = FakeNeo4jConnection()
        result = await _engine(conn).reconcile_turn(
            [_rel(predicate="NOT_A_PREDICATE")],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        assert result.flags and not conn.writes


class TestReviewAdditions:
    @pytest.mark.asyncio
    async def test_primary_append_is_the_last_write(self):
        # Commit-protocol ordering: closes/copies land before the probe-bearing
        # primary version, so partial application replays convergently.
        prior_row = _belief_row(edge_ref="ref-acme", target="acme")
        conn = FakeNeo4jConnection(query_responses={"t.id <> $target": [prior_row]})
        await _engine(conn).reconcile_turn(
            [_rel(predicate="WORKS_AT", target="initech")],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        last_query, last_params = conn.writes[-1]
        assert "MERGE (s)-[r:WORKS_AT {version_key: $vk}]->(t)" in last_query
        assert last_params["target"] == "initech"  # the primary, not the copy

    @pytest.mark.asyncio
    async def test_two_copies_in_one_turn_get_distinct_version_keys(self):
        # Cease over a true->false->true history: two same-fact latest versions
        # must clamp into TWO surviving copies, not silently MERGE into one.
        v1 = _belief_row(edge_ref="ref-v1", valid_from="2020-01-01T00:00:00+00:00")
        v2 = _belief_row(
            edge_ref="ref-v2",
            valid_from="2024-01-01T00:00:00+00:00",
            recorded_at="2024-01-01T00:00:00+00:00",
        )
        conn = FakeNeo4jConnection(query_responses={"t.id = $target": [v1, v2]})
        await _engine(conn).reconcile_turn(
            [{**_rel(), "assertion_kind": "cease"}],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        copy_keys = [p["vk"] for q, p in conn.writes if "version_key: $vk" in q]
        assert len(copy_keys) == 2 and len(set(copy_keys)) == 2

    @pytest.mark.asyncio
    async def test_symmetric_reverse_reassertion_reinforces(self):
        # B WORKS_WITH A after A WORKS_WITH B reinforces the canonical edge.
        canonical_row = _belief_row(edge_ref="ref-ww", target="zustand")
        conn = FakeNeo4jConnection(query_responses={"t.id = $target": [canonical_row]})
        result = await _engine(conn).reconcile_turn(
            [_rel(predicate="WORKS_WITH", source="zustand", target="react")],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        assert result.reinforced == 1 and result.appended == 0

    @pytest.mark.asyncio
    async def test_intra_turn_processing_is_sorted(self):
        conn = FakeNeo4jConnection()
        await _engine(conn).reconcile_turn(
            [_rel(predicate="USES", target="zz"), _rel(predicate="DISLIKES", target="aa")],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        appended_types = [q for q, _ in conn.writes if "version_key: $vk" in q]
        assert "DISLIKES" in appended_types[0] and "USES" in appended_types[1]


class TestSemanticsTableBehavior:
    """Regression guards for the v1.2.1 semantics-table corrections.

    Usage (behavior) is orthogonal to dislike (sentiment) and struggle
    (competence): asserting one must never retire the other. These drive
    the REAL ontology table through _fetch_existing, so they fail if the
    erasure pairs are ever reintroduced.
    """

    @pytest.mark.asyncio
    async def test_dislikes_does_not_retire_open_uses_fact(self):
        # "I hate jira" after a standing "I use jira": both must stay current.
        prior_uses = _belief_row(edge_ref="ref-uses", target="jira")
        conn = FakeNeo4jConnection(query_responses={"[r:USES]": [prior_uses]})
        result = await _engine(conn).reconcile_turn(
            [_rel(predicate="DISLIKES", target="jira")],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        assert result.appended == 1 and result.closed == 0

    @pytest.mark.asyncio
    async def test_uses_does_not_retire_open_dislikes_fact(self):
        # "I use jira" after a standing "I hate jira": the aversion survives.
        prior_dislikes = _belief_row(edge_ref="ref-dislikes", target="jira")
        conn = FakeNeo4jConnection(query_responses={"[r:DISLIKES]": [prior_dislikes]})
        result = await _engine(conn).reconcile_turn(
            [_rel(predicate="USES", target="jira")],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        assert result.appended == 1 and result.closed == 0

    @pytest.mark.asyncio
    async def test_uses_does_not_retire_open_struggles_with_fact(self):
        # "I've used Docker for a year and still struggle with it" -- co-true.
        prior_struggle = _belief_row(edge_ref="ref-struggle", target="docker")
        conn = FakeNeo4jConnection(query_responses={"[r:STRUGGLES_WITH]": [prior_struggle]})
        result = await _engine(conn).reconcile_turn(
            [_rel(predicate="USES", target="docker")],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        assert result.appended == 1 and result.closed == 0

    @pytest.mark.asyncio
    async def test_expert_in_supersedes_struggle_exactly_once(self):
        # STRUGGLES_WITH sits in both EXPERT_IN.contradicts and
        # .progression_supersedes; the duplicate fetch+candidate must collapse
        # to ONE clamped copy and ONE close (counters = actual writes).
        prior_struggle = _belief_row(edge_ref="ref-struggle", target="docker")
        conn = FakeNeo4jConnection(query_responses={"[r:STRUGGLES_WITH]": [prior_struggle]})
        result = await _engine(conn).reconcile_turn(
            [_rel(predicate="EXPERT_IN", target="docker")],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        assert result.appended == 2  # new version + exactly one clamped copy
        assert result.closed == 1
