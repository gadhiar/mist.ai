"""C2 ReconciliationEngine -- graph adapter over the pure planner."""

import pytest

from backend.knowledge.curation.graph_writer import RebuildStamps
from backend.knowledge.curation.reconciliation import (
    AssertionKind,
    EdgeAssertion,
    ReconciliationEngine,
    derive_assertion_kind,
)
from backend.knowledge.ontologies import EDGE_TYPES_BY_NAME
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
        # Close-only discriminator + forwarding (tests-quality-7): the old
        # 'recorded_until in query' filter also matched append queries, so
        # the assertion stayed green even if the close never ran.
        close_writes = [
            (q, p) for q, p in conn.writes if "SET r.recorded_until = $recorded_at" in q
        ]
        assert len(close_writes) == 1
        assert close_writes[0][1]["ref"] == "ref-acme"
        assert close_writes[0][1]["recorded_at"] == RECORDED_AT
        # Clamped-copy forwarding pins src=copy_of (prior's payload, not the
        # incoming assertion's).
        copies = [p for q, p in conn.writes if "version_key: $vk" in q and p["target"] == "acme"]
        assert len(copies) == 1
        assert copies[0]["valid_to"] == RECORDED_AT
        assert copies[0]["evidence"] == ["e0"]
        assert copies[0]["confidence"] == 0.8

    @pytest.mark.asyncio
    async def test_reinforce_forwards_prior_ref_and_assertion_confidence(self):
        # tests-quality-7: reinforcement must hit the PRIOR edge with the
        # NEW assertion's confidence and evidence id.
        prior_row = _belief_row(edge_ref="ref-uses-rust")
        conn = FakeNeo4jConnection(query_responses={"t.id = $target": [prior_row]})
        result = await _engine(conn).reconcile_turn(
            [_rel()], recorded_at=RECORDED_AT, event_id="e1", session_id="s1"
        )
        assert result.reinforced == 1
        reinforce_writes = [(q, p) for q, p in conn.writes if "$conf" in q]
        assert len(reinforce_writes) == 1
        _, params = reinforce_writes[0]
        assert params["ref"] == "ref-uses-rust"
        assert params["conf"] == 0.9
        assert params["eid"] == "e1"

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
        result = await _engine(conn).reconcile_turn(
            [_rel(predicate="WORKS_AT", target="initech")],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        # Scenario guard (tests-quality-6): if the conflict-fetch query
        # reshapes and the fake's substring key stops matching, this test
        # must FAIL rather than degrade to a trivially-true single-write
        # ordering assertion.
        assert result.appended == 2 and result.closed == 1
        assert len(conn.writes) == 3
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


class TestCrashReplayConvergence:
    """Commit-protocol crash-window walks (deep review tests-quality-2).

    The apply order is closes/clamped-copies FIRST, probe-bearing primary
    LAST, so a crash mid-commit must replay convergently: version_key MERGE
    keys absorb re-applied copies and nothing destructive repeats.
    """

    @pytest.mark.asyncio
    async def test_supersession_crash_after_copy_replays_to_one_close_and_same_vk(self):
        from backend.errors import Neo4jQueryError

        # Run 1: the clamped copy lands, then the transaction close dies.
        prior_row = _belief_row(edge_ref="ref-acme", target="acme")
        conn1 = FakeNeo4jConnection(
            query_responses={"t.id <> $target": [prior_row]},
            write_errors={"SET r.recorded_until = $recorded_at": Neo4jQueryError("die")},
        )
        with pytest.raises(Neo4jQueryError):
            await _engine(conn1).reconcile_turn(
                [_rel(predicate="WORKS_AT", target="initech")],
                recorded_at=RECORDED_AT,
                event_id="e1",
                session_id="s1",
            )
        copy_writes_1 = [p for q, p in conn1.writes if "version_key: $vk" in q]
        assert len(copy_writes_1) == 1, "the clamped copy must land before the close"
        crashed_copy_vk = copy_writes_1[0]["vk"]

        # Run 2 (replay): post-crash graph = open prior still latest PLUS the
        # landed copy row (is_latest_belief=true, valid_to clamped). The
        # primary never landed, so the (initech) probe misses.
        landed_copy_row = _belief_row(
            edge_ref="ref-acme-copy",
            target="acme",
            valid_to=RECORDED_AT,
            recorded_at=RECORDED_AT,
            source_utterance_id="e1",
        )
        conn2 = FakeNeo4jConnection(
            query_responses={"t.id <> $target": [prior_row, landed_copy_row]}
        )
        result = await _engine(conn2).reconcile_turn(
            [_rel(predicate="WORKS_AT", target="initech")],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )

        # The landed copy's clamped interval no longer overlaps the new open
        # version, so only the ORIGINAL prior is closed -- exactly once.
        close_writes = [
            (q, p) for q, p in conn2.writes if "SET r.recorded_until = $recorded_at" in q
        ]
        assert len(close_writes) == 1
        assert close_writes[0][1]["ref"] == "ref-acme"
        # The re-applied copy re-MERGEs onto the SAME version_key: no second
        # distinct vk for one logical copy across crash + replay.
        copy_writes_2 = [
            p for q, p in conn2.writes if "version_key: $vk" in q and p["target"] == "acme"
        ]
        assert len(copy_writes_2) == 1
        assert copy_writes_2[0]["vk"] == crashed_copy_vk
        # Primary lands last, replay converges to the intended end state.
        assert conn2.writes[-1][1]["target"] == "initech"
        assert result.closed == 1

    @pytest.mark.asyncio
    async def test_cease_crash_after_close_replays_non_destructively(self):
        from backend.errors import Neo4jQueryError

        # Run 1: CEASE closes the prior, then the clamped copy dies.
        prior_row = _belief_row(edge_ref="ref-uses", target="rust")
        conn1 = FakeNeo4jConnection(
            query_responses={"t.id = $target": [prior_row]},
            write_errors={"version_key: $vk": Neo4jQueryError("die")},
        )
        with pytest.raises(Neo4jQueryError):
            await _engine(conn1).reconcile_turn(
                [{**_rel(), "assertion_kind": "cease"}],
                recorded_at=RECORDED_AT,
                event_id="e1",
                session_id="s1",
            )
        assert any("SET r.recorded_until = $recorded_at" in q for q, _ in conn1.writes)

        # Run 2 (replay): the close landed, so the same-fact fetch (filtered
        # to latest beliefs) returns nothing -- the cease flags and writes
        # NOTHING. Non-destructive convergence.
        conn2 = FakeNeo4jConnection()
        result = await _engine(conn2).reconcile_turn(
            [{**_rel(), "assertion_kind": "cease"}],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        assert any("cease_without_prior" in f for f in result.flags)
        conn2.assert_no_writes()


class TestSingleWithMultipleOpenPriors:
    @pytest.mark.asyncio
    async def test_dirty_multi_prior_single_closes_every_prior(self):
        # Pre-invariant dirty data: TWO open WORKS_AT priors (legacy rows
        # predating the SINGLE invariant). A new assertion must close BOTH,
        # each with its own clamped copy (tests-quality-4).
        acme = _belief_row(edge_ref="r1", target="acme", valid_from=None)
        globex = _belief_row(edge_ref="r2", target="globex", valid_from=None)
        conn = FakeNeo4jConnection(query_responses={"t.id <> $target": [acme, globex]})
        result = await _engine(conn).reconcile_turn(
            [_rel(predicate="WORKS_AT", target="initech")],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        assert result.appended == 3  # new version + two clamped copies
        assert result.closed == 2
        close_refs = {
            p["ref"] for q, p in conn.writes if "SET r.recorded_until = $recorded_at" in q
        }
        assert close_refs == {"r1", "r2"}
        copy_targets = {
            p["target"]
            for q, p in conn.writes
            if "version_key: $vk" in q and p["target"] != "initech"
        }
        assert copy_targets == {"acme", "globex"}


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


class TestPastTenseMapping:
    """temporal_status='past' with no stated end maps ASSERT -> CEASE.

    A bare past-tense fact must not mint an OPEN current belief, reinforce
    the belief the user just said ended, or supersede the same-turn current
    fact of a SINGLE predicate (deep review recon-engine-2).
    """

    @pytest.mark.asyncio
    async def test_past_tense_fact_ceases_open_prior(self):
        # "I used to use jira" over a standing open USES jira: close it.
        prior = _belief_row(edge_ref="ref-uses", target="jira")
        conn = FakeNeo4jConnection(query_responses={"[r:USES]": [prior]})
        result = await _engine(conn).reconcile_turn(
            [_rel(predicate="USES", target="jira", props={"temporal_status": "past"})],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        assert result.closed == 1
        assert result.appended == 1  # the clamped copy, no open version
        assert any("past_tense_cease" in f for f in result.flags)

    @pytest.mark.asyncio
    async def test_past_tense_fact_does_not_reinforce_ended_belief(self):
        prior = _belief_row(edge_ref="ref-uses", target="jira")
        conn = FakeNeo4jConnection(query_responses={"[r:USES]": [prior]})
        result = await _engine(conn).reconcile_turn(
            [_rel(predicate="USES", target="jira", props={"temporal_status": "past"})],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        assert result.reinforced == 0

    @pytest.mark.asyncio
    async def test_fresh_past_tense_fact_flags_without_prior(self):
        conn = FakeNeo4jConnection()
        result = await _engine(conn).reconcile_turn(
            [_rel(predicate="USES", target="java", props={"temporal_status": "past"})],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        assert result.appended == 0 and result.closed == 0
        assert any("cease_without_prior" in f for f in result.flags)

    @pytest.mark.asyncio
    async def test_same_turn_employer_transition_keeps_new_employer(self):
        # "I left Titan and joined Acme": titan (past) must not supersede
        # acme (current) -- pre-fix the SINGLE machinery inverted the turn.
        conn = FakeNeo4jConnection()
        result = await _engine(conn).reconcile_turn(
            [
                _rel(predicate="WORKS_AT", target="acme"),
                _rel(predicate="WORKS_AT", target="titan", props={"temporal_status": "past"}),
            ],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        assert result.appended == 1  # acme open version only
        assert result.closed == 0  # titan-as-cease never touches acme
        open_appends = [
            p for q, p in conn.writes if "version_key: $vk" in q and p["valid_to"] is None
        ]
        assert len(open_appends) == 1 and open_appends[0]["target"] == "acme"

    @pytest.mark.asyncio
    async def test_past_with_stated_end_date_stays_assert(self):
        # The resolver filled end_date: the closed-assertion path handles it.
        conn = FakeNeo4jConnection()
        result = await _engine(conn).reconcile_turn(
            [
                _rel(
                    predicate="WORKS_AT",
                    target="titan",
                    props={"temporal_status": "past", "end_date": "2024-01-01"},
                )
            ],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        assert result.appended == 1  # closed historical version recorded
        _, params = conn.writes[0]
        assert params["valid_to"] is not None


class TestConfidenceForwarding:
    @pytest.mark.asyncio
    async def test_validator_shaped_confidence_reaches_edge(self):
        # Validator output carries confidence ONLY under properties; the old
        # top-level read flattened every edge to 0.8 (ontology-semantics-4).
        conn = FakeNeo4jConnection()
        rel = {
            "source": "user",
            "target": "rust",
            "type": "USES",
            "properties": {"confidence": 0.6},
        }
        await _engine(conn).reconcile_turn(
            [rel], recorded_at=RECORDED_AT, event_id="e1", session_id="s1"
        )
        _, params = conn.writes[0]
        assert params["confidence"] == 0.6


class TestDanglingAppends:
    @pytest.mark.asyncio
    async def test_missing_endpoint_append_flags_instead_of_counting(self):
        # MATCH..MATCH..MERGE writes zero rows when an endpoint node is
        # missing; counting that as applied reports success while the fact
        # silently vanishes (recon-engine-3a).
        conn = FakeNeo4jConnection(write_results=[{"n": 0}])
        result = await _engine(conn).reconcile_turn(
            [_rel()], recorded_at=RECORDED_AT, event_id="e1", session_id="s1"
        )
        assert result.appended == 0
        assert any(f.startswith("append_dangling:USES:user->rust") for f in result.flags)


class TestClampedCopyTemporalStatus:
    @pytest.mark.asyncio
    async def test_clamped_copy_written_as_past(self):
        # A copy with a closed valid_to is not a current state; inheriting
        # the prior's 'current' contradicts its own interval (recon-engine-7).
        prior_row = _belief_row(edge_ref="ref-acme", target="acme")
        conn = FakeNeo4jConnection(query_responses={"t.id <> $target": [prior_row]})
        await _engine(conn).reconcile_turn(
            [_rel(predicate="WORKS_AT", target="initech")],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        appends = [(q, p) for q, p in conn.writes if "version_key: $vk" in q]
        copies = [p for _, p in appends if p["valid_to"] is not None]
        opens = [p for _, p in appends if p["valid_to"] is None]
        assert copies and copies[0]["temporal_status"] == "past"
        assert opens and opens[0]["temporal_status"] == "current"


class TestExplicitKindGate:
    """C3: the interim past->CEASE mapping must apply only when the payload
    carried no explicit assertion_kind (pre-r2 events, parse drops). An
    explicit 'assert' on a past-tense EVENT must stay ASSERT so the engine
    accumulates it instead of coercing to CEASE and flagging it away.
    """

    USES = EDGE_TYPES_BY_NAME["USES"]

    def test_explicit_assert_suppresses_past_mapping(self):
        # Explicit assertion_kind='assert' + temporal_status='past' + no
        # end_date must stay ASSERT (past-tense EVENTs accumulate; coercion to
        # CEASE would FLAG_AMBIGUOUS and write nothing).
        rel = _rel(props={"temporal_status": "past", "assertion_kind": "assert"})
        a = EdgeAssertion.from_rel_dict(rel, self.USES)
        assert a.assertion_kind is AssertionKind.ASSERT
        assert a.past_mapped is False

    def test_absent_kind_past_still_maps_to_cease(self):
        # Pre-r2 payloads (no explicit kind) keep the interim past->CEASE map.
        rel = _rel(props={"temporal_status": "past"})
        a = EdgeAssertion.from_rel_dict(rel, self.USES)
        assert a.assertion_kind is AssertionKind.CEASE
        assert a.past_mapped is True

    def test_derive_assertion_kind_is_engine_truth(self):
        # The pure function and from_rel_dict agree on explicit/absent/case/
        # fallback. derive_assertion_kind reads rel-level first, then
        # properties, case-insensitive, including explicit 'assert'.
        cases = [
            ({"assertion_kind": "Cease", "properties": {}}, AssertionKind.CEASE, False),
            ({"properties": {"assertion_kind": "retract"}}, AssertionKind.RETRACT, False),
            ({"properties": {"temporal_status": "past"}}, AssertionKind.CEASE, True),
            (
                {"properties": {"temporal_status": "past", "end_date": "2026-01-01"}},
                AssertionKind.ASSERT,
                False,
            ),
            (
                {"properties": {"assertion_kind": "bogus", "temporal_status": "past"}},
                AssertionKind.CEASE,
                True,
            ),
            ({"temporal_status": "past", "properties": {}}, AssertionKind.CEASE, True),
        ]
        for rel, kind, mapped in cases:
            assert derive_assertion_kind(rel) == (kind, mapped)
