"""KG-125 (MIS-175): a seed belief keeps its seed origin when reconciliation clamps it.

Before KG-125 every row `_apply_append` wrote was stamped
`provenance='extraction'`, including the clamped copy it appends when a CEASE,
a SINGLE supersession, a contradiction or a progression retires a seed fact --
so the first conversation that retired a seed belief erased the fact that it
had ever been seeded. These tests drive each clamp path through the real
ontology table against a seed prior and pin what the copy carries:

- provenance='seed', source_type='stated', confidence=1.0 (Option A);
- seed_origin_version = the seed row's `seed_version` (D1), or the inherited
  `seed_origin_version` when the prior is itself a clamped copy;
- NO `seed_version` parameter or SET (D1: that property means "written by the
  applier" to the wipe, the gates and the cutover probe);
- the extraction stamps, so the seed-only cutover probe still refuses it.

And the negatives: a fresh assertion and a copy of an extraction row stay
'extraction' with no seed_origin_version.
"""

import pytest

from backend.knowledge.curation.graph_writer import RebuildStamps
from backend.knowledge.curation.reconciliation import (
    _BELIEF_RETURN,
    ActionKind,
    BeliefRow,
    ReconciliationEngine,
)
from backend.knowledge.seed.models import SEED_CONFIDENCE, SEED_PROVENANCE, SEED_SOURCE_TYPE
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection

STAMPS = RebuildStamps(ontology_version="1.2.0", extraction_version="v-test", model_hash="m-test")
RECORDED_AT = "2026-06-10T12:00:00+00:00"
SEED_V = "profile-v3"


def _engine(conn: FakeNeo4jConnection) -> ReconciliationEngine:
    return ReconciliationEngine(executor=FakeGraphExecutor(conn), rebuild_stamps=STAMPS)


def _rel(predicate, target, source="user", props=None, **top):
    return {
        "source": source,
        "target": target,
        "type": predicate,
        "confidence": 0.9,
        "properties": props or {},
        **top,
    }


def _seed_row(**overrides):
    """A belief row as `_BELIEF_RETURN` returns a KG-125 seed edge.

    confidence/source_type are what the applier writes; recorded_at falls back
    to '' because the applier writes neither recorded_at nor created_at on an
    edge; evidence and source_utterance_id are coalesce defaults.
    """
    row = {
        "edge_ref": "ref-seed",
        "target": "rust",
        "valid_from": None,
        "valid_to": None,
        "recorded_at": "",
        "confidence": 1.0,
        "source_type": "stated",
        "context": "",
        "temporal_status": "current",
        "evidence": [],
        "source_utterance_id": "",
        "provenance": "seed",
        "seed_version": SEED_V,
        "seed_origin_version": None,
    }
    row.update(overrides)
    return row


def _pre_w1_seed_row(**overrides):
    """A seed edge written before KG-125: no provenance/source_type/confidence.

    `_BELIEF_RETURN` coalesces the missing confidence/source_type to the
    extraction defaults (0.8, 'extracted'); provenance comes back NULL.
    """
    return _seed_row(provenance=None, confidence=0.8, source_type="extracted", **overrides)


def _extraction_row(**overrides):
    row = {
        "edge_ref": "ref-ext",
        "target": "rust",
        "valid_from": "2024-01-01T00:00:00+00:00",
        "valid_to": None,
        "recorded_at": "2024-01-01T00:00:00+00:00",
        "confidence": 0.7,
        "source_type": "extracted",
        "context": "",
        "temporal_status": "current",
        "evidence": ["e0"],
        "source_utterance_id": "e0",
        "provenance": "extraction",
        "seed_version": None,
        "seed_origin_version": None,
    }
    row.update(overrides)
    return row


def _appends(conn: FakeNeo4jConnection) -> list[tuple[str, dict]]:
    return [(q, p) for q, p in conn.writes if "version_key: $vk" in q]


def _copy_params(conn: FakeNeo4jConnection, target: str) -> dict:
    """The one clamped-copy append for `target` (valid_to set, prior's target)."""
    copies = [p for _, p in _appends(conn) if p["target"] == target and p["valid_to"] is not None]
    assert len(copies) == 1, f"expected exactly one clamped copy for {target!r}: {copies}"
    return copies[0]


def _assert_seed_copy(params: dict, *, origin: str) -> None:
    assert params["provenance"] == SEED_PROVENANCE == "seed"
    assert params["source_type"] == SEED_SOURCE_TYPE == "stated"
    assert params["confidence"] == SEED_CONFIDENCE == 1.0
    assert params["seed_origin_version"] == origin
    # D1: a copy is never "written by the applier".
    assert "seed_version" not in params
    # Stamps stay, so the seed-only cutover probe still refuses the graph.
    assert params["ontology_version"] == "1.2.0"
    assert params["extraction_version"] == "v-test"
    assert params["model_hash"] == "m-test"


class TestAppendQueryShape:
    @pytest.mark.asyncio
    async def test_append_sets_provenance_and_lineage_from_params_never_seed_version(self):
        # A copy of a seed row, so the query text is the one a seed copy uses.
        conn = FakeNeo4jConnection(query_responses={"t.id = $target": [_seed_row()]})
        await _engine(conn).reconcile_turn(
            [_rel("USES", "rust", assertion_kind="cease")],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        query, _ = _appends(conn)[0]
        assert "r.provenance = $provenance" in query
        assert "r.seed_origin_version = $seed_origin_version" in query
        assert "r.seed_version" not in query
        # The literal it replaced must be gone, or a seed copy would be
        # stamped 'extraction' by the query regardless of the parameter.
        assert "r.provenance = 'extraction'" not in query

    def test_belief_return_reads_the_three_properties_raw(self):
        # Raw, not coalesced: the D3 fallback lives in BeliefRow.is_seed only.
        assert "r.provenance AS provenance" in _BELIEF_RETURN
        assert "r.seed_version AS seed_version" in _BELIEF_RETURN
        assert "r.seed_origin_version AS seed_origin_version" in _BELIEF_RETURN
        assert "coalesce(r.provenance" not in _BELIEF_RETURN


class TestSeedClampPaths:
    @pytest.mark.asyncio
    async def test_cease_of_a_seed_row_keeps_seed_provenance(self):
        conn = FakeNeo4jConnection(query_responses={"t.id = $target": [_seed_row()]})
        result = await _engine(conn).reconcile_turn(
            [_rel("USES", "rust", assertion_kind="cease")],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        assert {a.reason for a in result.actions if a.kind is ActionKind.APPEND_CLOSED_COPY} == {
            "cease"
        }
        assert result.appended == 1 and result.closed == 1
        _assert_seed_copy(_copy_params(conn, "rust"), origin=SEED_V)

    @pytest.mark.asyncio
    async def test_single_supersession_of_a_seed_row_keeps_seed_provenance(self):
        conn = FakeNeo4jConnection(
            query_responses={"t.id <> $target": [_seed_row(edge_ref="ref-acme", target="acme")]}
        )
        result = await _engine(conn).reconcile_turn(
            [_rel("WORKS_AT", "initech")],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        assert {a.reason for a in result.actions if a.kind is ActionKind.APPEND_CLOSED_COPY} == {
            "single_supersession"
        }
        _assert_seed_copy(_copy_params(conn, "acme"), origin=SEED_V)
        # The new assertion itself is extraction, with no lineage.
        primary = [p for _, p in _appends(conn) if p["target"] == "initech"]
        assert len(primary) == 1
        assert primary[0]["provenance"] == "extraction"
        assert primary[0]["seed_origin_version"] is None
        assert primary[0]["confidence"] == 0.9

    @pytest.mark.asyncio
    async def test_contradiction_of_a_seed_row_keeps_seed_provenance(self):
        # DISLIKES contradicts INTERESTED_IN (and is not a progression of it).
        conn = FakeNeo4jConnection(
            query_responses={"[r:INTERESTED_IN]": [_seed_row(edge_ref="ref-int", target="jira")]}
        )
        result = await _engine(conn).reconcile_turn(
            [_rel("DISLIKES", "jira")],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        copies = [a for a in result.actions if a.kind is ActionKind.APPEND_CLOSED_COPY]
        assert [(a.reason, a.predicate) for a in copies] == [("contradiction", "INTERESTED_IN")]
        _assert_seed_copy(_copy_params(conn, "jira"), origin=SEED_V)

    @pytest.mark.asyncio
    async def test_progression_of_a_seed_row_keeps_seed_provenance(self):
        # EXPERT_IN supersedes LEARNING by progression (not a contradiction).
        conn = FakeNeo4jConnection(
            query_responses={"[r:LEARNING]": [_seed_row(edge_ref="ref-learn", target="rust")]}
        )
        result = await _engine(conn).reconcile_turn(
            [_rel("EXPERT_IN", "rust")],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        copies = [a for a in result.actions if a.kind is ActionKind.APPEND_CLOSED_COPY]
        assert [(a.reason, a.predicate) for a in copies] == [("progression", "LEARNING")]
        copy = [
            p
            for q, p in _appends(conn)
            if "[r:LEARNING {version_key: $vk}]" in q and p["valid_to"] is not None
        ]
        assert len(copy) == 1
        _assert_seed_copy(copy[0], origin=SEED_V)


class TestPreW1SeedRow:
    """D3: a seed edge written before KG-125 has no provenance but has seed_version."""

    def test_is_seed_falls_back_on_seed_version(self):
        row = BeliefRow(
            edge_ref="r",
            predicate="USES",
            source="user",
            target="rust",
            valid_from=None,
            valid_to=None,
            recorded_at="",
            confidence=0.8,
            source_type="extracted",
            context="",
            temporal_status="current",
            evidence=[],
            seed_version=SEED_V,
        )
        assert row.provenance is None
        assert row.is_seed
        assert row.seed_lineage == SEED_V

    def test_legacy_row_with_neither_is_not_seed(self):
        row = BeliefRow(
            edge_ref="r",
            predicate="USES",
            source="user",
            target="rust",
            valid_from=None,
            valid_to=None,
            recorded_at="",
            confidence=0.8,
            source_type="extracted",
            context="",
            temporal_status="current",
            evidence=[],
        )
        assert not row.is_seed
        assert row.seed_lineage is None

    @pytest.mark.asyncio
    async def test_copy_of_a_pre_w1_seed_row_gets_canonical_seed_values(self):
        # The row reads back as (0.8, 'extracted') through _BELIEF_RETURN's
        # coalesce defaults; the copy must carry the canonical seed values.
        conn = FakeNeo4jConnection(query_responses={"t.id = $target": [_pre_w1_seed_row()]})
        await _engine(conn).reconcile_turn(
            [_rel("USES", "rust", assertion_kind="cease")],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        _assert_seed_copy(_copy_params(conn, "rust"), origin=SEED_V)


class TestCopyOfACopy:
    @pytest.mark.asyncio
    async def test_copy_of_a_clamped_seed_copy_inherits_the_original_seed_version(self):
        # The prior is itself a clamped copy of a seed row (an earlier turn
        # closed it at 2026-01-01): provenance='seed', no seed_version (D1),
        # lineage in seed_origin_version. A cease stating an earlier end
        # shortens it again -- a copy of a copy.
        first_copy = _seed_row(
            edge_ref="ref-copy",
            seed_version=None,
            seed_origin_version="profile-v1",
            valid_from="2020-01-01T00:00:00+00:00",
            valid_to="2026-01-01T00:00:00+00:00",
            recorded_at="2025-06-01T00:00:00+00:00",
            temporal_status="past",
            evidence=["e0"],
            source_utterance_id="e0",
        )
        conn = FakeNeo4jConnection(query_responses={"t.id = $target": [first_copy]})
        result = await _engine(conn).reconcile_turn(
            [_rel("USES", "rust", props={"end_date": "2025-01-01"}, assertion_kind="cease")],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        assert result.appended == 1 and result.closed == 1
        params = _copy_params(conn, "rust")
        # parse_to_bound reads a stated date as an exclusive next-day bound.
        assert params["valid_to"] == "2025-01-02T00:00:00+00:00"
        _assert_seed_copy(params, origin="profile-v1")

    def test_seed_lineage_prefers_the_rows_own_seed_version(self):
        # An applier-written row names its own version even if (impossibly)
        # it also carried an inherited one.
        row = BeliefRow(
            edge_ref="r",
            predicate="USES",
            source="user",
            target="rust",
            valid_from=None,
            valid_to=None,
            recorded_at="",
            confidence=1.0,
            source_type="stated",
            context="",
            temporal_status="current",
            evidence=[],
            provenance="seed",
            seed_version="profile-v2",
            seed_origin_version="profile-v1",
        )
        assert row.seed_lineage == "profile-v2"


class TestNonSeedStaysExtraction:
    @pytest.mark.asyncio
    async def test_fresh_assertion_is_extraction_without_lineage(self):
        conn = FakeNeo4jConnection()
        await _engine(conn).reconcile_turn(
            [_rel("USES", "rust")], recorded_at=RECORDED_AT, event_id="e1", session_id="s1"
        )
        (_, params) = _appends(conn)[0]
        assert params["provenance"] == "extraction"
        assert params["seed_origin_version"] is None
        assert params["ontology_version"] == "1.2.0"

    @pytest.mark.asyncio
    async def test_copy_of_an_extraction_row_stays_extraction(self):
        conn = FakeNeo4jConnection(
            query_responses={"t.id <> $target": [_extraction_row(edge_ref="r-acme", target="acme")]}
        )
        await _engine(conn).reconcile_turn(
            [_rel("WORKS_AT", "initech")],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        params = _copy_params(conn, "acme")
        assert params["provenance"] == "extraction"
        assert params["seed_origin_version"] is None
        # The prior's own values are forwarded, not the seed canon.
        assert params["confidence"] == 0.7
        assert params["source_type"] == "extracted"
        assert params["model_hash"] == "m-test"

    @pytest.mark.asyncio
    async def test_copy_of_a_row_whose_provenance_is_extraction_ignores_seed_version(self):
        # D3 applies only when provenance is ABSENT: a row that says
        # 'extraction' is taken at its word.
        row = _extraction_row(edge_ref="r-acme", target="acme", seed_version=SEED_V)
        conn = FakeNeo4jConnection(query_responses={"t.id <> $target": [row]})
        await _engine(conn).reconcile_turn(
            [_rel("WORKS_AT", "initech")],
            recorded_at=RECORDED_AT,
            event_id="e1",
            session_id="s1",
        )
        params = _copy_params(conn, "acme")
        assert params["provenance"] == "extraction"
        assert params["seed_origin_version"] is None
