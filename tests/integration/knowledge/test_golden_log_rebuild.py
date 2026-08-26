r"""Graph-level rebuild-twice over the golden log (R1.4.5), staging-isolated.

`tests/unit/golden_log/test_replay.py` proves the replay executes and is deterministic in
the PAYLOAD STREAM it hands to curation. It cannot prove the reconciliation of that stream
is deterministic, because reconciling needs a graph. This does: it reconciles 87 authored
turns into staging through the real curation pipeline and diffs two independent rebuilds
with `assert_rebuild_twice_identical`.

Before the golden log existed this gate compared two graphs that were both just "the seed
applied twice" -- the live log has 0 turns, so the replay loop never ran and the comparison
proved almost nothing about replay.

Requires the staging Neo4j profile:

    docker compose -f docker-compose.yml -f docker-compose.staging-neo4j.yml \\
        --profile staging up -d mist-neo4j-staging

Staging wiring (endpoint discovery, the target-is-not-live guard, and the pipeline factory
call) is imported from `test_log_regenerator` rather than restated, so the two cannot drift.
"""

# ruff: noqa: F811 -- `staging_conn` is a pytest FIXTURE re-exported above so pytest
# registers it; every test then takes a parameter of the same name for injection. Ruff
# reads each parameter as redefining the import. It is the standard fixture pattern, not
# a redefinition defect, and it is file-wide rather than per-line so a new test does not
# have to rediscover it.

from __future__ import annotations

import pytest

from backend.knowledge.canonical_serialize import canonical_graph_form
from backend.knowledge.regeneration.log_regenerator import LogRegenerator
from backend.knowledge.regeneration.rebuild_gate import (
    assert_canonical_form_non_vacuous,
    assert_rebuild_twice_identical,
)
from backend.knowledge.regeneration.rebuild_journal import EventStoreRebuildJournal
from scripts.golden_log.generate import build_golden_turns, materialize_isolated
from tests.integration.knowledge.test_log_regenerator import (
    _ENDPOINT,
    _LIVE_URI,
    _build_staging_pipeline,
    _stage_components,
    _staging_uri,
)
from tests.integration.knowledge.test_log_regenerator import (
    staging_conn as staging_conn,  # noqa: F401 -- re-exported so pytest registers the fixture
)

EXPECTED_TURN_COUNT = 87

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        _ENDPOINT is None,
        reason=(
            "staging Neo4j not running (docker compose -f docker-compose.yml "
            "-f docker-compose.staging-neo4j.yml --profile staging up -d mist-neo4j-staging)"
        ),
    ),
]


async def _rebuild_into_staging(staging_conn, root):
    """Materialize the golden log under `root` and rebuild it into staging."""
    materialized = materialize_isolated(build_golden_turns(), root=root)
    regenerator = LogRegenerator(
        event_store=materialized.event_store,
        extraction_cache=materialized.extraction_cache,
        staging_curation_pipeline=_build_staging_pipeline(staging_conn),
        journal=EventStoreRebuildJournal(materialized.event_store),
        **_stage_components(),
    )
    report = await regenerator.rebuild(
        staging_uri=_staging_uri(),
        live_uri=_LIVE_URI,
        epoch=materialized.epoch,
        # `generate.SESSION_ORIGIN` is "test". `rebuild` now reads the origin
        # guard and defaults to ('real',), so replaying fixture traffic has to
        # be declared. See tests/unit/golden_log/test_replay.py.
        origins=("test",),
    )
    return report


class TestGoldenLogRebuild:
    @pytest.mark.asyncio
    async def test_rebuild_twice_is_byte_identical_over_a_log_with_real_turns(
        self, staging_conn, tmp_path
    ):
        # Act: first rebuild.
        report_a = await _rebuild_into_staging(staging_conn, tmp_path / "run-a")
        form_a = canonical_graph_form(staging_conn, include_provenance=False)

        # Wipe staging between runs (constraints and indexes survive DETACH DELETE).
        staging_conn.execute_write("MATCH (n) DETACH DELETE n", {})

        # Act: second rebuild from scratch.
        report_b = await _rebuild_into_staging(staging_conn, tmp_path / "run-b")
        form_b = canonical_graph_form(staging_conn, include_provenance=False)

        # Assert: fail closed on vacuity first -- an empty log would make the gate trivial.
        assert report_a.turns_processed == EXPECTED_TURN_COUNT
        assert report_b.turns_processed == EXPECTED_TURN_COUNT
        # Was `assert form_a.strip(), ...` -- inert. canonical_graph_form returns a
        # JSON envelope, so an empty graph is truthy after .strip() and this line
        # could never fail. See test_rebuild_gate_vacuity.py.
        assert_canonical_form_non_vacuous(form_a)
        assert_canonical_form_non_vacuous(form_b)

        assert_rebuild_twice_identical(form_a, form_b)

    @pytest.mark.asyncio
    async def test_gold_facts_land_in_the_rebuilt_graph(self, staging_conn, tmp_path):
        # Act
        report = await _rebuild_into_staging(staging_conn, tmp_path / "run")

        # Assert: ext-01 asserts user USES rust, so both endpoints must exist as entities.
        assert report.turns_processed == EXPECTED_TURN_COUNT
        rows = staging_conn.execute_query(
            "MATCH (n:__Entity__) WHERE n.id IN ['user', 'rust'] RETURN count(n) AS n", {}
        )
        assert rows[0]["n"] == 2

    @pytest.mark.asyncio
    async def test_edge_provenance_is_not_an_entity_type(self, staging_conn, tmp_path):
        # Assert: trap 1 checked where it would actually land -- on `r.source_type` in
        # Neo4j, after the whole translate -> cache -> replay -> reconcile path.
        await _rebuild_into_staging(staging_conn, tmp_path / "run")

        rows = staging_conn.execute_query(
            "MATCH (:__Entity__)-[r]->(:__Entity__) WHERE r.source_type IS NOT NULL "
            "RETURN DISTINCT r.source_type AS source_type",
            {},
        )
        assert rows, "no edge carries source_type; the corpus wrote no relationships"
        assert {row["source_type"] for row in rows} == {"extracted"}
