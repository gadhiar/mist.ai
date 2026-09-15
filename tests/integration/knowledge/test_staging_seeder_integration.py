"""Integration: `StagingSeeder` against a real Neo4j, composed as `rebuild()` calls it.

## The gap this closes

MIS-130 step B shipped `StagingSeeder` with unit coverage over a recording
connection and NO integration coverage at all. The merge commit and
`tests/mocks/seeder.py` both recorded that gap rather than leaving it to be
discovered: the three functions it composes each have R1.4 integration coverage,
but the COMPOSITION -- apply, then backfill, then RE-READ, in that order, against
one connection -- had none. The eval and staging Neo4j instances were down for
five weeks, and an integration test that has never been executed is a claim
rather than evidence.

The instances came up 2026-09-15 and this is the test that was owed.

## What only a real graph can show

The unit tests pin the composition's SHAPE against a fake connection that answers
every read with a fixed list. They cannot show that:

- `apply_seed_documents` actually routes a `partition: __SelfModel__` document's
  nodes onto `:__SelfModel__` rather than onto a colliding `:__Entity__` copy.
  That routing is `_assign_node_partitions`' job and it is invisible to a fake.
- The backfill's writes and `check_embeddings`' reads agree about WHICH nodes
  they are talking about -- they match on `seed_version` across a label union,
  and a mismatch there would make the gate examine zero nodes while reporting
  a pass.
- The embedding property survives the round trip at the declared dimension.

The last two are the ones that have cost this project live data. `seed/gates.py`
is explicit that `canonical_serialize` excludes `embedding`, so an unembedded
graph is byte-identical to a correct one and nothing downstream of the seeder can
tell them apart.

## Scope, stated so this is not read as more than it is

This exercises the seeder against a real graph. It does NOT run a rebuild, and
nothing here has been hydrated or gated. `rebuild()`'s use of the seeder --
ordering relative to the replay loop, the two floors -- is unit-pinned in
tests/unit/knowledge/regeneration/test_rebuild_applies_the_seed.py and unchanged
by this file.
"""

from __future__ import annotations

import socket

import pytest

from backend.knowledge.config import Neo4jConfig
from backend.knowledge.embeddings.embedding_generator import EmbeddingGenerator
from backend.knowledge.regeneration.staging_seeder import StagingSeeder
from backend.knowledge.seed.models import SeedDocument, SeedFact, SeedNode
from backend.knowledge.storage.neo4j_connection import Neo4jConnection

_CANDIDATES = (("mist-neo4j-staging", 7687), ("localhost", 7689))


def _staging_endpoint() -> tuple[str, int] | None:
    for host, port in _CANDIDATES:
        try:
            socket.create_connection((host, port), timeout=2).close()
            return host, port
        except OSError:
            continue
    return None


_ENDPOINT = _staging_endpoint()
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

_SEED_VERSION = "integration-seed-1"
_NOW = "2026-07-01T09:00:00+00:00"
_DIMENSION = 384


@pytest.fixture
def staging_conn():
    host, port = _ENDPOINT  # type: ignore[misc]
    conn = Neo4jConnection(
        Neo4jConfig(uri=f"bolt://{host}:{port}", username="neo4j", password="password")
    )
    conn.connect()
    conn.execute_write("MATCH (n) DETACH DELETE n", {})
    yield conn
    conn.execute_write("MATCH (n) DETACH DELETE n", {})
    conn.disconnect()


@pytest.fixture(scope="module")
def embedder():
    """The REAL generator. all-MiniLM-L6-v2 is deterministic for identical text."""
    return EmbeddingGenerator("all-MiniLM-L6-v2")


def _self_model_document() -> SeedDocument:
    """A `partition: __SelfModel__` document, the case the copy-forward used to serve."""
    return SeedDocument(
        seed_version=_SEED_VERSION,
        nodes=[
            SeedNode(id="mist-identity", type="MistIdentity"),
            SeedNode(id="mist-trait-warm", type="MistTrait"),
        ],
        facts=[SeedFact(subject="mist-identity", predicate="HAS_TRAIT", object="mist-trait-warm")],
        body="MIST is warm.",
        source_path="mist.md",
        partition="__SelfModel__",
    )


def _seeder(conn, embedder):
    return StagingSeeder(
        connection=conn,
        documents=[_self_model_document()],
        seed_version=_SEED_VERSION,
        embedding_generator=embedder,
        expected_dimension=_DIMENSION,
    )


class TestTheCompositionAgainstARealGraph:
    def test_the_apply_lands_on_the_self_model_partition(self, staging_conn, embedder):
        """`partition: __SelfModel__` must route to `:__SelfModel__`, not `:__Entity__`.

        Invisible to a fake connection: the routing happens inside
        `_assign_node_partitions` and only a real graph can report which label the
        node actually carries. A colliding `:__Entity__` copy would put seed
        content directly into the partition the gate compares.
        """
        result = _seeder(staging_conn, embedder).apply(now_iso=_NOW)

        assert result.nodes_written == 2
        rows = staging_conn.execute_query(
            "MATCH (n:__SelfModel__) RETURN n.id AS id ORDER BY n.id", {}
        )
        assert [r["id"] for r in rows] == ["mist-identity", "mist-trait-warm"]

        entity_rows = staging_conn.execute_query("MATCH (n:__Entity__) RETURN n.id AS id", {})
        assert entity_rows == [], (
            f"self-model seed content leaked onto :__Entity__ ({entity_rows}). That is "
            "the partition the live==rebuilt gate compares."
        )

    def test_the_fact_is_written_as_an_edge(self, staging_conn, embedder):
        result = _seeder(staging_conn, embedder).apply(now_iso=_NOW)

        assert result.facts_written == 1
        rows = staging_conn.execute_query(
            "MATCH (:__SelfModel__ {id: 'mist-identity'})-[r:HAS_TRAIT]->"
            "(:__SelfModel__ {id: 'mist-trait-warm'}) RETURN count(r) AS n",
            {},
        )
        assert rows[0]["n"] == 1

    def test_embeddings_are_present_at_the_declared_dimension(self, staging_conn, embedder):
        """The failure mode the canonical form is blind to BY DESIGN.

        `canonical_serialize` excludes `embedding` (`seed/gates.py:264-268`), so a
        seed-apply that skipped the backfill would certify byte-identical to one
        that did not. Only a direct read can tell them apart.
        """
        result = _seeder(staging_conn, embedder).apply(now_iso=_NOW)

        assert result.embedded == 2, "the backfill reported writing no vectors"
        rows = staging_conn.execute_query(
            "MATCH (n:__SelfModel__) WHERE n.embedding IS NOT NULL "
            "RETURN n.id AS id, size(n.embedding) AS dim ORDER BY n.id",
            {},
        )
        assert len(rows) == 2, f"only {len(rows)} of 2 seeded nodes carry an embedding"
        assert all(r["dim"] == _DIMENSION for r in rows), [r["dim"] for r in rows]

    def test_the_embedding_gate_examined_the_nodes_it_passed(self, staging_conn, embedder):
        """A pass having examined nothing is the shape of both historical live losses."""
        result = _seeder(staging_conn, embedder).apply(now_iso=_NOW)

        assert result.embedding_gate.passed, result.embedding_gate.failures
        assert result.embedding_gate.examined == 2, (
            f"the gate passed having examined {result.embedding_gate.examined} nodes. "
            "The backfill and the gate match on seed_version across a label union; a "
            "mismatch there makes the gate vacuous while reporting a pass."
        )

    def test_the_injected_timestamp_reaches_the_graph(self, staging_conn, embedder):
        """Determinism: two rebuilds of one epoch must stamp identically."""
        _seeder(staging_conn, embedder).apply(now_iso=_NOW)

        rows = staging_conn.execute_query(
            "MATCH (n:__SelfModel__ {id: 'mist-identity'}) "
            "RETURN n.created_at AS created, n.seed_version AS version",
            {},
        )
        assert rows[0]["created"] == _NOW
        assert rows[0]["version"] == _SEED_VERSION


class TestApplyingTwiceIsIdempotent:
    def test_a_second_apply_does_not_duplicate_nodes(self, staging_conn, embedder):
        """`rebuild()` runs twice per gate invocation, into a wiped staging each time.

        This asserts the weaker property that matters if the wipe is ever missed:
        the applier MERGEs, so a second pass over the same corpus must not double
        the partition.
        """
        seeder = _seeder(staging_conn, embedder)
        seeder.apply(now_iso=_NOW)
        seeder.apply(now_iso=_NOW)

        rows = staging_conn.execute_query("MATCH (n:__SelfModel__) RETURN count(n) AS n", {})
        assert rows[0]["n"] == 2


# ---------------------------------------------------------------------------
# There is deliberately NO live-refusal test in this file. Read this before adding one.
# ---------------------------------------------------------------------------
#
# The first version of this file had one. It built a real `Neo4jConnection` to
# `bolt://mist-neo4j:7687` and called `apply()` on it, expecting
# `_assert_seed_target_permitted` to refuse before any write. It passed.
#
# Then it was mutation-tested. The mutant set `allow_live=True` in
# `StagingSeeder.apply` -- disabling the very guard the test existed to prove --
# and the test did exactly what it was built to do: it ran `apply()` against the
# LIVE graph. `Neo4jConnection` connects lazily on `execute_write`, so never
# having called `.connect()` was not protection. It created `mist-trait-warm`,
# wired it to `mist-identity` with a `HAS_TRAIT` edge, and overwrote
# `mist-identity`'s `seed_version` and `updated_at`. Live went 32/30 -> 33/31 on
# a running system whose self-model then contained a trait nobody authored.
# Repaired the same day; no backup existed to repair from (MIS-140).
#
# The lesson is not "be careful with mutants". It is that a test whose ONLY
# protection against writing to production is the guard it is testing has no
# protection at all -- it is a loaded gun pointed at live, safe exactly while the
# thing under test works, which is the one condition a test may not assume.
#
# The refusal IS covered, safely, in
# tests/unit/knowledge/regeneration/test_staging_seeder.py::TestLiveTargetIsRefused.
# It uses `_RecordingConnection`, a pure fake that carries a live-looking
# `config.uri` so the guard has something to refuse but whose `execute_write`
# appends to a list and can reach no database at all. It asserts `conn.writes ==
# []`, so a disabled guard fails the test instead of reaching a graph. That is
# where this belongs, and it needs no real Neo4j.
#
# If you want integration-level coverage of the guard, the only acceptable shape
# is one that CANNOT write to live even with the guard removed. A real connection
# to a real live URI is not that shape.
