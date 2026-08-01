"""ADR-009 acceptance: seed produces clean :__Entity__ / :__SelfModel__ / :__Provenance__ split.

R1.4 Task 10: repointed from `scripts/seed_data.yaml` + `admin.apply_seed`
(deleted this task) onto the versioned seed source + `reseed`. This is not
a cosmetic swap of the load path -- the invariant itself inverted. The old
assertion ("seed yields ONLY :__Entity__ nodes") was already the wrong
shape for the live graph before this task: R1.0/R1.1 (2026-06-15) moved
self-model content into its own `:__SelfModel__` partition specifically so
it stops living under `:__Entity__`, and R1.4 Task 4's applier routes each
seed document to the partition its own `SeedDocument.partition` declares.
"Only :__Entity__" is therefore not a property the current design is even
trying to hold; the correct invariant is the fixed partition split the
source itself declares: 11 `:__Entity__` (anchor entities + the user) and
21 `:__SelfModel__` (identity + traits + capabilities + preferences),
0 `:__Provenance__`.

This test performs a FULL graph wipe (`reset_graph(include_derived=True)`).
Until R1.4 Task 14 it ran unguarded against whatever `NEO4J_URI` the
integration environment resolves to -- confirmed to be the same live
`mist-neo4j` instance `mist_admin.py seed` targets, not an isolated test
database, and confirmed to destroy real data: a live run against production
wiped the seeded profile's embeddings during this task. This hazard is
inherited from the original ADR-009 test (git history: `07f6aac`), not
introduced by R1.4 -- T10's rewrite changed the seed-load call and the
assertion counts and added a docstring warning, but the unguarded
`real_neo4j_connection` fixture itself was untouched; `CODEBASE.md`'s R1.3
entry already flagged the same hazard with the same proposed fix before this
sub-project started.

T14: `real_neo4j_connection` now SKIPS unless `MIST_EVAL_ISOLATION` is
active, and validates its target through
`backend.knowledge.eval_isolation.assert_neo4j_isolated` so a misconfigured
`NEO4J_URI` (e.g. still pointed at the live instance) refuses rather than
wiping the canonical graph. This test therefore no longer runs in the
default `tests/integration/` pass -- a real coverage loss, accepted because
the alternative is coverage purchased by periodically wiping the user's
memory. To run it: stand up the disposable eval instance
(`docker compose -f docker-compose.yml -f docker-compose.eval-neo4j.yml
--profile eval up -d mist-neo4j-eval`) and point `NEO4J_URI` at
`bolt://mist-neo4j-eval:7687` with `MIST_EVAL_ISOLATION=1`. Wiring that up
by default (so this test genuinely runs again) is follow-up work, not done
here.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from backend.knowledge import admin
from backend.knowledge.config import Neo4jConfig
from backend.knowledge.eval_isolation import assert_neo4j_isolated, is_eval_isolation_active
from backend.knowledge.seed.applier import reseed
from backend.knowledge.seed.loader import load_seed_documents
from backend.knowledge.storage.neo4j_connection import Neo4jConnection

pytestmark = pytest.mark.integration

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SEED_DIR = _REPO_ROOT / "mist-memory" / "seed"


@pytest.fixture
def real_neo4j_connection():
    """Provide a real Neo4jConnection targeting a disposable eval Neo4j instance.

    Fail-closed by default: this test performs a full graph wipe, so it
    SKIPS unless the caller has explicitly opted into eval isolation
    (`MIST_EVAL_ISOLATION=1`), and even then routes the resolved config
    through `assert_neo4j_isolated` -- a misconfigured `NEO4J_URI` that still
    resolves to the live instance refuses rather than wiping it. See the
    module docstring for the incident that motivated this and how to run
    the test deliberately.
    """
    if not is_eval_isolation_active():
        pytest.skip(
            "test_seed_label_split.py performs a full graph wipe "
            "(reset_graph(include_derived=True)) and must not run against the "
            "live graph. Set MIST_EVAL_ISOLATION=1 and point NEO4J_URI at a "
            "disposable instance (docker-compose.eval-neo4j.yml, --profile eval) "
            "to run it."
        )
    config = Neo4jConfig(
        uri=os.environ.get("NEO4J_URI", "bolt://localhost:7687"),
        username=os.environ.get("NEO4J_USER", "neo4j"),
        password=os.environ.get("NEO4J_PASSWORD", "password"),
    )
    assert_neo4j_isolated(config)
    conn = Neo4jConnection(config=config)
    conn.connect()
    yield conn
    conn.disconnect()


def _count_self_model_nodes(conn) -> int:
    rows = conn.execute_query("MATCH (n:__SelfModel__) RETURN count(n) AS c")
    return rows[0]["c"] if rows else 0


def _count_dual_labeled_nodes(conn) -> int:
    """Nodes carrying BOTH partition labels -- must always be 0 (Task 4's defect class)."""
    rows = conn.execute_query("MATCH (n:__Entity__:__SelfModel__) RETURN count(n) AS c")
    return rows[0]["c"] if rows else 0


def test_seed_yields_expected_entity_selfmodel_split(real_neo4j_connection):
    """After graph-reset --include-derived + seed: 11 :__Entity__, 21 :__SelfModel__,
    0 :__Provenance__, 0 dual-labeled (Task 4's partition-routing defect class).
    """
    conn = real_neo4j_connection

    # Full wipe -- removes :__Entity__, :__SelfModel__ (label-agnostic DETACH DELETE
    # is not used here; reset_graph only targets :__Entity__ + optionally
    # :__Provenance__, so self-model nodes survive this call untouched by
    # design -- see backend/knowledge/admin.py:reset_graph). The subsequent
    # reseed's wipe (scoped on seed_version) is what actually clears
    # :__SelfModel__ before recreating it, exercising the same wipe path
    # `mist_admin.py seed` uses live.
    admin.reset_graph(conn, include_derived=True)
    admin.ensure_schema(conn)

    documents = load_seed_documents(_SEED_DIR)
    seed_version = documents[0].seed_version
    now = datetime.now(UTC).isoformat()
    reseed(conn, documents, seed_version=seed_version, now_iso=now)

    # --- :__Entity__ count ---
    entity_rows = admin.count_nodes_by_type(conn)
    total_entities = sum(row["count"] for row in entity_rows)
    assert (
        total_entities == 11
    ), f"Expected 11 seeded :__Entity__ nodes, got {total_entities}: {entity_rows}"

    # --- :__SelfModel__ count ---
    total_self_model = _count_self_model_nodes(conn)
    assert (
        total_self_model == 21
    ), f"Expected 21 seeded :__SelfModel__ nodes, got {total_self_model}"

    # --- no node carries both partition labels (Task 4's defect class) ---
    dual_labeled = _count_dual_labeled_nodes(conn)
    assert dual_labeled == 0, f"Expected 0 dual-labeled nodes, got {dual_labeled}"

    # --- :__Provenance__ count (must be 0 post-seed) ---
    provenance_rows = admin.provenance_counts_by_type(conn)
    total_prov = sum(row["count"] for row in provenance_rows)
    assert (
        total_prov == 0
    ), f"Seed should create 0 :__Provenance__ nodes, got {total_prov}: {provenance_rows}"
