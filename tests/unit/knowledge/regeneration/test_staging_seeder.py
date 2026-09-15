"""The seed-apply step: what `rebuild()` uses instead of copying live forward.

## What this object is for

MIS-130 step B. `rebuild()` had no seed-apply of any kind; the `:__SelfModel__`
partition reached staging via `copy_self_model_partition`, which read from the
LIVE graph. Step A deleted that. `StagingSeeder` is what fills the gap, and it
has to do THREE things rather than one, because the live seed path is three
steps and skipping any of them produces a graph that passes every gate:

1. `apply_seed_documents` -- writes nodes and facts, stamped `seed_version`.
2. `_backfill_embeddings_for_seed` -- the applier NEVER writes `embedding`.
3. `check_embeddings` -- re-reads the graph, because the backfill's own return
   value counts rows it THOUGHT it wrote, reported by the same code that failed
   to write them.

Bundled behind one object rather than left as three calls in `rebuild()` so the
three cannot drift apart at the one call site that matters. `seed/gates.py:264`
states the consequence of drifting: `canonical_serialize` excludes `embedding`,
so an unembedded rebuild is byte-identical to an embedded one.

## Why `now_iso` is injected rather than read from the clock

`apply_seed_documents` takes `now_iso` explicitly so application is
byte-reproducible. `rebuild()` passes an epoch-derived value, so two rebuilds of
the same epoch stamp the same `created_at`/`updated_at` and
`assert_rebuild_twice_identical` compares content rather than wall-clock noise.
A `datetime.now()` here would make the determinism gate fail for a reason that
has nothing to do with determinism.

## Why the live-target refusal is tested here and not only at the write site

`_assert_seed_target_permitted` (`seed/applier.py:91`) is default-CLOSED at the
write site, and its docstring names this exact insertion point as the reason:
"at the R1.7 seed-apply insertion point, `source_conn` and `staging_conn` are
both in scope and differ by six characters". After step A that specific confusion
is structurally impossible -- `rebuild()` holds no live handle to confuse it
with -- but the seeder is constructed by a CALLER that still has both, so the
refusal is the thing standing between a typo and a wiped canonical graph. It is
worth a test that names it.
"""

from pathlib import Path

import pytest

from backend.knowledge.regeneration.staging_seeder import SeedApplyResult, StagingSeeder
from backend.knowledge.seed.models import SeedDocument, SeedFact, SeedNode

_SEED_VERSION = "test-seed-1"
_NOW = "2026-07-01T09:00:00+00:00"


def _document() -> SeedDocument:
    """One valid document. Types and predicate must exist in the ontology.

    `apply_seed_documents` validates both before any write, so an invented type
    would abort the apply and this test would pass for the wrong reason.
    """
    return SeedDocument(
        seed_version=_SEED_VERSION,
        nodes=[
            SeedNode(id="mist-identity", type="MistIdentity"),
            SeedNode(id="mist-trait-warm", type="MistTrait"),
        ],
        facts=[SeedFact(subject="mist-identity", predicate="HAS_TRAIT", object="mist-trait-warm")],
        body="MIST is warm.",
        source_path=Path("mist.md"),
        partition="__SelfModel__",
    )


class _Config:
    def __init__(self, uri):
        self.uri = uri


class _RecordingConnection:
    """Records writes; answers every read with `rows`.

    A real `EventStore`-style double is not available for Neo4j, and the
    mocking table's "external service" row prescribes a fake here.
    """

    def __init__(self, uri="bolt://mist-neo4j-staging:7687", rows=None):
        self.config = _Config(uri)
        self.writes = []
        self._rows = rows if rows is not None else []

    def execute_write(self, query, params=None):
        self.writes.append((query, params or {}))
        return []

    def execute_query(self, query, params=None):
        return list(self._rows)


class _Embedder:
    def __init__(self):
        self.calls = []

    def generate_embedding(self, text):
        self.calls.append(text)
        return [0.1] * 384


def _seeder(connection, documents=None):
    return StagingSeeder(
        connection=connection,
        documents=documents if documents is not None else [_document()],
        seed_version=_SEED_VERSION,
        embedding_generator=_Embedder(),
        expected_dimension=384,
    )


class TestSeedApplyIsDeterministic:
    def test_the_injected_now_reaches_every_write(self):
        """Determinism: two rebuilds of one epoch must stamp identical timestamps."""
        conn = _RecordingConnection()
        _seeder(conn).apply(now_iso=_NOW)

        stamped = [p for _, p in conn.writes if "now" in p]
        assert stamped, "no write carried a `now` parameter; the stamp is not reaching the graph"
        assert all(p["now"] == _NOW for p in stamped), (
            "a write used a timestamp other than the injected one. A wall-clock stamp "
            "here fails `assert_rebuild_twice_identical` for a reason that has nothing "
            "to do with determinism."
        )


class TestLiveTargetIsRefused:
    @pytest.mark.parametrize(
        "uri",
        ["bolt://mist-neo4j:7687", "bolt://localhost:7687", "bolt://127.0.0.1:7687"],
    )
    def test_a_live_connection_is_refused_before_any_write(self, uri):
        conn = _RecordingConnection(uri=uri)
        with pytest.raises(Exception, match="live graph"):
            _seeder(conn).apply(now_iso=_NOW)
        assert conn.writes == [], (
            "the seeder wrote to a LIVE connection before refusing it. The refusal "
            "must precede the first write, not follow it."
        )

    def test_a_staging_connection_is_permitted(self):
        conn = _RecordingConnection(uri="bolt://mist-neo4j-staging:7687")
        _seeder(conn).apply(now_iso=_NOW)
        assert conn.writes, "the seeder wrote nothing to a permitted staging target"


class TestResultReporting:
    def test_nodes_written_counts_the_seeded_nodes(self):
        result = _seeder(_RecordingConnection()).apply(now_iso=_NOW)
        assert isinstance(result, SeedApplyResult)
        assert result.nodes_written == 2

    def test_facts_written_counts_the_seeded_facts(self):
        result = _seeder(_RecordingConnection()).apply(now_iso=_NOW)
        assert result.facts_written == 1

    def test_a_failing_embedding_gate_is_reported_not_swallowed(self):
        """A graph that answers no read has no embeddings; say so rather than pass."""
        result = _seeder(_RecordingConnection(rows=[])).apply(now_iso=_NOW)
        assert result.embedding_gate.passed is False, (
            "the seeder reported a passing embedding gate against a graph that "
            "returned no rows. Swallowing this is how both historical live "
            "embedding losses stayed invisible."
        )


class TestConstructionRefusals:
    def test_an_empty_document_set_is_refused_at_construction(self):
        """Refuse before doing work, not after -- MIS-137's fail-closed shape."""
        with pytest.raises(ValueError, match="no seed documents"):
            _seeder(_RecordingConnection(), documents=[])
