"""F1: the seed applier takes a connection OBJECT, and no guard could inspect it.

Every isolation guard in the repo reasons about URI STRINGS
(`backend/knowledge/eval_isolation.py`). `apply_seed_documents` and `reseed`
take a `GraphConnection`, so none of them could be pointed at the thing that
actually issues the writes. At the R1.6/R1.7 seed-apply insertion point
(`log_regenerator.py:445`) `source_conn` and `staging_conn` are both in scope
and differ by six characters, and `reseed`'s wipe-then-recreate cycle is what
stripped all 32 live nodes on 2026-07-31.

The fix is default-CLOSED and lives at the write site rather than at the call
site. A guard the caller must remember to add is a guard that is absent exactly
when it matters; `allow_live=False` by default means the dangerous call is the
one that has to be spelled out. `cmd_seed` is the only caller that says
`allow_live=True`, and it says it in one place.

`FakeNeo4jConnection` has no `.config`, so a test double cannot be mistaken for
live and the ~60 existing applier tests are untouched. That is not a loophole:
the threat model is a REAL connection pointed at the canonical graph, and a
real `Neo4jConnection` always exposes `.config.uri`.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from backend.knowledge.eval_isolation import EvalIsolationError
from backend.knowledge.seed.applier import apply_seed_documents, reseed
from tests.mocks.neo4j import FakeNeo4jConnection

_NOW = "2026-08-26T00:00:00+00:00"


@dataclass
class _Config:
    uri: str


class _ConnectionWithUri(FakeNeo4jConnection):
    """A fake that DOES expose `.config.uri`, standing in for a real connection."""

    def __init__(self, uri: str, **kwargs):
        super().__init__(**kwargs)
        self.config = _Config(uri=uri)


@pytest.fixture
def docs():
    """Minimal valid seed input; the guard must fire before any validation."""
    return []


class TestApplySeedDocumentsLiveGuard:
    @pytest.mark.parametrize(
        "uri",
        [
            "bolt://mist-neo4j:7687",
            "bolt://localhost:7687",
            "bolt://127.0.0.1:7687",
        ],
    )
    def test_a_live_connection_is_refused_by_default(self, uri, docs):
        conn = _ConnectionWithUri(uri)
        with pytest.raises(EvalIsolationError, match="live"):
            apply_seed_documents(conn, docs, seed_version="v1", now_iso=_NOW)
        assert not conn.writes, "refusal must precede every write"

    def test_a_live_connection_is_permitted_when_explicitly_allowed(self, docs):
        """`cmd_seed` seeds the live graph on purpose; that must stay possible."""
        conn = _ConnectionWithUri("bolt://mist-neo4j:7687")
        apply_seed_documents(conn, docs, seed_version="v1", now_iso=_NOW, allow_live=True)

    @pytest.mark.parametrize(
        "uri",
        [
            "bolt://mist-neo4j-staging:7687",
            "bolt://mist-neo4j-dev:7687",
            "bolt://localhost:7689",
        ],
    )
    def test_non_live_connections_pass_without_the_flag(self, uri, docs):
        """The guard is a live denylist, not a staging allowlist.

        Seed-apply legitimately targets staging (the rebuild), dev (hydration),
        and live (`cmd_seed`). Only the last needs saying out loud.
        """
        apply_seed_documents(_ConnectionWithUri(uri), docs, seed_version="v1", now_iso=_NOW)

    def test_a_test_double_without_config_is_unaffected(self, docs):
        """The ~60 existing applier tests must not need editing.

        A double cannot reach live, so "cannot determine the URI" is not the
        dangerous case here -- it is proof the object is not a real connection.
        """
        apply_seed_documents(FakeNeo4jConnection(), docs, seed_version="v1", now_iso=_NOW)


class TestReseedLiveGuard:
    """`reseed` is the more dangerous of the two -- it WIPES before it applies."""

    def test_a_live_connection_is_refused_by_default(self, docs):
        conn = _ConnectionWithUri("bolt://mist-neo4j:7687")
        with pytest.raises(EvalIsolationError, match="live"):
            reseed(conn, docs, seed_version="v1", now_iso=_NOW)
        assert not conn.writes, "the wipe must not run; this is the 2026-07-31 loss path"

    def test_it_is_permitted_when_explicitly_allowed(self, docs):
        conn = _ConnectionWithUri("bolt://mist-neo4j:7687")
        reseed(conn, docs, seed_version="v1", now_iso=_NOW, allow_live=True)

    def test_reseed_guards_independently_of_apply(self, docs, monkeypatch):
        """Non-vacuity: `reseed` must not rely on its delegate to refuse.

        `reseed` calls `apply_seed_documents`, so a guard only in the delegate
        would look like it works. But the WIPE runs first -- by the time the
        delegate refuses, the graph is already empty. Neutering the delegate
        must still leave `reseed` refusing.
        """
        monkeypatch.setattr(
            "backend.knowledge.seed.applier.apply_seed_documents",
            lambda *a, **k: {},
        )
        conn = _ConnectionWithUri("bolt://mist-neo4j:7687")
        with pytest.raises(EvalIsolationError, match="live"):
            reseed(conn, docs, seed_version="v1", now_iso=_NOW)
        assert not conn.writes
