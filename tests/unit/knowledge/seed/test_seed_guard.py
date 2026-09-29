"""MIS-177 D1: the seed refuses a graph holding anything it could adopt.

`_MERGE_NODE` matches on id and `_MERGE_EDGE` on (subject, type, object)
alone, so a seed applied over a graph that already holds extraction-written or
other non-seed elements stamps them `seed_version`, and the next reseed's wipe
deletes them. `apply_seed_documents` and `reseed` therefore run the seed guard
before any write or wipe, with no override.

These tests answer the guard's two statements with chosen rows and pin what
the applier does with them: refuse before the first `execute_write` on any
nonzero count, fail closed on a missing or malformed row, and write on a clean
row. What each statement COUNTS in a real graph -- an empty graph and a
seed-only graph answering zero, each blocking element answering nonzero -- is
Cypher semantics, which a fake cannot evaluate; it is pinned against Neo4j in
`tests/integration/knowledge/test_seed_guard_eval.py`.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from backend.errors import Neo4jQueryError, SeedTargetNotSeedOnlyError
from backend.knowledge import admin
from backend.knowledge.seed import applier
from backend.knowledge.seed.applier import apply_seed_documents, reseed
from backend.knowledge.seed.models import SeedDocument, SeedFact, SeedNode
from backend.knowledge.storage.partitions import ENTITY_LABEL
from tests.mocks.neo4j import FakeNeo4jConnection
from tests.unit.knowledge.seed.seed_guard_rows import (
    CLEAN_RESET_GUARD_ROW,
    CLEAN_SEED_GUARD_ROW,
    seed_guard_router,
)

_SEED_VERSION = "profile-v1"
_NOW = "2026-09-29T00:00:00+00:00"


def _documents() -> list[SeedDocument]:
    return [
        SeedDocument(
            seed_version=_SEED_VERSION,
            nodes=[
                SeedNode(id="user", type="User"),
                SeedNode(id="slalom", type="Organization"),
            ],
            facts=[SeedFact(subject="user", predicate="WORKS_AT", object="slalom")],
            body="test body",
            source_path=Path("test.md"),
            partition=ENTITY_LABEL,
        )
    ]


class _OrderedConnection(FakeNeo4jConnection):
    """A fake that also records reads and writes in ONE sequence.

    `FakeNeo4jConnection` keeps reads and writes in separate lists, which
    cannot show that a read happened before a write.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.events: list[tuple[str, str]] = []

    def execute_query(self, query, params=None):
        self.events.append(("read", query))
        return super().execute_query(query, params)

    def execute_write(self, query, params=None):
        self.events.append(("write", query))
        return super().execute_write(query, params)


def _connection(
    *,
    reset_guard_rows: list[dict[str, Any]] | None = None,
    seed_guard_rows: list[dict[str, Any]] | None = None,
) -> _OrderedConnection:
    return _OrderedConnection(
        query_router=seed_guard_router(
            reset_guard_rows=reset_guard_rows, seed_guard_rows=seed_guard_rows
        )
    )


def _apply(connection: FakeNeo4jConnection) -> dict[str, int]:
    return apply_seed_documents(connection, _documents(), seed_version=_SEED_VERSION, now_iso=_NOW)


def _reseed(connection: FakeNeo4jConnection) -> dict[str, int]:
    return reseed(connection, _documents(), seed_version=_SEED_VERSION, now_iso=_NOW)


_ENTRY_POINTS: dict[str, Callable[[FakeNeo4jConnection], dict[str, int]]] = {
    "apply_seed_documents": _apply,
    "reseed": _reseed,
}


@pytest.fixture(params=sorted(_ENTRY_POINTS))
def seed(request) -> Callable[[FakeNeo4jConnection], dict[str, int]]:
    """Each test runs against both entry points."""
    return _ENTRY_POINTS[request.param]


def _reset_row(**counts: int) -> list[dict[str, int]]:
    return [{**CLEAN_RESET_GUARD_ROW, **counts}]


def _seed_row(**counts: int) -> list[dict[str, int]]:
    return [{**CLEAN_SEED_GUARD_ROW, **counts}]


# (case id, reset guard rows, seed guard rows, text the refusal must contain)
_BLOCKING = [
    (
        "non-seed :__Entity__ node",
        _reset_row(nodes=1),
        None,
        "1 non-seed :__Entity__ node(s) and 0 non-seed relationship(s)",
    ),
    (
        ":__Entity__ relationship without seed_version",
        _reset_row(relationships=2),
        None,
        "0 non-seed :__Entity__ node(s) and 2 non-seed relationship(s)",
    ),
    (
        "extraction-stamped :__SelfModel__ node",
        None,
        _seed_row(stamped_nodes=3),
        "3 node(s) and 0 relationship(s) in any partition carrying extraction_version",
    ),
    (
        "extraction-stamped relationship",
        None,
        _seed_row(stamped_relationships=4),
        "0 node(s) and 4 relationship(s) in any partition carrying extraction_version",
    ),
    (
        "provenance='extraction' node",
        None,
        _seed_row(extraction_nodes=5),
        "5 node(s) and 0 relationship(s) with provenance='extraction'",
    ),
    (
        "provenance='extraction' relationship",
        None,
        _seed_row(extraction_relationships=6),
        "0 node(s) and 6 relationship(s) with provenance='extraction'",
    ),
]


class TestRefusesBeforeAnyWrite:
    @pytest.mark.parametrize(
        ("reset_rows", "seed_rows", "named"),
        [case[1:] for case in _BLOCKING],
        ids=[case[0] for case in _BLOCKING],
    )
    def test_a_blocking_element_is_refused_with_its_count(self, seed, reset_rows, seed_rows, named):
        conn = _connection(reset_guard_rows=reset_rows, seed_guard_rows=seed_rows)

        with pytest.raises(SeedTargetNotSeedOnlyError) as excinfo:
            seed(conn)

        assert conn.writes == [], "the refusal must precede every write, the wipe included"
        assert named in str(excinfo.value)

    def test_the_refusal_names_every_count(self, seed):
        conn = _connection(
            reset_guard_rows=[{"nodes": 11, "relationships": 12}],
            seed_guard_rows=[
                {
                    "stamped_nodes": 13,
                    "extraction_nodes": 14,
                    "stamped_relationships": 15,
                    "extraction_relationships": 16,
                }
            ],
        )

        with pytest.raises(SeedTargetNotSeedOnlyError) as excinfo:
            seed(conn)

        message = str(excinfo.value)
        for count in range(11, 17):
            assert str(count) in message, message
        assert conn.writes == []


# (case id, reset guard rows, seed guard rows)
_UNUSABLE = [
    ("reset guard: no row", [], None),
    ("reset guard: missing column", [{"nodes": 0}], None),
    ("reset guard: null count", [{"nodes": None, "relationships": 0}], None),
    ("reset guard: non-numeric count", [{"nodes": "zero", "relationships": 0}], None),
    ("seed guard: no row", None, []),
    (
        "seed guard: missing column",
        None,
        [{k: v for k, v in CLEAN_SEED_GUARD_ROW.items() if k != "extraction_relationships"}],
    ),
    ("seed guard: null count", None, [{**CLEAN_SEED_GUARD_ROW, "stamped_nodes": None}]),
    ("seed guard: empty row", None, [{}]),
]


class TestFailsClosed:
    @pytest.mark.parametrize(
        ("reset_rows", "seed_rows"),
        [case[1:] for case in _UNUSABLE],
        ids=[case[0] for case in _UNUSABLE],
    )
    def test_a_missing_or_malformed_row_is_refused(self, seed, reset_rows, seed_rows):
        conn = _connection(reset_guard_rows=reset_rows, seed_guard_rows=seed_rows)

        with pytest.raises(Neo4jQueryError, match="guard query returned"):
            seed(conn)

        assert conn.writes == [], "an unreadable guard must stop the seed, not pass it"

    def test_an_unconfigured_fake_is_refused(self, seed):
        """A connection that answers every read with nothing is not a clean graph."""
        conn = FakeNeo4jConnection()

        with pytest.raises(Neo4jQueryError):
            seed(conn)

        conn.assert_no_writes()


class TestPassesACleanGraph:
    """Zero rows are what Neo4j returns for an empty graph and a seed-only one."""

    def test_a_clean_graph_is_seeded(self, seed):
        conn = _connection()

        counts = seed(conn)

        assert counts == {"nodes": 2, "facts": 1}
        assert any("MERGE (s)-[r:WORKS_AT]->(o)" in q for q, _ in conn.writes)

    def test_both_guard_statements_run_before_the_first_write(self, seed):
        conn = _connection()

        seed(conn)

        first_write = next(i for i, (kind, _) in enumerate(conn.events) if kind == "write")
        reads_before = [q for kind, q in conn.events[:first_write] if kind == "read"]
        assert admin.RESET_GUARD_CYPHER in reads_before
        assert applier.SEED_GUARD_CYPHER in reads_before


class TestReseedGuardsBeforeTheWipe:
    def test_the_wipe_does_not_run_on_refusal(self):
        conn = _connection(seed_guard_rows=_seed_row(stamped_relationships=1))

        with pytest.raises(SeedTargetNotSeedOnlyError):
            _reseed(conn)

        assert not any("DELETE" in q.upper() for q, _ in conn.writes)
        assert conn.writes == []

    def test_reseed_refuses_independently_of_its_delegate(self, monkeypatch):
        """Non-vacuity: a guard only in `apply_seed_documents` runs after the wipe.

        Neutering the delegate must still leave `reseed` refusing with no write.
        """
        monkeypatch.setattr(applier, "apply_seed_documents", lambda *a, **k: {})
        conn = _connection(reset_guard_rows=_reset_row(nodes=1))

        with pytest.raises(SeedTargetNotSeedOnlyError):
            _reseed(conn)

        assert conn.writes == []

    def test_the_guard_reads_precede_the_wipe(self):
        conn = _connection()

        _reseed(conn)

        wipe_index = next(
            i for i, (kind, q) in enumerate(conn.events) if kind == "write" and "DELETE" in q
        )
        reads_before_wipe = [q for kind, q in conn.events[:wipe_index] if kind == "read"]
        assert admin.RESET_GUARD_CYPHER in reads_before_wipe
        assert applier.SEED_GUARD_CYPHER in reads_before_wipe


class TestTheGuardStatement:
    def test_the_stamp_list_is_extraction_version_and_model_hash_only(self):
        """`ontology_version` is excluded: `ensure_mist_identity` writes it at startup."""
        assert applier.SEED_GUARD_STAMP_PROPERTIES == ("extraction_version", "model_hash")
        assert "ontology_version" not in applier.SEED_GUARD_CYPHER

    @pytest.mark.parametrize("stamp", ["extraction_version", "model_hash"])
    def test_every_stamp_is_tested_on_nodes_and_relationships(self, stamp):
        assert f"n.{stamp} IS NOT NULL" in applier.SEED_GUARD_CYPHER
        assert f"r.{stamp} IS NOT NULL" in applier.SEED_GUARD_CYPHER

    def test_provenance_extraction_is_tested_on_nodes_and_relationships(self):
        assert "n.provenance = 'extraction'" in applier.SEED_GUARD_CYPHER
        assert "r.provenance = 'extraction'" in applier.SEED_GUARD_CYPHER

    def test_it_matches_every_partition(self):
        """No label on either MATCH: a `:__SelfModel__` element counts as well."""
        assert applier.SEED_GUARD_CYPHER.startswith("MATCH (n) ")
        assert "OPTIONAL MATCH ()-[r]->() " in applier.SEED_GUARD_CYPHER
        assert "__Entity__" not in applier.SEED_GUARD_CYPHER
        assert "__SelfModel__" not in applier.SEED_GUARD_CYPHER

    def test_it_is_read_only(self):
        assert not re.search(
            r"\b(CREATE|MERGE|SET|DELETE|REMOVE|DETACH)\b", applier.SEED_GUARD_CYPHER
        )

    def test_it_returns_every_column_the_applier_reads(self):
        for column in applier.SEED_GUARD_COLUMNS:
            assert f"AS {column}" in applier.SEED_GUARD_CYPHER
