"""Synthetic live state and a graph fake, shared by the backup tests.

The container has no network and cannot reach the live stack, Neo4j, the vault
or any credential. Everything these tests read is built in the worktree.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.mocks.neo4j import FakeNeo4jConnection, FakeNeo4jRecord

# Two files that exist in the LIVE `./data` beside the real stores. They are
# here to be EXCLUDED: the negative test asserts a backup never sweeps them in.
STALE_BACKUP_FILENAMES = (
    "event_store.pre-r1.4-backup-2026-07-31.db",
    "event_store.pre-reset-backup-2026-06-09.db",
)


def make_store(path: Path, *, table: str, rows: int) -> None:
    """Create a WAL-mode SQLite store with `rows` rows in `table`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(f"CREATE TABLE {table} (id INTEGER PRIMARY KEY, payload TEXT)")
        conn.executemany(
            f"INSERT INTO {table} (payload) VALUES (?)",
            [(f"row-{index}",) for index in range(rows)],
        )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def state_root(tmp_path: Path) -> Path:
    """A stand-in for live `./data`: three real stores plus two stale ones."""
    root = tmp_path / "live-data"
    make_store(root / "event_store.db", table="conversation_turn_events", rows=3)
    make_store(root / "extraction_cache.db", table="extraction_cache", rows=2)
    make_store(root / "vault_sidecar.db", table="vault_chunks", rows=5)
    for filename in STALE_BACKUP_FILENAMES:
        make_store(root / filename, table="conversation_turn_events", rows=99)
    (root / "vector_store").mkdir()
    (root / "vector_store" / "chunks.lance").write_text("derived", encoding="utf-8")
    return root


@pytest.fixture
def vault_root(tmp_path: Path) -> Path:
    """A stand-in for `mist-memory/`, which is absent from every worktree."""
    root = tmp_path / "mist-memory"
    (root / "identity").mkdir(parents=True)
    (root / "identity" / "mist.md").write_text("# MIST\n", encoding="utf-8")
    (root / "users").mkdir()
    (root / "users" / "raj.md").write_text("# Raj\n", encoding="utf-8")
    return root


@pytest.fixture
def backup_root(tmp_path: Path) -> Path:
    """A destination outside the repository, as the guard requires."""
    root = tmp_path / "offsite"
    root.mkdir()
    return root


def make_graph_connection(
    *,
    nodes: list[dict] | None = None,
    relationships: list[dict] | None = None,
) -> FakeNeo4jConnection:
    """A `GraphConnection` fake answering the four reads the graph leg issues.

    `dump_full_graph_json` runs `MATCH (n)` then `MATCH (s)-[r]->(t)`, and
    `read_schema_ddl` runs `SHOW CONSTRAINTS` then `SHOW INDEXES`
    (`grep -n "SHOW CONSTRAINTS" backend/knowledge/admin.py` -> :889). Routing on
    query text is what the fake supports.
    """
    node_rows = [FakeNeo4jRecord(node) for node in (nodes or [])]
    rel_rows = [FakeNeo4jRecord(rel) for rel in (relationships or [])]

    def router(query: str, _params: dict | None) -> list | None:
        if "SHOW CONSTRAINTS" in query:
            return [FakeNeo4jRecord({"name": "c1", "createStatement": "CREATE CONSTRAINT c1"})]
        if "SHOW INDEXES" in query:
            return [FakeNeo4jRecord({"name": "i1", "createStatement": "CREATE INDEX i1"})]
        if "MATCH (s)-[r]->(t)" in query:
            return rel_rows
        if "MATCH (n)" in query:
            return node_rows
        return None

    return FakeNeo4jConnection(query_router=router)


@pytest.fixture
def graph_connection() -> FakeNeo4jConnection:
    """Two nodes, one edge, one 384-float embedding carried as exact floats."""
    return make_graph_connection(
        nodes=[
            {
                "id": "person-raj",
                "labels": ["__Entity__", "Person"],
                "properties": {"id": "person-raj", "name": "Raj", "embedding": [0.25, -0.5, 0.125]},
            },
            {
                "id": "project-mist",
                "labels": ["__Entity__", "Project"],
                "properties": {"id": "project-mist", "name": "MIST.AI"},
            },
        ],
        relationships=[
            {
                "source": "person-raj",
                "type": "WORKS_ON",
                "target": "project-mist",
                "properties": {"confidence": 0.9},
            }
        ],
    )


STAMPS = {
    "ontology_version": "1.0.0",
    "extraction_version": "v7",
    "model_hash": "abc123",
}
