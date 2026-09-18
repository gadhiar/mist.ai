"""Synthetic live state and a graph fake, shared by the backup tests.

The container has no network and cannot reach the live stack, Neo4j, the vault
or any credential. Everything these tests read is built in the worktree.
"""

from __future__ import annotations

import re
import sqlite3
import struct
from pathlib import Path
from typing import Any

import pytest

from scripts.backup.target import RESTORE_MARKER_FILENAME
from tests.mocks.neo4j import FakeNeo4jConnection, FakeNeo4jRecord

_SQLITE_VEC_AVAILABLE = False
try:
    import sqlite_vec as _sqlite_vec  # noqa: F401

    _SQLITE_VEC_AVAILABLE = True
except ImportError:
    pass

# Same convention as `tests/unit/test_factories_vault.py:45`: a platform without
# the extension SKIPS these tests rather than erroring in the fixture, because
# the fixture cannot even BUILD a vec0 store without it.
requires_sqlite_vec = pytest.mark.skipif(
    not _SQLITE_VEC_AVAILABLE,
    reason="sqlite_vec not available on this platform",
)

# The width `all-MiniLM-L6-v2` produces is 384; 4 is enough to exercise the
# module boundary and keeps the fixture store small.
VEC0_DIMENSIONS = 4

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


def make_vec0_sidecar(path: Path, *, rows: int) -> None:
    """Build a `vault_sidecar.db` shaped like the live one: a vec0 virtual table.

    `CREATE VIRTUAL TABLE ... USING vec0` mirrors
    `backend/vault/sidecar_index.py:849`, and brings four shadow tables with it
    (`_chunks`, `_info`, `_rowids`, `_vector_chunks00`). Building this needs
    sqlite_vec, which is why every test that uses it carries
    `requires_sqlite_vec`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    try:
        conn.enable_load_extension(True)
        _sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE vault_chunks (chunk_id INTEGER PRIMARY KEY, payload TEXT)")
        conn.execute(
            f"CREATE VIRTUAL TABLE vault_chunks_vec USING vec0(embedding float[{VEC0_DIMENSIONS}])"
        )
        for index in range(rows):
            chunk_id = index + 1
            conn.execute(
                "INSERT INTO vault_chunks (chunk_id, payload) VALUES (?, ?)",
                (chunk_id, f"chunk-{index}"),
            )
            packed = struct.pack(
                f"<{VEC0_DIMENSIONS}f",
                *[float(index + offset) for offset in range(VEC0_DIMENSIONS)],
            )
            conn.execute(
                "INSERT INTO vault_chunks_vec(rowid, embedding) VALUES (?, ?)",
                (chunk_id, packed),
            )
        conn.commit()
    finally:
        conn.close()


@pytest.fixture
def vec0_state_root(tmp_path: Path) -> Path:
    """A live-state stand-in whose sidecar holds vec0 tables, as the real one does.

    SEPARATE FROM `state_root` ON PURPOSE. Three existing assertions pin exact
    row-count dicts against that fixture (`test_stores.py:115`,
    `test_dump.py:154`, `test_restore.py:186`); adding a vec0 table there would
    break them and make them depend on whether the platform has the extension.
    """
    root = tmp_path / "live-data-vec0"
    make_store(root / "event_store.db", table="conversation_turn_events", rows=3)
    make_vec0_sidecar(root / "vault_sidecar.db", rows=2)
    return root


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

# A 384-float embedding, the width `all-MiniLM-L6-v2` produces
# (`grep -n "384-dim" CLAUDE.md`). Values are exact binary fractions plus one
# that is not, so an equality assertion over the restored list is a real check
# on the codec rather than one that passes because every value is 0.5.
EMBEDDING = [(index - 192) / 256.0 for index in range(383)] + [0.1]

# The MATCH statements `restore_graph_from_artifact` writes, parsed by the fake
# below. Labels and relationship types cannot be parameterized in Cypher, so
# they are interpolated into the statement and have to be read back out of it
# (`grep -n "def _label_pattern" backend/knowledge/admin.py` -> :996).
_NODE_CREATE = re.compile(r"CREATE \(n(?P<labels>:[^)]*)?\) SET n = row\.props")
_REL_CREATE = re.compile(r"CREATE \(a\)-\[r:`(?P<type>[^`]+)`\]->\(b\)")


class InMemoryGraphConnection(FakeNeo4jConnection):
    """A `GraphConnection` fake that HOLDS a graph, so a round trip is a real one.

    `FakeNeo4jConnection` alone returns canned rows for every write, which is
    enough to assert that a restore issued the right statements and not enough
    to assert that the graph coming out equals the graph that went in. That
    equality is the whole claim MIS-140 makes, so the fake executes the four
    write shapes `restore_graph_from_artifact` emits -- detach-delete, DDL,
    batched node CREATE, batched relationship CREATE -- against dicts.

    Values are stored by reference-free copy and never stringified, so an
    embedding that arrives as `list[float]` is compared as `list[float]`.
    """

    def __init__(
        self,
        *,
        nodes: list[dict[str, Any]] | None = None,
        relationships: list[dict[str, Any]] | None = None,
        constraints: dict[str, str] | None = None,
        indexes: dict[str, str] | None = None,
    ) -> None:
        super().__init__(query_router=self._route_read)
        self.nodes: list[dict[str, Any]] = [
            {"labels": sorted(n["labels"]), "properties": dict(n["properties"])}
            for n in (nodes or [])
        ]
        self.relationships: list[dict[str, Any]] = [dict(r) for r in (relationships or [])]
        self.constraints: dict[str, str] = dict(constraints or {})
        self.indexes: dict[str, str] = dict(indexes or {})

    def node_by_id(self, node_id: str) -> dict[str, Any]:
        """Return the single node carrying `node_id`, or raise if there is not one."""
        matches = [n for n in self.nodes if n["properties"].get("id") == node_id]
        if len(matches) != 1:
            raise AssertionError(f"expected exactly one node with id {node_id!r}, got {matches}")
        return matches[0]

    def snapshot(self) -> dict[str, Any]:
        """An order-independent view of the whole graph, comparable with `==`.

        Sorted by a string KEY rather than by the rows themselves: a property
        value may be a `list[float]` embedding, and ordering rows directly would
        compare a list against a string and raise. Values are copied, never
        rendered, so the comparison stays an exact float-list comparison.
        """
        return {
            "nodes": sorted(
                (
                    {"labels": list(n["labels"]), "properties": dict(n["properties"])}
                    for n in self.nodes
                ),
                key=lambda node: str(node["properties"].get("id")),
            ),
            "relationships": sorted(
                (
                    {
                        "source": r["source"],
                        "type": r["type"],
                        "target": r["target"],
                        "properties": dict(r.get("properties") or {}),
                    }
                    for r in self.relationships
                ),
                key=lambda rel: (str(rel["source"]), str(rel["type"]), str(rel["target"])),
            ),
        }

    def _route_read(self, query: str, _params: dict | None) -> list | None:
        if "SHOW CONSTRAINTS" in query:
            return [
                FakeNeo4jRecord({"name": name, "createStatement": statement})
                for name, statement in sorted(self.constraints.items())
            ]
        if "SHOW INDEXES" in query:
            return [
                FakeNeo4jRecord({"name": name, "createStatement": statement})
                for name, statement in sorted(self.indexes.items())
            ]
        if "MATCH (s)-[r]->(t)" in query:
            return [
                FakeNeo4jRecord(
                    {
                        "source": rel["source"],
                        "type": rel["type"],
                        "target": rel["target"],
                        "properties": dict(rel.get("properties") or {}),
                    }
                )
                for rel in self.relationships
            ]
        if "MATCH (n)" in query:
            return [
                FakeNeo4jRecord(
                    {
                        "id": node["properties"].get("id"),
                        "labels": list(node["labels"]),
                        "properties": dict(node["properties"]),
                    }
                )
                for node in self.nodes
            ]
        return None

    def execute_write(self, query, params=None):
        self.writes.append((query, params))

        if "DETACH DELETE" in query:
            deleted = len(self.nodes)
            self.nodes = []
            self.relationships = []
            return [FakeNeo4jRecord({"deleted": deleted})]

        if query.strip().upper().startswith("CREATE CONSTRAINT"):
            self.constraints[_ddl_name(query)] = query
            return []
        if "INDEX" in query.upper() and query.strip().upper().startswith("CREATE"):
            self.indexes[_ddl_name(query)] = query
            return []

        rows = (params or {}).get("rows", [])

        node_match = _NODE_CREATE.search(query)
        if node_match is not None:
            raw_labels = node_match.group("labels") or ""
            labels = sorted(re.findall(r"`([^`]+)`", raw_labels))
            for row in rows:
                self.nodes.append({"labels": labels, "properties": dict(row["props"])})
            return []

        rel_match = _REL_CREATE.search(query)
        if rel_match is not None:
            ids = {n["properties"].get("id") for n in self.nodes}
            created = 0
            for row in rows:
                if row["source"] in ids and row["target"] in ids:
                    self.relationships.append(
                        {
                            "source": row["source"],
                            "type": rel_match.group("type"),
                            "target": row["target"],
                            "properties": dict(row["props"]),
                        }
                    )
                    created += 1
            return [FakeNeo4jRecord({"created": created})]

        raise AssertionError(f"InMemoryGraphConnection got an unexpected write: {query!r}")


def _ddl_name(statement: str) -> str:
    """Pull the object name out of a captured `createStatement`, for the fake's registry."""
    parts = statement.split()
    for position, token in enumerate(parts):
        if token.upper() in {"CONSTRAINT", "INDEX"} and "`" not in token:
            return parts[position + 1].strip("`") if position + 1 < len(parts) else statement
    return statement


def make_populated_graph() -> InMemoryGraphConnection:
    """Three nodes, two edges, one 384-float embedding, one unlabelled node."""
    return InMemoryGraphConnection(
        nodes=[
            {
                "labels": ["__Entity__", "Person"],
                "properties": {"id": "person-raj", "name": "Raj", "embedding": list(EMBEDDING)},
            },
            {
                "labels": ["__Entity__", "Project"],
                "properties": {"id": "project-mist", "name": "MIST.AI", "started": 2025},
            },
            # A node with no labels is legal in Neo4j and appears in the live
            # graph as an unpartitioned node, so the round trip carries one.
            {"labels": [], "properties": {"id": "loose-note", "text": "no partition"}},
        ],
        relationships=[
            {
                "source": "person-raj",
                "type": "WORKS_ON",
                "target": "project-mist",
                "properties": {"confidence": 0.9},
            },
            {
                "source": "project-mist",
                "type": "MENTIONS",
                "target": "loose-note",
                "properties": {},
            },
        ],
        constraints={"c1": "CREATE CONSTRAINT `c1` FOR (n:__Entity__) REQUIRE n.id IS UNIQUE"},
        indexes={"i1": "CREATE VECTOR INDEX `i1` FOR (n:__Entity__) ON (n.embedding)"},
    )


@pytest.fixture
def source_graph() -> InMemoryGraphConnection:
    """The graph a backup is taken FROM."""
    return make_populated_graph()


@pytest.fixture
def target_graph() -> InMemoryGraphConnection:
    """The graph a restore writes INTO, holding state that must be replaced."""
    return InMemoryGraphConnection(
        nodes=[{"labels": ["__Entity__"], "properties": {"id": "stale", "name": "stale"}}],
        relationships=[],
    )


@pytest.fixture
def restore_target(tmp_path: Path) -> Path:
    """A marked, non-live target root holding its own state, as the dev stack does.

    Mirrors `docker-compose.dev-hydration.yml`, whose backend is configured with
    EVENT_STORE_DB_PATH=/app/dev-state/event_store.db (line 126) and
    MIST_VAULT_ROOT=/app/dev-state/vault (line 129).
    """
    root = tmp_path / "dev-state"
    make_store(root / "event_store.db", table="conversation_turn_events", rows=1)
    make_store(root / "vault_sidecar.db", table="vault_chunks", rows=1)
    (root / RESTORE_MARKER_FILENAME).write_text("", encoding="utf-8")
    (root / "vault").mkdir()
    (root / "vault" / "target-only.md").write_text("# target only\n", encoding="utf-8")
    return root
