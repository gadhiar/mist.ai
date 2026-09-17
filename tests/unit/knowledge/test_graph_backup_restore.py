"""Capture and restore around the codec: `admin` and the `graph-restore` guard.

Subjects: `dump_full_graph_artifact`, `read_schema_ddl`,
`restore_graph_from_artifact` in `backend/knowledge/admin.py`, and
`cmd_graph_restore` in `scripts/mist_admin.py`.

Shaped after `tests/unit/knowledge/test_admin_full_graph.py`, which tests
`dump_full_graph_json` through `FakeNeo4jConnection` for the same reason this
does: the unit tier cannot open Neo4j (`tests/unit/conftest.py` forces
`MIST_EVAL_ISOLATION=1`), and the properties under test here -- that a temporal
is TAGGED rather than stringified, and that a destructive restore refuses a
target outside the dev allowlist -- do not need a server to observe.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest
from neo4j.time import DateTime

from backend.knowledge.eval_isolation import EvalIsolationError
from backend.knowledge.graph_artifact import (
    GRAPH_ARTIFACT_FORMAT,
    GRAPH_ARTIFACT_VERSION,
    NEO4J_TYPE_TAG,
    GraphArtifactError,
    load_artifact,
)
from tests.mocks.neo4j import FakeNeo4jConnection

# scripts/ is not a package; insert repo root so mist_admin is importable.
_REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import mist_admin  # noqa: E402  -- after sys.path insertion

_CREATED_AT = DateTime(2026, 9, 17, 12, 0, 0, 123456789)


def _capture_conn(nodes, relationships, constraints=(), indexes=()):
    """Connection for the capture path.

    The four keys are mutually disjoint across the four reads
    `dump_full_graph_artifact` issues, so `FakeNeo4jConnection`'s
    first-match-wins substring matching cannot decide the answer.
    """
    return FakeNeo4jConnection(
        query_responses={
            "properties(n) AS properties": nodes,
            "MATCH (s)-[r]->(t)": relationships,
            "SHOW CONSTRAINTS YIELD name, createStatement": [
                {"name": f"c{i}", "createStatement": statement}
                for i, statement in enumerate(constraints)
            ],
            "SHOW INDEXES YIELD name, type": [
                {"name": f"i{i}", "createStatement": statement}
                for i, statement in enumerate(indexes)
            ],
        }
    )


def _restore_conn(write_results):
    """Connection for the restore path.

    `apply_schema_ddl` reads the EXISTING object names with a different
    projection than the capture path uses, so these keys are spelled in full to
    stay disjoint from `_capture_conn`'s.
    """
    return FakeNeo4jConnection(
        query_responses={
            "SHOW CONSTRAINTS YIELD name RETURN name": [],
            "SHOW INDEXES YIELD name RETURN name": [],
        },
        write_results=write_results,
    )


class TestDumpFullGraphArtifact:
    def test_a_temporal_is_tagged_not_stringified(self):
        """The defect this closes: `json.dumps(..., default=str)` on a DateTime.

        The old path wrote `"2026-09-17T12:00:00.123456789"` into the file and
        recorded nothing about the value having been a `DateTime`, so a restore
        would set a `str` where the captured property was a temporal.
        """
        conn = _capture_conn(
            [
                {
                    "id": "mist-identity",
                    "labels": ["__SelfModel__", "MistIdentity"],
                    "properties": {"id": "mist-identity", "created_at": _CREATED_AT},
                }
            ],
            [],
        )

        artifact = mist_admin_capture(conn)

        encoded = artifact["nodes"][0]["properties"]["created_at"]
        assert encoded[NEO4J_TYPE_TAG] == "DateTime"
        assert encoded["iso"] == "2026-09-17T12:00:00.123456789"

    def test_the_envelope_carries_format_version_source_and_stamps(self):
        artifact = mist_admin_capture(_capture_conn([], []))

        assert artifact["format"] == GRAPH_ARTIFACT_FORMAT
        assert artifact["format_version"] == GRAPH_ARTIFACT_VERSION
        assert artifact["source"] == {"uri": "bolt://mist-neo4j-dev:7687", "database": "neo4j"}
        assert artifact["stamps"] == {"ontology_version": "1.3.0"}

    def test_the_constraint_and_index_ddl_is_captured(self):
        """A restore with no constraints or vector index degrades retrieval silently."""
        conn = _capture_conn(
            [],
            [],
            constraints=["CREATE CONSTRAINT `entity_id` FOR (n:__Entity__) REQUIRE n.id IS UNIQUE"],
            indexes=["CREATE VECTOR INDEX `entity_embedding` FOR (n:__Entity__) ON (n.embedding)"],
        )

        artifact = mist_admin_capture(conn)

        assert artifact["schema"]["constraints"] == [
            "CREATE CONSTRAINT `entity_id` FOR (n:__Entity__) REQUIRE n.id IS UNIQUE"
        ]
        assert artifact["schema"]["indexes"] == [
            "CREATE VECTOR INDEX `entity_embedding` FOR (n:__Entity__) ON (n.embedding)"
        ]

    def test_the_hydration_scaffolding_index_is_excluded_by_name(self):
        """A backup taken after a hydration restore must not record `restore_key_tmp`.

        Excluded server-side in the SHOW INDEXES predicate, so the assertion is
        on the parameter the query was issued with.
        """
        conn = _capture_conn([], [])

        mist_admin_capture(conn)

        index_queries = [
            params for query, params in conn.queries if "SHOW INDEXES YIELD name, type" in query
        ]
        assert index_queries == [{"scaffolding": "restore_key_tmp"}]

    def test_counts_are_recorded_so_a_truncated_artifact_is_detectable(self):
        conn = _capture_conn(
            [{"id": "a", "labels": ["__Entity__"], "properties": {"id": "a"}}],
            [{"source": "a", "type": "USES", "target": "b", "properties": {}}],
        )

        artifact = mist_admin_capture(conn)

        assert artifact["counts"] == {"nodes": 1, "relationships": 1}

    def test_an_unencodable_property_refuses_the_whole_capture(self):
        """Half a backup is worse than a failed one: the graph is still readable."""
        conn = _capture_conn(
            [{"id": "a", "labels": ["__Entity__"], "properties": {"weird": {"a": 1}}}],
            [],
        )

        with pytest.raises(GraphArtifactError, match="weird"):
            mist_admin_capture(conn)


def mist_admin_capture(conn):
    """Call `dump_full_graph_artifact` with fixed source and stamps."""
    from backend.knowledge.admin import dump_full_graph_artifact

    return dump_full_graph_artifact(
        conn,
        source_uri="bolt://mist-neo4j-dev:7687",
        database="neo4j",
        stamps={"ontology_version": "1.3.0"},
    )


def _artifact(nodes, relationships, schema=None):
    """A loaded (decoded) artifact, as `restore_graph_from_artifact` expects."""
    return {
        "format": GRAPH_ARTIFACT_FORMAT,
        "format_version": GRAPH_ARTIFACT_VERSION,
        "schema": schema or {"constraints": [], "indexes": []},
        "nodes": nodes,
        "relationships": relationships,
    }


class TestRestoreGraphFromArtifact:
    def test_an_unresolvable_endpoint_refuses_before_anything_is_written(self):
        """Restore clears the target first, so validation must precede the wipe.

        A failure discovered half way through leaves neither the old graph nor
        the new one.
        """
        from backend.knowledge.admin import restore_graph_from_artifact

        conn = _restore_conn([{"deleted": 0, "created": 0}])
        artifact = _artifact(
            [{"id": "a", "labels": ["__Entity__"], "properties": {"id": "a"}}],
            [{"source": "a", "type": "USES", "target": "ghost", "properties": {}}],
        )

        with pytest.raises(GraphArtifactError, match="ghost"):
            restore_graph_from_artifact(conn, artifact)
        conn.assert_no_writes()

    def test_a_null_endpoint_refuses(self):
        """`dump_full_graph_json` emits None when an endpoint has no `id` property."""
        from backend.knowledge.admin import restore_graph_from_artifact

        conn = _restore_conn([{"deleted": 0, "created": 0}])
        artifact = _artifact(
            [{"id": "a", "labels": [], "properties": {"id": "a"}}],
            [{"source": "a", "type": "USES", "target": None, "properties": {}}],
        )

        with pytest.raises(GraphArtifactError, match="endpoint"):
            restore_graph_from_artifact(conn, artifact)
        conn.assert_no_writes()

    def test_duplicate_node_ids_refuse(self):
        """`MATCH (a {id: $id})` against two nodes would create every edge twice."""
        from backend.knowledge.admin import restore_graph_from_artifact

        conn = _restore_conn([{"deleted": 0, "created": 0}])
        artifact = _artifact(
            [
                {"id": "a", "labels": ["__Entity__"], "properties": {"id": "a"}},
                {"id": "a", "labels": ["__Entity__"], "properties": {"id": "a"}},
            ],
            [],
        )

        with pytest.raises(GraphArtifactError, match="duplicated node id"):
            restore_graph_from_artifact(conn, artifact)
        conn.assert_no_writes()

    def test_nodes_are_written_with_driver_native_property_values(self):
        """The point of the whole codec: a DateTime arrives as a DateTime."""
        from backend.knowledge.admin import restore_graph_from_artifact

        conn = _restore_conn([{"deleted": 0, "created": 1}])
        artifact = _artifact(
            [
                {
                    "id": "a",
                    "labels": ["MistIdentity", "__SelfModel__"],
                    "properties": {"id": "a", "created_at": _CREATED_AT},
                },
                {"id": "b", "labels": ["__Entity__"], "properties": {"id": "b"}},
            ],
            [{"source": "a", "type": "HAS_TRAIT", "target": "b", "properties": {}}],
        )

        report = restore_graph_from_artifact(conn, artifact)

        assert report["nodes"] == 2
        assert report["relationships"] == 1
        create = next((query, params) for query, params in conn.writes if "CREATE (n:" in query)
        assert "`MistIdentity`:`__SelfModel__`" in create[0]
        assert create[1]["rows"][0]["props"]["created_at"] == _CREATED_AT

    def test_a_short_relationship_batch_reports_a_partial_load(self):
        """Silence here would leave a graph missing edges and nothing saying so."""
        from backend.knowledge.admin import restore_graph_from_artifact

        conn = _restore_conn([{"deleted": 0, "created": 0}])
        artifact = _artifact(
            [
                {"id": "a", "labels": ["__Entity__"], "properties": {"id": "a"}},
                {"id": "b", "labels": ["__Entity__"], "properties": {"id": "b"}},
            ],
            [{"source": "a", "type": "USES", "target": "b", "properties": {}}],
        )

        with pytest.raises(GraphArtifactError, match="PARTIALLY LOADED"):
            restore_graph_from_artifact(conn, artifact)

    def test_an_unquotable_relationship_type_refuses(self):
        from backend.knowledge.admin import restore_graph_from_artifact

        conn = _restore_conn([{"deleted": 0, "created": 1}])
        artifact = _artifact(
            [
                {"id": "a", "labels": ["__Entity__"], "properties": {"id": "a"}},
                {"id": "b", "labels": ["__Entity__"], "properties": {"id": "b"}},
            ],
            [{"source": "a", "type": "US`ES", "target": "b", "properties": {}}],
        )

        with pytest.raises(GraphArtifactError, match="unquotable"):
            restore_graph_from_artifact(conn, artifact)


class TestGraphRestoreRefusesANonIsolatedTarget:
    @pytest.mark.parametrize(
        "uri",
        [
            pytest.param("bolt://mist-neo4j:7687", id="live-service-name"),
            pytest.param("bolt://localhost:7687", id="live-host-published-port"),
            pytest.param("bolt://some-other-host:7687", id="not-on-the-dev-allowlist"),
        ],
    )
    def test_the_guard_runs_before_the_artifact_is_even_read(self, uri, tmp_path):
        """A restore detach-deletes its target, so the URI is checked first.

        The artifact path given here does not exist: if the guard ran after the
        read, this would raise FileNotFoundError instead.
        """
        args = argparse.Namespace(
            artifact=str(tmp_path / "does-not-exist.json"), uri=uri, confirm=True
        )

        with pytest.raises(EvalIsolationError):
            mist_admin.cmd_graph_restore(args)

    def test_a_dev_endpoint_passes_the_guard(self, tmp_path):
        """The negative control: the guard is not refusing everything.

        Reaching the missing file proves the URI cleared the allowlist.
        """
        args = argparse.Namespace(
            artifact=str(tmp_path / "does-not-exist.json"),
            uri="bolt://mist-neo4j-dev:7687",
            confirm=True,
        )

        with pytest.raises(FileNotFoundError):
            mist_admin.cmd_graph_restore(args)

    def test_without_confirm_a_dev_target_only_validates_the_artifact(self, tmp_path, capsys):
        """Reading an artifact must be possible without risking a write."""
        artifact_path = tmp_path / "backup.json"
        artifact_path.write_text(
            json.dumps(
                {
                    "format": GRAPH_ARTIFACT_FORMAT,
                    "format_version": GRAPH_ARTIFACT_VERSION,
                    "captured_at": "2026-09-17T00:00:00Z",
                    "stamps": {"extraction_version": "1999-01-01-r0"},
                    "counts": {"nodes": 0, "relationships": 0},
                    "nodes": [],
                    "relationships": [],
                }
            ),
            encoding="utf-8",
        )
        args = argparse.Namespace(
            artifact=str(artifact_path), uri="bolt://mist-neo4j-dev:7687", confirm=False
        )

        assert mist_admin.cmd_graph_restore(args) == 0
        assert "--confirm" in capsys.readouterr().out

    def test_main_prints_the_refusal_and_exits_1_rather_than_tracing_back(self, tmp_path, capsys):
        """The refusal is the most important line this tool prints. It must be readable.

        `EvalIsolationError` is a bare `RuntimeError`, not a `MistError`
        (`grep -n "class EvalIsolationError" backend/knowledge/eval_isolation.py`
        -> `EvalIsolationError(RuntimeError)`), so `main`'s `MistError` arm did
        not cover it and a guarded command ended in a stack trace. `main` now
        names it explicitly alongside `MistError`.
        """
        exit_code = mist_admin.main(
            [
                "graph-restore",
                str(tmp_path / "does-not-exist.json"),
                "--uri",
                "bolt://mist-neo4j:7687",
                "--confirm",
            ]
        )

        assert exit_code == 1
        stderr = capsys.readouterr().err
        assert "[error] EvalIsolationError:" in stderr
        assert "live graph" in stderr

    def test_a_stale_extraction_version_does_not_block_the_load(self):
        """Stamps are recorded, never enforced. See `graph_version_stamps`."""
        loaded = load_artifact(
            {
                "format": GRAPH_ARTIFACT_FORMAT,
                "format_version": GRAPH_ARTIFACT_VERSION,
                "stamps": {"extraction_version": "1999-01-01-r0"},
                "nodes": [],
                "relationships": [],
            }
        )

        assert loaded["stamps"]["extraction_version"] == "1999-01-01-r0"
