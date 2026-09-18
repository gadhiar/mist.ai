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
import logging
import re
import sys
from pathlib import Path
from typing import Any

import pytest
from neo4j.time import DateTime

from backend.errors import Neo4jQueryError
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


# --- A target that already carries schema -------------------------------------
#
# MIS-140's restore leg failed with ConstraintAlreadyExists and left the target
# half-restored: stores and vault replaced, graph at 0 nodes. The cause was a
# property of the TARGET, not of the artifact -- it held a constraint named
# `rt_entity_id` over the same `(:__Entity__ {id})` schema the artifact's
# constraint covers, under a name that appears nowhere in this repository
# (`grep -rn "rt_entity_id" .` -> no hits), so skipping by name could not see it.

_ARTIFACT_CONSTRAINT = "CREATE CONSTRAINT `entity_id` FOR (n:__Entity__) REQUIRE n.id IS UNIQUE"
_ARTIFACT_INDEX = "CREATE VECTOR INDEX `entity_embedding` FOR (n:__Entity__) ON (n.embedding)"
_ARTIFACT_SCHEMA = {"constraints": [_ARTIFACT_CONSTRAINT], "indexes": [_ARTIFACT_INDEX]}

# Same constraint type, same label, same property -- different name.
_TARGET_EQUIVALENT_CONSTRAINT = (
    "CREATE CONSTRAINT `rt_entity_id` FOR (n:__Entity__) REQUIRE n.id IS UNIQUE"
)
# Same name as the artifact's, different constraint type.
_TARGET_DIVERGENT_CONSTRAINT = (
    "CREATE CONSTRAINT `entity_id` FOR (n:__Entity__) REQUIRE n.id IS NOT NULL"
)
# An object the artifact does not name at all.
_TARGET_ONLY_INDEX = "CREATE RANGE INDEX `legacy_name_idx` FOR (n:__Entity__) ON (n.name)"
# Server-created and server-maintained; excluded from capture, so dropping one
# would strip it permanently.
_TARGET_LOOKUP_INDEX = "CREATE LOOKUP INDEX `index_343aff4e` FOR (n) ON EACH labels(n)"

_CONSTRAINT_DDL = re.compile(
    r"CREATE CONSTRAINT `(?P<name>[^`]+)` FOR \(n:(?P<label>[^)]+)\) "
    r"REQUIRE n\.(?P<prop>\w+) IS (?P<type>[A-Z ]+?)\s*$"
)
_INDEX_DDL = re.compile(
    r"CREATE (?P<type>RANGE|VECTOR|TEXT|POINT) INDEX `(?P<name>[^`]+)` "
    r"FOR \(n:(?P<label>[^)]+)\) ON \(n\.(?P<prop>\w+)\)"
)
_LOOKUP_DDL = re.compile(r"CREATE LOOKUP INDEX `(?P<name>[^`]+)` FOR \(n\) ON EACH labels\(n\)")
_DROP_DDL = re.compile(r"DROP (?P<kind>CONSTRAINT|INDEX) `(?P<name>[^`]+)` IF EXISTS")


def _parse_schema_ddl(statement: str) -> dict[str, Any]:
    """Describe a `createStatement` the way a server catalogue would.

    `signature` is the (type, schema) pair Neo4j compares for equivalence, and
    it is deliberately independent of `name`: two statements can share a
    signature and differ in name, which is the whole shape of this defect.
    """
    match = _CONSTRAINT_DDL.search(statement)
    if match is not None:
        return {
            "name": match.group("name"),
            "kind": "CONSTRAINT",
            "type": match.group("type"),
            "signature": (match.group("type"), match.group("label"), match.group("prop")),
            "createStatement": statement,
            "owning_constraint": None,
        }
    match = _INDEX_DDL.search(statement)
    if match is not None:
        return {
            "name": match.group("name"),
            "kind": "INDEX",
            "type": match.group("type"),
            "signature": (match.group("type"), match.group("label"), match.group("prop")),
            "createStatement": statement,
            "owning_constraint": None,
        }
    match = _LOOKUP_DDL.search(statement)
    if match is not None:
        return {
            "name": match.group("name"),
            "kind": "INDEX",
            "type": "LOOKUP",
            "signature": ("LOOKUP", "", ""),
            "createStatement": statement,
            "owning_constraint": None,
        }
    raise AssertionError(f"the schema fake cannot parse {statement!r}")


class SchemaRejectingConnection(FakeNeo4jConnection):
    """A target that REFUSES a conflicting DDL statement, the way Neo4j 5 does.

    Every other graph fake in this repository accepts every statement silently.
    A target-carries-schema test written against one of those passes whether or
    not the defect is fixed, so it proves nothing -- which is why the rejection
    is modelled here rather than assumed.

    The rule is the Cypher manual's own, stated in its description of
    `CREATE ... IF NOT EXISTS`: that clause "will ensure that no error is thrown
    and that no constraint is created if any other constraint with the given
    name, or another constraint on the same constraint type and schema, or both,
    already exists". Inverted, a plain CREATE is rejected when

    - an object of that name already exists, or
    - an equivalent CONSTRAINT -- same constraint type, same schema -- already
      exists under any name.

    That the second arm is silent under `IF NOT EXISTS` even when the names
    differ is exactly why the restore path drops and recreates instead.

    Index-to-index equivalence is NOT modelled: the wording above is about
    constraints, and inventing a rule the manual does not state here would put a
    guess inside the thing whose job is to be the reference.

    Error text follows the codes the server emits -- 22N65 for an equivalent
    constraint, 22N67 for a duplicated name, and
    `Neo.ClientError.Schema.ConstraintAlreadyExists` on older servers. The code
    under test matches on none of them (that is deliberate: the code varies by
    server version), so only the raising is load-bearing.
    """

    def __init__(self, *, schema: tuple[str, ...] = (), nodes: int = 0) -> None:
        super().__init__()
        self.constraints: dict[str, dict[str, Any]] = {}
        self.indexes: dict[str, dict[str, Any]] = {}
        self._nodes = nodes
        for statement in schema:
            self._install(_parse_schema_ddl(statement))

    def _install(self, descriptor: dict[str, Any]) -> None:
        if descriptor["kind"] == "CONSTRAINT":
            self.constraints[descriptor["name"]] = descriptor
            # A constraint owns a backing index of the same name. Modelled
            # because it is why constraints must be dropped first.
            self.indexes[descriptor["name"]] = {
                **descriptor,
                "kind": "INDEX",
                "type": "RANGE",
                "owning_constraint": descriptor["name"],
            }
        else:
            self.indexes[descriptor["name"]] = descriptor

    def execute_query(self, query, params=None):
        self.queries.append((query, params))
        if "SHOW CONSTRAINTS" in query:
            return [
                {"name": name, "createStatement": descriptor["createStatement"]}
                for name, descriptor in sorted(self.constraints.items())
            ]
        if "SHOW INDEXES" in query:
            rows = sorted(self.indexes.items())
            # The two predicates the production queries carry, evaluated
            # literally: drop either one from the source and the rows change.
            if "owningConstraint IS NULL" in query:
                rows = [(n, d) for n, d in rows if d["owning_constraint"] is None]
            if "type <> 'LOOKUP'" in query:
                rows = [(n, d) for n, d in rows if d["type"] != "LOOKUP"]
            scaffolding = (params or {}).get("scaffolding")
            if scaffolding is not None:
                rows = [(n, d) for n, d in rows if n != scaffolding]
            return [{"name": n, "createStatement": d["createStatement"]} for n, d in rows]
        return []

    def execute_write(self, query, params=None):
        self.writes.append((query, params))

        if "DETACH DELETE" in query:
            deleted, self._nodes = self._nodes, 0
            return [{"deleted": deleted}]

        drop = _DROP_DDL.search(query)
        if drop is not None:
            self._drop(drop.group("kind"), drop.group("name"))
            return []

        if query.upper().startswith("CREATE "):
            self._create(query)
            return []

        if "CREATE (n" in query:
            return []
        return [{"created": len((params or {}).get("rows", []))}]

    def _create(self, statement: str) -> None:
        descriptor = _parse_schema_ddl(statement)
        name = descriptor["name"]
        if name in self.constraints or name in self.indexes:
            raise Neo4jQueryError(
                f"Write transaction failed: 22N67 duplicated name: an object named {name!r} "
                "already exists (Neo.ClientError.Schema.ConstraintAlreadyExists on Neo4j 5.x "
                "before the 22Nxx codes)"
            )
        if descriptor["kind"] == "CONSTRAINT":
            clash = next(
                (
                    other
                    for other in self.constraints.values()
                    if other["signature"] == descriptor["signature"]
                ),
                None,
            )
            if clash is not None:
                raise Neo4jQueryError(
                    "Write transaction failed: 22N65 an equivalent constraint already exists, "
                    f"named {clash['name']!r} (Neo.ClientError.Schema.ConstraintAlreadyExists "
                    "on Neo4j 5.x before the 22Nxx codes)"
                )
        self._install(descriptor)

    def _drop(self, kind: str, name: str) -> None:
        if kind == "CONSTRAINT":
            if self.constraints.pop(name, None) is not None:
                self.indexes.pop(name, None)
            return
        descriptor = self.indexes.get(name)
        if descriptor is None:
            return
        if descriptor["owning_constraint"] is not None:
            raise AssertionError(
                f"tried to drop {name!r}, which a constraint owns. Neo4j refuses this: "
                "drop the constraint and the index goes with it."
            )
        del self.indexes[name]


class TestTheFakeModelsNeo4jsRejection:
    """Negative controls for the fake itself.

    Without these the schema tests below could pass because nothing ever
    refuses anything.
    """

    def test_an_equivalent_constraint_under_another_name_is_rejected(self):
        from backend.knowledge.admin import apply_schema_ddl

        conn = SchemaRejectingConnection(schema=(_TARGET_EQUIVALENT_CONSTRAINT,))

        with pytest.raises(Neo4jQueryError, match="22N65"):
            apply_schema_ddl(conn, _ARTIFACT_SCHEMA)

    def test_a_duplicated_name_is_rejected(self):
        """Issued straight at the fake, not through `apply_schema_ddl`.

        That function skips this case by name, so going through it would assert
        nothing about what the server does with the statement.
        """
        conn = SchemaRejectingConnection(schema=(_TARGET_DIVERGENT_CONSTRAINT,))

        with pytest.raises(Neo4jQueryError, match="22N67"):
            conn.execute_write(_ARTIFACT_CONSTRAINT)

    def test_it_accepts_ddl_a_real_server_would_accept(self):
        """The other negative control: the fake is not refusing everything."""
        from backend.knowledge.admin import apply_schema_ddl, read_schema_ddl

        conn = SchemaRejectingConnection()

        assert apply_schema_ddl(conn, _ARTIFACT_SCHEMA) == 2
        assert read_schema_ddl(conn) == _ARTIFACT_SCHEMA


class TestRestoreOntoATargetThatAlreadyCarriesSchema:
    def test_an_equivalent_constraint_under_another_name_no_longer_fails_the_restore(self):
        """The MIS-140 failure, reproduced: `rt_entity_id` over `(:__Entity__ {id})`.

        Names are not repo-controlled, so no name-based skip can see this. The
        target's constraint has to be dropped, and afterwards the target's
        schema has to EQUAL the artifact's -- not contain it.
        """
        from backend.knowledge.admin import read_schema_ddl, restore_graph_from_artifact

        conn = SchemaRejectingConnection(schema=(_TARGET_EQUIVALENT_CONSTRAINT,))

        report = restore_graph_from_artifact(conn, _artifact([], [], _ARTIFACT_SCHEMA))

        assert read_schema_ddl(conn) == _ARTIFACT_SCHEMA
        assert "rt_entity_id" not in conn.constraints
        assert report["schema_statements"] == 2

    def test_a_same_name_different_definition_converges_and_is_reported(self, caplog):
        """Converge on the artifact, loudly. Refusing would block recovery.

        `CREATE CONSTRAINT ... IF NOT EXISTS` is the wrong tool for exactly this
        case: the manual says it throws nothing when a constraint of the given
        name already exists, so the target would silently keep its own
        definition. After the drop the target's definition exists nowhere, so
        the log is the only record that it differed.
        """
        from backend.knowledge.admin import read_schema_ddl, restore_graph_from_artifact

        conn = SchemaRejectingConnection(schema=(_TARGET_DIVERGENT_CONSTRAINT,))

        with caplog.at_level(logging.WARNING, logger="backend.knowledge.admin"):
            restore_graph_from_artifact(conn, _artifact([], [], _ARTIFACT_SCHEMA))

        assert read_schema_ddl(conn) == _ARTIFACT_SCHEMA
        warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
        assert any("entity_id" in message and "IS NOT NULL" in message for message in warnings)

    def test_an_object_the_artifact_does_not_have_is_gone_afterwards(self):
        """The second, unmeasured defect: `apply_schema_ddl` only ever ADDED.

        Schema accumulated across restores and could never converge back on the
        artifact, so an index the target happened to hold survived forever.
        """
        from backend.knowledge.admin import read_schema_ddl, restore_graph_from_artifact

        conn = SchemaRejectingConnection(schema=(_TARGET_ONLY_INDEX,))

        restore_graph_from_artifact(conn, _artifact([], [], _ARTIFACT_SCHEMA))

        assert "legacy_name_idx" not in conn.indexes
        assert read_schema_ddl(conn) == _ARTIFACT_SCHEMA

    def test_a_lookup_index_is_not_dropped(self):
        """Dropping one would strip it permanently: no artifact re-creates it.

        `read_schema_ddl` excludes LOOKUP from capture
        (`grep -n "type <> 'LOOKUP'" backend/knowledge/admin.py`), so a LOOKUP
        index dropped here is gone for good and the restored graph loses a
        server-maintained index -- degraded retrieval, not a failure.
        """
        from backend.knowledge.admin import restore_graph_from_artifact

        conn = SchemaRejectingConnection(
            schema=(_TARGET_LOOKUP_INDEX, _TARGET_EQUIVALENT_CONSTRAINT)
        )

        restore_graph_from_artifact(conn, _artifact([], [], _ARTIFACT_SCHEMA))

        assert "index_343aff4e" in conn.indexes
        assert not [query for query, _ in conn.writes if "index_343aff4e" in query]

    def test_constraints_are_dropped_before_indexes(self):
        """Dropping a constraint drops the index it owns; the other order fails."""
        from backend.knowledge.admin import restore_graph_from_artifact

        conn = SchemaRejectingConnection(schema=(_TARGET_EQUIVALENT_CONSTRAINT, _TARGET_ONLY_INDEX))

        restore_graph_from_artifact(conn, _artifact([], [], _ARTIFACT_SCHEMA))

        drops = [query for query, _ in conn.writes if query.startswith("DROP ")]
        assert drops == [
            "DROP CONSTRAINT `rt_entity_id` IF EXISTS",
            "DROP INDEX `legacy_name_idx` IF EXISTS",
        ]

    def test_an_unquotable_object_name_refuses_rather_than_escaping(self):
        """A backtick in a name would break out of the quoting in a DROP."""
        from backend.knowledge.admin import drop_graph_schema

        conn = SchemaRejectingConnection()
        conn.constraints["ba`d"] = {
            "name": "ba`d",
            "kind": "CONSTRAINT",
            "type": "UNIQUE",
            "signature": ("UNIQUE", "X", "id"),
            "createStatement": "CREATE CONSTRAINT `ba`d` FOR (n:X) REQUIRE n.id IS UNIQUE",
            "owning_constraint": None,
        }

        with pytest.raises(GraphArtifactError, match="unquotable"):
            drop_graph_schema(conn)
