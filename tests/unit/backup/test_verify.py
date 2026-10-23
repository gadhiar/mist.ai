"""Plan v1 T7, item 7: `python -m scripts.backup.verify` proves a restore matches its artifact.

Every test runs a real synthetic round trip: `run_dump` captures the synthetic source state,
`run_restore` writes it into a different target, and verify compares that target with the
artifact. The graph on both sides is an in-memory fake, so no test reaches Neo4j, the live stack
or any of Raj's data; the stores and the vault are built under `tmp_path`.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import pytest

from backend.knowledge.eval_isolation import (
    LIVE_INSTANCE_MARKER,
    LIVE_INSTANCES_FILE_ENV,
    EvalIsolationError,
)
from scripts.backup.dump import run_dump
from scripts.backup.restore import run_restore
from scripts.backup.verify import (
    EXIT_MATCH,
    EXIT_MISMATCH,
    EXIT_REFUSED,
    main,
    verify_restore,
)

from .conftest import STAMPS, InMemoryGraphConnection

TARGET_URI = "bolt://localhost:7690"
NEO4J_CLASS = "backend.knowledge.storage.neo4j_connection.Neo4jConnection"
ITEM_7_KEYS = {"result", "nodes", "relationships", "embeddings_equal", "stores", "vault_files"}

# Content the synthetic source carries that must never reach verify's output.
SECRET_TEXT = "private-note-body-7f3a"
SECRET_FILENAME = "private-journal-9c1e.md"


@pytest.fixture
def secret_vault_root(vault_root: Path) -> Path:
    (vault_root / "journal").mkdir()
    (vault_root / "journal" / SECRET_FILENAME).write_text(SECRET_TEXT, encoding="utf-8")
    return vault_root


@pytest.fixture
def artifact(backup_root, source_graph, state_root, secret_vault_root) -> Path:
    """One artifact taken by the real dump from the synthetic source state."""
    source_graph.nodes[2]["properties"]["text"] = SECRET_TEXT
    report = run_dump(
        backup_root=backup_root,
        connection=source_graph,
        source_uri="bolt://mist-neo4j:7687",
        database=None,
        stamps=dict(STAMPS),
        state_root=state_root,
        vault_root=secret_vault_root,
        label="source-artifact",
    )
    return report.artifact_dir


@pytest.fixture
def restored(artifact, restore_target, target_graph, backup_root) -> InMemoryGraphConnection:
    """The target after a real restore of `artifact`; returns the restored graph."""
    run_restore(
        artifact_dir=artifact,
        target_root=restore_target,
        target_graph_uri=TARGET_URI,
        connection=target_graph,
        confirm_token=str(restore_target.resolve()),
        stamps=dict(STAMPS),
        backup_root=backup_root,
    )
    return target_graph


def verify(artifact: Path, restore_target: Path, graph: InMemoryGraphConnection, **overrides):
    kwargs = {
        "artifact_dir": artifact,
        "graph_uri": TARGET_URI,
        "state_root": restore_target,
        "vault_root": None,
        "connection": graph,
    }
    kwargs.update(overrides)
    return verify_restore(**kwargs)


def cli(artifact: Path, restore_target: Path, *extra: str) -> list[str]:
    return [
        "--artifact",
        str(artifact),
        "--graph-uri",
        TARGET_URI,
        "--state-root",
        str(restore_target),
        *extra,
    ]


def delete_one_row(db_path: Path, table: str) -> None:
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(f"DELETE FROM {table} WHERE id = (SELECT MIN(id) FROM {table})")
        conn.commit()
    finally:
        conn.close()


def embedded_node(graph: InMemoryGraphConnection) -> dict:
    return next(node for node in graph.nodes if "embedding" in node["properties"])


# --- Exit 2: the isolation refusals --------------------------------------------------------


class TestRefusals:
    @pytest.mark.parametrize(
        "uri",
        [
            pytest.param("bolt://mist-neo4j:7687", id="live-in-compose"),
            pytest.param("bolt://localhost:7687", id="live-host-port"),
            pytest.param("bolt://example.invalid:7690", id="unknown-host"),
            pytest.param("not-a-uri", id="no-host-port"),
        ],
    )
    def test_a_graph_uri_outside_the_dev_allowlist_exits_2_before_any_connection(
        self, artifact, restore_target, monkeypatch, capsys, uri
    ):
        def no_connection(_config):
            raise AssertionError("verify connected before its URI guard")

        monkeypatch.setattr(NEO4J_CLASS, no_connection)
        argv = cli(artifact, restore_target)
        argv[argv.index("--graph-uri") + 1] = uri

        code = main(argv)

        assert code == EXIT_REFUSED
        assert capsys.readouterr().out == ""

    def test_a_state_root_under_a_marked_instance_exits_2(
        self, artifact, tmp_path, monkeypatch, capsys
    ):
        monkeypatch.setattr(NEO4J_CLASS, lambda _config: InMemoryGraphConnection())
        live = tmp_path / "daily"
        (live / "state").mkdir(parents=True)
        (live / LIVE_INSTANCE_MARKER).write_text('{"v": 1}', encoding="utf-8")

        code = main(cli(artifact, live / "state"))

        assert code == EXIT_REFUSED
        captured = capsys.readouterr()
        assert captured.out == ""
        assert LIVE_INSTANCE_MARKER in captured.err

    def test_a_marked_state_root_exits_2_beside_a_valid_vault_root(
        self, artifact, restore_target, tmp_path, monkeypatch
    ):
        """With `--vault-root` elsewhere, only the state-root guard can refuse this."""
        monkeypatch.setattr(NEO4J_CLASS, lambda _config: InMemoryGraphConnection())
        live = tmp_path / "daily"
        (live / "state").mkdir(parents=True)
        (live / LIVE_INSTANCE_MARKER).write_text('{"v": 1}', encoding="utf-8")
        vault = tmp_path / "standby-memory"
        vault.mkdir()

        code = main(cli(artifact, live / "state", "--vault-root", str(vault)))

        assert code == EXIT_REFUSED

    def test_a_state_root_registered_as_a_live_instance_exits_2(
        self, artifact, restore_target, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(NEO4J_CLASS, lambda _config: InMemoryGraphConnection())
        registry = tmp_path / "live-instances"
        registry.write_text(f"{restore_target.resolve()}\n", encoding="utf-8")
        monkeypatch.setenv(LIVE_INSTANCES_FILE_ENV, str(registry))

        assert main(cli(artifact, restore_target)) == EXIT_REFUSED

    def test_a_vault_root_under_a_marked_instance_exits_2(
        self, artifact, restore_target, tmp_path, monkeypatch
    ):
        monkeypatch.setattr(NEO4J_CLASS, lambda _config: InMemoryGraphConnection())
        live = tmp_path / "daily"
        (live / "mist-memory").mkdir(parents=True)
        (live / "mist-memory" / LIVE_INSTANCE_MARKER).write_text('{"v": 1}', encoding="utf-8")

        code = main(cli(artifact, restore_target, "--vault-root", str(live / "mist-memory")))

        assert code == EXIT_REFUSED

    def test_a_directory_without_a_manifest_exits_2(self, restore_target, tmp_path, monkeypatch):
        monkeypatch.setattr(NEO4J_CLASS, lambda _config: InMemoryGraphConnection())
        not_an_artifact = tmp_path / "not-an-artifact"
        not_an_artifact.mkdir()

        assert main(cli(not_an_artifact, restore_target)) == EXIT_REFUSED

    def test_the_library_entry_refuses_a_live_graph_uri_without_reading_the_graph(
        self, artifact, restore_target
    ):
        graph = InMemoryGraphConnection()

        with pytest.raises(EvalIsolationError, match="mist-neo4j:7687"):
            verify(artifact, restore_target, graph, graph_uri="bolt://mist-neo4j:7687")

        assert graph.queries == []


# --- What is compared ----------------------------------------------------------------------


class TestTheComparison:
    def test_a_faithful_restore_matches_on_every_count(self, artifact, restore_target, restored):
        report = verify(artifact, restore_target, restored)

        assert report.matched
        payload = report.to_dict()
        assert payload["result"] == "match"
        assert payload["nodes"] == {"artifact": 3, "restored": 3}
        assert payload["relationships"] == {"artifact": 2, "restored": 2}
        assert payload["embeddings_equal"] is True
        assert payload["stores"] == {
            "event_store.db": {"artifact": 3, "restored": 3},
            "extraction_cache.db": {"artifact": 2, "restored": 2},
            "vault_sidecar.db": {"artifact": 5, "restored": 5},
        }
        assert payload["vault_files"] == {"artifact": 3, "restored": 3}

    def test_a_missing_node_is_a_node_count_mismatch(self, artifact, restore_target, restored):
        restored.nodes = [n for n in restored.nodes if n["properties"]["id"] != "loose-note"]

        report = verify(artifact, restore_target, restored)

        assert not report.matched
        assert report.to_dict()["nodes"] == {"artifact": 3, "restored": 2}

    def test_a_missing_relationship_is_a_relationship_count_mismatch(
        self, artifact, restore_target, restored
    ):
        restored.relationships.pop()

        report = verify(artifact, restore_target, restored)

        assert not report.matched
        assert report.to_dict()["relationships"] == {"artifact": 2, "restored": 1}

    def test_an_embedding_one_ulp_away_is_not_equal(self, artifact, restore_target, restored):
        node = embedded_node(restored)
        vector = list(node["properties"]["embedding"])
        vector[-1] = math.nextafter(vector[-1], math.inf)
        node["properties"]["embedding"] = vector

        report = verify(artifact, restore_target, restored)

        assert not report.matched
        payload = report.to_dict()
        assert payload["embeddings_equal"] is False
        # Every count still agrees: only the exact comparison catches a rewritten vector.
        assert payload["nodes"]["artifact"] == payload["nodes"]["restored"]

    def test_a_dropped_embedding_is_not_equal(self, artifact, restore_target, restored):
        del embedded_node(restored)["properties"]["embedding"]

        assert verify(artifact, restore_target, restored).to_dict()["embeddings_equal"] is False

    def test_an_embedding_moved_to_another_node_is_not_equal(
        self, artifact, restore_target, restored
    ):
        node = embedded_node(restored)
        other = next(n for n in restored.nodes if n is not node)
        other["properties"]["embedding"] = node["properties"].pop("embedding")

        assert verify(artifact, restore_target, restored).to_dict()["embeddings_equal"] is False

    def test_a_store_missing_a_row_is_a_store_mismatch(self, artifact, restore_target, restored):
        delete_one_row(restore_target / "event_store.db", "conversation_turn_events")

        report = verify(artifact, restore_target, restored)

        assert not report.matched
        assert report.to_dict()["stores"]["event_store.db"] == {"artifact": 3, "restored": 2}

    def test_a_table_count_that_differs_is_a_mismatch_when_the_totals_agree(
        self, artifact, restore_target, restored
    ):
        store = restore_target / "event_store.db"
        delete_one_row(store, "conversation_turn_events")
        conn = sqlite3.connect(str(store))
        try:
            conn.execute("CREATE TABLE stray (id INTEGER PRIMARY KEY, payload TEXT)")
            conn.execute("INSERT INTO stray (payload) VALUES ('row')")
            conn.commit()
        finally:
            conn.close()

        report = verify(artifact, restore_target, restored)

        assert not report.matched
        # The totals the output reports still agree: only the per-table comparison differs.
        assert report.to_dict()["stores"]["event_store.db"] == {"artifact": 3, "restored": 3}

    def test_a_store_absent_from_the_target_is_a_mismatch(self, artifact, restore_target, restored):
        (restore_target / "extraction_cache.db").unlink()

        report = verify(artifact, restore_target, restored)

        assert not report.matched
        assert report.to_dict()["stores"]["extraction_cache.db"] == {
            "artifact": 2,
            "restored": None,
        }

    def test_a_vault_file_missing_from_the_target_is_a_vault_mismatch(
        self, artifact, restore_target, restored
    ):
        (restore_target / "vault" / "users" / "raj.md").unlink()

        report = verify(artifact, restore_target, restored)

        assert not report.matched
        assert report.to_dict()["vault_files"] == {"artifact": 3, "restored": 2}

    def test_an_explicit_vault_root_is_the_one_counted(
        self, artifact, restore_target, restored, tmp_path
    ):
        elsewhere = tmp_path / "standby-memory"
        (elsewhere / "only").mkdir(parents=True)
        (elsewhere / "only" / "one.md").write_text("# one\n", encoding="utf-8")

        report = verify(artifact, restore_target, restored, vault_root=elsewhere)

        assert report.to_dict()["vault_files"] == {"artifact": 3, "restored": 1}

    def test_the_artifact_store_copies_are_read_without_being_written(
        self, artifact, restore_target, restored
    ):
        stores = artifact / "stores"
        before = {path.name: path.read_bytes() for path in stores.iterdir()}

        verify(artifact, restore_target, restored)

        assert {path.name: path.read_bytes() for path in stores.iterdir()} == before

    def test_the_restored_graph_is_only_read(self, artifact, restore_target, restored):
        writes_before = len(restored.writes)

        verify(artifact, restore_target, restored)

        assert len(restored.writes) == writes_before


# --- The CLI: exit codes and output --------------------------------------------------------


class TestTheCli:
    def test_a_full_match_exits_0_with_the_item_7_json(
        self, artifact, restore_target, restored, monkeypatch, capsys
    ):
        monkeypatch.setattr(NEO4J_CLASS, lambda _config: restored)

        code = main(cli(artifact, restore_target))

        assert code == EXIT_MATCH
        payload = json.loads(capsys.readouterr().out)
        assert set(payload) == ITEM_7_KEYS
        assert payload["result"] == "match"

    def test_any_mismatch_exits_1_with_the_item_7_json(
        self, artifact, restore_target, restored, monkeypatch, capsys
    ):
        restored.relationships.pop()
        monkeypatch.setattr(NEO4J_CLASS, lambda _config: restored)

        code = main(cli(artifact, restore_target))

        assert code == EXIT_MISMATCH
        payload = json.loads(capsys.readouterr().out)
        assert set(payload) == ITEM_7_KEYS
        assert payload["result"] == "mismatch"

    def test_the_output_is_one_json_object_of_counts_and_flags_only(
        self, artifact, restore_target, restored, monkeypatch, capsys
    ):
        monkeypatch.setattr(NEO4J_CLASS, lambda _config: restored)

        main(cli(artifact, restore_target))

        out = capsys.readouterr().out
        assert out.count("\n") == 1
        payload = json.loads(out)
        for key in ("nodes", "relationships", "vault_files"):
            assert set(payload[key]) == {"artifact", "restored"}
            assert all(isinstance(value, int) for value in payload[key].values())
        assert isinstance(payload["embeddings_equal"], bool)
        for counts in payload["stores"].values():
            assert set(counts) == {"artifact", "restored"}
            assert all(isinstance(value, int) for value in counts.values())

    def test_the_output_carries_no_content_from_the_data(
        self, artifact, restore_target, restored, monkeypatch, capsys
    ):
        monkeypatch.setattr(NEO4J_CLASS, lambda _config: restored)

        main(cli(artifact, restore_target))

        out = capsys.readouterr().out
        for forbidden in (
            SECRET_TEXT,
            SECRET_FILENAME,
            "journal",
            "person-raj",
            "Raj",
            "WORKS_ON",
            "row-0",
            "conversation_turn_events",
            str(artifact),
            str(restore_target),
            "0.1",
        ):
            assert forbidden not in out, forbidden

    def test_the_connection_is_closed_after_the_comparison(
        self, artifact, restore_target, restored, monkeypatch
    ):
        monkeypatch.setattr(NEO4J_CLASS, lambda _config: restored)

        main(cli(artifact, restore_target))

        assert restored.is_connected() is False
