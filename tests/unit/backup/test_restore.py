"""MIS-140 T2: a full synthetic round trip, and every refusal that guards it.

The round trip is the point of the whole goal: dump synthetic stores, a vault
tree and a populated graph, restore them into a DIFFERENT target holding
different state, and assert the restored state equals the captured state.
Embeddings are compared with EXACT float-list equality, not cosine similarity --
a restore that returns vectors which are merely close has silently rewritten
them, and every similarity threshold in the codebase would hide it.

The container has no network and reaches neither the live stack nor Neo4j;
everything here is built in the worktree.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from backend.knowledge.eval_isolation import REPO_ROOT, EvalIsolationError
from backend.knowledge.graph_artifact import GraphArtifactError
from scripts.backup.destination import BACKUP_ROOT_ENV
from scripts.backup.dump import run_dump
from backend.errors import Neo4jQueryError
from scripts.backup.errors import (
    BackupDestinationError,
    BackupError,
    RestoreAbortedError,
    RestoreConfirmationError,
    RestorePreflightError,
    RestoreTargetError,
    RestoreTargetStateError,
)
from scripts.backup.manifest import MANIFEST_FILENAME, read_manifest, sha256_file
from scripts.backup.restore import (
    EXIT_REFUSED,
    RESTORE_PROGRESS_FILENAME,
    STAGING_SUFFIX,
    VAULT_PREVIOUS_SUFFIX,
    main,
    parse_ddl_object_name,
    remove_tree,
    restore_progress_marker_path,
    run_restore,
    stage_vault,
    staged_peak_bytes,
)
from scripts.backup.stores import STORES_DIRNAME
from scripts.backup.target import RESTORE_MARKER_FILENAME

from .conftest import EMBEDDING, STAMPS, InMemoryGraphConnection

TARGET_URI = "bolt://localhost:7690"


def capture(backup_root, connection, state_root, vault_root, label="source-artifact"):
    """Take one artifact from the synthetic source state."""
    return run_dump(
        backup_root=backup_root,
        connection=connection,
        source_uri="bolt://mist-neo4j:7687",
        database=None,
        stamps=dict(STAMPS),
        state_root=state_root,
        vault_root=vault_root,
        label=label,
    )


def restore(artifact_dir, restore_target, target_graph, backup_root, **overrides):
    """Run one restore with the synthetic fixtures wired in."""
    kwargs = {
        "artifact_dir": artifact_dir,
        "target_root": restore_target,
        "target_graph_uri": TARGET_URI,
        "connection": target_graph,
        "confirm_token": str(Path(restore_target).resolve()),
        "stamps": dict(STAMPS),
        "backup_root": backup_root,
    }
    kwargs.update(overrides)
    return run_restore(**kwargs)


def row_counts(db_path):
    """Row counts per user table, read straight from a restored store."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        names = [
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )
        ]
        return {
            name: conn.execute(f'SELECT COUNT(*) FROM "{name}"').fetchone()[0] for name in names
        }
    finally:
        conn.close()


def rewrite_graph_leg(artifact_dir, mutate):
    """Rewrite `graph.json` through `mutate` and KEEP the manifest digests valid.

    The point of the tests that use this: an artifact can be byte-for-byte
    intact, pass every sha256 check, and still be unreadable by this build. A
    tampered file that fails its digest proves nothing about that case.
    """
    path = artifact_dir / "graph.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    mutate(payload)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    manifest_path = artifact_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"]["graph.json"] = {
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return artifact_dir


def fingerprint(root):
    """Every file under `root` as {relative path: bytes}, for byte-identity assertions."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


@pytest.fixture
def artifact(backup_root, source_graph, state_root, vault_root):
    """One complete artifact taken from the synthetic source state."""
    return capture(backup_root, source_graph, state_root, vault_root).artifact_dir


class TestTheRoundTrip:
    def test_the_restored_graph_equals_the_captured_graph(
        self, artifact, restore_target, target_graph, backup_root, source_graph
    ):
        before = source_graph.snapshot()
        report = restore(artifact, restore_target, target_graph, backup_root)
        assert target_graph.snapshot() == before
        assert report.graph_nodes == 3
        assert report.graph_relationships == 2
        # The target's own node was detach-deleted rather than merged with.
        assert report.graph_deleted == 1
        assert all(n["properties"].get("id") != "stale" for n in target_graph.nodes)

    def test_the_restored_embedding_is_exactly_equal_float_for_float(
        self, artifact, restore_target, target_graph, backup_root
    ):
        restore(artifact, restore_target, target_graph, backup_root)
        restored = target_graph.node_by_id("person-raj")["properties"]["embedding"]
        # EXACT equality on the list, not cosine similarity. A vector that is
        # merely close has been rewritten, and every similarity check in the
        # codebase would still pass on it.
        assert restored == EMBEDDING
        assert len(restored) == 384
        assert all(isinstance(value, float) for value in restored)

    def test_an_unlabelled_node_survives_the_round_trip(
        self, artifact, restore_target, target_graph, backup_root
    ):
        restore(artifact, restore_target, target_graph, backup_root)
        assert target_graph.node_by_id("loose-note")["labels"] == []

    def test_the_captured_ddl_is_replayed_onto_the_target(
        self, artifact, restore_target, target_graph, backup_root
    ):
        report = restore(artifact, restore_target, target_graph, backup_root)
        assert report.schema_statements == 2
        assert "c1" in target_graph.constraints
        assert "i1" in target_graph.indexes

    def test_the_restored_stores_are_byte_identical_to_the_artifact(
        self, artifact, restore_target, target_graph, backup_root
    ):
        report = restore(artifact, restore_target, target_graph, backup_root)
        assert set(report.stores_restored) == {
            "event_store.db",
            "extraction_cache.db",
            "vault_sidecar.db",
        }
        for filename in report.stores_restored:
            captured = (artifact / STORES_DIRNAME / filename).read_bytes()
            assert (restore_target / filename).read_bytes() == captured

    def test_the_restored_stores_carry_the_captured_row_counts(
        self, artifact, restore_target, target_graph, backup_root
    ):
        # The target started with one row in event_store.db and the artifact
        # holds three, so an equal count is a replacement rather than a merge.
        restore(artifact, restore_target, target_graph, backup_root)
        assert row_counts(restore_target / "event_store.db") == {"conversation_turn_events": 3}
        assert row_counts(restore_target / "vault_sidecar.db") == {"vault_chunks": 5}

    def test_stale_wal_sidecars_beside_the_target_store_are_removed(
        self, artifact, restore_target, target_graph, backup_root
    ):
        # A restored .db paired with the TARGET's old -wal is a database SQLite
        # may open and apply stale frames into: wrong, and silent.
        for suffix in ("-wal", "-shm"):
            (restore_target / f"event_store.db{suffix}").write_bytes(b"stale")
        restore(artifact, restore_target, target_graph, backup_root)
        for suffix in ("-wal", "-shm"):
            assert not (restore_target / f"event_store.db{suffix}").exists()

    def test_the_vault_tree_is_replaced_rather_than_merged(
        self, artifact, restore_target, target_graph, backup_root
    ):
        report = restore(artifact, restore_target, target_graph, backup_root)
        vault = restore_target / "vault"
        assert report.vault_files == 2
        assert (vault / "identity" / "mist.md").read_text(encoding="utf-8") == "# MIST\n"
        # The target's own file is gone: a merge would leave a corpus that is
        # the union of two vaults and equal to neither.
        assert not (vault / "target-only.md").exists()

    def test_no_staging_or_marker_files_are_left_in_the_target(
        self, artifact, restore_target, target_graph, backup_root
    ):
        restore(artifact, restore_target, target_graph, backup_root)
        assert not list(restore_target.glob(f"*{STAGING_SUFFIX}"))
        assert not list(restore_target.glob(f"*{VAULT_PREVIOUS_SUFFIX}"))
        assert not restore_progress_marker_path(restore_target).exists()
        assert not list(restore_target.glob(f"{RESTORE_PROGRESS_FILENAME}*"))


class TestThePreRestoreBackup:
    def test_it_is_taken_before_anything_is_overwritten(
        self, artifact, restore_target, target_graph, backup_root
    ):
        report = restore(artifact, restore_target, target_graph, backup_root)
        assert report.pre_restore_artifact.is_dir()
        assert report.pre_restore_artifact.name.startswith("pre-restore-")
        manifest = read_manifest(report.pre_restore_artifact)
        # One row is the TARGET's pre-restore event store; the artifact being
        # restored holds three. This artifact therefore captured the state as it
        # was before the overwrite, which is the only way back.
        assert manifest.stores["event_store.db"]["row_counts"] == {"conversation_turn_events": 1}

    def test_it_captures_the_targets_vault_and_graph_too(
        self, artifact, restore_target, target_graph, backup_root
    ):
        report = restore(artifact, restore_target, target_graph, backup_root)
        pre = report.pre_restore_artifact
        assert (pre / "vault" / "target-only.md").is_file()
        manifest = read_manifest(pre)
        assert manifest.graph["nodes"] == 1

    def test_a_failing_pre_restore_backup_refuses_the_whole_restore(
        self, artifact, restore_target, target_graph, backup_root
    ):
        def failing_dump(**_kwargs):
            raise BackupError("synthetic capture failure")

        with pytest.raises(RestoreAbortedError) as refusal:
            restore(artifact, restore_target, target_graph, backup_root, dump=failing_dump)
        assert "synthetic capture failure" in str(refusal.value)
        # Nothing was overwritten: the target keeps its own state.
        assert row_counts(restore_target / "event_store.db") == {"conversation_turn_events": 1}
        assert (restore_target / "vault" / "target-only.md").is_file()
        assert target_graph.node_by_id("stale")["properties"]["name"] == "stale"

    def test_a_graph_artifact_error_in_the_capture_is_translated_not_leaked(
        self, artifact, restore_target, target_graph, backup_root
    ):
        # The likeliest real capture failure: the graph leg raises
        # `GraphArtifactError`, which is a RuntimeError rather than a MistError.
        # Untranslated it escaped as a traceback and exited with the code that
        # promises "the pre-restore artifact is the way back", when none existed.
        def failing_dump(**_kwargs):
            raise GraphArtifactError("a property value cannot be round-tripped")

        with pytest.raises(RestoreAbortedError) as refusal:
            restore(artifact, restore_target, target_graph, backup_root, dump=failing_dump)
        assert "cannot be round-tripped" in str(refusal.value)
        assert row_counts(restore_target / "event_store.db") == {"conversation_turn_events": 1}
        assert target_graph.writes == []

    def test_an_unset_backup_root_refuses_before_the_target_is_touched(
        self, artifact, restore_target, target_graph, monkeypatch
    ):
        monkeypatch.delenv(BACKUP_ROOT_ENV, raising=False)
        with pytest.raises(BackupDestinationError):
            restore(artifact, restore_target, target_graph, None)
        assert row_counts(restore_target / "event_store.db") == {"conversation_turn_events": 1}
        assert target_graph.writes == []


class TestTheFourGates:
    def test_it_refuses_without_a_confirmation_token(
        self, artifact, restore_target, target_graph, backup_root
    ):
        with pytest.raises(RestoreConfirmationError):
            restore(artifact, restore_target, target_graph, backup_root, confirm_token=None)
        assert row_counts(restore_target / "event_store.db") == {"conversation_turn_events": 1}
        assert target_graph.writes == []
        assert not list(backup_root.glob("pre-restore-*"))

    def test_it_refuses_a_token_that_is_not_the_resolved_target_path(
        self, artifact, restore_target, target_graph, backup_root, monkeypatch
    ):
        # The operator passes a relative path and confirms with the same
        # relative string. Equal to each other, not equal to where it lands.
        monkeypatch.chdir(restore_target.parent)
        with pytest.raises(RestoreConfirmationError):
            restore(
                artifact,
                "./dev-state",
                target_graph,
                backup_root,
                confirm_token="./dev-state",
            )
        assert target_graph.writes == []
        assert not list(backup_root.glob("pre-restore-*"))

    def test_it_refuses_a_target_that_lacks_the_handshake_marker(
        self, artifact, restore_target, target_graph, backup_root
    ):
        (restore_target / RESTORE_MARKER_FILENAME).unlink()
        with pytest.raises(RestoreTargetError) as refusal:
            restore(artifact, restore_target, target_graph, backup_root)
        assert RESTORE_MARKER_FILENAME in str(refusal.value)
        assert row_counts(restore_target / "event_store.db") == {"conversation_turn_events": 1}
        assert target_graph.writes == []

    def test_it_refuses_live_state_as_a_target(self, artifact, target_graph, backup_root):
        with pytest.raises(RestoreTargetError):
            restore(artifact, REPO_ROOT / "data", target_graph, backup_root)
        assert target_graph.writes == []

    def test_it_refuses_a_graph_uri_that_is_not_a_dev_endpoint(
        self, artifact, restore_target, target_graph, backup_root
    ):
        with pytest.raises(EvalIsolationError):
            restore(
                artifact,
                restore_target,
                target_graph,
                backup_root,
                target_graph_uri="bolt://mist-neo4j:7687",
            )
        assert target_graph.writes == []
        assert not list(backup_root.glob("pre-restore-*"))


class TestArtifactIntegrity:
    def test_a_directory_with_no_manifest_is_not_restorable(
        self, restore_target, target_graph, backup_root, tmp_path
    ):
        empty = tmp_path / "not-a-backup"
        empty.mkdir()
        with pytest.raises(BackupError):
            restore(empty, restore_target, target_graph, backup_root)
        assert target_graph.writes == []

    def test_a_corrupted_file_refuses_before_the_target_is_touched(
        self, artifact, restore_target, target_graph, backup_root
    ):
        store = artifact / STORES_DIRNAME / "event_store.db"
        store.write_bytes(store.read_bytes() + b"tampered")
        with pytest.raises(BackupError) as refusal:
            restore(artifact, restore_target, target_graph, backup_root)
        assert "sha256" in str(refusal.value)
        assert row_counts(restore_target / "event_store.db") == {"conversation_turn_events": 1}
        assert target_graph.writes == []
        assert not list(backup_root.glob("pre-restore-*"))

    def test_a_missing_file_refuses_before_the_target_is_touched(
        self, artifact, restore_target, target_graph, backup_root
    ):
        (artifact / "graph.json").unlink()
        with pytest.raises(BackupError):
            restore(artifact, restore_target, target_graph, backup_root)
        assert target_graph.writes == []

    def test_a_manifest_naming_an_unreadable_layout_is_refused(
        self, artifact, restore_target, target_graph, backup_root
    ):
        (artifact / MANIFEST_FILENAME).write_text('{"layout": "other", "layout_version": 1}')
        with pytest.raises(BackupError):
            restore(artifact, restore_target, target_graph, backup_root)


class TestAnUndecodableArtifactIsRefusedBeforeAnythingIsOverwritten:
    """The regression suite for the ordering defect this module shipped with.

    `load_artifact` used to run LAST, after the stores and the vault had already
    been replaced. So an artifact with perfectly valid digests and a
    `format_version` this build does not read produced a HALF-RESTORED target
    and a raw traceback. Digest validity and decodability are different
    properties; these tests assert the target is byte-for-byte unchanged, which
    is the only assertion that would have failed before the fix -- `raises` alone
    passed both before and after.
    """

    def test_an_unreadable_format_version_leaves_the_target_byte_for_byte_unchanged(
        self, artifact, restore_target, target_graph, backup_root
    ):
        rewrite_graph_leg(artifact, lambda payload: payload.update({"format_version": 2}))
        before = fingerprint(restore_target)

        with pytest.raises(RestorePreflightError) as refusal:
            restore(artifact, restore_target, target_graph, backup_root)

        assert fingerprint(restore_target) == before
        assert target_graph.writes == []
        assert "format_version 2" in str(refusal.value)
        assert "Nothing has been written to the target" in str(refusal.value)

    def test_it_refuses_before_the_pre_restore_backup_is_even_taken(
        self, artifact, restore_target, target_graph, backup_root
    ):
        rewrite_graph_leg(artifact, lambda payload: payload.update({"format_version": 2}))
        with pytest.raises(RestorePreflightError):
            restore(artifact, restore_target, target_graph, backup_root)
        # No capture was needed, because nothing was ever going to be destroyed.
        assert not list(backup_root.glob("pre-restore-*"))

    def test_a_relationship_endpoint_no_node_provides_is_refused_early(
        self, artifact, restore_target, target_graph, backup_root
    ):
        def orphan(payload):
            payload["relationships"][0]["target"] = "no-such-node"

        rewrite_graph_leg(artifact, orphan)
        before = fingerprint(restore_target)
        with pytest.raises(RestorePreflightError):
            restore(artifact, restore_target, target_graph, backup_root)
        assert fingerprint(restore_target) == before
        assert target_graph.writes == []

    def test_a_graph_leg_with_no_envelope_is_refused_early(
        self, artifact, restore_target, target_graph, backup_root
    ):
        rewrite_graph_leg(artifact, lambda payload: payload.pop("format_version"))
        before = fingerprint(restore_target)
        with pytest.raises(RestorePreflightError):
            restore(artifact, restore_target, target_graph, backup_root)
        assert fingerprint(restore_target) == before

    def test_a_graph_leg_that_is_not_json_is_refused_early(
        self, artifact, restore_target, target_graph, backup_root
    ):
        path = artifact / "graph.json"
        path.write_text("{not json", encoding="utf-8")
        manifest_path = artifact / MANIFEST_FILENAME
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["files"]["graph.json"] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        before = fingerprint(restore_target)
        with pytest.raises(RestorePreflightError):
            restore(artifact, restore_target, target_graph, backup_root)
        assert fingerprint(restore_target) == before

    def test_the_refusal_is_a_backup_error_so_callers_keep_one_type_to_catch(self):
        # `GraphArtifactError` is a RuntimeError, not a MistError
        # (`grep -n "class GraphArtifactError" backend/knowledge/graph_artifact.py`
        # -> :95), so it escaped every `except MistError` arm in this package as
        # a traceback. Translating it is what closes that hole.
        assert issubclass(RestorePreflightError, BackupError)
        assert not issubclass(GraphArtifactError, BackupError)


class TestAbsentLegs:
    def test_an_artifact_with_no_vault_leaves_the_targets_vault_alone(
        self, backup_root, source_graph, state_root, restore_target, target_graph, tmp_path
    ):
        # `mist-memory/` is absent from every fresh clone -- gitignored with
        # zero tracked files -- so artifacts with no vault leg are ordinary.
        # Replacing a tree with nothing is not a restore.
        report = capture(backup_root, source_graph, state_root, tmp_path / "no-vault")
        restore(report.artifact_dir, restore_target, target_graph, backup_root)
        assert (restore_target / "vault" / "target-only.md").is_file()

    def test_stage_vault_returns_none_when_the_artifact_has_no_vault_directory(
        self, tmp_path, restore_target
    ):
        assert stage_vault(tmp_path / "empty-artifact", restore_target / "vault") is None
        assert not list(restore_target.glob(f"*{STAGING_SUFFIX}"))

    def test_a_store_absent_at_capture_time_is_reported_not_silently_skipped(
        self, backup_root, source_graph, state_root, vault_root, restore_target, target_graph
    ):
        (state_root / "extraction_cache.db").unlink()
        report = capture(backup_root, source_graph, state_root, vault_root)
        restored = restore(report.artifact_dir, restore_target, target_graph, backup_root)
        assert restored.stores_absent_from_artifact == ("extraction_cache.db",)
        assert "extraction_cache.db" not in restored.stores_restored


class TestTheCli:
    def test_target_root_has_no_default(self):
        with pytest.raises(SystemExit):
            main(["--artifact", "x", "--target-graph-uri", TARGET_URI])

    def test_a_missing_token_is_a_refusal_exit_code_not_a_traceback(
        self, artifact, restore_target, backup_root, capsys, monkeypatch
    ):
        monkeypatch.setenv(BACKUP_ROOT_ENV, str(backup_root))
        code = main(
            [
                "--artifact",
                str(artifact),
                "--target-root",
                str(restore_target),
                "--target-graph-uri",
                "bolt://mist-neo4j:7687",
            ]
        )
        # The live URI is refused before a socket is opened, so this test needs
        # no Neo4j -- which the container could not provide anyway.
        assert code == EXIT_REFUSED
        assert "REFUSED" in capsys.readouterr().err
