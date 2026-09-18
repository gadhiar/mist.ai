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
import os
import shutil
import sqlite3
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.backup.restore as restore_module
from backend.errors import Neo4jQueryError
from backend.knowledge.eval_isolation import REPO_ROOT, EvalIsolationError
from backend.knowledge.graph_artifact import GraphArtifactError
from scripts.backup.destination import BACKUP_ROOT_ENV
from scripts.backup.dump import run_dump
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
    build_parser,
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


def live_fingerprint(root):
    """`fingerprint` over the target's OWN state only, for failures after phase 2.

    Three exclusions, each for a reason a plain `fingerprint` comparison would
    otherwise fail on for the wrong cause:

    - staged copies (`*.incoming`), because phase 3 creating them is the correct
      behaviour and they are not live state;
    - `restore.in-progress.json` and its `.tmp`, for the same reason;
    - `-wal`/`-shm` sidecars, because the PRE-RESTORE CAPTURE opens the target's
      stores and opening a WAL database creates them
      (`tests/unit/backup/test_dump.py:104-118` asserts exactly that). That is
      already the documented meaning of "the target was not overwritten".

    What remains is every byte a restore is supposed to replace. Comparing it
    across a failure is the assertion that caught the MIS-153 ordering bug when
    `pytest.raises` alone did not.
    """
    return {
        relative: payload
        for relative, payload in fingerprint(root).items()
        if STAGING_SUFFIX not in relative
        and not relative.startswith(RESTORE_PROGRESS_FILENAME)
        and not relative.endswith(("-wal", "-shm"))
    }


def read_marker(root):
    """The parsed `restore.in-progress.json` sitting in `root`."""
    return json.loads(restore_progress_marker_path(root).read_text(encoding="utf-8"))


def drop_store_from_artifact(artifact_dir, filename):
    """Delete one store file and its digest, while the manifest still calls it present.

    A genuine, non-mocked way to fail PHASE 3 at the SECOND store: the digest
    pass (`verify_artifact_files`) no longer names the file, so it passes, and
    staging refuses when it reaches a store the `stores` map records as present
    and the artifact does not hold.
    """
    (artifact_dir / STORES_DIRNAME / filename).unlink()
    manifest_path = artifact_dir / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"].pop(f"{STORES_DIRNAME}/{filename}")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return artifact_dir


def fail_rename_onto(monkeypatch, filename):
    """Make `os.replace` fail for one destination NAME and behave normally for the rest.

    Phase 5 is three atomic renames in a row and there is no cross-platform way
    to make the second one fail for real without root-dependent permission
    tricks, so this one injection point is a monkeypatch. It is scoped to a
    single destination name rather than to `os.replace` wholesale, because the
    progress marker is rewritten through `os.replace` too and breaking that
    would fail the test for the wrong reason.
    """
    real_replace = os.replace

    def fake_replace(source, destination):
        if Path(destination).name == filename:
            raise OSError(5, "synthetic rename failure")
        return real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fake_replace)


def failing_graph_write(_connection, _artifact):
    """Phase 4 dying the way the MIS-140 rehearsal's did."""
    raise GraphArtifactError("synthetic graph load failure")


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


class TestAFailureInEachPhaseLeavesExactlyTheDocumentedState:
    """The regression suite for the MIS-140 rehearsal, phase by phase.

    The rehearsal failed in the graph leg and left the target with all three
    stores and the vault replaced and the graph at 0 nodes, because the stage and
    the commit sat in one loop and the graph ran last. Under stage-then-swap each
    phase has a documented failure state, and these tests pin all three of them.
    """

    def test_a_phase_3_staging_failure_leaves_the_target_completely_untouched(
        self, artifact, restore_target, target_graph, backup_root
    ):
        # Fails at the SECOND store, so the first has already been staged. If
        # staging committed as it went -- the old behaviour -- event_store.db
        # would now hold the artifact's three rows.
        drop_store_from_artifact(artifact, "extraction_cache.db")
        before = live_fingerprint(restore_target)

        with pytest.raises(BackupError) as failure:
            restore(artifact, restore_target, target_graph, backup_root)

        assert live_fingerprint(restore_target) == before
        assert row_counts(restore_target / "event_store.db") == {"conversation_turn_events": 1}
        assert (restore_target / "vault" / "target-only.md").is_file()
        assert target_graph.node_by_id("stale")["properties"]["name"] == "stale"
        assert "Nothing in the target has been replaced" in str(failure.value)

    def test_a_phase_3_failure_leaves_no_orphaned_staged_copies(
        self, artifact, restore_target, target_graph, backup_root
    ):
        # Staging is consequence-free by design, and megabytes of orphaned
        # `.incoming` left in the target is the one way it could cost something.
        drop_store_from_artifact(artifact, "extraction_cache.db")
        with pytest.raises(BackupError):
            restore(artifact, restore_target, target_graph, backup_root)
        assert not list(restore_target.glob(f"*{STAGING_SUFFIX}"))

    def test_a_phase_4_graph_failure_costs_no_store_and_no_vault_file(
        self, artifact, restore_target, target_graph, backup_root
    ):
        # THE HEADLINE FIX. This is the rehearsal's exact failure, and it must
        # now leave the stores and the vault byte-for-byte the target's own.
        before = live_fingerprint(restore_target)

        with pytest.raises(GraphArtifactError):
            restore(
                artifact,
                restore_target,
                target_graph,
                backup_root,
                graph_writer=failing_graph_write,
            )

        assert live_fingerprint(restore_target) == before
        assert row_counts(restore_target / "event_store.db") == {"conversation_turn_events": 1}
        assert (restore_target / "vault" / "target-only.md").is_file()

    def test_a_phase_4_failure_leaves_the_staged_copies_in_place_and_says_so(
        self, artifact, restore_target, target_graph, backup_root
    ):
        with pytest.raises(GraphArtifactError):
            restore(
                artifact,
                restore_target,
                target_graph,
                backup_root,
                graph_writer=failing_graph_write,
            )
        assert (restore_target / f"event_store.db{STAGING_SUFFIX}").is_file()
        assert (restore_target / f"vault{STAGING_SUFFIX}" / "identity" / "mist.md").is_file()
        marker = read_marker(restore_target)
        assert marker["phases"]["staged"] is True
        assert marker["phases"]["graph"]["committed"] is False
        assert marker["phases"]["stores_committed"] == []
        assert marker["phases"]["vault_committed"] is False

    def test_a_phase_5_failure_commits_the_stores_before_it_and_no_others(
        self, artifact, restore_target, target_graph, backup_root, monkeypatch
    ):
        # `os.replace` is atomic PER STORE: three stores are three atomic
        # operations, not one, so the documented state after a failure at store
        # two is store one new and stores two and three old.
        fail_rename_onto(monkeypatch, "extraction_cache.db")

        with pytest.raises(BackupError) as failure:
            restore(artifact, restore_target, target_graph, backup_root)

        assert row_counts(restore_target / "event_store.db") == {"conversation_turn_events": 3}
        assert row_counts(restore_target / "vault_sidecar.db") == {"vault_chunks": 1}
        assert "['event_store.db']" in str(failure.value)

    def test_a_phase_5_failure_leaves_the_vault_untouched_and_the_marker_accurate(
        self, artifact, restore_target, target_graph, backup_root, monkeypatch
    ):
        fail_rename_onto(monkeypatch, "extraction_cache.db")
        with pytest.raises(BackupError):
            restore(artifact, restore_target, target_graph, backup_root)

        # The vault swap runs after every store, so it never started.
        assert (restore_target / "vault" / "target-only.md").is_file()
        assert not (restore_target / f"vault{VAULT_PREVIOUS_SUFFIX}").exists()
        marker = read_marker(restore_target)
        assert marker["phases"]["stores_committed"] == ["event_store.db"]
        assert marker["phases"]["graph"]["committed"] is True
        assert marker["phases"]["vault_committed"] is False
        assert marker["phases"]["vault_previous"] is None


class TestTheRestoreMarker:
    """Today a graph failure leaves nothing on disk saying so. The marker is that record."""

    def test_it_is_deleted_on_success(self, artifact, restore_target, target_graph, backup_root):
        restore(artifact, restore_target, target_graph, backup_root)
        assert not restore_progress_marker_path(restore_target).exists()

    def test_it_names_the_artifact_the_target_and_the_way_back(
        self, artifact, restore_target, target_graph, backup_root
    ):
        with pytest.raises(GraphArtifactError):
            restore(
                artifact,
                restore_target,
                target_graph,
                backup_root,
                graph_writer=failing_graph_write,
            )
        marker = read_marker(restore_target)
        assert marker["marker_version"] == 1
        assert marker["artifact_dir"] == str(artifact)
        assert marker["target_root"] == str(Path(restore_target).resolve())
        assert marker["artifact_label"] == "source-artifact"
        # The way back, and the first thing an operator reading this needs.
        assert Path(marker["pre_restore_artifact"]).is_dir()
        assert marker["started_utc"].endswith("Z")

    def test_it_records_the_graph_counts_once_phase_4_has_committed(
        self, artifact, restore_target, target_graph, backup_root, monkeypatch
    ):
        fail_rename_onto(monkeypatch, "event_store.db")
        with pytest.raises(BackupError):
            restore(artifact, restore_target, target_graph, backup_root)
        marker = read_marker(restore_target)
        assert marker["phases"]["graph"] == {
            "committed": True,
            "nodes": 3,
            "relationships": 2,
        }

    def test_its_own_rewrite_goes_through_a_tmp_file_that_never_survives(
        self, artifact, restore_target, target_graph, backup_root
    ):
        # A crash during a rewrite must leave the PREVIOUS marker intact rather
        # than a truncated one, which is what the `.tmp` plus `os.replace` buys.
        restore(artifact, restore_target, target_graph, backup_root)
        assert not (restore_target / f"{RESTORE_PROGRESS_FILENAME}.tmp").exists()


class TestAStaleMarkerRefuses:
    """A restore does NOT proceed over a half-applied one, and no flag makes it."""

    def test_it_refuses_at_exit_2_with_the_target_untouched(
        self, artifact, restore_target, target_graph, backup_root
    ):
        restore_progress_marker_path(restore_target).write_text(
            json.dumps(
                {
                    "marker_version": 1,
                    "pre_restore_artifact": "/offsite/pre-restore-20260917T030000Z",
                    "phases": {"staged": True, "graph": {"committed": False}},
                }
            ),
            encoding="utf-8",
        )
        before = fingerprint(restore_target)

        with pytest.raises(RestoreTargetStateError) as refusal:
            restore(artifact, restore_target, target_graph, backup_root)

        # Bit-for-bit, not merely "not overwritten": this refusal is raised
        # before the pre-restore capture, so not even a `-wal` sidecar appears.
        assert fingerprint(restore_target) == before
        assert target_graph.writes == []
        assert not list(backup_root.glob("pre-restore-*"))
        # The way back is printed, because it is the next thing the operator needs.
        assert "/offsite/pre-restore-20260917T030000Z" in str(refusal.value)
        assert "delete" in str(refusal.value).lower()

    def test_the_refusal_is_already_exit_2_without_the_tuple_in_main_being_edited(self):
        # `RestoreTargetStateError` subclasses `RestorePreflightError`, which
        # `main` already catches, so this refusal reached exit 2 without any
        # guard being widened.
        assert issubclass(RestoreTargetStateError, RestorePreflightError)
        assert issubclass(RestoreTargetStateError, BackupError)

    def test_there_is_no_override_flag_on_the_cli(self):
        # A `--force`-shaped flag would be a new bypass in a tool whose whole
        # design is that it cannot be run unattended. A manual delete cannot be
        # scripted into a cron job by accident.
        help_text = build_parser().format_help()
        for bypass in ("--force", "--ignore-marker", "--resume", "--no-marker"):
            assert bypass not in help_text

    def test_deleting_the_marker_by_hand_is_what_unblocks_the_re_run(
        self, artifact, restore_target, target_graph, backup_root
    ):
        restore_progress_marker_path(restore_target).write_text("{}", encoding="utf-8")
        with pytest.raises(RestoreTargetStateError):
            restore(artifact, restore_target, target_graph, backup_root)
        restore_progress_marker_path(restore_target).unlink()
        report = restore(artifact, restore_target, target_graph, backup_root)
        assert report.graph_nodes == 3

    def test_a_marker_this_build_cannot_parse_is_still_printed_in_full(
        self, artifact, restore_target, target_graph, backup_root
    ):
        restore_progress_marker_path(restore_target).write_text(
            "marker_version: 99 (not json at all)", encoding="utf-8"
        )
        with pytest.raises(RestoreTargetStateError) as refusal:
            restore(artifact, restore_target, target_graph, backup_root)
        assert "marker_version: 99 (not json at all)" in str(refusal.value)


class TestTheTargetPreflight:
    """Four read-only checks, all before the pre-restore backup is spent."""

    def test_a_target_graph_that_does_not_answer_refuses_before_any_dump(
        self, artifact, restore_target, backup_root
    ):
        class SilentGraph(InMemoryGraphConnection):
            def execute_query(self, query, params=None):
                if "SHOW " in query:
                    raise Neo4jQueryError("Query execution failed: connection reset")
                return super().execute_query(query, params)

        before = fingerprint(restore_target)
        with pytest.raises(RestoreTargetStateError) as refusal:
            restore(artifact, restore_target, SilentGraph(), backup_root)
        assert fingerprint(restore_target) == before
        assert not list(backup_root.glob("pre-restore-*"))
        assert "did not answer its schema reads" in str(refusal.value)

    def test_an_artifact_ddl_statement_whose_name_does_not_parse_is_refused(
        self, artifact, restore_target, target_graph, backup_root
    ):
        # An unparseable name is invisible to every name-keyed schema decision,
        # so it silently becomes "always execute" -- the rehearsal's direct cause.
        def unnameable(payload):
            payload["schema"]["constraints"] = ["ALTER CONSTRAINT whatever"]

        rewrite_graph_leg(artifact, unnameable)
        before = fingerprint(restore_target)
        with pytest.raises(RestoreTargetStateError) as refusal:
            restore(artifact, restore_target, target_graph, backup_root)
        assert fingerprint(restore_target) == before
        assert target_graph.writes == []
        assert "ALTER CONSTRAINT whatever" in str(refusal.value)

    def test_too_little_free_space_refuses_before_the_target_is_touched(
        self, artifact, restore_target, target_graph, backup_root, monkeypatch
    ):
        # One byte short of what the manifest says staging needs. The container's
        # own volume has plenty, so the shortfall has to be injected; the size
        # itself is the real, manifest-derived figure.
        required = staged_peak_bytes(read_manifest(artifact))
        usage = SimpleNamespace(total=required, used=required, free=required - 1)
        monkeypatch.setattr(restore_module.shutil, "disk_usage", lambda _path: usage)

        before = fingerprint(restore_target)
        with pytest.raises(RestoreTargetStateError) as refusal:
            restore(artifact, restore_target, target_graph, backup_root)
        assert fingerprint(restore_target) == before
        assert not list(backup_root.glob("pre-restore-*"))
        assert "staging needs about" in str(refusal.value)

    def test_the_space_requirement_is_sized_from_the_manifest_not_a_constant(self, artifact):
        manifest = read_manifest(artifact)
        stores_and_vault = sum(
            entry["bytes"]
            for relative, entry in manifest.files.items()
            if relative.startswith((f"{STORES_DIRNAME}/", "vault/"))
        )
        # Twice the artifact's own stores-plus-vault, and the graph leg -- which
        # is written into Neo4j, not into the target root -- is excluded.
        assert staged_peak_bytes(manifest) == stores_and_vault * 2
        assert manifest.files["graph.json"]["bytes"] > 0

    def test_the_real_free_space_check_passes_on_a_normal_target(
        self, artifact, restore_target, target_graph, backup_root
    ):
        # No monkeypatch: the container's own volume is measured, so the check
        # is exercised for real rather than only in its refusing direction.
        assert restore(artifact, restore_target, target_graph, backup_root).graph_nodes == 3


class TestDdlObjectNameParsing:
    def test_it_reads_the_name_at_whichever_token_neo4j_put_it(self):
        assert parse_ddl_object_name("CREATE CONSTRAINT `c1` FOR (n:X) REQUIRE n.id IS UNIQUE") == (
            "c1"
        )
        # Token 3, not token 2: reading token 2 unconditionally returns the
        # literal "INDEX" for every index.
        assert parse_ddl_object_name("CREATE RANGE INDEX `i1` FOR (n:X) ON (n.id)") == "i1"
        assert parse_ddl_object_name("CREATE VECTOR INDEX `i2` FOR (n:X) ON (n.embedding)") == "i2"

    def test_an_object_actually_named_index_is_not_read_as_the_keyword(self):
        assert parse_ddl_object_name("CREATE CONSTRAINT `index` FOR (n:X) REQUIRE n.id") == "index"

    def test_a_statement_it_cannot_name_returns_none_rather_than_a_guess(self):
        assert parse_ddl_object_name("DROP CONSTRAINT c1") is None
        assert parse_ddl_object_name("") is None
        assert parse_ddl_object_name("CREATE CONSTRAINT") is None


class TestARestoreOverAReadOnlyVault:
    """MIS-157: the live vault is a git repo, and git marks loose objects read-only."""

    def test_a_vault_holding_a_read_only_entry_restores(
        self, artifact, restore_target, target_graph, backup_root
    ):
        # Shaped like a git loose object: `-r--r--r--`, inside `.git/objects`.
        objects = restore_target / "vault" / ".git" / "objects" / "ab"
        objects.mkdir(parents=True)
        loose = objects / "cdef0123456789"
        loose.write_bytes(b"loose object")
        loose.chmod(stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)

        report = restore(artifact, restore_target, target_graph, backup_root)

        assert report.graph_nodes == 3
        assert (restore_target / "vault" / "identity" / "mist.md").is_file()
        assert not (restore_target / f"vault{VAULT_PREVIOUS_SUFFIX}").exists()

    def test_remove_tree_clears_a_permission_that_really_does_block_the_unlink(self, tmp_path):
        """The one genuine non-mocked reproduction this tier can offer, and its limit.

        The MIS-157 defect is Windows-specific: there a read-only FILE raises
        `WinError 5`. On Linux a read-only file is removable, so that case cannot
        be reproduced here at all. A read-only containing DIRECTORY can be, and
        it raises a real `PermissionError` through the same `onerror=` handler.

        So this proves the handler is wired in and that the retry succeeds once
        the attribute is cleared. It does NOT prove the Windows read-only-file
        behaviour; only the host rehearsal does.
        """
        tree = tmp_path / "vault.previous"
        locked = tree / "objects"
        locked.mkdir(parents=True)
        (locked / "loose").write_bytes(b"x")
        locked.chmod(0o555)

        # Without the handler this is the failure: plain rmtree cannot unlink an
        # entry out of a directory that denies write permission.
        with pytest.raises(PermissionError):
            shutil.rmtree(tree)

        remove_tree(tree)
        assert not tree.exists()


class TestSidecarsAfterTheCommit:
    def test_no_wal_or_shm_sits_beside_any_restored_store(
        self, artifact, restore_target, target_graph, backup_root
    ):
        # The artifact's copies carry no sidecars, because the capture leg
        # checkpoints and strips them, so the correct post-restore state is a
        # `.db` with nothing beside it.
        report = restore(artifact, restore_target, target_graph, backup_root)
        for filename in report.stores_restored:
            for suffix in ("-wal", "-shm"):
                assert not (restore_target / f"{filename}{suffix}").exists()

    def test_a_stale_sidecar_survives_a_phase_4_failure_because_its_store_did(
        self, artifact, restore_target, target_graph, backup_root
    ):
        # The sidecar clear belongs to the COMMIT phase and not to staging:
        # deleting the live `-wal` while staging would discard committed frames
        # from the OLD database, which is damage before this run has committed to
        # replacing anything.
        (restore_target / "event_store.db-wal").write_bytes(b"old frames")

        with pytest.raises(GraphArtifactError):
            restore(
                artifact,
                restore_target,
                target_graph,
                backup_root,
                graph_writer=failing_graph_write,
            )

        assert (restore_target / "event_store.db-wal").read_bytes() == b"old frames"


class TestATargetThatAlreadyCarriesSchema:
    def test_the_targets_own_constraints_and_indexes_are_replaced_by_the_artifacts(
        self, artifact, restore_target, target_graph_with_schema, backup_root
    ):
        # `rt_entity_id` is the name the rehearsal target actually carried over
        # the same schema as the artifact's `c1` under a different name.
        report = restore(artifact, restore_target, target_graph_with_schema, backup_root)
        assert report.graph_nodes == 3
        assert set(target_graph_with_schema.constraints) == {"c1"}
        assert set(target_graph_with_schema.indexes) == {"i1"}

    def test_the_drop_statements_are_actually_issued(
        self, artifact, restore_target, target_graph_with_schema, backup_root
    ):
        restore(artifact, restore_target, target_graph_with_schema, backup_root)
        written = [query for query, _params in target_graph_with_schema.writes]
        assert any(query.startswith("DROP CONSTRAINT") for query in written)
        assert any(query.startswith("DROP INDEX") for query in written)


class TestTheVaultFileCounts:
    def test_git_plumbing_is_counted_separately_from_the_corpus(
        self, backup_root, source_graph, state_root, vault_root, restore_target, target_graph
    ):
        # The rehearsal reported "117 vault files" as a corpus measure when it
        # was 13 notes and 104 git objects. `.git` is captured deliberately --
        # the live vault has no remote and its commits exist nowhere else -- so
        # the fix is two numbers, not a narrower capture.
        objects = vault_root / ".git" / "objects" / "ab"
        objects.mkdir(parents=True)
        for index in range(3):
            (objects / f"obj{index}").write_bytes(b"loose")

        captured = capture(backup_root, source_graph, state_root, vault_root, label="with-git")
        report = restore(captured.artifact_dir, restore_target, target_graph, backup_root)

        assert report.vault_corpus_files == 2
        assert report.vault_files == 5

    def test_a_vault_with_no_git_makes_the_two_counts_equal(
        self, artifact, restore_target, target_graph, backup_root
    ):
        report = restore(artifact, restore_target, target_graph, backup_root)
        assert report.vault_corpus_files == report.vault_files == 2


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
