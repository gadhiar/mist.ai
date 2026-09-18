"""MIS-140 T1: a full synthetic capture, and the refusals that precede one.

Everything here is built in the worktree: three WAL-mode SQLite stores, a vault
tree standing in for `mist-memory/` (absent from every worktree, because it is
gitignored with zero tracked files), and a graph fake. The container has no
network and reaches neither the live stack nor Neo4j.

`TestVaultCounts` is MIS-156 and covers the two vault numbers a capture now
reports: the corpus count, and the total that includes the `.git` plumbing the
dump captures on purpose.
"""

from __future__ import annotations

import json

import pytest

from backend.knowledge.eval_isolation import REPO_ROOT
from backend.knowledge.graph_artifact import load_artifact
from scripts.backup.destination import BACKUP_ROOT_ENV
from scripts.backup.dump import (
    EXIT_DESTINATION_REFUSED,
    GRAPH_FILENAME,
    VAULT_DIRNAME,
    _print_report,
    main,
    read_git_head,
    run_dump,
)
from scripts.backup.errors import BackupDestinationError, BackupError
from scripts.backup.manifest import (
    BACKUP_LAYOUT,
    BACKUP_LAYOUT_VERSION,
    MANIFEST_FILENAME,
    PARTIAL_SUFFIX,
    is_backup_artifact_dir,
    read_manifest,
    sha256_file,
)
from scripts.backup.stores import LIVE_STORE_FILENAMES, STORES_DIRNAME

from .conftest import STALE_BACKUP_FILENAMES, STAMPS


def dump(backup_root, connection, state_root, vault_root, **overrides):
    """Run one dump with the synthetic fixtures wired in."""
    kwargs = {
        "backup_root": backup_root,
        "connection": connection,
        "source_uri": "bolt://mist-neo4j:7687",
        "database": None,
        "stamps": dict(STAMPS),
        "state_root": state_root,
        "vault_root": vault_root,
        "label": "test-artifact",
    }
    kwargs.update(overrides)
    return run_dump(**kwargs)


class TestRoundTrip:
    def test_every_leg_lands_in_the_artifact(
        self, backup_root, graph_connection, state_root, vault_root
    ):
        report = dump(backup_root, graph_connection, state_root, vault_root)
        artifact = report.artifact_dir
        assert artifact == backup_root / "test-artifact"
        assert sorted(p.name for p in artifact.iterdir()) == [
            GRAPH_FILENAME,
            MANIFEST_FILENAME,
            STORES_DIRNAME,
            VAULT_DIRNAME,
        ]
        assert report.stores_present == len(LIVE_STORE_FILENAMES)
        assert report.stores_absent == ()
        assert report.graph_nodes == 2
        assert report.graph_relationships == 1
        assert report.vault_corpus_files == 2
        assert report.vault_files == 2

    def test_the_graph_leg_reads_back_through_the_shared_loader(
        self, backup_root, graph_connection, state_root, vault_root
    ):
        report = dump(backup_root, graph_connection, state_root, vault_root)
        payload = json.loads((report.artifact_dir / GRAPH_FILENAME).read_text(encoding="utf-8"))
        artifact = load_artifact(payload)
        assert artifact["counts"] == {"nodes": 2, "relationships": 1}
        # Embeddings travel as exact floats, not as strings.
        person = next(n for n in artifact["nodes"] if n["id"] == "person-raj")
        assert person["properties"]["embedding"] == [0.25, -0.5, 0.125]
        assert artifact["schema"]["constraints"] == ["CREATE CONSTRAINT c1"]

    def test_the_vault_tree_is_copied_whole(
        self, backup_root, graph_connection, state_root, vault_root
    ):
        report = dump(backup_root, graph_connection, state_root, vault_root)
        vault = report.artifact_dir / VAULT_DIRNAME
        assert (vault / "identity" / "mist.md").read_text(encoding="utf-8") == "# MIST\n"
        assert (vault / "users" / "raj.md").is_file()

    def test_live_store_bytes_are_untouched(
        self, backup_root, graph_connection, state_root, vault_root
    ):
        before = {p.name: p.read_bytes() for p in state_root.iterdir() if p.is_file()}
        dump(backup_root, graph_connection, state_root, vault_root)
        after = {p.name: p.read_bytes() for p in state_root.iterdir() if p.is_file()}
        assert {name: after[name] for name in before} == before

    def test_the_only_live_side_effect_is_wal_sidecars(
        self, backup_root, graph_connection, state_root, vault_root
    ):
        # Stated rather than asserted away. Opening a WAL database -- even
        # `mode=ro` -- makes SQLite create its `-shm` shared-memory file, so a
        # capture against a store the backend is not currently holding open
        # leaves empty sidecars behind. They are never deleted from LIVE state
        # by this tool: a `-wal` beside a running backend holds committed data
        # not yet checkpointed, and removing it would lose exactly the recent
        # turns this backup exists to keep.
        before = {p.name for p in state_root.iterdir()}
        dump(backup_root, graph_connection, state_root, vault_root)
        added = {p.name for p in state_root.iterdir()} - before
        assert all(name.endswith(("-wal", "-shm")) for name in added)
        assert before <= {p.name for p in state_root.iterdir()}

    def test_the_graph_leg_issues_no_writes(
        self, backup_root, graph_connection, state_root, vault_root
    ):
        dump(backup_root, graph_connection, state_root, vault_root)
        graph_connection.assert_no_writes()


class TestTheManifest:
    @pytest.fixture
    def manifest(self, backup_root, graph_connection, state_root, vault_root):
        report = dump(backup_root, graph_connection, state_root, vault_root)
        return read_manifest(report.artifact_dir)

    def test_it_identifies_the_directory_as_a_backup(self, backup_root, manifest):
        assert manifest.layout == BACKUP_LAYOUT
        assert manifest.layout_version == BACKUP_LAYOUT_VERSION
        assert is_backup_artifact_dir(backup_root / "test-artifact") is True

    def test_it_carries_an_utc_created_at_the_retention_leg_can_use(self, manifest):
        assert manifest.created_at.endswith("Z")
        assert manifest.created_at_datetime().utcoffset().total_seconds() == 0

    def test_it_records_per_file_digests_that_match_the_bytes(self, backup_root, manifest):
        artifact = backup_root / "test-artifact"
        assert MANIFEST_FILENAME not in manifest.files
        for relative, entry in manifest.files.items():
            path = artifact / relative
            assert entry["sha256"] == sha256_file(path)
            assert entry["bytes"] == path.stat().st_size
        assert f"{STORES_DIRNAME}/event_store.db" in manifest.files
        assert GRAPH_FILENAME in manifest.files

    def test_it_records_per_store_row_counts(self, manifest):
        assert manifest.stores["event_store.db"]["row_counts"] == {"conversation_turn_events": 3}
        assert manifest.stores["vault_sidecar.db"]["row_counts"] == {"vault_chunks": 5}

    def test_it_records_graph_and_vault_counts(self, manifest):
        assert manifest.graph["nodes"] == 2
        assert manifest.graph["relationships"] == 1
        assert manifest.graph["format_version"] == 1
        assert manifest.vault["corpus_file_count"] == 2
        assert manifest.vault["file_count"] == 2

    def test_it_records_the_stamps_without_gating_on_them(self, manifest):
        assert manifest.stamps == STAMPS

    def test_it_records_the_exclusion(self, manifest):
        assert manifest.excluded == ["vector_store"]

    def test_it_records_git_head_when_one_is_available(self, manifest):
        # Recorded, never enforced. None is a legitimate value when `git` is
        # absent, so this accepts either and only forbids a nonsense shape.
        assert manifest.git_head is None or len(manifest.git_head) == 40


class TestDestinationRefusals:
    def test_a_root_under_live_data_is_refused(self, graph_connection, state_root, vault_root):
        with pytest.raises(BackupDestinationError):
            dump(REPO_ROOT / "data" / "backups", graph_connection, state_root, vault_root)

    def test_a_root_under_the_repo_is_refused(self, graph_connection, state_root, vault_root):
        with pytest.raises(BackupDestinationError):
            dump(REPO_ROOT / "backups", graph_connection, state_root, vault_root)

    def test_a_refused_destination_writes_nothing_and_reads_nothing(
        self, graph_connection, state_root, vault_root
    ):
        with pytest.raises(BackupDestinationError):
            dump(REPO_ROOT / "backups", graph_connection, state_root, vault_root)
        assert not (REPO_ROOT / "backups").exists()
        assert graph_connection.queries == []

    def test_an_unset_env_refuses_before_the_graph_is_opened(self, monkeypatch, capsys):
        monkeypatch.delenv(BACKUP_ROOT_ENV, raising=False)
        assert main([]) == EXIT_DESTINATION_REFUSED
        captured = capsys.readouterr()
        assert "REFUSED" in captured.err
        assert BACKUP_ROOT_ENV in captured.err

    def test_the_cli_refuses_an_explicit_live_output(self, monkeypatch, capsys):
        monkeypatch.delenv(BACKUP_ROOT_ENV, raising=False)
        assert main(["--output", str(REPO_ROOT / "data")]) == EXIT_DESTINATION_REFUSED
        assert "REFUSED" in capsys.readouterr().err


class TestArtifactIntegrity:
    def test_stale_backup_files_are_absent_from_the_whole_artifact(
        self, backup_root, graph_connection, state_root, vault_root
    ):
        report = dump(backup_root, graph_connection, state_root, vault_root)
        names = {p.name for p in report.artifact_dir.rglob("*")}
        for stale in STALE_BACKUP_FILENAMES:
            assert stale not in names
        manifest = read_manifest(report.artifact_dir)
        assert set(manifest.stores) == set(LIVE_STORE_FILENAMES)

    def test_an_existing_artifact_is_never_overwritten(
        self, backup_root, graph_connection, state_root, vault_root
    ):
        dump(backup_root, graph_connection, state_root, vault_root)
        with pytest.raises(BackupError):
            dump(backup_root, graph_connection, state_root, vault_root)

    def test_a_leftover_partial_directory_blocks_a_reuse_of_its_label(
        self, backup_root, graph_connection, state_root, vault_root
    ):
        (backup_root / f"test-artifact{PARTIAL_SUFFIX}").mkdir()
        with pytest.raises(BackupError):
            dump(backup_root, graph_connection, state_root, vault_root)

    def test_a_failed_dump_leaves_no_manifest_bearing_directory(
        self, backup_root, graph_connection, vault_root, tmp_path
    ):
        broken = tmp_path / "broken-data"
        broken.mkdir()
        (broken / "event_store.db").write_text("not a database", encoding="utf-8")
        with pytest.raises(BackupError):
            dump(backup_root, graph_connection, broken, vault_root)
        assert (backup_root / f"test-artifact{PARTIAL_SUFFIX}").is_dir()
        assert not (backup_root / "test-artifact").exists()
        assert is_backup_artifact_dir(backup_root / f"test-artifact{PARTIAL_SUFFIX}") is False

    def test_the_default_label_is_the_utc_timestamp(
        self, backup_root, graph_connection, state_root, vault_root
    ):
        report = dump(backup_root, graph_connection, state_root, vault_root, label=None)
        manifest = read_manifest(report.artifact_dir)
        assert report.artifact_dir.name == manifest.label
        assert manifest.label.endswith("Z")


class TestAbsentSources:
    def test_an_absent_store_is_reported_rather_than_hidden(
        self, backup_root, graph_connection, state_root, vault_root
    ):
        (state_root / "extraction_cache.db").unlink()
        report = dump(backup_root, graph_connection, state_root, vault_root)
        assert report.stores_absent == ("extraction_cache.db",)
        manifest = read_manifest(report.artifact_dir)
        assert manifest.stores["extraction_cache.db"]["present"] is False

    def test_an_absent_vault_does_not_stop_the_dump(
        self, backup_root, graph_connection, state_root, tmp_path
    ):
        # `mist-memory/` is absent from every worktree, so a dump that refused
        # to run without it could never be exercised in this tier.
        report = dump(backup_root, graph_connection, state_root, tmp_path / "no-such-vault")
        manifest = read_manifest(report.artifact_dir)
        assert manifest.vault == {
            "directory": VAULT_DIRNAME,
            "source_present": False,
            "corpus_file_count": 0,
            "file_count": 0,
        }


class TestVaultCounts:
    """MIS-156: the two numbers a capture reports for the vault, and their gap.

    The rehearsal reported "117 vault files" as though that measured the corpus.
    On the host it was 13 notes and 104 git objects, so 89% of the headline
    number was plumbing and an operator could not tell from it whether their
    notes had come back. The fix is two counts, not a smaller one: `.git` stays
    captured because `mist-memory/` has no remote and its commits exist nowhere
    else.
    """

    # Three files under `.git/`, deliberately spread over a nested directory and
    # the repository root so the count cannot pass by matching only one depth.
    GIT_FILES = (".git/HEAD", ".git/config", ".git/objects/ab/cdef01")

    @pytest.fixture
    def vault_with_git(self, tmp_path):
        """A vault shaped like the live one: notes plus a `.git` directory.

        Built here rather than in `conftest.py` because the shared `vault_root`
        fixture is also the input to the restore tests (`test_restore.py`
        builds its artifact by calling the real `run_dump`), and giving every
        one of them a `.git` subtree would change what those tests measure.
        """
        root = tmp_path / "mist-memory-with-git"
        (root / "identity").mkdir(parents=True)
        (root / "identity" / "mist.md").write_text("# MIST\n", encoding="utf-8")
        (root / "users").mkdir()
        (root / "users" / "raj.md").write_text("# Raj\n", encoding="utf-8")
        for relative in self.GIT_FILES:
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"plumbing: {relative}\n", encoding="utf-8")
        return root

    def test_a_git_subtree_is_captured_whole_and_the_counts_differ_by_it(
        self, backup_root, graph_connection, state_root, vault_with_git
    ):
        report = dump(backup_root, graph_connection, state_root, vault_with_git)
        captured = report.artifact_dir / VAULT_DIRNAME

        # CAPTURED, not ignored. `mist-memory/` has no remote and no upstream,
        # so excluding `.git` would be the only copy of its history going
        # missing -- a new data-loss mode inside a data-loss fix.
        for relative in self.GIT_FILES:
            assert (captured / relative).is_file()
        assert (captured / "identity" / "mist.md").read_text(encoding="utf-8") == "# MIST\n"

        assert report.vault_corpus_files == 2
        assert report.vault_files == 2 + len(self.GIT_FILES)
        assert report.vault_files - report.vault_corpus_files == len(self.GIT_FILES)

    def test_a_vault_with_no_git_reports_the_two_counts_equal(
        self, backup_root, graph_connection, state_root, vault_root
    ):
        # The gap is the plumbing and nothing else: with no `.git` present the
        # corpus count must not be a different measure that happens to be lower.
        assert not (vault_root / ".git").exists()
        report = dump(backup_root, graph_connection, state_root, vault_root)
        assert report.vault_corpus_files == report.vault_files == 2

    def test_the_manifest_carries_both_counts(
        self, backup_root, graph_connection, state_root, vault_with_git
    ):
        # ADDED beside `file_count`, never redefining it: `BackupManifest`
        # carries no version field, so an old manifest and a new one would
        # disagree about what `file_count` means with nothing to tell them
        # apart. `file_count` therefore still counts every file.
        report = dump(backup_root, graph_connection, state_root, vault_with_git)
        manifest = read_manifest(report.artifact_dir)
        assert manifest.vault["corpus_file_count"] == 2
        assert manifest.vault["file_count"] == 5
        assert manifest.vault["source_present"] is True

    def test_the_dump_output_labels_both_counts(
        self, backup_root, graph_connection, state_root, vault_with_git, capsys
    ):
        report = dump(backup_root, graph_connection, state_root, vault_with_git)
        _print_report(report)
        printed = capsys.readouterr().out
        assert "2 vault corpus file(s)" in printed
        assert "5 including .git plumbing" in printed
        # Neither number may appear as a bare "N vault files", which is the
        # shape that produced the 117 figure.
        assert "5 vault files" not in printed
        assert "2 vault files" not in printed

    def test_a_vault_of_pure_plumbing_warns_that_no_notes_were_captured(
        self, backup_root, graph_connection, state_root, tmp_path, capsys
    ):
        # A state the single count could not express: files were captured, and
        # not one of them is a note. Under the old reporting this printed a
        # reassuring non-zero file count and no warning at all.
        root = tmp_path / "plumbing-only"
        (root / ".git").mkdir(parents=True)
        (root / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        report = dump(backup_root, graph_connection, state_root, root)
        assert report.vault_corpus_files == 0
        assert report.vault_files == 1
        _print_report(report)
        printed = capsys.readouterr().out
        assert "not one corpus note" in printed


class TestGitHead:
    def test_a_failing_git_call_yields_none_rather_than_failing_the_dump(self, tmp_path):
        # `git` is recorded for audit and must never be able to stop a backup.
        # A cwd that does not exist makes `subprocess.run` raise before exec,
        # which is the same arm an image without `git` takes.
        assert read_git_head(tmp_path / "does-not-exist") is None

    def test_a_real_checkout_yields_a_full_sha(self):
        head = read_git_head()
        assert head is None or (len(head) == 40 and all(c in "0123456789abcdef" for c in head))
