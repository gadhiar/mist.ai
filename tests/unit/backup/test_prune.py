"""MIS-140 T2: retention, and the three things it refuses to do.

The mandatory negative tests live here: a directory with no MIST.AI backup
manifest is never deleted, whatever its age or name, and `--retain 0` keeps one
artifact rather than none.
"""

from __future__ import annotations

import json

import pytest

from backend.knowledge.eval_isolation import REPO_ROOT
from scripts.backup.destination import BACKUP_ROOT_ENV
from scripts.backup.errors import BackupError
from scripts.backup.manifest import MANIFEST_FILENAME, PARTIAL_SUFFIX, BackupManifest
from scripts.backup.prune import (
    DEFAULT_RETAIN,
    EXIT_DESTINATION_REFUSED,
    EXIT_OK,
    MINIMUM_RETAIN,
    apply_prune,
    main,
    plan_prune,
    scan_artifacts,
)

from .conftest import STAMPS


def write_artifact(root, label, created_at, *, payload="payload"):
    """Create a directory that IS a MIST.AI backup artifact, dated by its manifest."""
    artifact = root / label
    artifact.mkdir(parents=True)
    (artifact / "graph.json").write_text(payload, encoding="utf-8")
    BackupManifest(
        layout="mist.backup",
        layout_version=1,
        created_at=created_at,
        label=label,
        git_head=None,
        stamps=dict(STAMPS),
        source={},
        files={},
        stores={},
        graph={},
        vault={},
        excluded=["vector_store"],
    ).write(artifact)
    return artifact


def names(candidates):
    """Directory names from a plan's keep/delete lists."""
    return [candidate.path.name for candidate in candidates]


@pytest.fixture
def populated_root(backup_root):
    """Ten artifacts whose manifest order is the REVERSE of their name order.

    Deliberately adversarial: sorting these by name, or by the mtime they were
    created in, keeps the wrong five. Only the manifest timestamp gives the
    intended answer, and `test_it_orders_by_the_manifest_timestamp` is what
    detects a regression to either.
    """
    for index in range(10):
        write_artifact(backup_root, f"artifact-{index:02d}", f"2026-09-{10 - index:02d}T03:00:00Z")
    return backup_root


class TestItNeverDeletesWhatItDoesNotRecognise:
    def test_a_directory_with_no_manifest_is_never_deleted(self, backup_root):
        # THE arm that stops a mis-pointed MIST_BACKUP_ROOT from eating
        # unrelated files. Pointed at a folder of someone's documents, prune
        # deletes nothing at all.
        for name in ("Taxes 2024", "photos", "thesis-final-v3"):
            folder = backup_root / name
            folder.mkdir()
            (folder / "important.txt").write_text("not a backup", encoding="utf-8")

        plan = plan_prune(backup_root, retain=1)
        assert plan.delete == ()
        assert sorted(p.name for p in plan.skipped) == ["Taxes 2024", "photos", "thesis-final-v3"]

        apply_prune(plan)
        assert sorted(p.name for p in backup_root.iterdir()) == [
            "Taxes 2024",
            "photos",
            "thesis-final-v3",
        ]
        assert (backup_root / "photos" / "important.txt").is_file()

    def test_a_foreign_manifest_json_does_not_make_a_directory_a_backup(self, backup_root):
        # `scripts/hydration/snapshot.py` writes a file of this name too
        # (`grep -n "MANIFEST_FILENAME =" scripts/hydration/manifest.py` -> :63),
        # so presence of the filename is not the test -- the layout string is.
        write_artifact(backup_root, "real", "2026-09-01T03:00:00Z")
        foreign = backup_root / "hydration-snapshot"
        foreign.mkdir()
        (foreign / MANIFEST_FILENAME).write_text(
            json.dumps({"layout": "mist.hydration.snapshot", "layout_version": 1}),
            encoding="utf-8",
        )
        plan = plan_prune(backup_root, retain=1)
        assert names(plan.delete) == []
        assert [p.name for p in plan.skipped] == ["hydration-snapshot"]

    def test_partial_directories_are_protected_because_they_carry_no_manifest(self, backup_root):
        # Protected for free by the manifest arm: a dump builds in
        # `<label>.partial` and renames only after the manifest is written, so a
        # partial has no manifest to be dated by.
        write_artifact(backup_root, "good", "2026-09-01T03:00:00Z")
        partial = backup_root / f"died-half-way{PARTIAL_SUFFIX}"
        partial.mkdir()
        (partial / "stores").mkdir()
        plan = plan_prune(backup_root, retain=1)
        apply_prune(plan)
        assert partial.is_dir()
        assert [p.name for p in plan.skipped] == [f"died-half-way{PARTIAL_SUFFIX}"]

    def test_an_unparseable_created_at_is_skipped_rather_than_ordered(self, backup_root):
        write_artifact(backup_root, "fine", "2026-09-05T03:00:00Z")
        broken = write_artifact(backup_root, "broken", "not-a-timestamp")
        candidates, skipped = scan_artifacts(backup_root)
        assert [c.path.name for c in candidates] == ["fine"]
        assert skipped == [broken]

    def test_loose_files_in_the_backup_root_are_ignored_entirely(self, backup_root):
        (backup_root / "notes.txt").write_text("hello", encoding="utf-8")
        write_artifact(backup_root, "good", "2026-09-01T03:00:00Z")
        plan = plan_prune(backup_root, retain=1)
        assert plan.skipped == ()
        apply_prune(plan)
        assert (backup_root / "notes.txt").is_file()


class TestItNeverPrunesBelowOne:
    def test_retain_zero_keeps_the_most_recent_artifact(self, populated_root):
        plan = plan_prune(populated_root, retain=0)
        assert plan.retain == MINIMUM_RETAIN
        assert plan.requested_retain == 0
        assert len(plan.keep) == 1
        assert names(plan.keep) == ["artifact-00"]
        apply_prune(plan)
        assert [p.name for p in populated_root.iterdir()] == ["artifact-00"]

    def test_a_negative_retain_is_clamped_too(self, populated_root):
        plan = plan_prune(populated_root, retain=-5)
        assert plan.retain == MINIMUM_RETAIN
        assert len(plan.keep) == 1

    def test_the_cli_says_out_loud_that_it_raised_the_number(
        self, populated_root, monkeypatch, capsys
    ):
        monkeypatch.setenv(BACKUP_ROOT_ENV, str(populated_root))
        assert main(["--retain", "0"]) == EXIT_OK
        out = capsys.readouterr().out
        assert "raised to 1" in out


class TestOrdering:
    def test_it_orders_by_the_manifest_timestamp_not_by_name_or_mtime(self, populated_root):
        # artifact-00 carries the NEWEST created_at and the alphabetically first
        # name, so keeping the newest three keeps 00, 01, 02.
        plan = plan_prune(populated_root, retain=3)
        assert names(plan.keep) == ["artifact-00", "artifact-01", "artifact-02"]
        assert len(plan.delete) == 7

    def test_touching_a_directory_does_not_change_what_is_kept(self, populated_root):
        before = names(plan_prune(populated_root, retain=3).keep)
        # An mtime bump is what a virus scanner, a copy to a new disk or an
        # rsync does. It must not promote an artifact.
        (populated_root / "artifact-09" / "touched.txt").write_text("x", encoding="utf-8")
        assert names(plan_prune(populated_root, retain=3).keep) == before

    def test_the_default_retains_seven(self, populated_root):
        plan = plan_prune(populated_root)
        assert plan.retain == DEFAULT_RETAIN
        assert len(plan.keep) == DEFAULT_RETAIN
        assert len(plan.delete) == 3

    def test_fewer_artifacts_than_retain_deletes_nothing(self, backup_root):
        write_artifact(backup_root, "only", "2026-09-01T03:00:00Z")
        plan = plan_prune(backup_root, retain=7)
        assert plan.delete == ()


class TestApplying:
    def test_planning_alone_deletes_nothing(self, populated_root):
        plan_prune(populated_root, retain=1)
        assert len(list(populated_root.iterdir())) == 10

    def test_applying_removes_exactly_the_planned_directories(self, populated_root):
        plan = plan_prune(populated_root, retain=2)
        removed = apply_prune(plan)
        assert sorted(p.name for p in removed) == [f"artifact-{i:02d}" for i in range(2, 10)]
        assert sorted(p.name for p in populated_root.iterdir()) == ["artifact-00", "artifact-01"]

    def test_a_directory_that_stopped_being_an_artifact_is_left_alone(self, populated_root):
        plan = plan_prune(populated_root, retain=2)
        # Between the plan and the application, something replaces a candidate.
        # The re-check refuses to delete on the strength of a stale observation.
        (populated_root / "artifact-09" / MANIFEST_FILENAME).unlink()
        removed = apply_prune(plan)
        assert (populated_root / "artifact-09").is_dir()
        assert populated_root / "artifact-09" not in removed


class TestTheCli:
    def test_it_prints_the_plan_and_deletes_nothing_without_confirm(
        self, populated_root, monkeypatch, capsys
    ):
        monkeypatch.setenv(BACKUP_ROOT_ENV, str(populated_root))
        assert main(["--retain", "2"]) == EXIT_OK
        assert len(list(populated_root.iterdir())) == 10
        out = capsys.readouterr().out
        assert "would delete 8" in out
        assert "Re-run with --confirm" in out

    def test_confirm_applies_the_plan(self, populated_root, monkeypatch, capsys):
        monkeypatch.setenv(BACKUP_ROOT_ENV, str(populated_root))
        assert main(["--retain", "2", "--confirm"]) == EXIT_OK
        assert sorted(p.name for p in populated_root.iterdir()) == ["artifact-00", "artifact-01"]
        assert "Deleted 8 artifact(s)" in capsys.readouterr().out

    def test_an_unset_backup_root_is_refused_rather_than_defaulted(self, monkeypatch, capsys):
        monkeypatch.delenv(BACKUP_ROOT_ENV, raising=False)
        assert main([]) == EXIT_DESTINATION_REFUSED
        assert BACKUP_ROOT_ENV in capsys.readouterr().err

    def test_a_root_inside_the_repository_is_refused(self, monkeypatch, capsys):
        monkeypatch.setenv(BACKUP_ROOT_ENV, str(REPO_ROOT / "data"))
        assert main(["--confirm"]) == EXIT_DESTINATION_REFUSED
        assert "REFUSED" in capsys.readouterr().err

    def test_a_root_of_unrecognised_directories_warns(self, backup_root, monkeypatch, capsys):
        (backup_root / "Documents").mkdir()
        monkeypatch.setenv(BACKUP_ROOT_ENV, str(backup_root))
        assert main(["--confirm"]) == EXIT_OK
        out = capsys.readouterr().out
        assert "no MIST.AI backup artifacts at all" in out
        assert (backup_root / "Documents").is_dir()


class TestAnAbsentRoot:
    def test_it_fails_rather_than_reporting_an_empty_plan(self, tmp_path):
        with pytest.raises(BackupError):
            plan_prune(tmp_path / "does-not-exist")
