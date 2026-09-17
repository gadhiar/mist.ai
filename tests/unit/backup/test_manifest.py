"""MIS-140 T1: the layout version is enforced and the producer stamps are not.

The distinction is the point of these tests. A backup that refuses to restore
because `EXTRACTION_VERSION` moved fails on the one day it is needed, so the
stamps are audit data. A backup whose FILE LAYOUT this build cannot parse is a
shape to refuse rather than to guess at.

The `is_backup_artifact_dir` cases exist for MIS-140 T2, which prunes by age and
must never delete a directory it cannot identify.
"""

from __future__ import annotations

import json

import pytest

from scripts.backup.errors import BackupManifestError
from scripts.backup.manifest import (
    BACKUP_LAYOUT,
    BACKUP_LAYOUT_VERSION,
    MANIFEST_FILENAME,
    BackupManifest,
    digest_artifact_files,
    is_backup_artifact_dir,
    read_manifest,
    sha256_file,
    utc_now_iso,
)


def make_manifest(**overrides) -> BackupManifest:
    """A minimal valid manifest, with fields replaceable per test."""
    fields = {
        "layout": BACKUP_LAYOUT,
        "layout_version": BACKUP_LAYOUT_VERSION,
        "created_at": "2026-09-17T04:05:06Z",
        "label": "20260917T040506Z",
        "git_head": "ebe1b0ddab5452ea52d1406b88bd5b1b809c2926",
        "stamps": {"ontology_version": "1.0.0", "extraction_version": "v7", "model_hash": "abc"},
        "source": {"graph_uri": "bolt://mist-neo4j:7687", "graph_database": None},
        "files": {},
        "stores": {},
        "graph": {"nodes": 0, "relationships": 0},
        "vault": {"file_count": 0},
        "excluded": ["vector_store"],
    }
    fields.update(overrides)
    return BackupManifest(**fields)


class TestLayoutVersionIsEnforced:
    def test_round_trip_through_disk(self, tmp_path):
        make_manifest().write(tmp_path)
        assert read_manifest(tmp_path).label == "20260917T040506Z"

    def test_an_unreadable_layout_version_is_refused(self, tmp_path):
        raw = make_manifest().to_dict()
        raw["layout_version"] = BACKUP_LAYOUT_VERSION + 1
        (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(BackupManifestError) as excinfo:
            read_manifest(tmp_path)
        assert "layout_version" in str(excinfo.value)

    def test_a_bool_layout_version_does_not_pass_as_one(self, tmp_path):
        # `isinstance(True, int)` is True and `True == 1`, so an untyped check
        # would accept `"layout_version": true` as version 1.
        raw = make_manifest().to_dict()
        raw["layout_version"] = True
        (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(raw), encoding="utf-8")
        with pytest.raises(BackupManifestError):
            read_manifest(tmp_path)

    def test_a_foreign_layout_is_refused(self, tmp_path):
        # A hydration artifact also writes `manifest.json`
        # (`grep -n "MANIFEST_FILENAME =" scripts/hydration/manifest.py` -> :63).
        (tmp_path / MANIFEST_FILENAME).write_text(
            json.dumps({"artifact_schema_version": 1, "created_at": "2026-09-17T00:00:00Z"}),
            encoding="utf-8",
        )
        with pytest.raises(BackupManifestError):
            read_manifest(tmp_path)

    def test_a_missing_manifest_is_refused(self, tmp_path):
        with pytest.raises(BackupManifestError) as excinfo:
            read_manifest(tmp_path)
        assert MANIFEST_FILENAME in str(excinfo.value)

    def test_malformed_json_is_refused(self, tmp_path):
        (tmp_path / MANIFEST_FILENAME).write_text("{not json", encoding="utf-8")
        with pytest.raises(BackupManifestError):
            read_manifest(tmp_path)


class TestStampsAreNeverEnforced:
    @pytest.mark.parametrize(
        "stamps",
        [
            pytest.param({}, id="no-stamps-at-all"),
            pytest.param({"extraction_version": "v1-from-2024"}, id="ancient-extraction"),
            pytest.param({"model_hash": "a-model-that-no-longer-exists"}, id="unknown-model"),
        ],
    )
    def test_drifted_stamps_still_read(self, tmp_path, stamps):
        # The failure this forbids: an artifact that self-invalidates on a
        # version bump is worse than no artifact, because it fails exactly when
        # it is reached for.
        make_manifest(stamps=stamps).write(tmp_path)
        assert read_manifest(tmp_path).stamps == stamps

    def test_stamps_are_preserved_verbatim_for_audit(self, tmp_path):
        stamps = {"ontology_version": "0.9.0", "extraction_version": "v2", "model_hash": "old"}
        make_manifest(stamps=stamps).write(tmp_path)
        assert read_manifest(tmp_path).stamps == stamps


class TestForTheRetentionLeg:
    def test_a_written_artifact_is_identified(self, tmp_path):
        make_manifest().write(tmp_path)
        assert is_backup_artifact_dir(tmp_path) is True

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param(None, id="no-manifest"),
            pytest.param("{not json", id="malformed-json"),
            pytest.param("[]", id="json-but-not-an-object"),
            pytest.param('{"artifact_schema_version": 1}', id="hydration-artifact"),
            pytest.param('{"layout": "mist.backup"}', id="no-layout-version"),
            pytest.param('{"layout": "mist.backup", "layout_version": 1}', id="no-created-at"),
        ],
    )
    def test_anything_else_is_not_a_backup_directory(self, tmp_path, content):
        if content is not None:
            (tmp_path / MANIFEST_FILENAME).write_text(content, encoding="utf-8")
        assert is_backup_artifact_dir(tmp_path) is False

    def test_a_newer_layout_is_still_recognised_as_ours(self, tmp_path):
        # Recognition and readability are different questions. A retention pass
        # asking "is this mine?" must say yes to an artifact from a newer build,
        # or it would treat it as foreign and leave it forever.
        raw = make_manifest().to_dict()
        raw["layout_version"] = BACKUP_LAYOUT_VERSION + 1
        (tmp_path / MANIFEST_FILENAME).write_text(json.dumps(raw), encoding="utf-8")
        assert is_backup_artifact_dir(tmp_path) is True

    def test_created_at_parses_as_utc(self, tmp_path):
        make_manifest().write(tmp_path)
        created = read_manifest(tmp_path).created_at_datetime()
        assert created.tzinfo is not None
        assert created.utcoffset().total_seconds() == 0
        assert (created.year, created.month, created.day) == (2026, 9, 17)

    def test_an_unparseable_created_at_raises_rather_than_defaulting(self):
        with pytest.raises(BackupManifestError):
            make_manifest(created_at="last tuesday").created_at_datetime()

    def test_utc_now_iso_is_parseable_by_the_same_reader(self):
        stamp = utc_now_iso()
        assert stamp.endswith("Z")
        assert make_manifest(created_at=stamp).created_at_datetime().tzinfo is not None


class TestFileDigests:
    def test_every_file_is_digested_with_posix_keys(self, tmp_path):
        (tmp_path / "stores").mkdir()
        (tmp_path / "stores" / "event_store.db").write_bytes(b"store-bytes")
        (tmp_path / "graph.json").write_text("{}", encoding="utf-8")
        digests = digest_artifact_files(tmp_path)
        assert set(digests) == {"stores/event_store.db", "graph.json"}
        assert digests["graph.json"]["bytes"] == 2
        assert digests["stores/event_store.db"]["sha256"] == sha256_file(
            tmp_path / "stores" / "event_store.db"
        )

    def test_the_manifest_does_not_digest_itself(self, tmp_path):
        (tmp_path / MANIFEST_FILENAME).write_text("{}", encoding="utf-8")
        (tmp_path / "graph.json").write_text("{}", encoding="utf-8")
        assert set(digest_artifact_files(tmp_path)) == {"graph.json"}
