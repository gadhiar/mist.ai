"""The backup status file: its exact keys, its atomic write, and what counts as corrupt.

`BackupProbe` (`/health`) reads this file in another process, so its key set is
a contract (plan v1, Interfaces and contracts item 6). The keys are spelled out
literally here rather than imported from `scripts.backup.status`, so a change to
the module's constant cannot silently change the contract this test pins.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from scripts.backup import status as status_module
from scripts.backup.status import (
    BackupStatus,
    SuccessCounts,
    read_status,
    write_status,
)

ITEM_6_KEYS = [
    "v",
    "last_attempt_at",
    "last_result",
    "last_exit_code",
    "last_success_at",
    "last_label",
    "last_duration_s",
    "retained",
    "last_success_counts",
]

ATTEMPT_AT = datetime(2026, 10, 23, 3, 0, 0, tzinfo=UTC)
SUCCESS_AT = datetime(2026, 10, 22, 3, 0, 5, tzinfo=UTC)


def make_status(**overrides: object) -> BackupStatus:
    """A status as a success leaves it; any field may be overridden."""
    fields: dict[str, object] = {
        "last_attempt_at": ATTEMPT_AT,
        "last_result": "ok",
        "last_exit_code": 0,
        "last_success_at": SUCCESS_AT,
        "last_label": "20261023T030000Z",
        "last_duration_s": 12.5,
        "retained": 7,
        "last_success_counts": SuccessCounts(
            graph_nodes=120,
            vault_corpus_files=13,
            stores=("event_store.db", "vault_sidecar.db"),
        ),
    }
    fields.update(overrides)
    return BackupStatus(**fields)  # type: ignore[arg-type]


class TestWrite:
    def test_written_file_has_exactly_the_item_6_keys_in_order(self, tmp_path: Path) -> None:
        path = tmp_path / "backup-status.json"

        write_status(path, make_status())

        assert list(json.loads(path.read_text(encoding="utf-8"))) == ITEM_6_KEYS

    def test_success_counts_carry_exactly_their_three_keys(self, tmp_path: Path) -> None:
        path = tmp_path / "backup-status.json"

        write_status(path, make_status())

        counts = json.loads(path.read_text(encoding="utf-8"))["last_success_counts"]
        assert counts == {
            "graph_nodes": 120,
            "vault_corpus_files": 13,
            "stores": ["event_store.db", "vault_sidecar.db"],
        }

    def test_timestamps_are_iso_8601_utc(self, tmp_path: Path) -> None:
        path = tmp_path / "backup-status.json"

        write_status(path, make_status())

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw["last_attempt_at"] == "2026-10-23T03:00:00Z"
        assert raw["last_success_at"] == "2026-10-22T03:00:05Z"
        assert raw["v"] == 1

    def test_a_status_before_any_event_writes_nulls(self, tmp_path: Path) -> None:
        path = tmp_path / "backup-status.json"

        write_status(path, BackupStatus())

        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw == {key: (1 if key == "v" else None) for key in ITEM_6_KEYS}

    def test_write_goes_through_a_temp_file_in_the_same_directory_then_os_replace(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "backup-status.json"
        replaced: list[tuple[Path, Path]] = []
        real_replace = status_module.os.replace

        def recording_replace(src: str, dst: str) -> None:
            replaced.append((Path(src), Path(dst)))
            real_replace(src, dst)

        monkeypatch.setattr(status_module.os, "replace", recording_replace)

        write_status(path, make_status())

        assert len(replaced) == 1
        source, destination = replaced[0]
        assert destination == path
        assert source.parent == path.parent
        assert source != path
        assert not source.exists()

    def test_a_failed_replace_keeps_the_previous_file_and_leaves_no_temp_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "backup-status.json"
        write_status(path, make_status(last_label="previous"))
        before = path.read_bytes()

        def failing_replace(src: str, dst: str) -> None:
            raise OSError("disk full")

        monkeypatch.setattr(status_module.os, "replace", failing_replace)

        with pytest.raises(OSError, match="disk full"):
            write_status(path, make_status(last_label="next"))

        assert path.read_bytes() == before
        assert sorted(p.name for p in tmp_path.iterdir()) == ["backup-status.json"]


class TestRead:
    def test_an_absent_file_reads_as_none(self, tmp_path: Path) -> None:
        assert read_status(tmp_path / "backup-status.json") is None

    def test_a_written_status_reads_back_equal(self, tmp_path: Path) -> None:
        path = tmp_path / "backup-status.json"
        written = make_status()

        write_status(path, written)

        assert read_status(path) == written

    @pytest.mark.parametrize(
        "content",
        [
            pytest.param("{not json", id="unparsable"),
            pytest.param("[]", id="not-an-object"),
            pytest.param(
                json.dumps({key: None for key in ITEM_6_KEYS if key != "retained"} | {"v": 1}),
                id="key-missing",
            ),
            pytest.param(
                json.dumps({key: None for key in ITEM_6_KEYS} | {"v": 2}),
                id="unknown-version",
            ),
            pytest.param(
                json.dumps(
                    {key: None for key in ITEM_6_KEYS}
                    | {"v": 1, "last_result": "ok", "last_success_at": "yesterday"}
                ),
                id="unparsable-timestamp",
            ),
            pytest.param(
                json.dumps({key: None for key in ITEM_6_KEYS} | {"v": 1, "last_result": "fine"}),
                id="unknown-result",
            ),
            pytest.param(
                json.dumps(
                    {key: None for key in ITEM_6_KEYS}
                    | {"v": 1, "last_success_counts": {"graph_nodes": "many"}}
                ),
                id="malformed-counts",
            ),
        ],
    )
    def test_a_corrupt_file_reads_as_none_and_is_logged(
        self, tmp_path: Path, content: str, caplog: pytest.LogCaptureFixture
    ) -> None:
        path = tmp_path / "backup-status.json"
        path.write_text(content, encoding="utf-8")

        with caplog.at_level(logging.WARNING, logger=status_module.__name__):
            result = read_status(path)

        assert result is None
        assert str(path) in caplog.text
