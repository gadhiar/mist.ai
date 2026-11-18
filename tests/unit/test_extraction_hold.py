"""Tests for the paused-extraction marker reader.

The marker is written by the host CLI (`scripts/daily/instance.py`) and read
here; neither side imports the other, so these tests pin the backend's half
of the contract in plan v4's own words: the file name `extraction-paused` and
the JSON `{"v": 1, "since": <UTC>, "reason": <r>, "until": <UTC or null>}`.
The wall clock is injected, so no test waits for a marker to expire.
"""

import json
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from backend.extraction_hold import (
    EXTRACTION_PAUSED_MARKER,
    HOLD_REASONS,
    ExtractionHold,
    ExtractionHoldReader,
    build_extraction_hold_reader,
)

NOW = datetime(2026, 11, 15, 14, 30, 0, tzinfo=UTC)


def _utc(moment: datetime) -> str:
    """The form the host CLI writes (`_format_utc` in `scripts/daily/instance.py`)."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _payload(**overrides) -> dict:
    body = {"v": 1, "since": _utc(NOW - timedelta(hours=4)), "reason": "window", "until": None}
    body.update(overrides)
    return body


def _write(tmp_path: Path, body) -> Path:
    marker = tmp_path / EXTRACTION_PAUSED_MARKER
    text = body if isinstance(body, str) else json.dumps(body)
    marker.write_text(text, encoding="utf-8")
    return marker


def _reader(marker: Path, now: datetime = NOW) -> ExtractionHoldReader:
    return ExtractionHoldReader(marker, wall_clock=lambda: now)


def test_the_marker_name_and_reasons_are_the_contract():
    """The host CLI writes this file under the same name; a rename on one side breaks both."""
    assert EXTRACTION_PAUSED_MARKER == "extraction-paused"
    assert frozenset({"window", "mode", "gaming", "starting", "unconfigured"}) == HOLD_REASONS


def test_no_marker_is_no_hold(tmp_path):
    assert _reader(tmp_path / EXTRACTION_PAUSED_MARKER).current() is None


@pytest.mark.parametrize("reason", sorted(HOLD_REASONS))
def test_a_marker_without_until_is_a_hold(tmp_path, reason):
    since = NOW - timedelta(hours=4)
    marker = _write(tmp_path, _payload(reason=reason, since=_utc(since), until=None))

    hold = _reader(marker).current()

    assert hold == ExtractionHold(since=since, reason=reason, until=None)


def test_a_marker_whose_until_is_ahead_is_a_hold(tmp_path):
    until = NOW + timedelta(minutes=10)
    marker = _write(tmp_path, _payload(reason="starting", until=_utc(until)))

    hold = _reader(marker).current()

    assert hold is not None
    assert (hold.reason, hold.until) == ("starting", until)


@pytest.mark.parametrize(
    "until",
    [pytest.param(NOW, id="exactly-now"), pytest.param(NOW - timedelta(seconds=1), id="past")],
)
def test_a_marker_past_its_until_is_absent(tmp_path, until, caplog):
    """A hold that should have ended (a start that hung) must not hide the service."""
    marker = _write(tmp_path, _payload(reason="starting", until=_utc(until)))

    with caplog.at_level(logging.WARNING, logger="backend.extraction_hold"):
        assert _reader(marker).current() is None

    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_the_reader_reads_the_clock_on_every_call(tmp_path):
    until = NOW + timedelta(minutes=1)
    marker = _write(tmp_path, _payload(until=_utc(until)))
    now = {"value": NOW}
    reader = ExtractionHoldReader(marker, wall_clock=lambda: now["value"])

    before = reader.current()
    now["value"] = until + timedelta(seconds=1)
    after = reader.current()

    assert before is not None
    assert after is None


def test_a_timestamp_with_an_offset_or_none_is_read_as_utc(tmp_path):
    marker = _write(
        tmp_path,
        _payload(since="2026-11-15T10:30:00+00:00", until="2026-11-15T15:00:00"),
    )

    hold = _reader(marker).current()

    assert hold is not None
    assert hold.since == datetime(2026, 11, 15, 10, 30, tzinfo=UTC)
    assert hold.until == datetime(2026, 11, 15, 15, 0, tzinfo=UTC)


MALFORMED = [
    pytest.param("not json at all", id="not-json"),
    pytest.param("[1, 2]", id="not-an-object"),
    pytest.param(_payload(v=2), id="unknown-version"),
    pytest.param(_payload(v="1"), id="version-as-text"),
    pytest.param(_payload(reason="holiday"), id="unknown-reason"),
    pytest.param(_payload(reason=None), id="null-reason"),
    pytest.param(_payload(reason=["window"]), id="reason-a-list"),
    pytest.param(_payload(v=True), id="version-as-bool"),
    pytest.param(_payload(since=None), id="null-since"),
    pytest.param(_payload(since="yesterday"), id="since-not-iso"),
    pytest.param(_payload(until="soon"), id="until-not-iso"),
    pytest.param(_payload(until=1731681000), id="until-a-number"),
    pytest.param(_payload(since="0001-01-01T00:00:00+05:00"), id="since-before-utc-range"),
    pytest.param(_payload(until="9999-12-31T23:59:59-05:00"), id="until-past-utc-range"),
    pytest.param({"v": 1, "since": _utc(NOW), "reason": "mode"}, id="until-missing"),
    pytest.param({"v": 1, "reason": "mode", "until": None}, id="since-missing"),
]


@pytest.mark.parametrize("body", MALFORMED)
def test_a_malformed_marker_is_absent_with_a_warning(tmp_path, body, caplog):
    """Claiming a pause nobody can be shown to have meant would hide a down service."""
    marker = _write(tmp_path, body)

    with caplog.at_level(logging.WARNING, logger="backend.extraction_hold"):
        assert _reader(marker).current() is None

    assert [r.levelno for r in caplog.records] == [logging.WARNING]


def test_an_undecodable_marker_is_absent_with_a_warning(tmp_path, caplog):
    marker = tmp_path / EXTRACTION_PAUSED_MARKER
    marker.write_bytes(b"\xff\xfe\x00not utf-8")

    with caplog.at_level(logging.WARNING, logger="backend.extraction_hold"):
        assert _reader(marker).current() is None

    assert [r.levelno for r in caplog.records] == [logging.WARNING]


def test_an_unreadable_marker_is_absent_with_a_warning(tmp_path, caplog):
    """A directory where the file should be stands in for any read error."""
    marker = tmp_path / EXTRACTION_PAUSED_MARKER
    marker.mkdir()

    with caplog.at_level(logging.WARNING, logger="backend.extraction_hold"):
        assert _reader(marker).current() is None

    assert [r.levelno for r in caplog.records] == [logging.WARNING]


def test_a_lasting_malformed_marker_warns_once_until_it_changes(tmp_path, caplog):
    """The probe reads every 10 s; one bad file must not fill the log."""
    marker = _write(tmp_path, "not json")
    reader = _reader(marker)

    with caplog.at_level(logging.WARNING, logger="backend.extraction_hold"):
        for _ in range(3):
            reader.current()
        marker.unlink()
        reader.current()
        _write(tmp_path, "still not json")
        reader.current()

    assert len(caplog.records) == 2


@pytest.mark.parametrize("ops_dir", [None, "", "   "])
def test_no_ops_dir_builds_no_reader(ops_dir):
    """The dev stack sets no `MIST_OPS_DIR`, and its probe must behave as before."""
    assert build_extraction_hold_reader(ops_dir) is None


def test_an_ops_dir_builds_a_reader_on_the_marker(tmp_path):
    reader = build_extraction_hold_reader(f"  {tmp_path}  ", wall_clock=lambda: NOW)
    _write(tmp_path, _payload(reason="mode"))

    assert reader is not None
    hold = reader.current()
    assert hold is not None and hold.reason == "mode"
