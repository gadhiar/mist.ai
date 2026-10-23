"""Tests for `GET /usage/turns-per-day`, the counts-only read of the 5-of-7 gate.

Every store is synthetic, built in `tmp_path` from the real `schema.sql`, and
`EVENT_STORE_DB_PATH` always points at it, so no test reaches `~/.mist`.

`TestClient` is constructed without the `with` block, as in
`tests/unit/test_server_health_wiring.py`: entering it would run the ASGI
lifespan and load the model stack.

The route reads the wall clock, so turns are stamped relative to
`datetime.now(UTC)` at whole-day offsets and the assertions are on counts and
the gate rather than on a particular date. If the test straddles UTC midnight,
each offset shifts by one day, which keeps every turn inside the 7-day window.
"""

from __future__ import annotations

import logging
import re
import sqlite3
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import server
from backend.event_store import usage

_SCHEMA_PATH = Path(usage.__file__).parent / "schema.sql"

_SESSION_ID = "session-7f3a-private"
_UTTERANCE = "remind me about the dentist on thursday"
_RESPONSE = "I will remind you on Thursday morning"


def _make_store(path: Path) -> Path:
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA_PATH.read_text(encoding="utf-8"))
    conn.commit()
    conn.close()
    return path


def _add_session(path: Path, session_id: str, origin: str) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO conversation_sessions (session_id, started_at, origin) VALUES (?, ?, ?)",
        (session_id, "2026-01-01T00:00:00+00:00", origin),
    )
    conn.commit()
    conn.close()


def _add_turn(path: Path, session_id: str, days_ago: int) -> None:
    stamp = (datetime.now(UTC) - timedelta(days=days_ago)).isoformat()
    conn = sqlite3.connect(path)
    conn.execute(
        "INSERT INTO conversation_turn_events "
        "(event_id, session_id, turn_index, timestamp, user_utterance, system_response) "
        "VALUES (?, ?, 0, ?, ?, ?)",
        (str(uuid.uuid4()), session_id, stamp, _UTTERANCE, _RESPONSE),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = _make_store(tmp_path / "event_store.db")
    monkeypatch.setenv("EVENT_STORE_DB_PATH", str(path))
    monkeypatch.setenv("MIST_USAGE_TIMEZONE", "UTC")
    return path


@pytest.fixture
def client() -> TestClient:
    return TestClient(server.app)


def test_returns_the_contract_shape(store: Path, client: TestClient) -> None:
    response = client.get("/usage/turns-per-day")

    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"timezone", "origin", "days", "gate"}
    assert body["timezone"] == "UTC"
    assert body["origin"] == "real"
    assert len(body["days"]) == 7
    assert all(set(day) == {"date", "real_turns"} for day in body["days"])
    assert set(body["gate"]) == {"window_days", "required_days", "days_used", "met"}
    assert (body["gate"]["window_days"], body["gate"]["required_days"]) == (7, 5)


def test_days_are_dense_oldest_first_and_end_today(store: Path, client: TestClient) -> None:
    before = datetime.now(UTC).date()
    body = client.get("/usage/turns-per-day", params={"days": 10}).json()
    after = datetime.now(UTC).date()

    dates = [datetime.strptime(d["date"], "%Y-%m-%d").date() for d in body["days"]]
    assert len(dates) == 10
    assert all(b - a == timedelta(days=1) for a, b in zip(dates, dates[1:], strict=False))
    assert dates[-1] in {before, after}


def test_counts_only_real_origin_turns(store: Path, client: TestClient) -> None:
    _add_session(store, "s-real", "real")
    _add_session(store, "s-test", "test")
    _add_session(store, "s-seed", "seed")
    for days_ago in (1, 2):
        _add_turn(store, "s-real", days_ago)
        _add_turn(store, "s-test", days_ago)
        _add_turn(store, "s-seed", days_ago)
    _add_turn(store, "s-no-session-row", 3)

    body = client.get("/usage/turns-per-day").json()

    assert sum(day["real_turns"] for day in body["days"]) == 3
    assert body["gate"]["days_used"] == 3


def test_gate_is_met_at_five_used_days(store: Path, client: TestClient) -> None:
    for days_ago in (0, 1, 2, 3, 4):
        _add_turn(store, "s-real", days_ago)

    gate = client.get("/usage/turns-per-day").json()["gate"]

    assert (gate["days_used"], gate["met"]) == (5, True)


def test_gate_is_not_met_at_four_used_days(store: Path, client: TestClient) -> None:
    for days_ago in (0, 1, 3, 4):
        _add_turn(store, "s-real", days_ago)
    # Outside the 7-day window whatever `days` is asked for.
    _add_turn(store, "s-real", 12)

    body = client.get("/usage/turns-per-day", params={"days": 30}).json()

    assert sum(day["real_turns"] for day in body["days"]) == 5
    assert (body["gate"]["days_used"], body["gate"]["met"]) == (4, False)


def test_gate_covers_seven_days_even_when_one_day_is_asked(store: Path, client: TestClient) -> None:
    for days_ago in (0, 1, 2, 3, 4):
        _add_turn(store, "s-real", days_ago)

    body = client.get("/usage/turns-per-day", params={"days": 1}).json()

    assert len(body["days"]) == 1
    assert body["gate"]["window_days"] == 7
    assert body["gate"]["days_used"] == 5


@pytest.mark.parametrize("days", [1, 90])
def test_days_at_the_bounds_are_accepted(store: Path, client: TestClient, days: int) -> None:
    response = client.get("/usage/turns-per-day", params={"days": days})

    assert response.status_code == 200
    assert len(response.json()["days"]) == days


@pytest.mark.parametrize("days", ["0", "-1", "91", "1000", "seven"])
def test_days_outside_one_to_ninety_is_422(store: Path, client: TestClient, days: str) -> None:
    response = client.get("/usage/turns-per-day", params={"days": days})

    assert response.status_code == 422


def test_missing_store_is_503_and_is_not_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, client: TestClient
) -> None:
    missing = tmp_path / "nowhere" / "event_store.db"
    monkeypatch.setenv("EVENT_STORE_DB_PATH", str(missing))

    response = client.get("/usage/turns-per-day")

    assert response.status_code == 503
    assert response.json() == {"error": "store_unavailable"}
    assert not missing.exists()


def test_locked_store_is_503(store: Path, client: TestClient) -> None:
    holder = sqlite3.connect(store, isolation_level=None)
    holder.execute("PRAGMA locking_mode=EXCLUSIVE")
    holder.execute("BEGIN EXCLUSIVE")
    try:
        response = client.get("/usage/turns-per-day")
    finally:
        holder.execute("ROLLBACK")
        holder.close()

    assert response.status_code == 503
    assert response.json() == {"error": "store_unavailable"}


def test_unknown_zone_falls_back_to_utc_and_is_logged(
    store: Path,
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("MIST_USAGE_TIMEZONE", "Atlantis/Capital")

    with caplog.at_level(logging.WARNING, logger=usage.__name__):
        response = client.get("/usage/turns-per-day")

    assert response.status_code == 200
    assert response.json()["timezone"] == "UTC"
    assert "Atlantis/Capital" in caplog.text


def test_configured_zone_is_reported(
    store: Path, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("MIST_USAGE_TIMEZONE", "Europe/Berlin")

    assert client.get("/usage/turns-per-day").json()["timezone"] == "Europe/Berlin"


def test_response_holds_no_session_id_utterance_or_time_of_day(
    store: Path, client: TestClient
) -> None:
    _add_session(store, _SESSION_ID, "real")
    _add_turn(store, _SESSION_ID, 0)
    _add_turn(store, _SESSION_ID, 2)

    response = client.get("/usage/turns-per-day", params={"days": 90})
    text = response.text

    assert response.status_code == 200
    for secret in (_SESSION_ID, _UTTERANCE, _RESPONSE, "session", "utterance", "timestamp"):
        assert secret not in text
    # Nothing finer than a date: no clock time and no ISO `T` separator.
    assert not re.search(r"\d{1,2}:\d{2}", text)
    assert not re.search(r"\d{4}-\d{2}-\d{2}T", text)
    assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", d["date"]) for d in response.json()["days"])


def test_route_docstring_states_the_counts_only_contract() -> None:
    doc = " ".join((server.usage_turns_per_day.__doc__ or "").split())

    assert "counts only" in doc.lower()
    assert "without opening the event store" in doc
    assert "gate" in doc
