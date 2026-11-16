"""Tests for `GET /capabilities/usage`: capability use and alarm events per local day.

The host is real, over a real action log and a real alarm event trail in a temporary
capabilities database, so each test reads what the view reads in production; only the clock is
fixed. The endpoint tests use `TestClient` without the `with` block, so the ASGI lifespan does
not run and the capability globals are the ones each test installs.
"""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend import server
from backend.capabilities.action_log import ActionLog, ActionLogEntry
from backend.capabilities.alarms.capability import SchedulerAlarmControl
from backend.capabilities.alarms.records import AlarmRecords
from backend.capabilities.db import CapabilitiesDb
from backend.capabilities.decoder import ActionDecoder
from backend.capabilities.host import CapabilityHost, HostSettings
from backend.capabilities.policy import PolicyTable
from backend.capabilities.power import FixedPower
from backend.capabilities.router import RouterConfig, ToolRouter
from backend.capabilities.secrets import Secret, SecretRegistry
from backend.chat.context_budget import ApproximateTokenCounter
from backend.event_store import usage
from backend.factories import CapabilityBuild
from tests.unit.capabilities.fakes import FakeCapability, FakeClock, RecordingEventSink

NOW = datetime(2026, 1, 5, 7, 0, tzinfo=UTC)
TAURI = "http://tauri.localhost"
SECRET = "ha-secret-token-0123456789"  # noqa: S105 - a fake token for the redaction check
ARGUMENT = "the-kitchen-lamp-by-the-window"
AHEAD = "Asia/Tokyo"
"""Nine hours ahead of UTC all year, so the local midnight case needs no DST rule."""

EMPTY_ALARMS = {
    "fired": 0,
    "stopped": 0,
    "timed_out": 0,
    "snoozed": 0,
    "missed_late": 0,
    "missed_unheard": 0,
    "dismissed": 0,
    "reported": 0,
    "fallback": 0,
}


def _row(at: datetime, **changes: object) -> ActionLogEntry:
    fields: dict[str, object] = {
        "at": at,
        "session_id": "s1",
        "turn_id": "t1",
        "tool_call_id": f"call-{at.isoformat()}",
        "tool": "lights.power",
        "capability": "lights",
        "source": "fastpath",
        "status": "ok",
        "args_summary": "on",
        "summary": "Done.",
        "reason": None,
        "proposal_id": None,
        "duration_ms": 3,
        "power_level": "full",
        "turn_ms": 100,
    }
    fields.update(changes)
    return ActionLogEntry(**fields)  # type: ignore[arg-type]


class Rig:
    """A real host over one temporary capabilities database, installed as the server's build."""

    def __init__(self, path: str, registry: SecretRegistry, *, alarms: bool, now: datetime):
        self.db = CapabilitiesDb(path)
        self.log = ActionLog(self.db)
        self.records = AlarmRecords(self.db)
        capability = FakeCapability("fake")
        counter = ApproximateTokenCounter()
        router = ToolRouter(
            [capability], RouterConfig(token_budget=1200, reserved_tokens=100), counter, None
        )
        control = None
        if alarms:
            # Production migrates in `AlarmsCapability.start`; the rig is a started host.
            self.records.migrate()
            # The usage pass-throughs read the store only, so the scheduler, settings and zone
            # the control keeps for `alarm_state` and client actions are never reached here.
            control = SchedulerAlarmControl(
                self.records,
                SimpleNamespace(),  # type: ignore[arg-type]
                FakeClock(now),
                SimpleNamespace(),  # type: ignore[arg-type]
                UTC,
            )
        self.host = CapabilityHost(
            [capability],
            router=router,
            counter=counter,
            policy=PolicyTable.from_specs(capability.tools(), None),
            decoder=ActionDecoder(),
            secrets=registry,
            action_log=self.log,
            event_sink=RecordingEventSink(),
            clock=FakeClock(now),
            settings=HostSettings(),
            alarm_control=control,
        )

    def actions(self, *rows: ActionLogEntry) -> None:
        async def write() -> None:
            for row in rows:
                await self.log.record(row)

        asyncio.run(write())

    def events(self, *events: tuple[datetime, str, dict[str, object] | None]) -> None:
        """Append alarm events as `AlarmRecords` stores them: UTC times with microseconds."""
        self.records.migrate()

        def write(conn) -> None:
            for at, kind, detail in events:
                conn.execute(
                    "INSERT INTO alarm_events (record_id, occurrence_id, at, kind, source,"
                    " endpoint, late_by_s, detail) VALUES ('r1', 'o1', ?, ?, 'scheduler', NULL,"
                    " NULL, ?)",
                    (
                        at.astimezone(UTC).isoformat(timespec="microseconds"),
                        kind,
                        json.dumps(detail) if detail is not None else None,
                    ),
                )

        asyncio.run(self.db.run(write))


@pytest.fixture
def endpoint(monkeypatch, tmp_path):
    """Install a build per test, and return a client, the secret registry and a rig factory."""
    registry = SecretRegistry()
    monkeypatch.setattr(server, "_capability_secrets", registry)
    monkeypatch.setattr(server, "_capability_build", None)
    monkeypatch.setattr(server, "_capability_config", None)
    monkeypatch.setattr(server, "_capability_power", FixedPower("full"))
    monkeypatch.delenv("MIST_ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv(usage.TIMEZONE_ENV, raising=False)
    rigs: list[Rig] = []

    def rig(*, alarms: bool = True, now: datetime = NOW, path: str | None = None) -> Rig:
        built = Rig(path or str(tmp_path / "capabilities.db"), registry, alarms=alarms, now=now)
        rigs.append(built)
        monkeypatch.setattr(
            server, "_capability_build", CapabilityBuild(host=built.host, disabled_reason=None)
        )
        return built

    def install(build: CapabilityBuild | None) -> None:
        monkeypatch.setattr(server, "_capability_build", build)

    yield SimpleNamespace(
        client=TestClient(server.app), registry=registry, rig=rig, install=install
    )
    for built in rigs:
        built.db.close()


def _day(date: str, **changes: object) -> dict[str, object]:
    day: dict[str, object] = {
        "date": date,
        "invocations": 0,
        "by_capability": {},
        "by_source": {"fastpath": 0, "model": 0, "proposal": 0},
        "by_level": {"full": 0, "low": 0, "off": 0},
        "by_status": {},
        "turn_ms": {"p50": None, "p90": None},
        "turn_ms_by_level": _levels(),
    }
    day.update(changes)
    return day


def _levels(**changes: dict[str, int | None]) -> dict[str, dict[str, int | None]]:
    """A day's `turn_ms_by_level`: every level null unless changed."""
    null = {"p50": None, "p90": None}
    return {"full": null, "low": null, "off": null} | changes


def _alarms(date: str, **changes: int) -> dict[str, object]:
    return {"date": date} | EMPTY_ALARMS | changes


# -- Shape ---------------------------------------------------------------------------------------


class TestShape:
    def test_a_seeded_log_and_trail_answer_the_planned_shape(self, endpoint):
        rig = endpoint.rig()
        rig.actions(
            # Outside the two-day window, and the oldest row: `since` and nothing else.
            _row(datetime(2026, 1, 1, 10, 0, tzinfo=UTC), capability="clock", source="model"),
            _row(datetime(2026, 1, 4, 23, 59, tzinfo=UTC), capability="alarms",
                 source="proposal", status="pending_confirmation", power_level="off",
                 turn_ms=2500),
            _row(datetime(2026, 1, 5, 6, 0, tzinfo=UTC), turn_ms=120),
            _row(datetime(2026, 1, 5, 6, 30, tzinfo=UTC), source="model", power_level="low",
                 turn_ms=900),
        )  # fmt: skip
        rig.events(
            # The oldest event, outside the window and not a counted kind: `alarms_since` only.
            (datetime(2026, 1, 3, 7, 0, tzinfo=UTC), "set", None),
            (datetime(2026, 1, 4, 7, 0, tzinfo=UTC), "missed", {"reason": "unheard"}),
            (datetime(2026, 1, 4, 8, 0, tzinfo=UTC), "missed", {"reason": "late"}),
            (datetime(2026, 1, 4, 8, 0, tzinfo=UTC), "fallback", None),
            (datetime(2026, 1, 4, 9, 0, tzinfo=UTC), "reported", None),
            (datetime(2026, 1, 4, 9, 30, tzinfo=UTC), "snoozed", None),
            (datetime(2026, 1, 4, 10, 0, tzinfo=UTC), "timed_out", None),
            (datetime(2026, 1, 4, 11, 0, tzinfo=UTC), "dismissed", None),
            (datetime(2026, 1, 5, 6, 0, tzinfo=UTC), "fired", None),
            (datetime(2026, 1, 5, 6, 1, tzinfo=UTC), "stopped", None),
        )

        response = endpoint.client.get("/capabilities/usage", params={"days": 2})

        assert response.status_code == 200
        assert response.json() == {
            "tz": "UTC",
            "since": "2026-01-01T10:00:00+00:00",
            "alarms_since": "2026-01-03T07:00:00+00:00",
            "days": [
                _day(
                    "2026-01-05",
                    invocations=2,
                    by_capability={"lights": 2},
                    by_source={"fastpath": 1, "model": 1, "proposal": 0},
                    by_level={"full": 1, "low": 1, "off": 0},
                    by_status={"ok": 2},
                    turn_ms={"p50": 120, "p90": 900},
                    turn_ms_by_level=_levels(
                        full={"p50": 120, "p90": 120}, low={"p50": 900, "p90": 900}
                    ),
                ),
                _day(
                    "2026-01-04",
                    invocations=1,
                    by_capability={"alarms": 1},
                    by_source={"fastpath": 0, "model": 0, "proposal": 1},
                    by_level={"full": 0, "low": 0, "off": 1},
                    by_status={"pending_confirmation": 1},
                    turn_ms={"p50": 2500, "p90": 2500},
                    turn_ms_by_level=_levels(off={"p50": 2500, "p90": 2500}),
                ),
            ],
            "alarms": [
                _alarms("2026-01-05", fired=1, stopped=1),
                _alarms(
                    "2026-01-04",
                    timed_out=1,
                    snoozed=1,
                    missed_late=1,
                    missed_unheard=1,
                    dismissed=1,
                    reported=1,
                    fallback=1,
                ),
            ],
        }

    def test_empty_stores_answer_every_day_and_null_sinces(self, endpoint):
        endpoint.rig()

        body = endpoint.client.get("/capabilities/usage").json()

        assert body["since"] is None
        assert body["alarms_since"] is None
        assert "alarm_tz" not in body
        dates = ["2026-01-05", "2026-01-04", "2026-01-03", "2026-01-02", "2026-01-01",
                 "2025-12-31", "2025-12-30"]  # fmt: skip
        assert body["days"] == [_day(date) for date in dates]
        assert body["alarms"] == [_alarms(date) for date in dates]

    def test_without_an_alarm_control_the_alarm_days_are_zero(self, endpoint):
        rig = endpoint.rig(alarms=False)
        rig.actions(_row(datetime(2026, 1, 5, 6, 0, tzinfo=UTC)))

        body = endpoint.client.get("/capabilities/usage", params={"days": 1}).json()

        assert body["alarms_since"] is None
        assert body["alarms"] == [_alarms("2026-01-05")]
        assert body["days"][0]["invocations"] == 1

    def test_each_level_answers_its_own_p50_and_p90(self, endpoint):
        rig = endpoint.rig()
        rig.actions(
            _row(datetime(2026, 1, 5, 6, 0, tzinfo=UTC), power_level="full", turn_ms=1000),
            _row(datetime(2026, 1, 5, 6, 10, tzinfo=UTC), power_level="low", turn_ms=500),
            _row(datetime(2026, 1, 5, 6, 20, tzinfo=UTC), power_level="full", turn_ms=100),
        )

        body = endpoint.client.get("/capabilities/usage", params={"days": 1}).json()

        # A swapped p50 and p90, or one level's rows read as another's, changes this body.
        assert body["days"][0]["turn_ms_by_level"] == {
            "full": {"p50": 100, "p90": 1000},
            "low": {"p50": 500, "p90": 500},
            "off": {"p50": None, "p90": None},
        }


# -- Time zone -----------------------------------------------------------------------------------


class TestTimeZone:
    # 00:05 on 6 Jan in Tokyo, still 5 Jan in UTC.
    JUST_AFTER_LOCAL_MIDNIGHT = datetime(2026, 1, 5, 15, 5, tzinfo=UTC)

    def _seed(self, rig: Rig) -> None:
        rig.actions(
            _row(datetime(2026, 1, 5, 14, 50, tzinfo=UTC)),  # 23:50 on 5 Jan in Tokyo
            _row(datetime(2026, 1, 5, 15, 1, tzinfo=UTC)),  # 00:01 on 6 Jan in Tokyo
        )
        rig.events((datetime(2026, 1, 5, 15, 2, tzinfo=UTC), "fired", None))

    def test_a_zone_ahead_of_utc_dates_a_call_after_local_midnight_locally(
        self, endpoint, monkeypatch
    ):
        if usage.resolve_timezone(AHEAD)[1] == "UTC":
            pytest.skip("tzdata absent until H1")
        monkeypatch.setenv(usage.TIMEZONE_ENV, AHEAD)
        self._seed(endpoint.rig(now=self.JUST_AFTER_LOCAL_MIDNIGHT))

        body = endpoint.client.get("/capabilities/usage", params={"days": 2}).json()

        assert body["tz"] == AHEAD
        assert [(d["date"], d["invocations"]) for d in body["days"]] == [
            ("2026-01-06", 1),
            ("2026-01-05", 1),
        ]
        assert [(a["date"], a["fired"]) for a in body["alarms"]] == [
            ("2026-01-06", 1),
            ("2026-01-05", 0),
        ]

    def test_unset_the_dates_are_utc_dates(self, endpoint):
        self._seed(endpoint.rig(now=self.JUST_AFTER_LOCAL_MIDNIGHT))

        body = endpoint.client.get("/capabilities/usage", params={"days": 2}).json()

        assert body["tz"] == "UTC"
        assert [(d["date"], d["invocations"]) for d in body["days"]] == [
            ("2026-01-05", 2),
            ("2026-01-04", 0),
        ]
        assert [(a["date"], a["fired"]) for a in body["alarms"]] == [
            ("2026-01-05", 1),
            ("2026-01-04", 0),
        ]


# -- Days, Origin, no host and store failures ----------------------------------------------------


class TestRequest:
    @pytest.mark.parametrize("days", ["0", "91", "-1", "seven"])
    def test_days_outside_one_to_ninety_gets_422(self, endpoint, days):
        endpoint.rig()

        response = endpoint.client.get("/capabilities/usage", params={"days": days})

        assert response.status_code == 422

    @pytest.mark.parametrize("days", [1, 90])
    def test_days_at_each_bound_answers_that_many_days(self, endpoint, days):
        endpoint.rig()

        body = endpoint.client.get("/capabilities/usage", params={"days": days}).json()

        assert len(body["days"]) == days
        assert len(body["alarms"]) == days

    def test_an_origin_the_allowlist_refuses_gets_403(self, endpoint, monkeypatch):
        monkeypatch.setenv("MIST_ALLOWED_ORIGINS", TAURI)
        endpoint.rig().actions(_row(datetime(2026, 1, 5, 6, 0, tzinfo=UTC)))

        response = endpoint.client.get(
            "/capabilities/usage", headers={"Origin": "https://evil.example"}
        )

        assert response.status_code == 403
        assert "lights" not in response.text

    @pytest.mark.parametrize("headers", [{"Origin": TAURI}, {"Origin": TAURI.upper()}, {}])
    def test_a_listed_origin_or_none_gets_the_body(self, endpoint, monkeypatch, headers):
        monkeypatch.setenv("MIST_ALLOWED_ORIGINS", TAURI)
        endpoint.rig()

        response = endpoint.client.get("/capabilities/usage", headers=headers)

        assert response.status_code == 200
        assert len(response.json()["days"]) == 7

    def test_with_no_host_the_disabled_body_comes_back(self, endpoint):
        endpoint.registry.register(Secret(SECRET))
        endpoint.install(
            CapabilityBuild(host=None, disabled_reason=f"capabilities disabled {SECRET}")
        )

        response = endpoint.client.get("/capabilities/usage")

        assert response.status_code == 200
        assert response.json() == {"enabled": False, "reason": "capabilities disabled ***"}

    def test_before_lifespan_the_disabled_body_says_not_started(self, endpoint):
        endpoint.install(None)

        response = endpoint.client.get("/capabilities/usage")

        assert response.json() == {"enabled": False, "reason": "capabilities not started"}

    def test_an_unreadable_store_gets_503(self, endpoint, tmp_path):
        blocker = tmp_path / "not-a-directory"
        blocker.write_text("a regular file", encoding="utf-8")
        endpoint.rig(alarms=False, path=str(blocker / "capabilities.db"))

        response = endpoint.client.get("/capabilities/usage")

        assert response.status_code == 503
        assert response.json() == {"error": "store_unavailable"}


# -- Redaction -----------------------------------------------------------------------------------


class TestRedaction:
    def test_no_secret_or_argument_text_reaches_the_body(self, endpoint):
        endpoint.registry.register(Secret(SECRET))
        rig = endpoint.rig()
        rig.actions(
            _row(
                datetime(2026, 1, 5, 6, 0, tzinfo=UTC),
                args_summary=f"turn on {ARGUMENT} with Bearer {SECRET}",
                summary=f"Turned on {ARGUMENT}.",
                reason=f"token {SECRET}",
            ),
            # A capability name is the one text field the body carries, so it is redacted too.
            _row(datetime(2026, 1, 5, 6, 1, tzinfo=UTC), capability=f"lights-{SECRET}"),
        )

        response = endpoint.client.get("/capabilities/usage", params={"days": 1})

        assert response.status_code == 200
        assert SECRET not in response.text
        assert ARGUMENT not in response.text
        assert response.json()["days"][0]["by_capability"] == {"lights": 1, "lights-***": 1}
