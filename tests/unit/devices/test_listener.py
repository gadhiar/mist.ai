"""Tests for `backend/devices/listener.py`: who reaches a session, and how sessions end.

Each test drives the real `DeviceApp` through ASGI on the test's own loop: `AsgiDevice` is the
server's side of one connection, so a close before accept is visible as the close code uvicorn
would turn into an HTTP 403. The registry is a real `DeviceRegistry` over a file under
`tmp_path`; the limiter and the revocation check run on a fake clock.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
from pathlib import Path

import pytest
import pytest_asyncio

from backend.devices.limiter import AuthLimiter
from backend.devices.listener import (
    CLOSE_POLICY,
    CLOSE_REPLACED,
    CLOSE_REVOKED,
    DEVICE_PATH,
    REVOCATION_POLL_S,
    DeviceSocket,
    build_device_app,
)
from backend.devices.registry import DeviceRegistry

pytestmark = pytest.mark.asyncio

LISTENER_LOGGER = "backend.devices.listener"
CREATED = "2026-11-17T09:00:00+00:00"
PEER = ("10.0.0.5", 50123)


def secret_for(seed: int) -> bytes:
    return bytes([seed]) * 32


def token_of(name: str, secret: bytes) -> str:
    return f"{name}.{base64.urlsafe_b64encode(secret).decode().rstrip('=')}"


def entry(name: str, secret: bytes, *, revoked: bool = False) -> dict[str, object]:
    return {
        "name": name,
        "area": "example-area",
        "token_sha256": hashlib.sha256(secret).hexdigest(),
        "revoked": revoked,
        "created": CREATED,
    }


def write_registry(path: Path, entries: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"version": 1, "devices": entries}), encoding="utf-8")


ROOM = token_of("example-room", secret_for(1))
OTHER = token_of("other-room", secret_for(2))
ROTATED_ROOM = token_of("example-room", secret_for(4))
"""`example-room`'s token after `devices.py rotate`."""


class FakeClock:
    """Monotonic seconds the test moves; `sleep` returns once the time has been advanced."""

    def __init__(self) -> None:
        self.now = 1000.0
        self._sleepers: list[tuple[float, asyncio.Future[None]]] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._sleepers.append((self.now + seconds, future))
        await future

    async def advance(self, seconds: float) -> None:
        self.now += seconds
        for due, future in list(self._sleepers):
            if due <= self.now and not future.done():
                future.set_result(None)
                self._sleepers.remove((due, future))
        await settle()


async def settle(rounds: int = 30) -> None:
    for _ in range(rounds):
        await asyncio.sleep(0)


class AsgiDevice:
    """One connection to the app, as uvicorn would hand it over, and what the app sent back."""

    def __init__(self, app, authorization: str | None) -> None:
        headers = [] if authorization is None else [(b"authorization", authorization.encode())]
        self.scope = {
            "type": "websocket",
            "asgi": {"version": "3.0"},
            "scheme": "wss",
            "path": DEVICE_PATH,
            "raw_path": DEVICE_PATH.encode(),
            "root_path": "",
            "query_string": b"",
            "headers": headers,
            "client": PEER,
            "server": ("127.0.0.1", 8443),
            "subprotocols": [],
        }
        self.inbox: asyncio.Queue[dict[str, object]] = asyncio.Queue()
        self.inbox.put_nowait({"type": "websocket.connect"})
        self.sent: list[dict[str, object]] = []
        self.accepted = False
        self.closed_with: int | None = None
        self.task = asyncio.get_running_loop().create_task(
            app(self.scope, self._receive, self._send)
        )

    async def _receive(self) -> dict[str, object]:
        return await self.inbox.get()

    async def _send(self, message: dict[str, object]) -> None:
        self.sent.append(message)
        if message["type"] == "websocket.accept":
            self.accepted = True
        elif message["type"] == "websocket.close":
            self.closed_with = int(message.get("code", 1000))
            # The server's side: the close handshake ends with a disconnect.
            self.inbox.put_nowait({"type": "websocket.disconnect", "code": self.closed_with})

    def text(self, payload: str) -> None:
        self.inbox.put_nowait({"type": "websocket.receive", "text": payload})

    def leave(self, code: int = 1000) -> None:
        self.inbox.put_nowait({"type": "websocket.disconnect", "code": code})

    def texts(self) -> list[str]:
        return [str(m["text"]) for m in self.sent if m["type"] == "websocket.send" and "text" in m]

    async def ended(self) -> None:
        await asyncio.wait_for(self.task, timeout=5)


class GatedDevice(AsgiDevice):
    """A connection whose accept is held until the test opens `gate`.

    The app has verified the token by the time it sends the accept, so holding the accept opens
    the window between verification and the live set that a registry read can land in.
    """

    def __init__(self, app, authorization: str | None) -> None:
        self.gate = asyncio.Event()
        super().__init__(app, authorization)

    async def _send(self, message: dict[str, object]) -> None:
        if message["type"] == "websocket.accept":
            await self.gate.wait()
        await super()._send(message)


class RecordingServe:
    """The per-connection serve: records who it served, greets, and echoes until disconnect."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.sockets: list[DeviceSocket] = []
        self.ended: list[str] = []

    async def __call__(self, socket: DeviceSocket, name: str, kind: str) -> None:
        self.calls.append((name, kind))
        self.sockets.append(socket)
        await socket.send_text(f"hello {name}")
        while True:
            message = await socket.receive()
            if message["type"] == "websocket.disconnect":
                break
            await socket.send_text(f"echo {message.get('text')}")
        self.ended.append(name)


class LingeringServe(RecordingServe):
    """A serve that outlives its disconnect until `release` is set, as committing a turn does.

    It holds a replacing connection in the listener's wait for the session it replaces.
    """

    def __init__(self) -> None:
        super().__init__()
        self.release = asyncio.Event()

    async def __call__(self, socket: DeviceSocket, name: str, kind: str) -> None:
        await super().__call__(socket, name, kind)
        await self.release.wait()


class Rig:
    def __init__(self, path: Path, serve: RecordingServe | None = None) -> None:
        self.path = path
        self.clock = FakeClock()
        self.limiter = AuthLimiter(self.clock)
        self.serve = serve if serve is not None else RecordingServe()
        self.registry = DeviceRegistry.load(path)
        self.app = build_device_app(self.registry, self.serve, self.limiter, self.clock)
        self.devices: list[AsgiDevice] = []

    async def connect(self, authorization: str | None) -> AsgiDevice:
        device = AsgiDevice(self.app, authorization)
        self.devices.append(device)
        await settle()
        return device

    async def connect_gated(self, authorization: str | None) -> GatedDevice:
        device = GatedDevice(self.app, authorization)
        self.devices.append(device)
        await settle()
        return device


@pytest.fixture
def registry_path(tmp_path) -> Path:
    path = tmp_path / "registry.json"
    write_registry(
        path,
        [
            entry("example-room", secret_for(1)),
            entry("other-room", secret_for(2)),
            entry("old-room", secret_for(3), revoked=True),
        ],
    )
    return path


async def disconnect_all(built: Rig) -> None:
    for device in built.devices:
        if not device.task.done():
            device.leave()
    await settle()
    for device in built.devices:
        if not device.task.done():
            device.task.cancel()


@pytest_asyncio.fixture
async def rig(registry_path):
    built = Rig(registry_path)
    yield built
    await disconnect_all(built)


@pytest_asyncio.fixture
async def lingering_rig(registry_path):
    serve = LingeringServe()
    built = Rig(registry_path, serve)
    yield built
    serve.release.set()
    await disconnect_all(built)


def failures(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if "device_auth_failed" in r.getMessage()]


# -- Authentication ------------------------------------------------------------------------


class TestAuthentication:
    async def test_a_valid_token_is_accepted_and_served_as_a_room_bound_to_its_name(self, rig):
        device = await rig.connect(f"Bearer {ROOM}")

        assert device.accepted
        assert rig.serve.calls == [("example-room", "room")]
        assert device.texts() == ["hello example-room"]
        assert rig.app.connected() == ["example-room"]

    async def test_the_scheme_is_matched_case_insensitively(self, rig):
        device = await rig.connect(f"bearer {ROOM}")

        assert device.accepted

    @pytest.mark.parametrize(
        ("authorization", "reason", "shown"),
        [
            pytest.param(None, "no_token", "?", id="no-header"),
            pytest.param("Basic abc", "no_token", "?", id="not-bearer"),
            pytest.param("Bearer nodot", "malformed", "?", id="no-name"),
            pytest.param(
                "Bearer " + token_of("example-room", secret_for(9)),
                "bad_secret",
                "example-room",
                id="wrong-secret",
            ),
            pytest.param("Bearer example-room.short", "bad_secret", "example-room", id="short"),
            pytest.param(
                "Bearer " + token_of("nobody", secret_for(1)),
                "unknown_device",
                "nobody",
                id="unknown",
            ),
            pytest.param(
                "Bearer " + token_of("old-room", secret_for(3)), "revoked", "old-room", id="revoked"
            ),
            pytest.param(
                "Bearer " + token_of("Not A Name", secret_for(1)),
                "malformed",
                "?",
                id="invalid-name",
            ),
        ],
    )
    async def test_each_refusal_closes_1008_before_accept_and_logs_its_reason(
        self, rig, caplog, authorization, reason, shown
    ):
        with caplog.at_level(logging.WARNING, logger=LISTENER_LOGGER):
            device = await rig.connect(authorization)
            await device.ended()

        assert not device.accepted
        assert device.closed_with == CLOSE_POLICY
        assert [m["type"] for m in device.sent] == ["websocket.close"]
        assert rig.serve.calls == []
        assert failures(caplog) == [
            f"device_auth_failed device={shown} peer={PEER[0]}:{PEER[1]} reason={reason}"
        ]

    async def test_no_log_line_carries_the_secret(self, rig, caplog):
        secret_part = token_of("example-room", secret_for(9)).partition(".")[2]

        with caplog.at_level(logging.DEBUG):
            device = await rig.connect("Bearer " + token_of("example-room", secret_for(9)))
            await device.ended()
            await rig.connect(f"Bearer {ROOM}")

        assert secret_part not in caplog.text
        assert ROOM.partition(".")[2] not in caplog.text


# -- Limits --------------------------------------------------------------------------------


def wrong(name: str) -> str:
    return "Bearer " + token_of(name, secret_for(99))


class TestLimits:
    async def test_five_failures_block_that_id_even_with_its_right_token(self, rig, caplog):
        for _ in range(5):
            await (await rig.connect(wrong("example-room"))).ended()

        with caplog.at_level(logging.WARNING, logger=LISTENER_LOGGER):
            right = await rig.connect(f"Bearer {ROOM}")
            await right.ended()

        assert (right.accepted, right.closed_with) == (False, CLOSE_POLICY)
        assert failures(caplog)[-1].endswith(
            "device=example-room peer=10.0.0.5:50123 reason=blocked"
        )

    async def test_another_id_is_still_accepted_while_one_is_blocked(self, rig):
        for _ in range(5):
            await (await rig.connect(wrong("example-room"))).ended()

        other = await rig.connect(f"Bearer {OTHER}")

        assert other.accepted

    async def test_the_block_ends_after_300_s(self, rig):
        for _ in range(5):
            await (await rig.connect(wrong("example-room"))).ended()
        await rig.clock.advance(300)

        right = await rig.connect(f"Bearer {ROOM}")

        assert right.accepted

    async def test_31_failures_of_made_up_ids_pause_everyone_for_60_s(self, rig, caplog):
        with caplog.at_level(logging.WARNING):
            for index in range(31):
                await (await rig.connect(wrong(f"made-up-{index}"))).ended()
            during = await rig.connect(f"Bearer {ROOM}")
            await during.ended()
        await rig.clock.advance(60)
        after = await rig.connect(f"Bearer {ROOM}")

        assert (during.accepted, during.closed_with) == (False, CLOSE_POLICY)
        assert failures(caplog)[-1].endswith("reason=paused")
        paused = [r.getMessage() for r in caplog.records if "device_auth_paused" in r.getMessage()]
        assert paused == ["device_auth_paused failures=31"]
        assert after.accepted

    async def test_a_refusal_during_the_pause_is_not_checked_or_counted(self, rig):
        for index in range(31):
            await (await rig.connect(wrong(f"made-up-{index}"))).ended()
        for _ in range(10):
            await (await rig.connect(wrong("example-room"))).ended()
        await rig.clock.advance(60)

        right = await rig.connect(f"Bearer {ROOM}")

        assert right.accepted


# -- One session per device ----------------------------------------------------------------


class TestReplacement:
    async def test_a_second_connection_replaces_the_first_with_4001(self, rig, caplog):
        first = await rig.connect(f"Bearer {ROOM}")

        with caplog.at_level(logging.INFO, logger=LISTENER_LOGGER):
            second = await rig.connect(f"Bearer {ROOM}")
            await first.ended()

        assert first.closed_with == CLOSE_REPLACED
        assert second.accepted and second.closed_with is None
        assert rig.serve.ended == ["example-room"]
        assert rig.app.connected() == ["example-room"]
        assert "device_disconnected device=example-room code=4001" in caplog.text

    async def test_the_first_session_ends_before_the_second_is_served(self, rig):
        await rig.connect(f"Bearer {ROOM}")

        await rig.connect(f"Bearer {ROOM}")

        assert rig.serve.ended == ["example-room"]
        assert len(rig.serve.calls) == 2

    async def test_a_device_leaving_is_logged_with_its_own_code(self, rig, caplog):
        device = await rig.connect(f"Bearer {ROOM}")

        with caplog.at_level(logging.INFO, logger=LISTENER_LOGGER):
            device.leave(1001)
            await device.ended()

        assert device.closed_with is None
        assert rig.app.connected() == []
        assert "device_disconnected device=example-room code=1001" in caplog.text

    async def test_messages_reach_the_session_in_order(self, rig):
        device = await rig.connect(f"Bearer {ROOM}")

        device.text("one")
        device.text("two")
        await settle()

        assert device.texts() == ["hello example-room", "echo one", "echo two"]


# -- Revocation ----------------------------------------------------------------------------


class TestRevocation:
    async def test_a_revoked_device_is_closed_4003_within_10_s(self, rig, caplog):
        device = await rig.connect(f"Bearer {ROOM}")
        other = await rig.connect(f"Bearer {OTHER}")

        write_registry(
            rig.path,
            [
                entry("example-room", secret_for(1), revoked=True),
                entry("other-room", secret_for(2)),
            ],
        )
        with caplog.at_level(logging.WARNING, logger=LISTENER_LOGGER):
            await rig.clock.advance(REVOCATION_POLL_S)
            await rig.clock.advance(REVOCATION_POLL_S)
            await device.ended()

        assert REVOCATION_POLL_S * 2 <= 10
        assert device.closed_with == CLOSE_REVOKED
        assert other.closed_with is None
        assert rig.app.connected() == ["other-room"]
        assert caplog.text.count("device_revoked_closed device=example-room") == 1

    async def test_a_removed_device_is_closed_4003(self, rig):
        device = await rig.connect(f"Bearer {ROOM}")

        write_registry(rig.path, [entry("other-room", secret_for(2))])
        await rig.clock.advance(REVOCATION_POLL_S)
        await device.ended()

        assert device.closed_with == CLOSE_REVOKED

    async def test_a_revocation_seen_first_by_an_attempt_still_closes_the_session(self, rig):
        device = await rig.connect(f"Bearer {ROOM}")
        write_registry(
            rig.path,
            [
                entry("example-room", secret_for(1), revoked=True),
                entry("other-room", secret_for(2)),
            ],
        )

        await rig.connect(f"Bearer {OTHER}")
        await device.ended()

        assert device.closed_with == CLOSE_REVOKED

    async def test_the_revoked_device_cannot_reconnect(self, rig):
        write_registry(rig.path, [entry("example-room", secret_for(1), revoked=True)])

        device = await rig.connect(f"Bearer {ROOM}")
        await device.ended()

        assert (device.accepted, device.closed_with) == (False, CLOSE_POLICY)

    async def test_with_no_device_connected_nothing_watches_the_registry(self, rig):
        device = await rig.connect(f"Bearer {ROOM}")
        device.leave()
        await device.ended()
        await settle()

        watches = [t for t in asyncio.all_tasks() if t.get_name() == "device-registry-watch"]
        assert watches == []


def closures(caplog) -> list[tuple[str, str]]:
    return [
        (r.levelname, r.getMessage())
        for r in caplog.records
        if r.getMessage().startswith("device_revoked_closed")
    ]


def rotated_registry() -> list[dict[str, object]]:
    return [entry("example-room", secret_for(4)), entry("other-room", secret_for(2))]


CHANGES = [
    pytest.param(rotated_registry(), "rotated", id="rotated"),
    pytest.param(
        [entry("example-room", secret_for(1), revoked=True), entry("other-room", secret_for(2))],
        "revoked",
        id="revoked",
    ),
    pytest.param([entry("other-room", secret_for(2))], "removed", id="removed"),
]


class TestClosingReasons:
    @pytest.mark.parametrize(("entries", "reason"), CHANGES)
    async def test_each_closing_is_logged_once_at_warning_with_its_reason(
        self, rig, caplog, entries, reason
    ):
        device = await rig.connect(f"Bearer {ROOM}")

        write_registry(rig.path, entries)
        with caplog.at_level(logging.WARNING, logger=LISTENER_LOGGER):
            await rig.clock.advance(REVOCATION_POLL_S)
            await rig.clock.advance(REVOCATION_POLL_S)
            await device.ended()

        assert device.closed_with == CLOSE_REVOKED
        assert closures(caplog) == [
            ("WARNING", f"device_revoked_closed device=example-room reason={reason}")
        ]

    async def test_an_unreadable_registry_closes_every_session_as_removed(self, rig, caplog):
        device = await rig.connect(f"Bearer {ROOM}")
        other = await rig.connect(f"Bearer {OTHER}")

        rig.path.write_text("not json", encoding="utf-8")
        with caplog.at_level(logging.WARNING, logger=LISTENER_LOGGER):
            await rig.clock.advance(REVOCATION_POLL_S)
            await device.ended()
            await other.ended()

        assert (device.closed_with, other.closed_with) == (CLOSE_REVOKED, CLOSE_REVOKED)
        assert sorted(message for _, message in closures(caplog)) == [
            "device_revoked_closed device=example-room reason=removed",
            "device_revoked_closed device=other-room reason=removed",
        ]

    async def test_the_check_runs_after_a_read_that_finds_the_file_unchanged(self, rig, caplog):
        device = await rig.connect(f"Bearer {ROOM}")
        write_registry(rig.path, rotated_registry())
        # Another reader takes the change first, so the listener's own read finds nothing new.
        assert rig.registry.reload_if_changed()

        with caplog.at_level(logging.WARNING, logger=LISTENER_LOGGER):
            await rig.clock.advance(REVOCATION_POLL_S)
            await device.ended()

        assert device.closed_with == CLOSE_REVOKED
        assert closures(caplog) == [
            ("WARNING", "device_revoked_closed device=example-room reason=rotated")
        ]


class TestRotation:
    async def test_a_session_on_the_old_secret_is_closed_4003_within_10_s(self, rig, caplog):
        device = await rig.connect(f"Bearer {ROOM}")

        write_registry(rig.path, rotated_registry())
        with caplog.at_level(logging.WARNING, logger=LISTENER_LOGGER):
            await rig.clock.advance(REVOCATION_POLL_S)
            await rig.clock.advance(REVOCATION_POLL_S)
            await device.ended()

        assert REVOCATION_POLL_S * 2 <= 10
        assert device.closed_with == CLOSE_REVOKED
        assert rig.serve.ended == ["example-room"]
        assert rig.app.connected() == []
        assert closures(caplog) == [
            ("WARNING", "device_revoked_closed device=example-room reason=rotated")
        ]

    async def test_a_new_token_arriving_before_the_poll_ends_the_old_session_and_is_served(
        self, rig, caplog
    ):
        old = await rig.connect(f"Bearer {ROOM}")
        write_registry(rig.path, rotated_registry())

        with caplog.at_level(logging.WARNING, logger=LISTENER_LOGGER):
            new = await rig.connect(f"Bearer {ROTATED_ROOM}")
            await old.ended()
            for _ in range(3):
                await rig.clock.advance(REVOCATION_POLL_S)

        assert old.closed_with == CLOSE_REVOKED
        assert new.accepted and new.closed_with is None
        assert new.texts() == ["hello example-room"]
        assert rig.serve.calls == [("example-room", "room"), ("example-room", "room")]
        assert rig.app.connected() == ["example-room"]
        assert closures(caplog) == [
            ("WARNING", "device_revoked_closed device=example-room reason=rotated")
        ]

    async def test_a_session_on_the_current_secret_survives_another_devices_change(
        self, rig, caplog
    ):
        device = await rig.connect(f"Bearer {ROOM}")
        other = await rig.connect(f"Bearer {OTHER}")

        write_registry(
            rig.path, [entry("example-room", secret_for(1)), entry("other-room", secret_for(5))]
        )
        with caplog.at_level(logging.WARNING, logger=LISTENER_LOGGER):
            await rig.clock.advance(REVOCATION_POLL_S)
            await other.ended()
            await rig.clock.advance(REVOCATION_POLL_S)

        assert other.closed_with == CLOSE_REVOKED
        assert device.closed_with is None
        assert rig.app.connected() == ["example-room"]
        assert closures(caplog) == [
            ("WARNING", "device_revoked_closed device=other-room reason=rotated")
        ]


class TestAcceptRace:
    @pytest.mark.parametrize(("entries", "reason"), CHANGES)
    async def test_a_connection_verified_before_a_change_is_closed_as_it_enters(
        self, rig, caplog, entries, reason
    ):
        other = await rig.connect(f"Bearer {OTHER}")
        late = await rig.connect_gated(f"Bearer {ROOM}")
        assert not late.accepted

        write_registry(rig.path, entries)
        with caplog.at_level(logging.WARNING, logger=LISTENER_LOGGER):
            await rig.clock.advance(REVOCATION_POLL_S)
            late.gate.set()
            await late.ended()

        assert late.closed_with == CLOSE_REVOKED
        assert other.closed_with is None
        assert rig.app.connected() == ["other-room"]
        assert closures(caplog) == [
            ("WARNING", f"device_revoked_closed device=example-room reason={reason}")
        ]

    async def test_a_late_connection_on_the_old_secret_does_not_replace_the_new_one(
        self, rig, caplog
    ):
        late = await rig.connect_gated(f"Bearer {ROOM}")
        write_registry(rig.path, rotated_registry())
        new = await rig.connect(f"Bearer {ROTATED_ROOM}")

        with caplog.at_level(logging.WARNING, logger=LISTENER_LOGGER):
            late.gate.set()
            await late.ended()
            await rig.clock.advance(REVOCATION_POLL_S)

        assert late.closed_with == CLOSE_REVOKED
        assert new.closed_with is None
        assert rig.app.connected() == ["example-room"]
        assert closures(caplog) == [
            ("WARNING", "device_revoked_closed device=example-room reason=rotated")
        ]

    async def test_a_connection_ended_during_the_replace_wait_is_never_served(
        self, lingering_rig, caplog
    ):
        rig = lingering_rig
        first = await rig.connect(f"Bearer {ROOM}")
        with caplog.at_level(logging.INFO, logger=LISTENER_LOGGER):
            second = await rig.connect(f"Bearer {ROOM}")
            # The first session still commits its turn, so the second waits to replace it.
            assert first.closed_with is None and not second.task.done()

            write_registry(rig.path, rotated_registry())
            await rig.clock.advance(REVOCATION_POLL_S)
            rig.serve.release.set()
            await first.ended()
            await second.ended()

        assert first.closed_with == CLOSE_REPLACED
        assert second.closed_with == CLOSE_REVOKED
        assert rig.serve.calls == [("example-room", "room")]
        assert second.texts() == []
        assert rig.app.connected() == []
        assert caplog.text.count("device_connected device=example-room") == 0
        assert closures(caplog) == [
            ("WARNING", "device_revoked_closed device=example-room reason=rotated")
        ]


# -- The app's surface ---------------------------------------------------------------------


async def test_the_app_has_the_device_route_only(registry_path):
    app = build_device_app(
        DeviceRegistry.load(registry_path), RecordingServe(), AuthLimiter(FakeClock()), FakeClock()
    )

    assert [route.path for route in app.routes] == [DEVICE_PATH]
