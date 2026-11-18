"""Tests for the spike listener: a fake device over loopback TLS, with a fake clock.

The server's certificate and key are made by `openssl` in pytest's temporary directory, since the
repository refuses committed keys, and the registry is written in `scripts/daily/devices.py`'s
format with the synthetic device names `example-room` and `example-old`. The fake clock stands in
for both the listener's clock and its tone pacer's sleep, so the 60 s tone and the 30 minute
memory window take no real time; a test moves the clock between the device's sends and waits for
the listener to count each send before it moves it again.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import ssl
import subprocess  # nosec B404 - runs a fixed openssl argument list, never a shell
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import pytest
from websockets.asyncio.client import ClientConnection, connect
from websockets.exceptions import InvalidStatus

from backend.realtime.protocol import (
    DOWNLINK_SAMPLE_RATE,
    UPLINK_SAMPLE_RATE,
    Direction,
    Frame,
    MsgType,
    SessionStartedMessage,
    StateCycleMessage,
    StreamStartMessage,
    VoiceDecisionWakeRejected,
    VoiceStateMessage,
    decode_frame,
    encode_frame,
    parse_server_message,
)
from scripts.daily.devices import REGISTRY_FILE, SERVER_CERT, SERVER_KEY, TLS_DIR
from scripts.room_endpoint import spike_listener
from scripts.room_endpoint.spike_listener import (
    RESULT_FILE_SCHEMA,
    ResultFile,
    SpikeListener,
    load_devices_dir,
    read_commands,
)

DEVICE = "example-room"
REVOKED = "example-old"
UPLINK_PAYLOAD = bytes(640)  # 20 ms of 16 kHz PCM16 silence
WAIT_S = 10.0


def _token(name: str, fill: int) -> tuple[str, str]:
    """A bearer token and the registry digest of its secret."""
    secret = bytes([fill]) * 32
    encoded = base64.urlsafe_b64encode(secret).rstrip(b"=").decode()
    return f"{name}.{encoded}", hashlib.sha256(secret).hexdigest()


TOKEN, TOKEN_DIGEST = _token(DEVICE, 7)
REVOKED_TOKEN, REVOKED_DIGEST = _token(REVOKED, 9)
WRONG_SECRET_TOKEN, _ = _token(DEVICE, 8)
UNKNOWN_TOKEN, _ = _token("example-none", 7)


@pytest.fixture(scope="module")
def tls_pair(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Path]:
    """A throwaway self-signed server certificate for 127.0.0.1, and its key."""
    directory = tmp_path_factory.mktemp("tls")
    key, cert = directory / SERVER_KEY, directory / SERVER_CERT
    subprocess.run(  # nosec B603 B607 - fixed argument list, no shell
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "ec",
            "-pkeyopt",
            "ec_paramgen_curve:P-256",
            "-nodes",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-days",
            "1",
            "-subj",
            "/CN=spike-test",
            "-addext",
            "subjectAltName=IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
    return cert, key


@pytest.fixture
def devices_dir(tmp_path: Path, tls_pair: tuple[Path, Path]) -> Path:
    """A devices dir as `devices.py` lays it out: `registry.json` and `tls/`."""
    directory = tmp_path / "devices"
    (directory / TLS_DIR).mkdir(parents=True)
    cert, key = tls_pair
    (directory / TLS_DIR / SERVER_CERT).write_bytes(cert.read_bytes())
    (directory / TLS_DIR / SERVER_KEY).write_bytes(key.read_bytes())
    registry = {
        "version": 1,
        "devices": [
            {
                "name": DEVICE,
                "area": "Example",
                "token_sha256": TOKEN_DIGEST,
                "revoked": False,
                "created": "2026-11-17T00:00:00+00:00",
            },
            {
                "name": REVOKED,
                "area": "Example",
                "token_sha256": REVOKED_DIGEST,
                "revoked": True,
                "created": "2026-11-17T00:00:00+00:00",
            },
        ],
    }
    (directory / REGISTRY_FILE).write_text(json.dumps(registry), encoding="utf-8")
    return directory


@pytest.fixture
def client_ssl(tls_pair: tuple[Path, Path]) -> ssl.SSLContext:
    """What the device pins: the server's own certificate as the trust anchor."""
    context = ssl.create_default_context(cafile=str(tls_pair[0]))
    return context


class FakeClock:
    """The listener's clock and the tone pacer's sleep: a sleep moves the clock, not time."""

    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += max(seconds, 0.0)
        await asyncio.sleep(0)


async def until(predicate: Callable[[], bool]) -> None:
    """Wait in real time until the listener has handled what the device sent."""
    deadline = asyncio.get_running_loop().time() + WAIT_S
    while not predicate():
        if asyncio.get_running_loop().time() > deadline:
            raise AssertionError("the listener did not get there in time")
        await asyncio.sleep(0.005)


@dataclass
class Running:
    listener: SpikeListener
    port: int
    out: Path
    clock: FakeClock


@asynccontextmanager
async def running(
    devices_dir: Path, out: Path, clock: FakeClock, *, fresh: bool = False
) -> AsyncIterator[Running]:
    """A listener on a free loopback port, stopped (and its result written) on exit."""
    registry, server_ssl = load_devices_dir(devices_dir)
    listener = SpikeListener(
        registry=registry,
        ssl_context=server_ssl,
        result=ResultFile.open(out, fresh=fresh),
        clock=clock,
        sleep=clock.sleep,
    )
    port = await listener.start("127.0.0.1", 0)
    try:
        yield Running(listener, port, out, clock)
    finally:
        await listener.stop()


class FakeDevice:
    """The device's side of `/v2/device`: one `seq` sequence per connection, as item 4 says."""

    def __init__(self, ws: ClientConnection) -> None:
        self.ws = ws
        self.seq = 0

    @classmethod
    @asynccontextmanager
    async def connected(
        cls, port: int, context: ssl.SSLContext, token: str = TOKEN
    ) -> AsyncIterator[FakeDevice]:
        async with connect(
            f"wss://127.0.0.1:{port}/v2/device",
            ssl=context,
            additional_headers={"Authorization": f"Bearer {token}"},
        ) as ws:
            device = cls(ws)
            started = await device.recv()
            assert isinstance(started, SessionStartedMessage)
            yield device

    async def send_json(self, payload: Mapping[str, object]) -> None:
        await self.ws.send(json.dumps(payload))

    async def voice_start(self, req: int = 1, muted: bool = False) -> VoiceStateMessage:
        await self.send_json(
            {
                "type": "voice_start",
                "req": req,
                "mode": "wake",
                "sample_rate": 16000,
                "frame_ms": 20,
                "aec": "device",
                "playback": True,
                "endpoint": "room",
                "spotter": "endpoint",
                "muted": muted,
            }
        )
        answer = await self.recv()
        assert isinstance(answer, VoiceStateMessage)
        return answer

    async def wake(self, backlog_ms: int = 5000, score: float = 0.9) -> None:
        await self.send_json(
            {"type": "wake", "keyword": "mist", "score": score, "backlog_ms": backlog_ms}
        )

    async def frames(self, count: int, *, skip_before: int = 0) -> None:
        self.seq += skip_before
        for _ in range(count):
            frame = Frame(MsgType.MIC_AUDIO, 0, self.seq, UPLINK_SAMPLE_RATE, UPLINK_PAYLOAD)
            await self.ws.send(encode_frame(frame))
            self.seq += 1

    async def stats(self, *, heap: int = 60000, psram: int = 4_000_000, underruns: int = 0) -> None:
        await self.send_json(
            {
                "type": "device_stats",
                "heap_free_min": heap,
                "psram_free": psram,
                "mic_overruns": 0,
                "ring_overruns": 0,
                "playback_underruns": underruns,
                "uptime_s": 100,
                "reconnects": 0,
            }
        )

    async def recv(self) -> object:
        message = await asyncio.wait_for(self.ws.recv(), WAIT_S)
        if isinstance(message, bytes):
            return decode_frame(message, direction=Direction.DOWN)
        return parse_server_message(message)


def _read(out: Path) -> dict:
    return json.loads(out.read_text(encoding="utf-8"))


# -- Serving and authentication ----------------------------------------------------------------


@pytest.mark.asyncio
async def test_serves_v2_device_from_the_devices_dir_and_answers_voice_start(
    tmp_path: Path, devices_dir: Path, client_ssl: ssl.SSLContext
) -> None:
    async with (
        running(devices_dir, tmp_path / "out" / "spike_result.json", FakeClock()) as run,
        FakeDevice.connected(run.port, client_ssl) as device,
    ):
        answer = await device.voice_start(req=3)
    assert answer.cause == "answer"
    assert answer.req == 3
    assert answer.active is True
    assert answer.mode == "wake"
    assert answer.refused is None
    assert answer.muted is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "header",
    [
        f"Bearer {UNKNOWN_TOKEN}",
        f"Bearer {WRONG_SECRET_TOKEN}",
        f"Bearer {REVOKED_TOKEN}",
        TOKEN,
        "",
    ],
    ids=["unregistered", "wrong-secret", "revoked", "no-scheme", "no-header"],
)
async def test_refuses_any_token_but_a_registered_unrevoked_one(
    tmp_path: Path, devices_dir: Path, client_ssl: ssl.SSLContext, header: str
) -> None:
    out = tmp_path / "out" / "spike_result.json"
    async with running(devices_dir, out, FakeClock()) as run:
        headers = {"Authorization": header} if header else {}
        with pytest.raises(InvalidStatus) as refused:
            async with connect(
                f"wss://127.0.0.1:{run.port}/v2/device", ssl=client_ssl, additional_headers=headers
            ):
                pass
        snapshot = run.listener.snapshot()
    assert refused.value.response.status_code == 403
    assert snapshot["auth_refused"] == 1
    assert snapshot["connections"] == 0
    secret = header.rpartition(".")[2]
    if secret:
        assert secret not in out.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_serves_only_the_device_path(
    tmp_path: Path, devices_dir: Path, client_ssl: ssl.SSLContext
) -> None:
    async with running(devices_dir, tmp_path / "spike_result.json", FakeClock()) as run:
        with pytest.raises(InvalidStatus) as refused:
            async with connect(
                f"wss://127.0.0.1:{run.port}/ws",
                ssl=client_ssl,
                additional_headers={"Authorization": f"Bearer {TOKEN}"},
            ):
                pass
    assert refused.value.response.status_code == 404


@pytest.mark.asyncio
async def test_an_invalid_voice_start_is_answered_inactive(
    tmp_path: Path, devices_dir: Path, client_ssl: ssl.SSLContext
) -> None:
    async with (
        running(devices_dir, tmp_path / "spike_result.json", FakeClock()) as run,
        FakeDevice.connected(run.port, client_ssl) as device,
    ):
        await device.send_json({"type": "voice_start", "req": 5, "mode": "push"})
        answer = await device.recv()
        snapshot = run.listener.snapshot()
    assert isinstance(answer, VoiceStateMessage)
    assert (answer.req, answer.active, answer.refused) == (5, False, "invalid")
    assert snapshot["protocol_errors"] == 1
    assert snapshot["first_voice_start_ms"] is None


# -- Operator commands -------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_sends_state_cycle_and_wake_rejected_on_command(
    tmp_path: Path, devices_dir: Path, client_ssl: ssl.SSLContext
) -> None:
    async with (
        running(devices_dir, tmp_path / "spike_result.json", FakeClock()) as run,
        FakeDevice.connected(run.port, client_ssl) as device,
    ):
        await device.voice_start()
        await device.wake(score=0.75)
        await until(lambda: run.listener.recorder.wakes_opened == 1)
        reader = asyncio.StreamReader()
        reader.feed_data(b"state speak\nreject\nreject self_media\nstate bogus\nfly\n")
        reader.feed_eof()
        replies: list[str] = []
        await read_commands(reader, run.listener, replies.append)
        received = [await device.recv() for _ in range(3)]
        snapshot = run.listener.snapshot()
    speak, rejected, media = received
    assert isinstance(speak, StateCycleMessage)
    assert speak.state == "speak"
    assert isinstance(rejected, VoiceDecisionWakeRejected)
    assert rejected.reason == "not_addressed"
    assert rejected.detail.source == "endpoint"
    assert rejected.detail.score == 0.75
    assert isinstance(media, VoiceDecisionWakeRejected)
    assert media.reason == "self_media"
    assert replies[3].startswith("refused")
    assert replies[4].startswith("unknown command")
    assert snapshot["wakes"][0]["closed_by"] == "state_cycle speak"


@pytest.mark.asyncio
async def test_a_command_with_no_device_connected_is_refused(
    tmp_path: Path, devices_dir: Path
) -> None:
    async with running(devices_dir, tmp_path / "spike_result.json", FakeClock()) as run:
        reply = await run.listener.command("state idle")
        quit_reply = await run.listener.command("quit")
        assert run.listener.stop_requested.is_set()
    assert reply == "refused: no device connected"
    assert quit_reply == "stopping"


# -- The tone ----------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streams_a_generated_24_khz_tone_as_tagged_frames_for_60_s(
    tmp_path: Path, devices_dir: Path, client_ssl: ssl.SSLContext
) -> None:
    clock = FakeClock(10.0)
    async with (
        running(devices_dir, tmp_path / "spike_result.json", clock) as run,
        FakeDevice.connected(run.port, client_ssl) as device,
    ):
        await device.voice_start()
        assert (await run.listener.command("tone")).startswith("tone started")
        start = await device.recv()
        chunks: list[Frame] = []
        while True:
            received = await device.recv()
            assert isinstance(received, Frame)
            if received.msg_type is MsgType.AUDIO_COMPLETE:
                complete = received
                break
            chunks.append(received)
        await until(lambda: not run.listener.tone_playing)
        snapshot = run.listener.snapshot()
    assert isinstance(start, StreamStartMessage)
    assert start.audio_tag is not None
    assert len(chunks) == 1500
    assert {chunk.msg_type for chunk in chunks} == {MsgType.AUDIO_CHUNK}
    assert {chunk.turn_tag for chunk in chunks} == {start.audio_tag}
    assert {chunk.sample_rate for chunk in chunks} == {DOWNLINK_SAMPLE_RATE}
    assert {len(chunk.payload) for chunk in chunks} == {1920}  # 40 ms of 24 kHz PCM16
    assert [chunk.seq for chunk in chunks] == list(range(1500))
    assert any(chunks[0].payload)
    assert complete.turn_tag == start.audio_tag
    assert clock.now >= 10.0 + 59.7  # paced over the tone's 60 s, on the fake clock
    playback = snapshot["playbacks"][0]
    assert playback["frames_sent"] == 1500
    assert playback["completed"] is True
    assert playback["audio_tag"] == start.audio_tag


async def _play_with_uplink(
    run: Running, device: FakeDevice, *, skip_at: int | None = None, underruns_after: int = 0
) -> None:
    """Wake, then hold the uplink open through a tone: two 20 ms frames per 40 ms tone frame."""
    await device.voice_start()
    await device.wake()
    await until(lambda: run.listener.recorder.wakes_opened == 1)
    await device.frames(250)
    await device.stats(underruns=2)
    await until(lambda: run.listener.recorder.stats_samples == 1)
    await run.listener.command("tone")
    assert isinstance(await device.recv(), StreamStartMessage)
    sent = 0
    while True:
        frame = await device.recv()
        assert isinstance(frame, Frame)
        if frame.msg_type is MsgType.AUDIO_COMPLETE:
            break
        await device.frames(2, skip_before=1 if sent == skip_at else 0)
        sent += 2
    await until(lambda: run.listener.recorder.uplink_frames == 250 + sent)
    run.clock.now += 61.0
    await device.stats(underruns=2 + underruns_after)
    await until(lambda: run.listener.recorder.stats_samples == 2)


@pytest.mark.asyncio
async def test_records_60_s_of_live_uplink_during_playback(
    tmp_path: Path, devices_dir: Path, client_ssl: ssl.SSLContext
) -> None:
    async with (
        running(devices_dir, tmp_path / "spike_result.json", FakeClock(50.0)) as run,
        FakeDevice.connected(run.port, client_ssl) as device,
    ):
        await _play_with_uplink(run, device)
        snapshot = run.listener.snapshot()
    playback = snapshot["playbacks"][0]
    assert playback["live_frames"] == 3000
    assert playback["seq_gaps"] == 0
    assert playback["underruns"] == 0
    assert playback["pass"] is True


@pytest.mark.asyncio
async def test_a_seq_gap_or_an_underrun_during_playback_fails_it(
    tmp_path: Path, devices_dir: Path, client_ssl: ssl.SSLContext
) -> None:
    async with (
        running(devices_dir, tmp_path / "spike_result.json", FakeClock(50.0)) as run,
        FakeDevice.connected(run.port, client_ssl) as device,
    ):
        await _play_with_uplink(run, device, skip_at=100, underruns_after=3)
        snapshot = run.listener.snapshot()
    playback = snapshot["playbacks"][0]
    assert playback["seq_gaps"] == 1
    assert playback["missing_frames"] == 1
    assert playback["underruns"] == 3
    assert playback["pass"] is False


# -- Wakes -------------------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_records_a_wake_its_backlog_timing_and_live_seq(
    tmp_path: Path, devices_dir: Path, client_ssl: ssl.SSLContext
) -> None:
    clock = FakeClock(100.0)
    async with (
        running(devices_dir, tmp_path / "spike_result.json", clock) as run,
        FakeDevice.connected(run.port, client_ssl) as device,
    ):
        await device.voice_start()
        await device.frames(0)
        await device.wake(backlog_ms=5000)
        await until(lambda: run.listener.recorder.wakes_opened == 1)
        clock.now = 100.4
        await device.frames(250)
        await until(lambda: run.listener.recorder.uplink_frames == 250)
        clock.now = 101.0
        await device.frames(10)
        await device.frames(5, skip_before=3)
        await until(lambda: run.listener.recorder.uplink_frames == 265)
        snapshot = run.listener.snapshot()
    wake = snapshot["wakes"][0]
    assert wake["keyword"] == "mist"
    assert wake["backlog_ms"] == 5000
    assert wake["backlog_frames_expected"] == 250
    assert wake["backlog_frames"] == 250
    assert wake["backlog_seq_contiguous"] is True
    assert wake["wake_to_last_backlog_ms"] == 400
    assert wake["live_frames"] == 15
    assert wake["live_seq_gaps"] == 1
    assert wake["live_missing_frames"] == 3
    assert wake["pass"] is True
    assert wake["failures"] == []


@pytest.mark.asyncio
async def test_a_short_broken_or_late_backlog_fails_the_wake(
    tmp_path: Path, devices_dir: Path, client_ssl: ssl.SSLContext
) -> None:
    clock = FakeClock(100.0)
    async with (
        running(devices_dir, tmp_path / "spike_result.json", clock) as run,
        FakeDevice.connected(run.port, client_ssl) as device,
    ):
        await device.voice_start()
        await device.wake(backlog_ms=4000)
        await until(lambda: run.listener.recorder.wakes_opened == 1)
        clock.now = 101.2
        await device.frames(100)
        await device.frames(100, skip_before=1)
        await until(lambda: run.listener.recorder.uplink_frames == 200)
        await device.wake(backlog_ms=5000)
        await until(lambda: run.listener.recorder.wakes_opened == 2)
        await device.frames(120)
        await until(lambda: run.listener.recorder.uplink_frames == 320)
        snapshot = run.listener.snapshot()
    first, second = snapshot["wakes"]
    assert first["backlog_frames"] == 200
    assert first["backlog_seq_contiguous"] is False
    assert first["wake_to_last_backlog_ms"] == 1200
    assert first["pass"] is False
    assert set(first["failures"]) == {"backlog_ms", "backlog_seq", "backlog_late"}
    assert second["backlog_frames"] == 120
    assert second["wake_to_last_backlog_ms"] is None
    assert second["failures"] == ["backlog_incomplete"]


@pytest.mark.asyncio
async def test_counts_uplink_frames_sent_without_a_wake(
    tmp_path: Path, devices_dir: Path, client_ssl: ssl.SSLContext
) -> None:
    async with (
        running(devices_dir, tmp_path / "spike_result.json", FakeClock()) as run,
        FakeDevice.connected(run.port, client_ssl) as device,
    ):
        await device.voice_start()
        await device.frames(4)
        await until(lambda: run.listener.recorder.uplink_frames == 4)
        snapshot = run.listener.snapshot()
    assert snapshot["frames_without_wake"] == 4
    assert snapshot["wakes"] == []


@pytest.mark.asyncio
async def test_records_the_device_stats_series(
    tmp_path: Path, devices_dir: Path, client_ssl: ssl.SSLContext
) -> None:
    clock = FakeClock(0.0)
    async with (
        running(devices_dir, tmp_path / "spike_result.json", clock) as run,
        FakeDevice.connected(run.port, client_ssl) as device,
    ):
        await device.voice_start()
        for index, heap in enumerate((50000, 41000, 45000)):
            clock.now = 10.0 * (index + 1)
            await device.stats(heap=heap, psram=3_000_000 - index)
            await until(lambda n=index + 1: run.listener.recorder.stats_samples == n)
        snapshot = run.listener.snapshot()
    series = snapshot["device_stats"]
    assert [sample["t_s"] for sample in series] == [10.0, 20.0, 30.0]
    assert [sample["heap_free_min"] for sample in series] == [50000, 41000, 45000]
    assert series[2]["psram_free"] == 2_999_998
    assert set(series[0]) == {
        "t_s",
        "heap_free_min",
        "psram_free",
        "mic_overruns",
        "ring_overruns",
        "playback_underruns",
        "uptime_s",
        "reconnects",
    }
    assert snapshot["memory"]["heap_free_min"] == 41000
    assert snapshot["memory"]["window_s"] == 20.0


# -- The result file ---------------------------------------------------------------------------


async def _full_run(run: Running, client_ssl: ssl.SSLContext, *, heap: int = 60000) -> None:
    """Twenty passing wakes, one tone with live uplink, and stats over 30 minutes."""
    async with FakeDevice.connected(run.port, client_ssl) as device:
        run.clock.now += 2.0
        await device.voice_start()
        await device.stats(heap=heap)
        await until(lambda: run.listener.recorder.stats_samples == 1)
        for index in range(19):
            await device.wake()
            await until(lambda n=index + 1: run.listener.recorder.wakes_opened == n)
            run.clock.now += 0.5
            await device.frames(250)
            await until(lambda n=index + 1: run.listener.recorder.uplink_frames == 250 * n)
            await run.listener.command("state idle")
            assert isinstance(await device.recv(), StateCycleMessage)
            run.clock.now += 10.0
        await device.wake()
        await until(lambda: run.listener.recorder.wakes_opened == 20)
        run.clock.now += 0.5
        await device.frames(250)
        await until(lambda: run.listener.recorder.uplink_frames == 250 * 20)
        await run.listener.command("tone")
        assert isinstance(await device.recv(), StreamStartMessage)
        while True:
            frame = await device.recv()
            assert isinstance(frame, Frame)
            if frame.msg_type is MsgType.AUDIO_COMPLETE:
                break
            await device.frames(2)
        await until(lambda: run.listener.recorder.uplink_frames == 250 * 20 + 3000)
        run.clock.now += 61.0
        await device.stats(heap=heap)
        await until(lambda: run.listener.recorder.stats_samples == 2)
        run.clock.now += 1800.0
        await device.stats(heap=heap)
        await until(lambda: run.listener.recorder.stats_samples == 3)


async def _restart_run(run: Running, client_ssl: ssl.SSLContext, after_s: float) -> None:
    async with FakeDevice.connected(run.port, client_ssl) as device:
        run.clock.now += after_s
        await device.voice_start()
        await until(lambda: run.listener.snapshot()["first_voice_start_ms"] is not None)


@pytest.mark.asyncio
async def test_writes_spike_result_with_each_g1_check_passing(
    tmp_path: Path, devices_dir: Path, client_ssl: ssl.SSLContext
) -> None:
    out = tmp_path / "result" / "spike_result.json"
    clock = FakeClock(1000.0)
    async with running(devices_dir, out, clock) as run:
        await _full_run(run, client_ssl)
    clock.now += 300.0
    async with running(devices_dir, out, clock) as run:
        await _restart_run(run, client_ssl, after_s=4.5)
    result = _read(out)
    assert result["schema"] == RESULT_FILE_SCHEMA
    checks = result["g1"]["checks"]
    assert set(checks) == {
        "heap_free_min",
        "psram_free_min",
        "wakes",
        "uplink_during_playback",
        "reconnection",
    }
    assert {name: check["pass"] for name, check in checks.items()} == dict.fromkeys(checks, True)
    assert result["g1"]["pass"] is True
    assert checks["wakes"]["recorded"] == 20
    assert checks["wakes"]["passing"] == 20
    assert checks["reconnection"]["restarts_ms"] == [4500]
    assert "voice_start" in checks["reconnection"]["measured_as"]
    assert checks["heap_free_min"]["observed_b"] == 60000
    assert result["g3"]["frames_without_wake"] == 0
    assert result["g3"]["pass"] is True
    assert len(result["runs"]) == 2
    assert result["runs"][0]["first_voice_start_ms"] == 2000


@pytest.mark.asyncio
async def test_each_g1_check_fails_on_its_own_miss(
    tmp_path: Path, devices_dir: Path, client_ssl: ssl.SSLContext
) -> None:
    out = tmp_path / "result" / "spike_result.json"
    clock = FakeClock(1000.0)
    async with running(devices_dir, out, clock) as run:
        await _full_run(run, client_ssl, heap=29999)
    clock.now += 300.0
    async with running(devices_dir, out, clock) as run:
        await _restart_run(run, client_ssl, after_s=15.5)
    checks = _read(out)["g1"]["checks"]
    assert checks["heap_free_min"]["pass"] is False
    assert checks["psram_free_min"]["pass"] is True
    assert checks["reconnection"]["pass"] is False
    assert checks["reconnection"]["restarts_ms"] == [15500]
    assert _read(out)["g1"]["pass"] is False


@pytest.mark.asyncio
async def test_a_single_short_run_fails_every_g1_check(
    tmp_path: Path, devices_dir: Path, client_ssl: ssl.SSLContext
) -> None:
    out = tmp_path / "spike_result.json"
    async with (
        running(devices_dir, out, FakeClock()) as run,
        FakeDevice.connected(run.port, client_ssl) as device,
    ):
        await device.voice_start()
        await device.stats()
        await until(lambda: run.listener.recorder.stats_samples == 1)
    result = _read(out)
    assert {name: check["pass"] for name, check in result["g1"]["checks"].items()} == {
        "heap_free_min": False,
        "psram_free_min": False,
        "wakes": False,
        "uplink_during_playback": False,
        "reconnection": False,
    }
    assert result["g1"]["checks"]["reconnection"]["restarts_ms"] == []


@pytest.mark.asyncio
async def test_writes_no_file_but_the_json_and_keeps_no_audio(
    tmp_path: Path, devices_dir: Path, client_ssl: ssl.SSLContext
) -> None:
    out_dir = tmp_path / "result"
    out = out_dir / "spike_result.json"
    devices_before = sorted(path.relative_to(devices_dir) for path in devices_dir.rglob("*"))
    async with running(devices_dir, out, FakeClock(5.0)) as run:
        async with FakeDevice.connected(run.port, client_ssl) as device:
            await _play_with_uplink(run, device)
            await device.frames(20)
            await until(lambda: run.listener.recorder.uplink_frames == 3270)
        run.listener.write_result()
    assert [path.name for path in out_dir.rglob("*")] == ["spike_result.json"]
    assert sorted(path.relative_to(devices_dir) for path in devices_dir.rglob("*")) == (
        devices_before
    )
    text = out.read_text(encoding="utf-8")
    assert len(text) < 20000  # counts and timings, never the 3270 frames' 2 MB of samples
    assert "payload" not in text


@pytest.mark.asyncio
async def test_fresh_discards_earlier_runs(tmp_path: Path, devices_dir: Path) -> None:
    out = tmp_path / "spike_result.json"
    async with running(devices_dir, out, FakeClock()):
        pass
    async with running(devices_dir, out, FakeClock()):
        pass
    assert len(_read(out)["runs"]) == 2
    async with running(devices_dir, out, FakeClock(), fresh=True):
        pass
    assert len(_read(out)["runs"]) == 1


# -- The command line --------------------------------------------------------------------------


def test_cli_defaults_bind_all_interfaces_on_8443() -> None:
    args = spike_listener.build_parser().parse_args(["--devices", "d", "--out", "o.json"])
    assert (args.host, args.port, args.fresh) == ("0.0.0.0", 8443, False)


def test_cli_refuses_a_devices_dir_without_the_registry(tmp_path: Path) -> None:
    code = spike_listener.main(
        ["--devices", str(tmp_path / "missing"), "--out", str(tmp_path / "spike_result.json")]
    )
    assert code == 2
    assert not (tmp_path / "spike_result.json").exists()


def test_cli_refuses_an_out_file_that_is_not_a_spike_result(
    tmp_path: Path, devices_dir: Path
) -> None:
    out = tmp_path / "spike_result.json"
    out.write_text('{"something": "else"}', encoding="utf-8")
    code = spike_listener.main(["--devices", str(devices_dir), "--out", str(out)])
    assert code == 2
    assert out.read_text(encoding="utf-8") == '{"something": "else"}'
