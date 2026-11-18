"""The rig the end-to-end WebSocket voice scenarios share (T16).

`voice_server` runs the real `backend.server` lifespan behind Starlette's `TestClient`, so every
scenario goes through `/ws` as a client would. `build_mist_runtime` is replaced by `VoiceStack`,
which builds a real `MistRuntime` and `SessionServices` over fakes: the energy VAD, a scripted
transcriber, a scripted name spotter, the fake synthesizer, inline lanes and a `FakeResponder`.
Every builder that would open the vault, Neo4j, the model stack or the network returns None or a
fake, and the extraction service's health client answers from an in-process transport.

Time is the session clock: a `FakeClock` the rig advances 20 ms per uplink frame, so endpointing,
partial cadence and the controller's timers run without a wall-clock wait. The wake arbiter's
window runs on a `FakeCallLater` a scenario fires by hand. The lifespan's own periodic tasks
(heartbeat, health probes) still run on the loop; `VoiceClient.voice` leaves their messages out.

`TestClient` is entered with `with`, so its one portal (one loop) serves the lifespan and every
connection: the registry, the arbiter and the sink are shared the way they are in production.
Each frame is sent alone and the loop settled before the next, because ingress runs at most one
name-spotter job per session and skips a hop that finds one running.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypeVar

import anyio
import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession

from backend import server
from backend.realtime.budget import BargeInConfig, EngineOption, VoiceSettings
from backend.realtime.power import PowerWatcher
from backend.realtime.protocol import (
    Direction,
    Frame,
    MsgType,
    PowerLevel,
    decode_frame,
    encode_frame,
)
from backend.realtime.registry import SessionRegistry
from backend.realtime.runtime import (
    MIB,
    EngineLoaders,
    GpuEngines,
    MistRuntime,
    load_speech_engines,
)
from backend.realtime.session import InferenceLanes, SessionServices, SpeechEngines
from backend.realtime.speech import NameHit
from tests.fakes.scripted_audio import AudioPart, scripted_audio, to_pcm16
from tests.fakes.speech import (
    EnergyVad,
    FakeSynthesizer,
    ScriptedNameSpotter,
    ScriptedTranscriber,
    ScriptedTurnEnd,
)

from .turn_fakes import (
    TEST_PLAYBACK,
    TEST_SEGMENTER,
    FakeCallLater,
    FakeClock,
    FakeResponder,
    IdCounter,
    InlineLane,
    floor_placement,
    settle,
)

FRAME_MS = 20
FRAME_SAMPLES = 320

_E = TypeVar("_E")

PROCESS_WIDE = frozenset(
    {
        "heartbeat",
        "health_status",
        "extraction_status",
        "system_status",
        "capabilities_status",
        "alarm_state",
    }
)
"""Messages the lifespan's own tasks and the capability host send every session; not voice."""


# -- The process around the runtime --------------------------------------------------------------


class FakeLLMProvider:
    """The base provider the health probe and the model tier read; never asked for a token."""

    async def health_check(self) -> bool:
        return True


class FakeHandler:
    """The handler attributes `lifespan` reads; no event store, so no extraction backlog."""

    def __init__(self) -> None:
        self.event_store = None
        self._tool_usage_tracker = None
        self._provider = None

    def attach_capability_host(self, host: object, *, tiers: object = None) -> None:
        return None

    async def aclose(self) -> None:
        return None


class FakeScheduler:
    async def start(self) -> bool:
        return False

    async def stop(self) -> None:
        return None


class FakePersona:
    """`PersonaRegister` in place: no profile file is read."""

    def __init__(self, profile: str) -> None:
        self.profile = profile

    @classmethod
    def load(cls, root: Path, profile: str) -> FakePersona:
        return cls(profile)

    def prompt_line(self) -> str:
        return "register"

    def address_form(self) -> str:
        return "you"


class FakeCuda:
    """`torch.cuda` as the GPU control reads it: 2 GiB in the cache until it is emptied."""

    def __init__(self) -> None:
        self.reserved = 2048 * MIB

    def is_available(self) -> bool:
        return True

    def memory_reserved(self) -> int:
        return self.reserved

    def memory_allocated(self) -> int:
        return self.reserved

    def max_memory_reserved(self) -> int:
        return self.reserved

    def reset_peak_memory_stats(self) -> None:
        return None

    def empty_cache(self) -> None:
        self.reserved = 0


class FakeTorch:
    def __init__(self) -> None:
        self.cuda = FakeCuda()


def _named(engine: _E, model_id: str) -> _E:
    """The chain walk records each engine's `model_id`; the VAD and EOT fakes carry none."""
    engine.model_id = model_id  # type: ignore[attr-defined]
    return engine


async def _never(seconds: float) -> None:
    """The power watcher's sleep: it never wakes, so a scenario reads the level by hand."""
    await asyncio.Event().wait()


def _extraction_health(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"status": "ok", "llm_reachable": True, "uptime_s": 1.0})


# -- The runtime ---------------------------------------------------------------------------------


@dataclass
class VoiceStack:
    """`build_mist_runtime` in place: a real `MistRuntime` over the fakes a scenario scripts.

    `transcripts` are taken one per opened utterance, across every session, in order; `kws`
    False builds no name spotter, as an unset `MIST_VOICE_KWS_MODEL` does.
    """

    transcripts: Sequence[str] = ()
    eot: Sequence[float] = (0.95,)
    kws: bool = True
    stt_devices: tuple[str, ...] | None = None
    stt_load_errors: list[BaseException] = field(default_factory=list)
    markers: dict[bytes, float] = field(default_factory=dict)
    settings: VoiceSettings = field(
        default_factory=lambda: VoiceSettings(
            playback=TEST_PLAYBACK,
            segmenter=TEST_SEGMENTER,
            barge_in=BargeInConfig(arm_delay_ms=0),
        )
    )
    clock: FakeClock = field(default_factory=FakeClock)
    ids: IdCounter = field(default_factory=IdCounter)
    call_later: FakeCallLater = field(default_factory=FakeCallLater)
    handler: FakeHandler = field(default_factory=FakeHandler)
    responder: FakeResponder = field(init=False)
    transcriber: ScriptedTranscriber = field(init=False)
    spotter: ScriptedNameSpotter | None = field(init=False)
    synthesizer: FakeSynthesizer = field(init=False)
    runtime: MistRuntime | None = None
    torch: FakeTorch = field(default_factory=lambda: FakeTorch())
    retry_event: asyncio.Event | None = None

    def __post_init__(self) -> None:
        self.responder = FakeResponder(clock=self.clock)
        self.transcriber = ScriptedTranscriber(list(self.transcripts), ms_per_word=20)
        self.spotter = ScriptedNameSpotter(markers=self.markers) if self.kws else None
        self.synthesizer = FakeSynthesizer()

    def __call__(self, config: object, **kwargs: object) -> MistRuntime:
        gate = kwargs["power_gate"]
        gpu = None
        if self.stt_devices is None:
            engines = SpeechEngines(
                vad=EnergyVad(),
                turn_end=ScriptedTurnEnd(self.eot),
                transcriber=self.transcriber,
                synthesizer=self.synthesizer,
                spotter=self.spotter,
            )
        else:
            engines, gpu = self._placed_engines(gate.level())  # type: ignore[attr-defined]
        lanes = InferenceLanes(
            vad=InlineLane(), eot=InlineLane(), stt=InlineLane(), tts=InlineLane(), kws=InlineLane()
        )
        services = SessionServices(
            responder=self.responder,  # type: ignore[arg-type]
            engines=engines,
            lanes=lanes,
            settings=self.settings,
            placement=floor_placement,
            power_level=gate.level,  # type: ignore[attr-defined]
            clock_factory=lambda: self.clock,
            utterance_ids=lambda: self.ids,
            power_gate=gate,  # type: ignore[arg-type]
            alarm_ringing=kwargs["alarm_ringing"],  # type: ignore[arg-type]
            on_capability_message=kwargs["on_capability_message"],  # type: ignore[arg-type]
        )
        self.runtime = MistRuntime(
            llm_provider=kwargs["llm_provider"],
            conversation_handler=self.handler,
            responder=self.responder,  # type: ignore[arg-type]
            engines=engines,
            lanes=lanes,
            settings=self.settings,
            services=services,
            agent_available=True,
            gpu=gpu,
        )
        return self.runtime

    # Gaming mode: engines placed by the runtime's own chain walk, swapped by `GpuEngines`.

    def _placed_engines(self, level: PowerLevel) -> tuple[SpeechEngines, GpuEngines]:
        """STT on `stt_devices` in chain order, TTS on the CPU (its only recorded arm)."""
        assert self.stt_devices is not None
        stt_engine = {"cuda": "nemotron-160", "cpu": "parakeet-cpu"}
        placement = dataclasses.replace(
            self.settings.placement,
            stt_chain=tuple(EngineOption(stt_engine[d], d) for d in self.stt_devices),
            tts_chain=(EngineOption("pocket-clone", "cpu"),),
            kws_model="scripted-kws" if self.kws else None,
        )
        self.settings = dataclasses.replace(self.settings, placement=placement)
        loaders = EngineLoaders(
            chains={
                "nemotron-160": self._load_stt,
                "parakeet-cpu": self._load_stt,
                "pocket-clone": lambda settings, device: self.synthesizer,
            },
            vad=lambda placement: _named(EnergyVad(), "energy-vad"),
            eot=lambda placement, endpointing: _named(ScriptedTurnEnd(self.eot), "scripted-eot"),
            kws=lambda placement: self.spotter,  # type: ignore[arg-type, return-value]
        )
        engines = load_speech_engines(self.settings, loaders, power_level=level)
        gpu = GpuEngines(
            engines,
            self.settings,
            loaders,
            torch_loader=lambda: self.torch,
            sleep=self._retry_sleep,
        )
        return engines, gpu

    def _load_stt(self, settings: VoiceSettings, device: str) -> ScriptedTranscriber:
        """A fresh engine per load, continuing the one script and its streams."""
        if device == "cuda" and self.stt_load_errors:
            raise self.stt_load_errors.pop(0)
        loaded = ScriptedTranscriber([], ms_per_word=self.transcriber.ms_per_word)
        loaded.script = self.transcriber.script
        loaded.streams = self.transcriber.streams
        self.transcriber = loaded
        return loaded

    async def _retry_sleep(self, delay: float) -> None:
        """`GpuEngines`' wait between reload retries: until `retry_now`."""
        self.retry_event = asyncio.Event()
        await self.retry_event.wait()

    def retry_now(self) -> None:
        """End the reload retry's wait; runs on the loop."""
        if self.retry_event is not None:
            self.retry_event.set()


# -- A connection ----------------------------------------------------------------------------------


def voice_start(req: int, **overrides: object) -> dict[str, object]:
    """A desktop `voice_start`: wake mode, the server's spotter, a headset."""
    message: dict[str, object] = {
        "type": "voice_start",
        "req": req,
        "mode": "wake",
        "sample_rate": 16000,
        "frame_ms": 20,
        "aec": "headset",
        "playback": True,
        "endpoint": "desktop",
        "spotter": "server",
    }
    message.update(overrides)
    return message


def pcm(parts: Sequence[AudioPart], *, seed: int = 0) -> bytes:
    return to_pcm16(scripted_audio(parts, seed=seed))


class VoiceClient:
    """One `/ws` connection: what it sent, everything it received, decoded, in order.

    `received` holds a dict per text message and a `Frame` per binary one. `pos` is the uplink
    position the next microphone frame takes, which is where a scenario scripts a name hit.
    """

    def __init__(self, server_rig: VoiceServer, ws: WebSocketTestSession) -> None:
        self._rig = server_rig
        self.ws = ws
        self.received: list[object] = []
        self.voice_starts: list[object] = []
        self.closed = False
        self.pos = 0
        self.seq = 0

    # Sending

    def send(self, *payloads: dict[str, object] | str) -> None:
        for payload in payloads:
            if isinstance(payload, dict) and payload.get("type") == "voice_start":
                self.voice_starts.append(payload.get("req"))
            self.ws.send_text(payload if isinstance(payload, str) else json.dumps(payload))
            self._rig.settle()

    def send_bytes(self, data: bytes) -> None:
        self.ws.send_bytes(data)
        self._rig.settle()

    def frames(
        self,
        audio: bytes,
        *,
        msg_type: MsgType = MsgType.MIC_AUDIO,
        reference: Callable[[int, bytes], bytes | None] | None = None,
    ) -> None:
        """Send `audio` as 20 ms microphone frames, the session clock moving 20 ms per frame.

        `reference(seq, payload)` may return a `0x11` payload to send just before each frame.
        """
        size = FRAME_SAMPLES * 2
        for start in range(0, len(audio) - size + 1, size):
            payload = audio[start : start + size]
            if reference is not None:
                ref = reference(self.seq, payload)
                if ref is not None:
                    self.ws.send_bytes(
                        encode_frame(Frame(MsgType.REF_AUDIO, 0, self.seq, 16000, ref))
                    )
            self.ws.send_bytes(encode_frame(Frame(msg_type, 0, self.seq, 16000, payload)))
            if msg_type == MsgType.MIC_AUDIO:
                self.pos += 1
            self.seq += 1
            self._rig.advance(FRAME_MS)

    def say(
        self,
        parts: Sequence[AudioPart],
        *,
        name_at_ms: int | None = None,
        seed: int = 0,
        reference: Callable[[int, bytes], bytes | None] | None = None,
    ) -> int | None:
        """Send scripted audio; with `name_at_ms`, the spotter hears "mist" ending there.

        Returns the uplink position of the scripted hit.
        """
        hit_pos = None
        if name_at_ms is not None:
            hit_pos = self.pos + name_at_ms // FRAME_MS
            spotter = self._rig.stack.spotter
            assert spotter is not None
            spotter.hits.append(NameHit(end_pos=hit_pos, keyword="mist", score=0.9))
        self.frames(pcm(parts, seed=seed), reference=reference)
        return hit_pos

    # Receiving

    def drain(self) -> list[object]:
        """Take everything the server has sent so far, without waiting."""

        def take() -> list[dict[str, object]]:
            out: list[dict[str, object]] = []
            while True:
                try:
                    out.append(self.ws._send_rx.receive_nowait())
                except (anyio.WouldBlock, anyio.ClosedResourceError, anyio.EndOfStream):
                    return out

        if self.closed:
            return []
        self._rig.settle()
        fresh: list[object] = []
        for message in self.ws.portal.call(take):
            if message.get("type") != "websocket.send":
                fresh.append(message)
                continue
            if message.get("bytes") is not None:
                fresh.append(decode_frame(message["bytes"], direction=Direction.DOWN))
                continue
            text = message["text"]
            try:
                fresh.append(json.loads(text))
            except ValueError:
                fresh.append(text)
        self.received.extend(fresh)
        return fresh

    @property
    def voice(self) -> list[object]:
        """Everything received but the process-wide messages."""
        self.drain()
        return [
            item
            for item in self.received
            if not (isinstance(item, dict) and item.get("type") in PROCESS_WIDE)
        ]

    def of(self, kind: str, **match: object) -> list[dict[str, object]]:
        """The received messages of `type` (or `voice_decision` `kind`) matching `match`."""
        out = []
        for item in self.voice:
            if not isinstance(item, dict):
                continue
            if item.get("type") != kind and not (
                item.get("type") == "voice_decision" and item.get("kind") == kind
            ):
                continue
            if all(item.get(key) == value for key, value in match.items()):
                out.append(item)
        return out

    def names(self) -> list[str]:
        """Each received voice item as one word: a type, `decision:<kind>`, or `0x01`..`0x03`."""
        out = []
        for item in self.voice:
            if isinstance(item, Frame):
                out.append(f"0x{int(item.msg_type):02x}")
            elif isinstance(item, dict):
                kind = item.get("type")
                if kind == "voice_decision":
                    out.append(f"decision:{item['kind']}")
                elif kind == "state_cycle":
                    out.append(f"state:{item['state']}")
                else:
                    out.append(str(kind))
            else:
                out.append(repr(item))
        return out

    def session_id(self) -> str:
        (started,) = self.of("session_started")
        return str(started["session_id"])

    def finish_playback(self, turn_id: str, played_ms: int = 1000) -> None:
        self.send({"type": "playback_done", "turn_id": turn_id, "played_ms": played_ms})


class VoiceServer:
    """The running server: its fakes, its loop through the portal, and its connections."""

    def __init__(self, client: TestClient, stack: VoiceStack) -> None:
        self.client = client
        self.stack = stack
        self.clients: list[VoiceClient] = []
        self.watchers: list[PowerWatcher] = []

    @property
    def clock(self) -> FakeClock:
        return self.stack.clock

    @property
    def responder(self) -> FakeResponder:
        return self.stack.responder

    def _call(self, fn: Callable[..., object], *args: object) -> object:
        portal = self.client.portal
        assert portal is not None
        return portal.call(fn, *args)

    def run(self, fn: Callable[..., object], *args: object) -> object:
        """Run `fn` on the server's loop and settle after it."""
        result = self._call(fn, *args)
        self.settle()
        return result

    def settle(self) -> None:
        self._call(settle)

    def advance(self, ms: int) -> None:
        self._call(self.clock.advance, ms)

    def fire_arbiter(self) -> None:
        """End the wake arbiter's window: its oldest pending timer runs."""
        self.run(self.stack.call_later.fire)

    # Gaming mode

    def poll_gpu_yield(self) -> None:
        """One poll of the marker, then one read of the level by the power watcher."""
        gpu_yield = server._gpu_yield
        assert gpu_yield is not None
        self.run(gpu_yield.poll_once)
        (watcher,) = self.watchers
        self.run(watcher.check)

    def until(self, done: Callable[[], bool], timeout_s: float = 5.0) -> None:
        """Settle until `done()`; for work a worker thread finishes, such as a reload retry."""
        deadline = time.monotonic() + timeout_s
        while not done():
            assert time.monotonic() < deadline, "timed out waiting on a worker thread"
            self._call(asyncio.sleep, 0.01)
        self.settle()

    @contextmanager
    def connect(self) -> Iterator[VoiceClient]:
        with self.client.websocket_connect("/ws") as ws:
            voice = VoiceClient(self, ws)
            self.clients.append(voice)
            voice.drain()
            try:
                yield voice
            finally:
                voice.drain()
                voice.closed = True
                # Disconnect and let the session commit and end before the test client cancels
                # the connection's task.
                ws.close(1000)
                self.settle()


@pytest.fixture
def voice_server(monkeypatch: pytest.MonkeyPatch) -> Callable[..., object]:
    """The real lifespan behind `TestClient`, with nothing behind it but a `VoiceStack`.

    Call the fixture's value with a stack, as a context manager, to start the server.
    """
    for name in (
        "runtime",
        "session_registry",
        "curation_scheduler",
        "log_handler",
        "extraction_dispatcher",
        "vault_writer",
        "vault_sidecar",
        "vault_filewatcher",
        "vault_invalidation_bus",
        "_broadcaster_state",
        "_health_registry",
        "_health_probe_executor",
        "_extraction_health_client",
        "_extraction_dispatcher_start_failed",
        "_gpu_yield",
        "_models_ready",
        "_capability_build",
        "_capability_config",
        "_capability_power",
        "_capability_sink",
    ):
        monkeypatch.setattr(server, name, getattr(server, name))
    monkeypatch.setattr(server, "message_queue", asyncio.Queue())
    monkeypatch.setattr(server, "configure_file_logging", lambda log_dir: None)
    # The record factory is process-global; installing it here would outlive the test.
    monkeypatch.setattr(server, "install_log_redaction", lambda registry: None)
    monkeypatch.setattr(server, "build_vault_writer", lambda *a, **k: None)
    monkeypatch.setattr(server, "build_sidecar_index", lambda *a, **k: None)
    monkeypatch.setattr(server, "build_phase3_components", lambda *a, **k: None)
    monkeypatch.setattr(server, "build_llm_provider", lambda config, **kwargs: FakeLLMProvider())
    monkeypatch.setattr(server, "PersonaRegister", FakePersona)
    monkeypatch.setattr(server, "build_curation_scheduler", lambda config, **k: FakeScheduler())
    monkeypatch.setattr(
        server,
        "_new_extraction_health_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(_extraction_health)),
    )
    for name in (
        "MIST_OPS_DIR",
        "MIST_BACKUP_STATUS_PATH",
        "MIST_GPU_YIELD_POLL_S",
        "MIST_VOICE_PROFILE",
        "MIST_CAPABILITIES_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)
    system_status = server.config.system_status.model_copy(update={"enabled": False})
    monkeypatch.setattr(
        server,
        "config",
        server.config.model_copy(update={"tts_enabled": True, "system_status": system_status}),
    )

    @contextmanager
    def start(stack: VoiceStack, *, ops_dir: Path | None = None) -> Iterator[VoiceServer]:
        """With `ops_dir`, gaming mode watches its marker; a scenario polls it by hand."""
        watchers: list[PowerWatcher] = []

        def watcher(*args: object, **kwargs: object) -> PowerWatcher:
            built = PowerWatcher(*args, sleep=_never, **kwargs)  # type: ignore[arg-type]
            watchers.append(built)
            return built

        monkeypatch.setattr(server, "build_mist_runtime", stack)
        monkeypatch.setattr(server, "PowerWatcher", watcher)
        monkeypatch.setattr(
            server,
            "SessionRegistry",
            lambda services, **kwargs: SessionRegistry(
                services, call_later=stack.call_later, **kwargs
            ),
        )
        if ops_dir is not None:
            monkeypatch.setenv("MIST_OPS_DIR", str(ops_dir))
            # The poll loop's first poll runs at boot; every later one is the scenario's.
            monkeypatch.setenv("MIST_GPU_YIELD_POLL_S", "3600")
        with TestClient(server.app) as client:
            running = VoiceServer(client, stack)
            running.watchers = watchers
            yield running

    return start
