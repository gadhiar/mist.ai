"""T22: the engine slots, the GPU release and reload in gaming mode, and the power watcher.

Every engine is a fake built by a recording loader, torch is a fake whose `memory_reserved`
drops when the cache is emptied, and the watcher and the retries run on fake sleeps. The
release and reload run on a worker thread, as `GpuYield` runs them, while the test's loop serves
what they post to it.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
import pytest_asyncio

from backend.capabilities.power import FixedTiers, GpuYieldPower, PowerLevel, SingleModelTiers
from backend.chat.stream_events import Token
from backend.gpu_yield import GpuYield
from backend.realtime import runtime as runtime_module
from backend.realtime.budget import EngineOption, PlacementConfig, VoiceSettings
from backend.realtime.errors import RealtimeError, SpeechEngineError
from backend.realtime.lanes import LaneJobCancelled
from backend.realtime.power import (
    NO_MODEL_MESSAGE,
    PowerGate,
    PowerWatcher,
    SynthesizerSlot,
    TranscriberSlot,
)
from backend.realtime.protocol import VoiceDecisionPlacement
from backend.realtime.runtime import (
    EngineArm,
    EngineLoaders,
    GpuEngines,
    MistRuntime,
    load_speech_engines,
)
from backend.realtime.speech import CancelToken, SynthChunk, TextSegment, TranscriptUpdate
from backend.realtime.turn import Interrupt
from tests.fakes.speech import FakeSynthesizer, ScriptedTranscriber

from .test_session import (
    Rig,
    _stop_rigs,  # noqa: F401 - stops the session-cut rigs
    spoken_turn,
)
from .turn_fakes import Pause, settle

MIB = 1024 * 1024
RUNTIME_LOGGER = "backend.realtime.runtime"
REPO = Path(__file__).resolve().parents[3]
SEGMENT = TextSegment("Hello.", 0, 6)


# -- Fakes -------------------------------------------------------------------------------------


class FakeCuda:
    """`torch.cuda`: `memory_reserved` falls by `freed_mib` at each `empty_cache`."""

    def __init__(self, reserved_mib: int = 2000, freed_mib: int = 900) -> None:
        self.reserved = reserved_mib * MIB
        self.freed = freed_mib * MIB
        self.empty_calls = 0
        self.error: BaseException | None = None

    def is_available(self) -> bool:
        return True

    def memory_reserved(self) -> int:
        return self.reserved

    def empty_cache(self) -> None:
        self.empty_calls += 1
        if self.error is not None:
            raise self.error
        self.reserved -= self.freed


class Loaders:
    """Loaders that record each load; `fail(option, *errors)` makes the next loads raise."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self.failures: dict[str, list[BaseException]] = {}
        self.made: dict[str, list[object]] = {}
        self.factories: dict[str, Callable[[], object]] = {
            "nemotron-160": lambda: ScriptedTranscriber([], model_id="nemotron-model"),
            "parakeet-cpu": lambda: ScriptedTranscriber([], model_id="parakeet-model"),
            "pocket-clone": lambda: FakeSynthesizer(model_id="pocket-model"),
            "cuda-tts": lambda: FakeSynthesizer(model_id="cuda-tts-model", engine_id="cuda-tts"),
        }

    def fail(self, option: str, *errors: BaseException) -> None:
        self.failures.setdefault(option, []).extend(errors)

    def _make(self, option: str, factory: Callable[[], object]) -> object:
        self.calls.append(option)
        queue = self.failures.get(option)
        if queue:
            raise queue.pop(0)
        engine = factory()
        self.made.setdefault(option, []).append(engine)
        return engine

    def build(self) -> EngineLoaders:
        def chain(engine_id: str) -> Callable[[VoiceSettings, str], object]:
            return lambda settings, device: self._make(
                f"{engine_id}@{device}", self.factories[engine_id]
            )

        return EngineLoaders(
            chains={engine_id: chain(engine_id) for engine_id in self.factories},
            vad=lambda placement: SimpleNamespace(model_id="vad-model"),  # type: ignore[arg-type,return-value]
            eot=lambda placement, endpointing: SimpleNamespace(model_id="eot-model"),  # type: ignore[arg-type,return-value]
            kws=lambda placement: SimpleNamespace(model_id="kws-model"),  # type: ignore[arg-type,return-value]
        )


class FakeHost:
    """The registry's broadcast side; records each message and the thread that sent it."""

    def __init__(self) -> None:
        self.sent: list[object] = []
        self.threads: set[int] = set()

    def broadcast(self, message: object, *, endpoint: object = None) -> None:
        self.threads.add(threading.get_ident())
        self.sent.append(message)

    def broadcast_voice_state(self) -> None:
        self.threads.add(threading.get_ident())
        self.sent.append("voice_state")

    def placements(self) -> list[tuple[str, int | None, str, str]]:
        return [
            (m.reason, m.detail.released_mib, m.detail.engines.stt, m.detail.engines.tts)
            for m in self.sent
            if isinstance(m, VoiceDecisionPlacement)
        ]


class FakeSleep:
    """Records each delay; `hold()` makes every later sleep wait until `release()`."""

    def __init__(self) -> None:
        self.delays: list[float] = []
        self.gate: asyncio.Event | None = None

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)
        if self.gate is not None:
            await self.gate.wait()
        await asyncio.sleep(0)

    def hold(self) -> None:
        self.gate = asyncio.Event()


class StuckStream:
    """A transcript stream whose final decode ignores `close` until the test lets it end."""

    def __init__(self, records: list[tuple[str, int]]) -> None:
        self.records = records
        self.ended = threading.Event()
        self.started = threading.Event()

    def push(self, pcm: object) -> None:
        self.records.append(("push", threading.get_ident()))

    def decode_partial(self, token: CancelToken) -> TranscriptUpdate:
        self.records.append(("decode_partial", threading.get_ident()))
        return TranscriptUpdate(text="", stable_chars=0, is_final=False, audio_ms=0)

    def decode_final(self, token: CancelToken) -> TranscriptUpdate:
        self.started.set()
        self.ended.wait(5)
        return TranscriptUpdate(text="late", stable_chars=4, is_final=True, audio_ms=0)

    def close(self) -> None:
        self.records.append(("close", threading.get_ident()))


class RecordingTranscriber(ScriptedTranscriber):
    """A CUDA transcriber whose streams record each call with its thread."""

    def __init__(self) -> None:
        super().__init__([], model_id="nemotron-model")
        self.records: list[tuple[str, int]] = []
        self.opened: list[StuckStream] = []

    def open_stream(self) -> StuckStream:  # type: ignore[override]
        stream = StuckStream(self.records)
        self.opened.append(stream)
        return stream


class Built(SimpleNamespace):
    loaders: Loaders
    engines: object
    gpu: GpuEngines
    host: FakeHost
    cuda: FakeCuda
    sleep: FakeSleep
    collects: list[int]


def build(
    loaders: Loaders | None = None,
    *,
    stt_chain: tuple[EngineOption, ...] | None = None,
    tts_chain: tuple[EngineOption, ...] | None = None,
    power_level: PowerLevel = "full",
    timeout_s: float = 1.0,
) -> Built:
    loaders = loaders or Loaders()
    placement = PlacementConfig(release_timeout_s=timeout_s)
    if stt_chain is not None:
        placement = PlacementConfig(
            stt_chain=stt_chain, tts_chain=placement.tts_chain, release_timeout_s=timeout_s
        )
    if tts_chain is not None:
        placement = PlacementConfig(
            stt_chain=placement.stt_chain, tts_chain=tts_chain, release_timeout_s=timeout_s
        )
    settings = VoiceSettings(placement=placement)
    engines = load_speech_engines(settings, loaders.build(), power_level=power_level)
    cuda = FakeCuda()
    sleep = FakeSleep()
    collects: list[int] = []
    gpu = GpuEngines(
        engines,
        settings,
        loaders.build(),
        torch_loader=lambda: SimpleNamespace(cuda=cuda),
        collect=lambda: collects.append(1),
        sleep=sleep,
    )
    host = FakeHost()
    gpu.attach(asyncio.get_running_loop(), host)
    built = Built(
        loaders=loaders,
        engines=engines,
        gpu=gpu,
        host=host,
        cuda=cuda,
        sleep=sleep,
        collects=collects,
    )
    BUILT.append(built)
    return built


BUILT: list[Built] = []


@pytest_asyncio.fixture(autouse=True)
async def _stop_retries():
    yield
    while BUILT:
        BUILT.pop().gpu.stop()
    await asyncio.sleep(0)


@pytest.fixture
def cuda_tts(monkeypatch: pytest.MonkeyPatch) -> None:
    """An engine id with a CUDA arm for TTS; the default table has none yet."""
    monkeypatch.setitem(
        runtime_module.ENGINE_TABLE, "cuda-tts", EngineArm("tts", ("cuda",))  # type: ignore[index]
    )


def where(built: Built) -> tuple[str, str]:
    statuses = built.engines.statuses  # type: ignore[attr-defined]
    return statuses.stt.location, statuses.tts.location


async def release(built: Built) -> int:
    released = await asyncio.to_thread(built.gpu.release)
    await settle()
    return released


async def reload(built: Built) -> None:
    await asyncio.to_thread(built.gpu.reload)
    await settle()


# -- Slots -------------------------------------------------------------------------------------


class TestSlots:
    def test_a_stream_opened_before_a_swap_keeps_its_engine_until_it_closes(self):
        old = ScriptedTranscriber(["old words"], model_id="old")
        new = ScriptedTranscriber(["new words"], model_id="new")
        slot = TranscriberSlot(old)
        before = slot.open_stream()

        assert slot.swap(new) is old
        after = slot.open_stream()

        assert (len(old.streams), len(new.streams)) == (1, 1)
        assert slot.model_id == "new"
        assert slot.busy(old)
        before.close()
        assert not slot.busy(old)
        after.close()

    def test_a_segment_in_synthesis_finishes_on_the_old_synthesizer(self):
        old = FakeSynthesizer(model_id="old")
        new = FakeSynthesizer(model_id="new")
        slot = SynthesizerSlot(old)
        chunks = slot.synthesize(SEGMENT, CancelToken())
        first = next(chunks)

        slot.swap(new)
        rest = list(chunks)

        assert isinstance(first, SynthChunk)
        assert all(isinstance(chunk, SynthChunk) for chunk in rest)
        assert (old.segments, new.segments) == ([SEGMENT], [])
        assert slot.model_id == "new"

    def test_wait_idle_reports_both_an_open_stream_and_a_segment_in_synthesis(self):
        stt = ScriptedTranscriber(["words"])
        tts = FakeSynthesizer()
        stt_slot, tts_slot = TranscriberSlot(stt), SynthesizerSlot(tts)
        stream = stt_slot.open_stream()
        chunks = tts_slot.synthesize(SEGMENT, CancelToken())
        next(chunks)

        assert stt_slot.wait_idle(stt, 0.01) is False
        assert tts_slot.wait_idle(tts, 0.01) is False
        stream.close()
        list(chunks)
        assert stt_slot.wait_idle(stt, 0.01) is True
        assert tts_slot.wait_idle(tts, 0.01) is True

    def test_an_empty_slot_refuses_work_with_a_speech_engine_error(self):
        stream = TranscriberSlot(None).open_stream()
        stream.push(b"")
        with pytest.raises(SpeechEngineError):
            stream.decode_final(CancelToken())
        with pytest.raises(SpeechEngineError):
            SynthesizerSlot(None).synthesize(SEGMENT, CancelToken())


# -- Release -----------------------------------------------------------------------------------


@pytest.mark.asyncio
class TestRelease:
    async def test_release_drops_the_cuda_stt_engine_and_moves_it_to_the_cpu(self):
        built = build()
        nemotron = built.engines.transcriber  # type: ignore[attr-defined]
        pocket = built.engines.synthesizer  # type: ignore[attr-defined]
        vad = built.engines.vad  # type: ignore[attr-defined]

        released = await release(built)

        assert released == 900
        assert nemotron.releases == 1
        assert built.engines.transcriber.model_id == "parakeet-model"  # type: ignore[attr-defined]
        assert built.engines.stt_slot.engine is built.engines.transcriber  # type: ignore[attr-defined]
        assert built.engines.synthesizer is pocket  # type: ignore[attr-defined]
        assert pocket.releases == 0
        assert built.engines.vad is vad  # type: ignore[attr-defined]
        assert where(built) == ("cpu", "cpu")
        assert (built.cuda.empty_calls, len(built.collects)) == (1, 1)
        assert built.host.placements() == [("power_off", 900, "cpu", "cpu")]
        assert built.host.sent.count("voice_state") == 1

    async def test_release_and_reload_are_idempotent(self):
        built = build()
        await release(built)

        assert await release(built) == 0
        await reload(built)
        calls = list(built.loaders.calls)
        await reload(built)

        assert built.loaders.calls == calls
        assert built.host.placements() == [("power_off", 900, "cpu", "cpu")]
        assert where(built) == ("cuda", "cpu")

    async def test_engine_release_runs_before_the_drop_once_and_a_failure_is_logged(self, caplog):
        built = build()
        nemotron = built.engines.transcriber  # type: ignore[attr-defined]
        nemotron.release_error = SpeechEngineError("cuda said: hello there")

        with caplog.at_level(logging.ERROR, logger=RUNTIME_LOGGER):
            released = await release(built)

        assert nemotron.releases == 1
        assert released == 900
        errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
        assert any("nemotron-model" in message for message in errors)
        assert not any("hello there" in message for message in errors)
        assert where(built) == ("cpu", "cpu")

    async def test_release_closes_open_stt_streams_on_the_loop_thread(self):
        loaders = Loaders()
        recording = RecordingTranscriber()
        loaders.factories["nemotron-160"] = lambda: recording
        built = build(loaders)
        stream = built.engines.stt_slot.open_stream()  # type: ignore[attr-defined]
        stream.push(b"")
        loop_thread = threading.get_ident()

        await release(built)

        assert ("close", loop_thread) in recording.records
        assert {thread for _call, thread in recording.records} == {loop_thread}
        with pytest.raises(SpeechEngineError):
            stream.decode_partial(CancelToken())

    async def test_a_stuck_job_times_the_release_out_then_moves_the_slot_when_it_ends(self):
        loaders = Loaders()
        recording = RecordingTranscriber()
        loaders.factories["nemotron-160"] = lambda: recording
        built = build(loaders, timeout_s=0.2)
        stream = built.engines.stt_slot.open_stream()  # type: ignore[attr-defined]
        decode = threading.Thread(target=lambda: _final(stream))
        decode.start()
        recording.opened[0].started.wait(5)

        released = await release(built)

        assert released == 0
        assert where(built) == ("off", "cpu")
        assert built.host.placements() == [("release_timeout", None, "off", "cpu")]
        empties = built.cuda.empty_calls
        recording.opened[0].ended.set()
        decode.join(5)
        for finisher in list(built.gpu.finishers):
            await asyncio.to_thread(finisher.join, 5)
        await settle()

        assert built.cuda.empty_calls == empties + 1
        assert recording.releases == 1
        assert where(built) == ("cpu", "cpu")
        assert built.engines.transcriber.model_id == "parakeet-model"  # type: ignore[attr-defined]
        assert built.host.placements()[-1] == ("power_off", 900, "cpu", "cpu")

    async def test_release_cancels_the_tts_jobs_of_a_cuda_synthesizer(self, cuda_tts):
        built = build(tts_chain=(EngineOption("cuda-tts", "cuda"),))
        slot = built.engines.tts_slot  # type: ignore[attr-defined]
        token = CancelToken()
        chunks = slot.synthesize(SEGMENT, token)
        next(chunks)
        cuts: list[int] = []
        slot.subscribe(lambda: cuts.append(threading.get_ident()))

        await release(built)

        assert token.is_set()
        assert cuts == [threading.get_ident()]
        assert where(built) == ("cpu", "off")
        assert built.host.placements() == [("power_off", 900, "cpu", "off")]

    async def test_messages_from_the_thread_are_sent_on_the_loop(self):
        built = build()
        built.sleep.hold()
        built.loaders.fail("nemotron-160@cuda", SpeechEngineError("no"))

        await release(built)
        await reload(built)

        assert built.host.threads == {threading.get_ident()}
        assert [p[0] for p in built.host.placements()] == ["power_off", "reload_failed"]

    @pytest.mark.parametrize("error", [RealtimeError("x"), RuntimeError("x"), OSError("x")])
    async def test_an_error_inside_the_release_is_logged_and_not_raised(self, error, caplog):
        built = build()
        built.cuda.error = error

        with caplog.at_level(logging.ERROR, logger=RUNTIME_LOGGER):
            await release(built)

        assert any(r.levelno == logging.ERROR for r in caplog.records)


def _final(stream: object) -> None:
    with contextlib.suppress(SpeechEngineError, LaneJobCancelled):
        stream.decode_final(CancelToken())  # type: ignore[attr-defined]


# -- Reload ------------------------------------------------------------------------------------


@pytest.mark.asyncio
class TestReload:
    async def test_reload_restores_each_slot_one_engine_at_a_time(self):
        built = build()
        await release(built)
        built.loaders.calls.clear()

        await reload(built)

        assert built.loaders.calls == ["nemotron-160@cuda"]
        assert where(built) == ("cuda", "cpu")
        assert built.engines.transcriber.model_id == "nemotron-model"  # type: ignore[attr-defined]

    async def test_a_slot_skipped_at_an_off_boot_reloads_to_its_first_option(self):
        built = build(power_level="off")
        assert where(built) == ("cpu", "cpu")

        await reload(built)

        assert where(built) == ("cuda", "cpu")
        assert built.host.placements() == []

    async def test_a_failing_cuda_load_leaves_each_slot_on_its_next_option(self, cuda_tts):
        built = build(tts_chain=(EngineOption("cuda-tts", "cuda"),))
        built.sleep.hold()
        await release(built)
        built.loaders.fail("nemotron-160@cuda", SpeechEngineError("oom"))
        built.loaders.fail("cuda-tts@cuda", SpeechEngineError("oom"))

        await reload(built)

        assert where(built) == ("cpu", "off")
        assert built.host.placements()[-1] == ("reload_failed", None, "cpu", "off")
        assert [p[0] for p in built.host.placements()].count("reload_failed") == 1

    @pytest.mark.parametrize("error", [RealtimeError("x"), RuntimeError("x"), OSError("x")])
    async def test_an_error_inside_the_reload_is_logged_and_not_raised(self, error, caplog):
        built = build()
        built.sleep.hold()
        await release(built)
        built.loaders.fail("nemotron-160@cuda", error)

        with caplog.at_level(logging.ERROR, logger=RUNTIME_LOGGER):
            await reload(built)

        assert any(r.levelno == logging.ERROR for r in caplog.records)
        assert where(built) == ("cpu", "cpu")

    async def test_a_real_gpu_yield_is_normal_after_a_failing_reload(self, tmp_path, caplog):
        built = build()
        built.sleep.hold()
        marker = tmp_path / "gpu-yielded"

        async def tick(_delay: float) -> None:
            await asyncio.sleep(0)

        gpu_yield = GpuYield(marker, built.gpu.release, built.gpu.reload, poll_s=0.01, sleep=tick)
        marker.write_text("{}")
        await gpu_yield.poll_once()
        assert gpu_yield.state() == "yielded"
        built.loaders.fail("nemotron-160@cuda", RuntimeError("CUDA out of memory"))
        marker.unlink()

        await gpu_yield.poll_once()

        assert gpu_yield.state() == "normal"
        task = asyncio.get_running_loop().create_task(gpu_yield.run_forever())
        await settle()
        assert not task.done()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

    async def test_a_slot_below_its_device_is_retried_until_it_loads(self, caplog):
        built = build()
        await release(built)
        built.loaders.fail("nemotron-160@cuda", SpeechEngineError("oom"), SpeechEngineError("oom"))

        with caplog.at_level(logging.INFO, logger=RUNTIME_LOGGER):
            await reload(built)
            for _ in range(20):
                await settle()
                if where(built)[0] == "cuda":
                    break
                await asyncio.sleep(0.01)

        assert where(built) == ("cuda", "cpu")
        assert built.sleep.delays[:2] == [2.0, 4.0]
        assert built.host.placements()[-1] == ("power_restored", None, "cuda", "cpu")
        assert built.host.sent[-1] == "voice_state"
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert len(errors) == 1

    async def test_the_retry_delay_doubles_up_to_sixty_seconds(self):
        built = build()
        await release(built)
        built.loaders.fail("nemotron-160@cuda", *[SpeechEngineError("oom")] * 9)

        await reload(built)
        for _ in range(40):
            await settle()
            if len(built.sleep.delays) >= 8:
                break
            await asyncio.sleep(0.01)

        assert built.sleep.delays[:8] == [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0, 60.0]

    async def test_a_release_cancels_the_retries(self):
        built = build()
        await release(built)
        built.sleep.hold()
        built.loaders.fail("nemotron-160@cuda", SpeechEngineError("oom"))
        await reload(built)
        assert built.gpu.retrying

        await release(built)

        assert not built.gpu.retrying


# -- The session's cut -------------------------------------------------------------------------


@pytest.mark.asyncio
class TestSessionCut:
    async def test_a_tts_turn_in_synthesis_is_cut_whole_and_recorded_interrupted(self):
        pause = Pause()
        rig = Rig([Token("Hello there. "), pause])
        slot = SynthesizerSlot(rig.engines.synthesizer)
        rig.engines.tts_slot = slot
        await rig.start()
        await spoken_turn(rig)
        await settle()

        slot.cancel_jobs(rig.engines.synthesizer)
        await settle()

        assert rig.posted(Interrupt)
        assert [c.outcome for c in rig.responder.commits] == ["interrupted"]

    async def test_a_typed_turn_streaming_at_the_drop_is_not_cut(self):
        pause = Pause()
        rig = Rig([Token("Hello there. "), pause])
        slot = SynthesizerSlot(rig.engines.synthesizer)
        rig.engines.tts_slot = slot
        await rig.start()
        await rig.send({"type": "text", "text": "hi"})

        slot.cancel_jobs(rig.engines.synthesizer)
        await settle()

        assert rig.posted(Interrupt) == []
        pause.release()
        await settle()
        assert [c.outcome for c in rig.responder.commits] == ["completed"]


# -- The watcher -------------------------------------------------------------------------------


class Level:
    def __init__(self, level: PowerLevel) -> None:
        self.value: PowerLevel = level

    def level(self) -> PowerLevel:
        return self.value


def gate(level: Level) -> PowerGate:
    return PowerGate(level, FixedTiers({}), lambda: NO_MODEL_MESSAGE)


class TestPowerWatcher:
    def test_a_drop_to_off_is_reported_on_the_next_read_once(self):
        level, changes = Level("full"), []
        watcher = PowerWatcher(gate(level), lambda: "yielded", changes.append)

        watcher.check()
        level.value = "off"
        watcher.check()
        watcher.check()

        assert changes == ["off"]

    def test_a_rise_is_reported_only_once_gpu_state_is_normal(self):
        level, changes, state = Level("off"), [], ["yielded"]
        watcher = PowerWatcher(gate(level), lambda: state[0], changes.append)
        watcher.check()
        level.value = "full"

        watcher.check()
        assert changes == []
        state[0] = "normal"
        watcher.check()

        assert changes == ["full"]

    @pytest.mark.asyncio
    async def test_run_reads_the_level_every_interval(self):
        level, changes = Level("full"), []
        sleeps: list[float] = []

        async def sleep(delay: float) -> None:
            sleeps.append(delay)
            if len(sleeps) == 2:
                level.value = "off"
            await asyncio.sleep(0)

        watcher = PowerWatcher(gate(level), lambda: "normal", changes.append, sleep=sleep)
        task = asyncio.get_running_loop().create_task(watcher.run())
        await settle()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task

        assert set(sleeps) == {0.25}
        assert changes == ["off"]

    @pytest.mark.asyncio
    async def test_each_change_sends_voice_state_and_a_rise_power_restored(self):
        built = build()

        built.gpu.level_changed("off")
        built.gpu.level_changed("full")

        assert built.host.sent.count("voice_state") == 2
        assert built.host.placements() == [("power_restored", None, "cuda", "cpu")]

    @pytest.mark.asyncio
    async def test_with_gpu_yield_each_message_is_sent_once_with_its_mib(self):
        built = build()
        fake = SimpleNamespace(value="normal")
        fake.state = lambda: fake.value
        power = GpuYieldPower(fake)  # type: ignore[arg-type]
        watcher = PowerWatcher(
            PowerGate(power, SingleModelTiers(lambda: object(), power), lambda: ""),  # type: ignore[arg-type]
            fake.state,
            built.gpu.level_changed,
        )
        watcher.check()

        fake.value = "yielded"
        watcher.check()
        await asyncio.to_thread(built.gpu.release)
        await settle()
        await asyncio.to_thread(built.gpu.reload)
        watcher.check()
        fake.value = "normal"
        watcher.check()
        await settle()

        assert built.host.placements() == [
            ("power_off", 900, "cpu", "cpu"),
            ("power_restored", None, "cuda", "cpu"),
        ]


# -- The runtime and the zone ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mist_runtime_delegates_release_and_reload():
    built = build()
    runtime = MistRuntime(
        llm_provider=None,
        conversation_handler=None,
        responder=None,  # type: ignore[arg-type]
        engines=built.engines,  # type: ignore[arg-type]
        lanes=None,  # type: ignore[arg-type]
        settings=VoiceSettings(),
        services=None,  # type: ignore[arg-type]
        agent_available=False,
        gpu=built.gpu,
    )

    assert await asyncio.to_thread(runtime.release_gpu_engines) == 900
    await asyncio.to_thread(runtime.reload_gpu_engines)
    assert where(built) == ("cuda", "cpu")


def test_a_runtime_without_gpu_control_releases_nothing():
    runtime = MistRuntime(
        llm_provider=None,
        conversation_handler=None,
        responder=None,  # type: ignore[arg-type]
        engines=None,  # type: ignore[arg-type]
        lanes=None,  # type: ignore[arg-type]
        settings=VoiceSettings(),
        services=None,  # type: ignore[arg-type]
        agent_available=False,
    )

    assert runtime.release_gpu_engines() == 0
    assert runtime.reload_gpu_engines() is None


ZONE = (
    "backend/realtime/power.py",
    "backend/realtime/session.py",
    "backend/realtime/runtime.py",
)


def test_no_broad_except_in_the_zone():
    for name in ZONE:
        text = (REPO / name).read_text(encoding="utf-8")
        assert not re.search(r"except\s+(Exception|BaseException)\b|except\s*:", text), name


WIRE_SOURCE = 'source: Literal["manual", "fullscreen"]'
"""The `gpu_yield` message's schema names the client's source; it reads no full-screen state."""


def test_nothing_in_realtime_reads_fullscreen_notification_or_vram_pressure_state():
    pattern = re.compile(r"fullscreen|SHQueryUserNotificationState|mem_get_info")
    root = REPO / "backend" / "realtime"
    hits = [
        f"{path.relative_to(root).as_posix()}: {line.strip()}"
        for path in root.rglob("*.py")
        for line in path.read_text(encoding="utf-8").splitlines()
        if pattern.search(line)
    ]

    assert hits == [f"protocol.py: {WIRE_SOURCE}"]
