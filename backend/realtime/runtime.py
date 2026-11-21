"""The composed realtime runtime, and the walk that loads each speech slot from its chain.

`build_mist_runtime` in `backend/factories.py` builds `MistRuntime` once at startup; this module
holds what that composition produces and the engine walk it runs, so the walk is testable with
fake loaders and no factory import. No library is imported here: every engine loads through an
injected loader, and the GPU meter imports torch only when a CUDA option loads.
"""

from __future__ import annotations

import asyncio
import gc
import logging
import math
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

from backend.errors import MistError
from backend.gpu_yield import DEFAULT_POLL_S, MAX_RETRY_DELAY_S

from .budget import EndpointingConfig, EngineOption, PlacementConfig, VoiceSettings
from .errors import RealtimeError
from .power import LEVEL_RANK, SynthesizerSlot, TranscriberSlot
from .protocol import (
    EnginePlacement,
    EngineStatus,
    EngineStatuses,
    PlacementDetail,
    PowerLevel,
    VoiceCapabilities,
    VoiceDecisionPlacement,
)
from .session import OFF, InferenceLanes, SessionServices, SpeechEngines
from .speech import NameSpotter, TurnEndDetector, VoiceActivityDetector, WakeVerifier
from .turn import TurnPlacement, TurnResponder
from .voice_assets import VoiceAssetError

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
Slot = Literal["stt", "tts"]
PlacementReason = Literal["power_off", "power_restored", "reload_failed", "release_timeout"]
MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class EngineArm:
    """One engine id's row in the engine table: its slot and the devices its arm recorded."""

    slot: Slot
    devices: tuple[str, ...]


ENGINE_TABLE: Mapping[str, EngineArm] = {
    "pocket-clone": EngineArm("tts", ("cpu",)),
    "nemotron-160": EngineArm("stt", ("cuda",)),
    "parakeet-cpu": EngineArm("stt", ("cpu",)),
}
"""The engines a chain may name. An option outside it, or on a device its arm did not record,
was never benchmarked there, so the walk skips it rather than guess."""


@dataclass(frozen=True, slots=True)
class EngineLoaders:
    """How each engine loads. Each callable imports its library only when called.

    `chains` maps an engine id to `(settings, device) -> engine`; `vad`, `eot`, `kws` and
    `verify` load the single-device engines from the placement. With `verify` None no verifier
    can load, so the spotter is not loaded either.
    """

    chains: Mapping[str, Callable[[VoiceSettings, str], object]]
    vad: Callable[[PlacementConfig], VoiceActivityDetector]
    eot: Callable[[PlacementConfig, EndpointingConfig], TurnEndDetector]
    kws: Callable[[PlacementConfig], NameSpotter]
    verify: Callable[[PlacementConfig], WakeVerifier] | None = None


class GpuMeter(Protocol):
    """Measures the VRAM one engine's load takes."""

    def measure(self, load: Callable[[], _T]) -> tuple[_T, int]:
        """Run `load` and return its engine with the reserved peak in MiB."""
        ...


class TorchGpuMeter:
    """`torch.cuda`'s allocator figures around a load; torch is imported on first use.

    The peak is reset before the load, so `max_memory_reserved` after it counts the allocator's
    cached pool the engine caused. It leaves out the CUDA context and anything torch does not
    allocate.
    """

    def __init__(self, torch_loader: Callable[[], object]) -> None:
        self._torch_loader = torch_loader

    def measure(self, load: Callable[[], _T]) -> tuple[_T, int]:
        """Run `load` between a peak reset and a read of the reserved peak."""
        cuda = self._torch_loader().cuda  # type: ignore[attr-defined]
        if not cuda.is_available():
            # No device: the load raises its own `SpeechEngineError`, and the walk goes on.
            return load(), 0
        before = cuda.memory_allocated()
        cuda.reset_peak_memory_stats()
        engine = load()
        reserved = cuda.max_memory_reserved()
        logger.info(
            "speech engine VRAM: allocated %d MiB, reserved peak %d MiB",
            math.ceil((cuda.memory_allocated() - before) / MIB),
            math.ceil(reserved / MIB),
        )
        return engine, math.ceil(reserved / MIB)


def _status(location: str, model_id: str, vram_mib: int | None = None) -> EngineStatus:
    return EngineStatus(loaded=True, location=location, model_id=model_id, vram_mib=vram_mib)


def _on_gpu(device: str) -> bool:
    return device.startswith("cuda")


def _chain(settings: VoiceSettings, slot: Slot) -> tuple[EngineOption, ...]:
    placement = settings.placement
    return placement.stt_chain if slot == "stt" else placement.tts_chain


class _Walk:
    """One run of engine loads, one at a time: at startup, or under a GPU release or reload.

    At startup a failed load logs WARNING with its error, as T12 shipped. Under a release or
    reload, `failure_level` is set: a failure logs the error's class only at that level, and a
    `RuntimeError` or `OSError` from a loader (CUDA's out-of-memory is a `RuntimeError`) moves
    the walk on as a `MistError` does. `gpu_failed` runs once after each failed CUDA load,
    before the next option: the GPU control empties the CUDA cache there, so a partial
    allocation goes back to the game rather than sitting in torch's pool.
    """

    def __init__(
        self,
        settings: VoiceSettings,
        loaders: EngineLoaders,
        gpu_meter: GpuMeter | None,
        gpu_allowed: bool,
        *,
        failure_level: int | None = None,
        gpu_failed: Callable[[], object] | None = None,
    ) -> None:
        self.settings = settings
        self.loaders = loaders
        self.gpu_meter = gpu_meter
        self.gpu_allowed = gpu_allowed
        self.failure_level = failure_level
        self.gpu_failed = gpu_failed
        self.tolerated: tuple[type[BaseException], ...] = (
            (MistError,) if failure_level is None else (MistError, RuntimeError, OSError)
        )
        self.vram_mib = 0
        self.skipped_gpu: dict[Slot, int] = {}

    def _load(self, device: str, load: Callable[[], _T]) -> tuple[_T, int | None]:
        if _on_gpu(device) and self.gpu_meter is not None:
            engine, vram = self.gpu_meter.measure(load)
            self.vram_mib += vram
            return engine, vram
        return load(), None

    def chain(self, slot: Slot) -> tuple[object | None, EngineStatus, int | None]:
        """The first option of the slot's chain that loads with its index, or None and `off`."""
        engine, status, index = self.walk(slot)
        if engine is None and _chain(self.settings, slot):
            logger.warning("%s: no option of the chain loaded; the slot is off", slot)
        return engine, status, index

    def walk(
        self, slot: Slot, start: int = 0, stop: int | None = None
    ) -> tuple[object | None, EngineStatus, int | None]:
        """The first of the chain's options `start` to `stop` that loads, with its index.

        A CUDA option skipped because the GPU is not allowed is noted in `skipped_gpu`, so the
        slot knows the option a reload brings it back to.
        """
        options = _chain(self.settings, slot)
        end = len(options) if stop is None else stop
        for index in range(start, end):
            option = options[index]
            arm = ENGINE_TABLE.get(option.engine)
            loader = self.loaders.chains.get(option.engine)
            if arm is None or arm.slot != slot or loader is None:
                logger.error("%s chain: unknown engine %r skipped", slot, option.engine)
                continue
            if option.device not in arm.devices:
                logger.error(
                    "%s chain: %s has no recorded arm on %s; skipped",
                    slot,
                    option.engine,
                    option.device,
                )
                continue
            if _on_gpu(option.device) and not self.gpu_allowed:
                logger.info(
                    "%s chain: %s@%s skipped, GPU yielded", slot, option.engine, option.device
                )
                self.skipped_gpu.setdefault(slot, index)
                continue
            try:
                engine, vram = self._load(
                    option.device,
                    lambda loader=loader, option=option: loader(self.settings, option.device),
                )
            except VoiceAssetError as error:
                logger.error("%s chain: %s@%s: %s", slot, option.engine, option.device, error)
                self._after_failure(option.device)
                continue
            except self.tolerated as error:
                if self.failure_level is None:
                    logger.warning(
                        "%s chain: %s@%s failed to load: %s",
                        slot,
                        option.engine,
                        option.device,
                        error,
                    )
                else:
                    logger.log(
                        self.failure_level,
                        "%s chain: %s@%s failed to load (%s)",
                        slot,
                        option.engine,
                        option.device,
                        type(error).__name__,
                    )
                self._after_failure(option.device)
                continue
            model_id = engine.model_id  # type: ignore[attr-defined]
            logger.info("%s: %s loaded on %s (%s)", slot, option.engine, option.device, model_id)
            return engine, _status(option.device, model_id, vram), index
        return None, OFF, None

    def _after_failure(self, device: str) -> None:
        """Return what a failed CUDA load allocated before the walk tries the next option."""
        if _on_gpu(device) and self.gpu_failed is not None:
            self.gpu_failed()

    def single(
        self, name: str, device: str, load: Callable[[], _T]
    ) -> tuple[_T | None, EngineStatus]:
        """A single-device engine: loaded, or off with a WARNING."""
        if _on_gpu(device) and not self.gpu_allowed:
            logger.info("%s on %s skipped, GPU yielded", name, device)
            return None, OFF
        try:
            engine, vram = self._load(device, load)
        except MistError as error:
            logger.warning("%s failed to load on %s: %s", name, device, error)
            return None, OFF
        model_id = engine.model_id  # type: ignore[attr-defined]
        return engine, _status(device, model_id, vram)


def load_speech_engines(
    settings: VoiceSettings,
    loaders: EngineLoaders,
    *,
    power_level: PowerLevel = "full",
    gpu_meter: GpuMeter | None = None,
) -> SpeechEngines:
    """Load every speech slot, one engine at a time.

    STT first: with it configured off (an empty chain), no utterance can be transcribed, so the
    VAD, end-of-turn model, wake verifier and name spotter are not loaded. A configured chain
    loads them even when no option loaded, so a boot while yielded with a CUDA-only chain has
    them ready when the reload brings STT up.

    The verifier loads before the spotter, and the spotter loads only beside a verifier trained
    against it (`spotter_model_id` equal to `kws_model`): without one no wake can be accepted,
    so the spotter stays off and one WARNING says why (fail closed).

    Each chain keeps its first option that loads; at power `off` the GPU belongs to something
    else, so every CUDA option is skipped and the walk goes on down the chain. A sum of measured VRAM over `vram_budget_mib` logs a WARNING and sets
    `vram_over_budget`; nothing is unloaded.

    STT and TTS each get a slot the GPU release and reload swap. Its `home` is the option a
    reload brings it back to: the one loaded, or at `off` the first CUDA option the walk
    skipped. A slot none of whose options can load gets no slot, so T12's behaviour for an off
    engine stands.
    """
    placement = settings.placement
    walk = _Walk(settings, loaders, gpu_meter, gpu_allowed=power_level != "off")
    transcriber, stt, stt_index = walk.chain("stt")
    vad = turn_end = spotter = None
    verifier: WakeVerifier | None = None
    vad_status = eot_status = kws_status = OFF
    if placement.stt_chain:
        vad, vad_status = walk.single("vad", placement.vad_device, lambda: loaders.vad(placement))
        turn_end, eot_status = walk.single(
            "eot", placement.eot_device, lambda: loaders.eot(placement, settings.endpointing)
        )
        load_verifier = loaders.verify
        if placement.verify_model is not None and load_verifier is not None:
            verifier, _ = walk.single("verify", "cpu", lambda: load_verifier(placement))
        kws_model = placement.kws_model
        if kws_model is not None:
            if placement.verify_model is None:
                logger.warning(
                    "kws: %s not loaded: no wake verifier is set (MIST_VOICE_VERIFY_MODEL), so "
                    "no wake could be accepted",
                    kws_model,
                )
            elif verifier is None:
                logger.warning(
                    "kws: %s not loaded: the wake verifier %s did not load, so no wake could be "
                    "accepted",
                    kws_model,
                    placement.verify_model,
                )
            elif verifier.spotter_model_id != kws_model:
                logger.warning(
                    "kws: %s not loaded: the wake verifier %s was trained against spotter %s",
                    kws_model,
                    verifier.model_id,
                    verifier.spotter_model_id,
                )
            else:
                spotter, kws_status = walk.single("kws", "cpu", lambda: loaders.kws(placement))
    synthesizer, tts, tts_index = walk.chain("tts")
    stt_home = walk.skipped_gpu.get("stt", stt_index)
    tts_home = walk.skipped_gpu.get("tts", tts_index)
    over = walk.vram_mib > placement.vram_budget_mib
    if over:
        logger.warning(
            "speech engines take %d MiB of VRAM, over the %d MiB budget",
            walk.vram_mib,
            placement.vram_budget_mib,
        )
    return SpeechEngines(
        vad=vad,
        turn_end=turn_end,
        transcriber=transcriber,  # type: ignore[arg-type]
        synthesizer=synthesizer,  # type: ignore[arg-type]
        spotter=spotter,
        statuses=EngineStatuses(vad=vad_status, eot=eot_status, stt=stt, tts=tts, kws=kws_status),
        vram_over_budget=over,
        stt_slot=(
            None
            if stt_home is None
            else TranscriberSlot(transcriber, option=stt_index, home=stt_home)  # type: ignore[arg-type]
        ),
        tts_slot=(
            None
            if tts_home is None
            else SynthesizerSlot(synthesizer, option=tts_index, home=tts_home)  # type: ignore[arg-type]
        ),
        verifier=verifier,
    )


def voice_capabilities(engines: SpeechEngines) -> VoiceCapabilities:
    """What `session_started` announces: the modes and spotters the loaded engines serve.

    Every wake needs the verifier, so `endpoint` is offered only with one, `server` only with
    the spotter as well, and `wake` only while some spotter is offered: a client is never
    offered a mode nothing can accept.
    """
    listens = engines.vad is not None and engines.transcriber is not None
    spotters: list[Literal["server", "endpoint"]] = []
    if listens and engines.verifier is not None:
        spotters.append("endpoint")
        if engines.spotter is not None:
            spotters.append("server")
    return VoiceCapabilities(
        protocol=2,
        modes=["wake"] if spotters else [],
        spotters=spotters,
        reference=listens and engines.spotter is not None,
        uplink_sample_rate=16000,
        downlink_sample_rate=24000,
        barge_in="keyword",
        tts=engines.synthesizer is not None,
        stt=engines.transcriber is not None,
        eot=engines.turn_end is not None,
    )


def primary_placement(engines: SpeechEngines, settings: VoiceSettings) -> TurnPlacement:
    """The placement a turn reads at its open until T22's arbiter replaces the provider."""
    return TurnPlacement(
        mode="primary",
        stt=engines.statuses.stt.location,
        tts=engines.statuses.tts.location,
        budget=settings.budget,
    )


class PowerHost(Protocol):
    """Where the GPU control sends what it causes: `SessionRegistry` in production."""

    def broadcast(self, message: object) -> None:
        """Queue a process-wide message for every session."""
        ...

    def broadcast_voice_state(self) -> None:
        """Send each session its own `voice_state`."""
        ...


_SLOTS: tuple[Slot, ...] = ("stt", "tts")
_RELEASE_ERRORS = (RealtimeError, RuntimeError, OSError)
_FINISHER_POLL_S = 0.1


@dataclass(slots=True)
class _Moving:
    """A slot a release took off CUDA, and the engine it holds until the drop."""

    name: Slot
    slot: TranscriberSlot | SynthesizerSlot
    engine: object | None
    option: int | None


class GpuEngines:
    """Gaming mode's release and reload of the STT and TTS engines placed on CUDA.

    `GpuYield` runs `release` and `reload` on a worker thread, one at a time; both take the
    release's lock, as each retry does, so one engine loads at a time. The closes and the cut a
    release starts run on the loop, and the thread waits for them; every message either causes
    reaches the loop through `call_soon_threadsafe`. Neither raises: a `RealtimeError`, a
    `RuntimeError` or an `OSError` inside is logged at ERROR, because `GpuYield` would read a
    raised error as a failed resume and refuse every turn while the CPU engines could serve.

    Args:
        engines: The shared engines, whose slots and statuses are swapped in place.
        settings: The voice settings; the chains and `release_timeout_s` are read.
        loaders: How each engine loads.
        torch_loader: Imports torch on first use, for `torch.cuda`.
        collect: `gc.collect`.
        sleep: Awaitable sleep between retries.
        gpu_meter: Measures the VRAM a CUDA load takes; None records none.
    """

    def __init__(
        self,
        engines: SpeechEngines,
        settings: VoiceSettings,
        loaders: EngineLoaders,
        *,
        torch_loader: Callable[[], object],
        collect: Callable[[], object] = gc.collect,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        gpu_meter: GpuMeter | None = None,
    ) -> None:
        self._engines = engines
        self._settings = settings
        self._loaders = loaders
        self._torch_loader = torch_loader
        self._collect = collect
        self._sleep = sleep
        self._gpu_meter = gpu_meter
        self._lock = threading.Lock()
        self._released = False
        self._level: PowerLevel = "full"
        self._loop: asyncio.AbstractEventLoop | None = None
        self._host: PowerHost | None = None
        self._retry: asyncio.Task[None] | None = None
        self._stopped = threading.Event()
        self.finishers: list[threading.Thread] = []

    def attach(self, loop: asyncio.AbstractEventLoop, host: PowerHost) -> None:
        """The loop the closes, the cut and the messages run on, and where messages go."""
        self._loop = loop
        self._host = host

    @property
    def retrying(self) -> bool:
        """Whether a slot left below its device is being retried."""
        task = self._retry
        return task is not None and not task.done()

    def stop(self) -> None:
        """Stop the retries and any late finisher; the loop is shutting down."""
        self._stopped.set()
        task, self._retry = self._retry, None
        if task is not None:
            task.cancel()

    # -- Release ---------------------------------------------------------------------------

    def release(self) -> int:
        """Drop every STT and TTS engine on CUDA and return the MiB the cache gave back.

        Each slot on CUDA goes `off` for new work, the jobs on its engine are cancelled on the
        loop, and the release waits at most `release_timeout_s` for them to stop. Then each
        engine's `release()` runs, the engine is dropped, `gc.collect()` and
        `torch.cuda.empty_cache()` run, and each slot moves to its next CPU option, and
        `power_off` carries the MiB. A job that outlives the timeout leaves the slots `off` with
        `release_timeout`; a finisher thread drops them once it ends, and `power_off` follows.
        A second release before a reload does nothing and returns 0.
        """
        with self._lock:
            if self._released:
                return 0
            self._released = True
            self._cancel_retries()
            moving = self._take_cuda_slots()
            if moving:
                self._on_loop(lambda: self._cancel_jobs(moving))
            deadline = time.monotonic() + self._settings.placement.release_timeout_s
            stuck = [
                entry.name
                for entry in moving
                if entry.engine is not None
                and not entry.slot.wait_idle(entry.engine, max(0.0, deadline - time.monotonic()))
            ]
            if stuck:
                logger.warning(
                    "GPU release: a job on %s outlived %.1f s; the slots stay off until it ends",
                    ", ".join(stuck),
                    self._settings.placement.release_timeout_s,
                )
                self._send_placement("release_timeout", None)
                self._post(self._broadcast_voice_state)
                finisher = threading.Thread(
                    target=self._finish_late, args=(moving,), name="gpu-release-finisher"
                )
                finisher.daemon = True
                self.finishers.append(finisher)
                finisher.start()
                return 0
            released = self._drop(moving)
            self._send_placement("power_off", released)
            self._post(self._broadcast_voice_state)
            return released

    def _take_cuda_slots(self) -> list[_Moving]:
        moving: list[_Moving] = []
        for name in _SLOTS:
            slot = self._slot(name)
            if slot is None or slot.option is None:
                continue
            if not _on_gpu(_chain(self._settings, name)[slot.option].device):
                continue
            engine = slot.swap(None)
            moving.append(_Moving(name=name, slot=slot, engine=engine, option=slot.option))
            slot.option = None
            self._set(name, None, OFF)
        return moving

    @staticmethod
    def _cancel_jobs(moving: list[_Moving]) -> None:
        for entry in moving:
            if entry.engine is not None:
                entry.slot.cancel_jobs(entry.engine)

    def _finish_late(self, moving: list[_Moving]) -> None:
        """Wait for the stuck jobs, then drop their engines and send `power_off`."""
        for entry in moving:
            while entry.engine is not None and not entry.slot.wait_idle(
                entry.engine, _FINISHER_POLL_S
            ):
                if self._stopped.is_set():
                    return
        with self._lock:
            released = self._drop(moving)
            if self._released:
                self._send_placement("power_off", released)
                self._post(self._broadcast_voice_state)

    def _drop(self, moving: list[_Moving]) -> int:
        """Release and drop each engine, empty the cache, then move each slot to the CPU."""
        self._release_engines(moving)
        self._collect()
        released = self._empty_cache()
        if self._released:
            for entry in moving:
                if entry.slot.option is None:
                    self._move_to_cpu(entry)
        return released

    def _release_engines(self, moving: list[_Moving]) -> None:
        for entry in moving:
            engine, entry.engine = entry.engine, None
            if engine is None:
                continue
            try:
                engine.release()  # type: ignore[attr-defined]
            except _RELEASE_ERRORS as error:
                logger.error(
                    "GPU release: %s engine %s failed to release (%s); dropped anyway",
                    entry.name,
                    getattr(engine, "model_id", "?"),
                    type(error).__name__,
                )

    def _empty_cache(self) -> int:
        try:
            cuda = self._torch_loader().cuda  # type: ignore[attr-defined]
            if not cuda.is_available():
                return 0
            before = cuda.memory_reserved()
            cuda.empty_cache()
            after = cuda.memory_reserved()
        except _RELEASE_ERRORS as error:
            logger.error("GPU release: emptying the CUDA cache failed (%s)", type(error).__name__)
            return 0
        return max(0, math.ceil((before - after) / MIB))

    def _move_to_cpu(self, entry: _Moving) -> None:
        start = 0 if entry.option is None else entry.option + 1
        walk = self._walk(gpu_allowed=False, failure_level=logging.ERROR)
        engine, status, index = walk.walk(entry.name, start)
        self._swap(entry.name, entry.slot, engine, status, index)

    # -- Reload ----------------------------------------------------------------------------

    def reload(self) -> None:
        """Bring each slot back to its chain position from before the release.

        One engine loads at a time. A slot whose load fails stays on the option it is on; one
        `reload_failed` says where each slot settled, and the slots below their device are
        retried. A rise's `power_restored` is the watcher's, sent after this returns.
        """
        with self._lock:
            self._released = False
            failed = [
                name
                for name in _SLOTS
                if self._below_home(self._slot(name))
                and not self._raise_slot(name, failure_level=logging.ERROR)
            ]
            if not failed:
                return
            statuses = self._engines.statuses
            logger.warning(
                "GPU reload: %s did not load on its device; STT is on %s and TTS on %s, retrying",
                ", ".join(failed),
                statuses.stt.location,
                statuses.tts.location,
            )
            self._send_placement("reload_failed", None)
            self._post(self._start_retries)

    def _raise_slot(self, name: Slot, *, failure_level: int) -> bool:
        """Walk the slot from its home up to the option it is on; True once it is home."""
        slot = self._slot(name)
        assert slot is not None and slot.home is not None  # noqa: S101 - `_below_home` held
        stop = len(_chain(self._settings, name)) if slot.option is None else slot.option
        walk = self._walk(gpu_allowed=True, failure_level=failure_level)
        engine, status, index = walk.walk(name, slot.home, stop)
        if engine is not None:
            self._swap(name, slot, engine, status, index)
        return not self._below_home(slot)

    def _start_retries(self) -> None:
        if self._released or self._stopped.is_set() or self.retrying:
            return
        assert self._loop is not None  # noqa: S101 - only posted once attached
        self._retry = self._loop.create_task(self._retry_loop(), name="gpu-engine-retry")

    async def _retry_loop(self) -> None:
        delay = DEFAULT_POLL_S
        while True:
            await self._sleep(delay)
            if await asyncio.to_thread(self._retry_once):
                return
            delay = min(delay * 2, MAX_RETRY_DELAY_S)

    def _retry_once(self) -> bool:
        """One engine's try under the release's lock; True when nothing is left to retry."""
        with self._lock:
            if self._released or self._stopped.is_set():
                return True
            for name in _SLOTS:
                slot = self._slot(name)
                if not self._below_home(slot):
                    continue
                before = slot.option  # type: ignore[union-attr]
                self._raise_slot(name, failure_level=logging.INFO)
                if slot.option != before:  # type: ignore[union-attr]
                    logger.info("GPU reload retry: %s loaded on its device again", name)
                    self._send_placement("power_restored", None)
                    self._post(self._broadcast_voice_state)
                break
            return not any(self._below_home(self._slot(name)) for name in _SLOTS)

    def _cancel_retries(self) -> None:
        task, self._retry = self._retry, None
        if task is not None:
            self._post(task.cancel)

    # -- The level -------------------------------------------------------------------------

    def level_changed(self, level: PowerLevel) -> None:
        """The watcher's report: each session gets its `voice_state`; a rise `power_restored`."""
        previous, self._level = self._level, level
        if LEVEL_RANK[level] > LEVEL_RANK[previous]:
            self._send_placement("power_restored", None, level=level)
        self._post(self._broadcast_voice_state)

    # -- Plumbing --------------------------------------------------------------------------

    def _slot(self, name: Slot) -> TranscriberSlot | SynthesizerSlot | None:
        engines = self._engines
        return engines.stt_slot if name == "stt" else engines.tts_slot

    @staticmethod
    def _below_home(slot: TranscriberSlot | SynthesizerSlot | None) -> bool:
        if slot is None or slot.home is None:
            return False
        return slot.option is None or slot.option > slot.home

    def _walk(self, *, gpu_allowed: bool, failure_level: int) -> _Walk:
        return _Walk(
            self._settings,
            self._loaders,
            self._gpu_meter,
            gpu_allowed,
            failure_level=failure_level,
            gpu_failed=self._empty_cache,
        )

    def _swap(
        self,
        name: Slot,
        slot: TranscriberSlot | SynthesizerSlot,
        engine: object | None,
        status: EngineStatus,
        index: int | None,
    ) -> None:
        slot.swap(engine)
        slot.option = index
        self._set(name, engine, status)

    def _set(self, name: Slot, engine: object | None, status: EngineStatus) -> None:
        engines = self._engines
        if name == "stt":
            engines.transcriber = engine  # type: ignore[assignment]
        else:
            engines.synthesizer = engine  # type: ignore[assignment]
        engines.statuses = engines.statuses.model_copy(update={name: status})

    def _send_placement(
        self, reason: PlacementReason, released_mib: int | None, *, level: PowerLevel | None = None
    ) -> None:
        if level is None:
            level = "off" if reason in ("power_off", "release_timeout") else "full"
        statuses = self._engines.statuses
        message = VoiceDecisionPlacement(
            reason=reason,
            detail=PlacementDetail(
                power_level=level,
                engines=EnginePlacement(stt=statuses.stt.location, tts=statuses.tts.location),
                released_mib=released_mib,
            ),
        )
        self._post(lambda: self._broadcast(message))

    def _broadcast(self, message: object) -> None:
        if self._host is not None:
            self._host.broadcast(message)

    def _broadcast_voice_state(self) -> None:
        if self._host is not None:
            self._host.broadcast_voice_state()

    def _on_loop_thread(self) -> bool:
        try:
            return asyncio.get_running_loop() is self._loop
        except RuntimeError:
            return False

    def _post(self, call: Callable[[], object]) -> None:
        """Run `call` on the loop: now when already on it, else through `call_soon_threadsafe`."""
        loop = self._loop
        if loop is None or self._on_loop_thread():
            call()
            return
        try:
            loop.call_soon_threadsafe(call)
        except RuntimeError:
            logger.warning("GPU control: the loop is closed; a message was not sent")

    def _on_loop(self, call: Callable[[], None]) -> None:
        """Run `call` on the loop and wait for it, at most the release timeout."""
        loop = self._loop
        if loop is None or self._on_loop_thread():
            call()
            return

        async def run() -> None:
            call()

        try:
            future = asyncio.run_coroutine_threadsafe(run(), loop)
            future.result(timeout=self._settings.placement.release_timeout_s)
        except _RELEASE_ERRORS as error:
            logger.error(
                "GPU release: cancelling jobs on the loop failed (%s)", type(error).__name__
            )


@dataclass(frozen=True, slots=True)
class MistRuntime:
    """What startup built, shared by every session.

    `engines` is mutable on purpose: T22's `release_gpu_engines` and `reload_gpu_engines` swap
    its slots and statuses, and sessions read them at the next `voice_start` or `voice_state`.
    `agent_available` is false when turns are answered by `DirectResponder`. `gpu` is the GPU
    control `GpuYield`'s callbacks reach; None (T13 wires it) releases nothing.
    """

    llm_provider: object
    conversation_handler: object | None
    responder: TurnResponder
    engines: SpeechEngines
    lanes: InferenceLanes
    settings: VoiceSettings
    services: SessionServices
    agent_available: bool
    gpu: GpuEngines | None = None

    def release_gpu_engines(self) -> int:
        """`GpuYield`'s `on_yield`: drop the CUDA speech engines; the MiB freed."""
        return 0 if self.gpu is None else self.gpu.release()

    def reload_gpu_engines(self) -> None:
        """`GpuYield`'s `on_resume`: bring the slots back to their devices; never raises."""
        if self.gpu is not None:
            self.gpu.reload()

    def voice_capabilities(self) -> VoiceCapabilities:
        """The `capabilities.voice` of `session_started`."""
        return voice_capabilities(self.engines)

    def close(self) -> None:
        """Stop each lane's worker thread, then close the wake verifier.

        The lanes go first: a shutdown fails the waiting jobs and waits for the running one, so
        no new verify job starts once the verifier is closed.
        """
        lanes = self.lanes
        for lane in (lanes.vad, lanes.eot, lanes.stt, lanes.tts, lanes.kws, lanes.verify):
            shutdown = getattr(lane, "shutdown", None)
            if shutdown is not None:
                shutdown()
        verifier = self.engines.verifier
        if verifier is not None:
            verifier.close()
