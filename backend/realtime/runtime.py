"""The composed realtime runtime, and the walk that loads each speech slot from its chain.

`build_mist_runtime` in `backend/factories.py` builds `MistRuntime` once at startup; this module
holds what that composition produces and the engine walk it runs, so the walk is testable with
fake loaders and no factory import. No library is imported here: every engine loads through an
injected loader, and the GPU meter imports torch only when a CUDA option loads.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol, TypeVar

from backend.errors import MistError

from .budget import EndpointingConfig, PlacementConfig, VoiceSettings
from .protocol import EngineStatus, EngineStatuses, PowerLevel, VoiceCapabilities
from .session import OFF, InferenceLanes, SessionServices, SpeechEngines
from .speech import NameSpotter, TurnEndDetector, VoiceActivityDetector
from .turn import TurnPlacement, TurnResponder
from .voice_assets import VoiceAssetError

logger = logging.getLogger(__name__)

_T = TypeVar("_T")
Slot = Literal["stt", "tts"]
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

    `chains` maps an engine id to `(settings, device) -> engine`; `vad`, `eot` and `kws` load
    the single-device engines from the placement.
    """

    chains: Mapping[str, Callable[[VoiceSettings, str], object]]
    vad: Callable[[PlacementConfig], VoiceActivityDetector]
    eot: Callable[[PlacementConfig, EndpointingConfig], TurnEndDetector]
    kws: Callable[[PlacementConfig], NameSpotter]


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


class _Walk:
    """One startup's engine loads, one at a time."""

    def __init__(
        self,
        settings: VoiceSettings,
        loaders: EngineLoaders,
        gpu_meter: GpuMeter | None,
        gpu_allowed: bool,
    ) -> None:
        self.settings = settings
        self.loaders = loaders
        self.gpu_meter = gpu_meter
        self.gpu_allowed = gpu_allowed
        self.vram_mib = 0

    def _load(self, device: str, load: Callable[[], _T]) -> tuple[_T, int | None]:
        if _on_gpu(device) and self.gpu_meter is not None:
            engine, vram = self.gpu_meter.measure(load)
            self.vram_mib += vram
            return engine, vram
        return load(), None

    def chain(self, slot: Slot) -> tuple[object | None, EngineStatus]:
        """The first option of the slot's chain that loads, or None and `off`."""
        placement = self.settings.placement
        options = placement.stt_chain if slot == "stt" else placement.tts_chain
        for option in options:
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
                continue
            try:
                engine, vram = self._load(
                    option.device,
                    lambda loader=loader, option=option: loader(self.settings, option.device),
                )
            except VoiceAssetError as error:
                logger.error("%s chain: %s@%s: %s", slot, option.engine, option.device, error)
                continue
            except MistError as error:
                logger.warning(
                    "%s chain: %s@%s failed to load: %s", slot, option.engine, option.device, error
                )
                continue
            model_id = engine.model_id  # type: ignore[attr-defined]
            logger.info("%s: %s loaded on %s (%s)", slot, option.engine, option.device, model_id)
            return engine, _status(option.device, model_id, vram)
        if options:
            logger.warning("%s: no option of the chain loaded; the slot is off", slot)
        return None, OFF

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

    STT first: with it off, no utterance can be transcribed, so the VAD, end-of-turn model and
    name spotter are not loaded. Each chain keeps its first option that loads; at power `off`
    the GPU belongs to something else, so every CUDA option is skipped and the walk goes on down
    the chain. A sum of measured VRAM over `vram_budget_mib` logs a WARNING and sets
    `vram_over_budget`; nothing is unloaded.
    """
    placement = settings.placement
    walk = _Walk(settings, loaders, gpu_meter, gpu_allowed=power_level != "off")
    transcriber, stt = walk.chain("stt")
    vad = turn_end = spotter = None
    vad_status = eot_status = kws_status = OFF
    if transcriber is not None:
        vad, vad_status = walk.single(
            "vad", placement.vad_device, lambda: loaders.vad(placement)
        )
        turn_end, eot_status = walk.single(
            "eot", placement.eot_device, lambda: loaders.eot(placement, settings.endpointing)
        )
        if placement.kws_model is not None:
            spotter, kws_status = walk.single("kws", "cpu", lambda: loaders.kws(placement))
    synthesizer, tts = walk.chain("tts")
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
    )


def voice_capabilities(engines: SpeechEngines) -> VoiceCapabilities:
    """What `session_started` announces: the modes and spotters the loaded engines serve."""
    listens = engines.vad is not None and engines.transcriber is not None
    spotters: list[Literal["server", "endpoint"]] = []
    if listens:
        spotters.append("endpoint")
        if engines.spotter is not None:
            spotters.append("server")
    return VoiceCapabilities(
        protocol=2,
        modes=["wake"] if listens else [],
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


@dataclass(frozen=True, slots=True)
class MistRuntime:
    """What startup built, shared by every session.

    `engines` is mutable on purpose: T22's `release_gpu_engines` and `reload_gpu_engines` swap
    its slots and statuses, and sessions read them at the next `voice_start` or `voice_state`.
    `agent_available` is false when turns are answered by `DirectResponder`.
    """

    llm_provider: object
    conversation_handler: object | None
    responder: TurnResponder
    engines: SpeechEngines
    lanes: InferenceLanes
    settings: VoiceSettings
    services: SessionServices
    agent_available: bool

    def voice_capabilities(self) -> VoiceCapabilities:
        """The `capabilities.voice` of `session_started`."""
        return voice_capabilities(self.engines)

    def close(self) -> None:
        """Stop each lane's worker thread."""
        lanes = self.lanes
        for lane in (lanes.vad, lanes.eot, lanes.stt, lanes.tts, lanes.kws):
            shutdown = getattr(lane, "shutdown", None)
            if shutdown is not None:
                shutdown()
