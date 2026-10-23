"""Voice latency budget, voice configuration and the loader that reads them from the environment.

The budget is a set of per-stage targets that changes no behaviour: `turn_metrics` reports each
measured stage against it, and the latency bench compares against it. Behaviour lives in the
configuration dataclasses (`EndpointingConfig`, `BargeInConfig` and the rest). Both are read
once, at startup, by `load_voice_settings`, which raises `ValueError` naming the variable for any
malformed value, so a typo in `.env` stops the backend with its name instead of surfacing later
as an engine failure.
"""

import dataclasses
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, TypeVar
from urllib.parse import urlsplit

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class VoiceLatencyBudget:
    """Per-stage latency targets in milliseconds, with the name of the preset they came from.

    `total_p50_ms` is the sum of the seven stages from `end_of_turn_ms` to
    `transport_playback_ms` (`STAGE_FIELDS`); `total_p90_ms`, `tool_call_ms` and
    `barge_in_reaction_ms` are separate targets outside that sum. `preset` travels with the
    figures so `turn_metrics.budget.preset` can say which preset a turn was judged against.
    """

    preset: str
    end_of_turn_ms: int
    stt_final_ms: int
    context_build_ms: int
    llm_first_token_ms: int
    first_segment_ms: int
    tts_first_chunk_ms: int
    transport_playback_ms: int
    total_p50_ms: int
    total_p90_ms: int
    tool_call_ms: int
    barge_in_reaction_ms: int


BUDGET_FIELDS: tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(VoiceLatencyBudget) if f.name != "preset"
)
"""The budget table's eleven rows, in table order."""

STAGE_FIELDS: tuple[str, ...] = BUDGET_FIELDS[:7]
"""The seven serial stages whose sum is `total_p50_ms`."""


def _preset(name: str, *rows: int) -> VoiceLatencyBudget:
    return VoiceLatencyBudget(name, *rows)


# The Design doc's Latency budget table. `floor` is what the 4070 SUPER host can do with no game
# running, from measured rates where they exist and estimates where they do not; `floor_yield`
# is STT and TTS on the CPU or a remote host while a game shares the card (estimates); `memo` is
# the model-host memo's P50 budget sized for a larger GPU. Measured P50s replace the estimates
# through `MIST_VOICE_BUDGET` overrides, not code changes.
BUDGET_PRESETS: Mapping[str, VoiceLatencyBudget] = MappingProxyType(
    {
        "floor": _preset("floor", 250, 250, 0, 250, 120, 500, 100, 1470, 2100, 450, 700),
        "floor_yield": _preset(
            "floor_yield", 250, 600, 0, 500, 240, 1200, 130, 2920, 4100, 900, 1200
        ),
        "memo": _preset("memo", 250, 50, 0, 200, 100, 150, 100, 850, 1200, 270, 700),
    }
)

DEFAULT_BUDGET = "floor"
DEFAULT_YIELD_BUDGET = "floor_yield"


def parse_budget(spec: str) -> VoiceLatencyBudget:
    """Build a budget from a preset name optionally followed by row overrides.

    `"floor"` returns the preset; `"floor,stt_final_ms=300"` replaces that row and recomputes
    `total_p50_ms` as the stage sum, so the total can never disagree with its stages, which is
    also why the total itself cannot be overridden.

    Raises:
        ValueError: The preset or a row name is unknown, an override names `total_p50_ms`, or
            an override is not `<row>=<non-negative integer>`. The message names the offender.
    """
    name, *items = (part.strip() for part in spec.split(","))
    if name not in BUDGET_PRESETS:
        raise ValueError(
            f"unknown budget preset {name!r}; expected one of {', '.join(sorted(BUDGET_PRESETS))}"
        )
    overrides: dict[str, int] = {}
    for item in items:
        row, sep, raw = (piece.strip() for piece in item.partition("="))
        if row == "total_p50_ms":
            raise ValueError("total_p50_ms cannot be overridden: it is the sum of the stages")
        if row not in BUDGET_FIELDS:
            raise ValueError(f"unknown budget row {row!r}")
        if not sep:
            raise ValueError(f"budget override {row!r} needs a value: {row}=<milliseconds>")
        try:
            value = int(raw)
        except ValueError as exc:
            raise ValueError(f"budget override {row} must be an integer, got {raw!r}") from exc
        if value < 0:
            raise ValueError(f"budget override {row} must not be negative, got {value}")
        overrides[row] = value
    budget = dataclasses.replace(BUDGET_PRESETS[name], **overrides)
    total = sum(getattr(budget, stage) for stage in STAGE_FIELDS)
    return dataclasses.replace(budget, total_p50_ms=total)


@dataclass(frozen=True, slots=True)
class EndpointingConfig:
    """How ingress decides an utterance started and ended (Design doc, Ingress and endpointing).

    Speech starts after `speech_start_windows` consecutive 512-sample VAD windows at or above
    `vad_threshold`, keeping `pre_roll_ms` of audio before them. Partials are requested every
    `stt_partial_interval_ms`, or `stt_partial_interval_speaking_ms` while MIST speaks, for
    keyword detection. After `endpoint_silence_ms` of silence the final decode and the end-of-turn
    score of the last `eot_window_ms` start together; a probability above `eot_threshold` ends
    the turn, otherwise it is re-checked every `eot_recheck_ms` until `eot_max_silence_ms`. An
    utterance under `min_utterance_ms` with an empty final is noise. A final with
    `no_speech_prob` above `no_speech_threshold` and `avg_logprob` at or below
    `logprob_threshold` is dropped as non-speech, the thresholds openai-whisper uses. Audio
    queued past `max_backlog_ms` is dropped oldest first.
    """

    vad_threshold: float = 0.5
    speech_start_windows: int = 3
    pre_roll_ms: int = 300
    stt_partial_interval_ms: int = 400
    stt_partial_interval_speaking_ms: int = 250
    endpoint_silence_ms: int = 200
    eot_threshold: float = 0.5
    eot_recheck_ms: int = 300
    eot_max_silence_ms: int = 1500
    eot_window_ms: int = 8000
    min_utterance_ms: int = 250
    max_backlog_ms: int = 2000
    speculative_start: bool = True
    no_speech_threshold: float = 0.6
    logprob_threshold: float = -1.0


@dataclass(frozen=True, slots=True)
class BargeInConfig:
    """What user speech during MIST's speech means (Design doc, Barge-in policy).

    The keyword policy arms `arm_delay_ms` after a turn's first audio reached the client, matches
    `keywords` (normalized: lower case, single spaces) as whole words in the first
    `keyword_window_words` words, and ignores a hit heard in TTS text played within
    `echo_guard_ms`. `wake_arbitration_ms` is the window in which wake reports from several
    endpoints are compared and only the strongest answered.
    """

    keywords: tuple[str, ...] = ("stop", "wait", "hold on", "cancel", "never mind")
    arm_delay_ms: int = 1000
    keyword_window_words: int = 3
    echo_guard_ms: int = 1500
    wake_arbitration_ms: int = 250


@dataclass(frozen=True, slots=True)
class PlaybackConfig:
    """Downlink pacing and the client's playback buffer."""

    client_prebuffer_ms: int = 80
    max_lead_ms: int = 500
    fade_ms: int = 30
    progress_interval_ms: int = 100
    frame_ms: int = 40


@dataclass(frozen=True, slots=True)
class SegmenterConfig:
    """How reply text is cut into TTS segments.

    `first_segment_min_chars` is not read from the environment here: the runtime copies the
    handler's `LLMConfig.voice_first_segment_min_chars`, so the handler's first emission and the
    first TTS segment cut at the same point.
    """

    first_segment_min_chars: int = 40
    max_segment_chars: int = 200


# The TTS model each variant loads when `PlacementConfig.tts_model` is `None`.
TTS_MODEL_PINS: Mapping[str, str] = MappingProxyType(
    {
        "turbo": "ResembleAI/chatterbox-turbo@749d1c1a46eb10492095d68fbcf55691ccf137cd",
        "original": "ResembleAI/chatterbox@5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18",
    }
)


@dataclass(frozen=True, slots=True)
class PlacementConfig:
    """Where each engine runs and which weights it loads.

    A location is `cpu`, `cuda`, `cuda:<n>`, `remote` or `off`; a placement is an ordered tuple
    of locations of which the first that loads wins, with `off` (no engine) only last. STT and
    TTS have a placement for each GPU mode, primary and yield; VAD and end of turn run on one
    device in both. Model settings are `<repo>@<revision>` for TTS and
    `<repo>@<revision>:<file>` for the ONNX engines, so every download is pinned.
    """

    vad_device: str = "cpu"
    eot_device: str = "cpu"
    stt_primary: tuple[str, ...] = ("cuda", "cpu")
    stt_yield: tuple[str, ...] = ("remote", "cpu")
    tts_primary: tuple[str, ...] = ("cuda", "remote", "off")
    tts_yield: tuple[str, ...] = ("remote", "off")
    vad_model: str = (
        "onnx-community/silero-vad@e71cae966052b992a7eca6b17738916ce0eca4ec:onnx/model.onnx"
    )
    eot_model: str = (
        "pipecat-ai/smart-turn-v3@f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
        ":smart-turn-v3.2-cpu.onnx"
    )
    stt_model: str = "base"
    tts_variant: Literal["turbo", "original"] = "turbo"
    tts_model: str | None = None
    speech_service_url: str | None = None
    vram_budget_mib: int = 4500


@dataclass(frozen=True, slots=True)
class VoiceReferenceConfig:
    """Where the active voice profile's reference audio is read from."""

    profiles_dir: str = "data/voice_profiles"
    profile: str = "jarvis"


@dataclass(frozen=True, slots=True)
class GpuYieldConfig:
    """When the voice engines give the GPU up to a game and take it back.

    `triggers` may hold `manual`, `fullscreen` and `pressure`; pressure yields after free VRAM
    stays under `free_mib` for `yield_after_s`, polled every `poll_s`.
    """

    triggers: frozenset[str] = frozenset({"manual", "fullscreen"})
    free_mib: int = 512
    yield_after_s: float = 3.0
    resume_after_s: float = 30.0
    switch_timeout_s: float = 10.0
    poll_s: float = 1.0


@dataclass(frozen=True, slots=True)
class TurnConfig:
    """Turn teardown limits: how long a dispatched tool may finish after a cancel."""

    tool_cancel_grace_ms: int = 2000


@dataclass(frozen=True, slots=True)
class VoiceSettings:
    """Every voice setting, as `load_voice_settings` read it at startup."""

    budget: VoiceLatencyBudget = field(default_factory=lambda: BUDGET_PRESETS[DEFAULT_BUDGET])
    yield_budget: VoiceLatencyBudget = field(
        default_factory=lambda: BUDGET_PRESETS[DEFAULT_YIELD_BUDGET]
    )
    endpointing: EndpointingConfig = field(default_factory=EndpointingConfig)
    barge_in: BargeInConfig = field(default_factory=BargeInConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    placement: PlacementConfig = field(default_factory=PlacementConfig)
    voice_reference: VoiceReferenceConfig = field(default_factory=VoiceReferenceConfig)
    gpu_yield: GpuYieldConfig = field(default_factory=GpuYieldConfig)
    turn: TurnConfig = field(default_factory=TurnConfig)


_LOCATION = re.compile(r"cpu|cuda(?::\d+)?|remote|off")
_DEVICE = re.compile(r"cpu|cuda(?::\d+)?")
_TTS_MODEL = re.compile(r"[^@:\s]+@[^@:\s]+")
_ONNX_MODEL = re.compile(r"[^@:\s]+@[^@:\s]+:[^@:\s]+")
_TRIGGERS = frozenset({"manual", "fullscreen", "pressure"})
_VARIANTS = ("turbo", "original")
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off"})


class _Env:
    """Typed, validated reads of one environment mapping; each error names its variable."""

    def __init__(self, environ: Mapping[str, str]) -> None:
        self._environ = environ

    def _read(self, name: str, default: _T, parse: Callable[[str, str], _T]) -> _T:
        raw = self._environ.get(name)
        return default if raw is None else parse(name, raw)

    def integer(self, name: str, default: int) -> int:
        return self._read(name, default, _parse_non_negative_int)

    def number(self, name: str, default: float, *, allow_negative: bool = False) -> float:
        def parse(var: str, raw: str) -> float:
            return _parse_float(var, raw, allow_negative=allow_negative)

        return self._read(name, default, parse)

    def probability(self, name: str, default: float) -> float:
        return self._read(name, default, _parse_probability)

    def boolean(self, name: str, default: bool) -> bool:
        return self._read(name, default, _parse_bool)

    def text(self, name: str, default: str) -> str:
        return self._read(name, default, _parse_non_empty)

    def placement(self, name: str, default: tuple[str, ...]) -> tuple[str, ...]:
        return self._read(name, default, _parse_placement)

    def device(self, name: str, default: str) -> str:
        return self._read(name, default, _parse_device)

    def pattern(self, name: str, default: _T, pattern: re.Pattern[str], form: str) -> str | _T:
        def parse(var: str, raw: str) -> str:
            value = raw.strip()
            if not pattern.fullmatch(value):
                raise ValueError(f"{var}: expected {form}, got {raw!r}")
            return value

        return self._read(name, default, parse)

    def choice(self, name: str, default: str, choices: tuple[str, ...]) -> str:
        def parse(var: str, raw: str) -> str:
            value = raw.strip()
            if value not in choices:
                raise ValueError(f"{var}: expected one of {', '.join(choices)}, got {raw!r}")
            return value

        return self._read(name, default, parse)

    def budget(self, name: str, default: str) -> VoiceLatencyBudget:
        def parse(var: str, raw: str) -> VoiceLatencyBudget:
            try:
                return parse_budget(raw)
            except ValueError as exc:
                raise ValueError(f"{var}: {exc}") from exc

        return self._read(name, BUDGET_PRESETS[default], parse)

    def keywords(self, name: str, default: tuple[str, ...]) -> tuple[str, ...]:
        return self._read(name, default, _parse_keywords)

    def triggers(self, name: str, default: frozenset[str]) -> frozenset[str]:
        return self._read(name, default, _parse_triggers)

    def url(self, name: str, default: str | None) -> str | None:
        return self._read(name, default, _parse_base_url)


def _parse_non_negative_int(name: str, raw: str) -> int:
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name}: expected a non-negative integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name}: expected a non-negative integer, got {raw!r}")
    return value


def _parse_float(name: str, raw: str, *, allow_negative: bool) -> float:
    try:
        value = float(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name}: expected a number, got {raw!r}") from exc
    if not math.isfinite(value):
        raise ValueError(f"{name}: expected a finite number, got {raw!r}")
    if value < 0 and not allow_negative:
        raise ValueError(f"{name}: expected a non-negative number, got {raw!r}")
    return value


def _parse_probability(name: str, raw: str) -> float:
    value = _parse_float(name, raw, allow_negative=False)
    if value > 1:
        raise ValueError(f"{name}: expected a probability between 0 and 1, got {raw!r}")
    return value


def _parse_bool(name: str, raw: str) -> bool:
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ValueError(f"{name}: expected true or false, got {raw!r}")


def _parse_non_empty(name: str, raw: str) -> str:
    value = raw.strip()
    if not value:
        raise ValueError(f"{name}: must not be empty")
    return value


def _split_list(raw: str) -> list[str]:
    return [item.strip() for item in raw.split(",") if item.strip()]


def _parse_placement(name: str, raw: str) -> tuple[str, ...]:
    locations = _split_list(raw)
    if not locations:
        raise ValueError(f"{name}: expected a comma-separated list of locations, got {raw!r}")
    for location in locations:
        if not _LOCATION.fullmatch(location):
            raise ValueError(
                f"{name}: unknown location {location!r}; "
                "expected cpu, cuda, cuda:<n>, remote or off"
            )
    if len(set(locations)) != len(locations):
        raise ValueError(f"{name}: a location is repeated in {raw!r}")
    if "off" in locations[:-1]:
        raise ValueError(f"{name}: off may only be the last location, got {raw!r}")
    return tuple(locations)


def _parse_device(name: str, raw: str) -> str:
    value = raw.strip()
    if not _DEVICE.fullmatch(value):
        raise ValueError(f"{name}: expected one device, cpu, cuda or cuda:<n>, got {raw!r}")
    return value


def _parse_keywords(name: str, raw: str) -> tuple[str, ...]:
    keywords = tuple(" ".join(item.lower().split()) for item in _split_list(raw))
    if not keywords:
        raise ValueError(f"{name}: expected at least one comma-separated keyword, got {raw!r}")
    return keywords


def _parse_triggers(name: str, raw: str) -> frozenset[str]:
    triggers = frozenset(_split_list(raw))
    unknown = sorted(triggers - _TRIGGERS)
    if unknown:
        raise ValueError(
            f"{name}: unknown trigger {unknown[0]!r}; expected manual, fullscreen or pressure"
        )
    return triggers


def _parse_base_url(name: str, raw: str) -> str:
    value = raw.strip()
    parts = urlsplit(value)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValueError(f"{name}: expected an http:// or https:// base URL, got {raw!r}")
    if parts.query or parts.fragment:
        raise ValueError(f"{name}: a base URL takes no query or fragment, got {raw!r}")
    return value


def load_voice_settings(environ: Mapping[str, str]) -> VoiceSettings:
    """Read every voice setting from an environment mapping, defaulting what is unset.

    Not read here: `MIST_VOICE_THINKING_BUDGET_TOKENS` and `MIST_VOICE_FIRST_SEGMENT_MIN_CHARS`
    (the handler's `LLMConfig` owns them, and the handler never imports this package) and
    `MIST_VOICE_PROFILES_HOST_DIR` (only Docker Compose substitutes it, into a mount).

    Raises:
        ValueError: A set variable is malformed; the message starts with the variable's name.
    """
    env = _Env(environ)
    defaults = EndpointingConfig()
    endpointing = EndpointingConfig(
        vad_threshold=env.probability("MIST_VOICE_VAD_THRESHOLD", defaults.vad_threshold),
        endpoint_silence_ms=env.integer(
            "MIST_VOICE_ENDPOINT_SILENCE_MS", defaults.endpoint_silence_ms
        ),
        eot_threshold=env.probability("MIST_VOICE_EOT_THRESHOLD", defaults.eot_threshold),
        eot_max_silence_ms=env.integer(
            "MIST_VOICE_EOT_MAX_SILENCE_MS", defaults.eot_max_silence_ms
        ),
        speculative_start=env.boolean("MIST_VOICE_SPECULATIVE_START", defaults.speculative_start),
        no_speech_threshold=env.probability(
            "MIST_VOICE_NO_SPEECH_THRESHOLD", defaults.no_speech_threshold
        ),
        # A log probability is negative by nature, and so is its default.
        logprob_threshold=env.number(
            "MIST_VOICE_LOGPROB_THRESHOLD", defaults.logprob_threshold, allow_negative=True
        ),
    )
    barge_defaults = BargeInConfig()
    barge_in = BargeInConfig(
        keywords=env.keywords("MIST_VOICE_BARGE_IN_KEYWORDS", barge_defaults.keywords),
        arm_delay_ms=env.integer("MIST_VOICE_BARGE_IN_ARM_DELAY_MS", barge_defaults.arm_delay_ms),
        echo_guard_ms=env.integer("MIST_VOICE_ECHO_GUARD_MS", barge_defaults.echo_guard_ms),
        wake_arbitration_ms=env.integer(
            "MIST_VOICE_WAKE_ARBITRATION_MS", barge_defaults.wake_arbitration_ms
        ),
    )
    playback = PlaybackConfig(
        client_prebuffer_ms=env.integer(
            "MIST_VOICE_CLIENT_PREBUFFER_MS", PlaybackConfig().client_prebuffer_ms
        ),
    )
    segmenter = SegmenterConfig(
        max_segment_chars=env.integer(
            "MIST_VOICE_MAX_SEGMENT_CHARS", SegmenterConfig().max_segment_chars
        ),
    )
    place = PlacementConfig()
    tts_variant = env.choice("MIST_VOICE_TTS_VARIANT", place.tts_variant, _VARIANTS)
    placement = PlacementConfig(
        vad_device=env.device("MIST_VOICE_VAD_DEVICE", place.vad_device),
        eot_device=env.device("MIST_VOICE_EOT_DEVICE", place.eot_device),
        stt_primary=env.placement("MIST_VOICE_STT_DEVICE", place.stt_primary),
        stt_yield=env.placement("MIST_VOICE_STT_YIELD_DEVICE", place.stt_yield),
        tts_primary=env.placement("MIST_VOICE_TTS_DEVICE", place.tts_primary),
        tts_yield=env.placement("MIST_VOICE_TTS_YIELD_DEVICE", place.tts_yield),
        vad_model=env.pattern(
            "MIST_VOICE_VAD_MODEL", place.vad_model, _ONNX_MODEL, "<repo>@<revision>:<file>"
        ),
        eot_model=env.pattern(
            "MIST_VOICE_EOT_MODEL", place.eot_model, _ONNX_MODEL, "<repo>@<revision>:<file>"
        ),
        stt_model=env.text("MIST_VOICE_STT_MODEL", place.stt_model),
        tts_variant="original" if tts_variant == "original" else "turbo",
        tts_model=env.pattern(
            "MIST_VOICE_TTS_MODEL", place.tts_model, _TTS_MODEL, "<repo>@<revision>"
        ),
        speech_service_url=env.url("MIST_SPEECH_SERVICE_URL", place.speech_service_url),
        vram_budget_mib=env.integer("MIST_VOICE_VRAM_BUDGET_MIB", place.vram_budget_mib),
    )
    reference = VoiceReferenceConfig()
    voice_reference = VoiceReferenceConfig(
        profiles_dir=env.text("MIST_VOICE_PROFILES_DIR", reference.profiles_dir),
        profile=env.text("VOICE_PROFILE", reference.profile),
    )
    gpu = GpuYieldConfig()
    gpu_yield = GpuYieldConfig(
        triggers=env.triggers("MIST_VOICE_GPU_YIELD_TRIGGERS", gpu.triggers),
        free_mib=env.integer("MIST_VOICE_GPU_YIELD_FREE_MIB", gpu.free_mib),
        yield_after_s=env.number("MIST_VOICE_GPU_YIELD_AFTER_S", gpu.yield_after_s),
        resume_after_s=env.number("MIST_VOICE_GPU_RESUME_AFTER_S", gpu.resume_after_s),
    )
    turn = TurnConfig(
        tool_cancel_grace_ms=env.integer(
            "MIST_VOICE_TOOL_CANCEL_GRACE_MS", TurnConfig().tool_cancel_grace_ms
        ),
    )
    return VoiceSettings(
        budget=env.budget("MIST_VOICE_BUDGET", DEFAULT_BUDGET),
        yield_budget=env.budget("MIST_VOICE_YIELD_BUDGET", DEFAULT_YIELD_BUDGET),
        endpointing=endpointing,
        barge_in=barge_in,
        playback=playback,
        segmenter=segmenter,
        placement=placement,
        voice_reference=voice_reference,
        gpu_yield=gpu_yield,
        turn=turn,
    )
