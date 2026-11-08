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
import posixpath
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeVar

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class VoiceLatencyBudget:
    """Per-stage latency targets in milliseconds, with the name of the preset they came from.

    `total_p50_ms` is the sum of the seven stages from `end_of_turn_ms` to
    `transport_playback_ms` (`STAGE_FIELDS`). Every row after it is a separate target outside
    that sum: `total_p90_ms`, `tool_call_ms` and `barge_in_reaction_ms`, then the five screen
    and caption targets from `wake_screen_ms` to `event_to_pixel_ms`, which time what the user
    sees rather than a step of the reply. `preset` travels with the figures so
    `turn_metrics.budget.preset` can say which preset a turn was judged against.
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
    wake_screen_ms: int
    caption_lag_ms: int
    speech_screen_skew_ms: int
    screen_command_ms: int
    event_to_pixel_ms: int


BUDGET_FIELDS: tuple[str, ...] = tuple(
    f.name for f in dataclasses.fields(VoiceLatencyBudget) if f.name != "preset"
)
"""The budget table's sixteen rows, in table order."""

STAGE_FIELDS: tuple[str, ...] = BUDGET_FIELDS[:7]
"""The seven serial stages whose sum is `total_p50_ms`."""

# The screen and caption targets, the same in every preset: wake to screen, caption lag behind
# speech, speech-to-screen skew, a spoken screen command, and a client event to its pixels.
_SCREEN_TARGETS = (300, 500, 150, 600, 50)


def _preset(name: str, *rows: int) -> VoiceLatencyBudget:
    return VoiceLatencyBudget(name, *rows, *_SCREEN_TARGETS)


# The Design doc's Latency budget table. `floor` is what the 4070 SUPER host can do, from
# measured rates where they exist and estimates where they do not; `memo` is the model-host
# memo's P50 budget sized for a larger GPU. Measured P50s replace the estimates through
# `MIST_VOICE_BUDGET` overrides, not code changes.
BUDGET_PRESETS: Mapping[str, VoiceLatencyBudget] = MappingProxyType(
    {
        "floor": _preset("floor", 250, 250, 0, 250, 120, 500, 100, 1470, 2100, 450, 700),
        "memo": _preset("memo", 250, 50, 0, 200, 100, 150, 100, 850, 1200, 270, 700),
    }
)

DEFAULT_BUDGET = "floor"


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
    keyword detection and captions. After `endpoint_silence_ms` of silence the final decode and
    the end-of-turn score of the last `eot_window_ms` start together; a probability above
    `eot_threshold` ends the turn, otherwise it is re-checked every `eot_recheck_ms` until
    `eot_max_silence_ms`. An utterance under `min_utterance_ms` with an empty final is noise. A
    final with `no_speech_prob` above `no_speech_threshold` and `avg_logprob` at or below
    `logprob_threshold` is dropped as non-speech, the thresholds openai-whisper uses. Audio
    queued past `max_backlog_ms` is dropped oldest first.
    """

    vad_threshold: float = 0.5
    speech_start_windows: int = 3
    pre_roll_ms: int = 300
    stt_partial_interval_ms: int = 250
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
    endpoints are compared and only the strongest answered. `alarm_stop_keywords` are the
    keywords that also stop a ringing alarm; each is one of `keywords`, so a word that stops an
    alarm always stops MIST's speech too.
    """

    keywords: tuple[str, ...] = ("stop", "wait", "hold on", "cancel", "never mind")
    arm_delay_ms: int = 1000
    keyword_window_words: int = 3
    echo_guard_ms: int = 1500
    wake_arbitration_ms: int = 250
    alarm_stop_keywords: tuple[str, ...] = ("stop",)


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


@dataclass(frozen=True, slots=True)
class PlacementConfig:
    """Where each engine runs, which engine and weights it loads, and how the GPU is released.

    A location is `cpu`, `cuda`, `cuda:<n>` or `off`; a placement is an ordered tuple of
    locations of which the first that loads wins, with `off` (no engine) only last. VAD and end
    of turn run on one device. An engine id is `[a-z0-9][a-z0-9-]*`, a step 0 engine; `None`
    loads nothing, so that capability is off until the adapters set a default. Model settings
    are `<repo>@<revision>` for STT and TTS and `<repo>@<revision>:<file>` for the ONNX engines,
    so every weight is pinned, and every weight is read under `models_dir`. `release_timeout_s`
    bounds how long a GPU release waits for running jobs to stop.
    """

    vad_device: str = "cpu"
    eot_device: str = "cpu"
    eot_threads: int = 4
    stt_device: tuple[str, ...] = ("cuda", "cpu")
    tts_device: tuple[str, ...] = ("cuda", "off")
    vad_model: str = (
        "onnx-community/silero-vad@e71cae966052b992a7eca6b17738916ce0eca4ec:onnx/model.onnx"
    )
    eot_model: str = (
        "pipecat-ai/smart-turn-v3@f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
        ":smart-turn-v3.2-cpu.onnx"
    )
    stt_engine: str | None = None
    stt_model: str | None = None
    tts_engine: str | None = None
    tts_model: str | None = None
    tts_voice: str | None = None
    vram_budget_mib: int = 4500
    release_timeout_s: float = 10.0
    models_dir: str = "/models"


@dataclass(frozen=True, slots=True)
class TurnConfig:
    """Turn teardown limits: how long a dispatched tool may finish after a cancel."""

    tool_cancel_grace_ms: int = 2000


@dataclass(frozen=True, slots=True)
class VoiceSettings:
    """Every voice setting, as `load_voice_settings` read it at startup."""

    budget: VoiceLatencyBudget = field(default_factory=lambda: BUDGET_PRESETS[DEFAULT_BUDGET])
    endpointing: EndpointingConfig = field(default_factory=EndpointingConfig)
    barge_in: BargeInConfig = field(default_factory=BargeInConfig)
    playback: PlaybackConfig = field(default_factory=PlaybackConfig)
    segmenter: SegmenterConfig = field(default_factory=SegmenterConfig)
    placement: PlacementConfig = field(default_factory=PlacementConfig)
    turn: TurnConfig = field(default_factory=TurnConfig)
    profile: str = "mist"


_LOCATION = re.compile(r"cpu|cuda(?::\d+)?|off")
_DEVICE = re.compile(r"cpu|cuda(?::\d+)?")
_ENGINE_ID = re.compile(r"[a-z0-9][a-z0-9-]*")
_REPO_MODEL = re.compile(r"[^@:\s]+@[^@:\s]+")
_ONNX_MODEL = re.compile(r"[^@:\s]+@[^@:\s]+:[^@:\s]+")
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

    def positive_integer(self, name: str, default: int) -> int:
        return self._read(name, default, _parse_positive_int)

    def number(self, name: str, default: float, *, allow_negative: bool = False) -> float:
        def parse(var: str, raw: str) -> float:
            return _parse_float(var, raw, allow_negative=allow_negative)

        return self._read(name, default, parse)

    def positive_number(self, name: str, default: float) -> float:
        return self._read(name, default, _parse_positive_float)

    def probability(self, name: str, default: float) -> float:
        return self._read(name, default, _parse_probability)

    def boolean(self, name: str, default: bool) -> bool:
        return self._read(name, default, _parse_bool)

    def text(self, name: str, default: _T) -> str | _T:
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

    def engine_id(self, name: str, default: str | None) -> str | None:
        return self.pattern(name, default, _ENGINE_ID, "an engine id matching [a-z0-9][a-z0-9-]*")

    def absolute_path(self, name: str, default: str) -> str:
        return self._read(name, default, _parse_absolute_path)

    def budget(self, name: str, default: str) -> VoiceLatencyBudget:
        def parse(var: str, raw: str) -> VoiceLatencyBudget:
            try:
                return parse_budget(raw)
            except ValueError as exc:
                raise ValueError(f"{var}: {exc}") from exc

        return self._read(name, BUDGET_PRESETS[default], parse)

    def keywords(self, name: str, default: tuple[str, ...]) -> tuple[str, ...]:
        return self._read(name, default, _parse_keywords)


def _parse_non_negative_int(name: str, raw: str) -> int:
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name}: expected a non-negative integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{name}: expected a non-negative integer, got {raw!r}")
    return value


def _parse_positive_int(name: str, raw: str) -> int:
    try:
        value = int(raw.strip())
    except ValueError as exc:
        raise ValueError(f"{name}: expected a positive integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{name}: expected a positive integer, got {raw!r}")
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


def _parse_positive_float(name: str, raw: str) -> float:
    value = _parse_float(name, raw, allow_negative=False)
    if value <= 0:
        raise ValueError(f"{name}: expected a positive number, got {raw!r}")
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


def _parse_absolute_path(name: str, raw: str) -> str:
    # The backend runs in a Linux container, so the weights root is a POSIX path.
    value = raw.strip()
    if not posixpath.isabs(value):
        raise ValueError(f"{name}: expected an absolute path, got {raw!r}")
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
                f"{name}: unknown location {location!r}; expected cpu, cuda, cuda:<n> or off"
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


def _alarm_stop_keywords(env: _Env, name: str, barge_in: BargeInConfig) -> tuple[str, ...]:
    keywords = env.keywords(name, barge_in.alarm_stop_keywords)
    for keyword in keywords:
        if keyword not in barge_in.keywords:
            raise ValueError(
                f"{name}: {keyword!r} is not a barge-in keyword; expected one of "
                f"{', '.join(barge_in.keywords)}"
            )
    return keywords


def load_voice_settings(environ: Mapping[str, str]) -> VoiceSettings:
    """Read every voice setting from an environment mapping, defaulting what is unset.

    Not read here: `MIST_VOICE_THINKING_BUDGET_TOKENS` and `MIST_VOICE_FIRST_SEGMENT_MIN_CHARS`,
    which the handler's `LLMConfig` owns (the handler never imports this package). The variables
    the GPU yield, the remote speech service and the cloned voice used are not read either, so a
    stale `.env` entry for one of them changes nothing.

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
    barge_in = dataclasses.replace(
        barge_in,
        alarm_stop_keywords=_alarm_stop_keywords(env, "MIST_VOICE_ALARM_STOP_KEYWORDS", barge_in),
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
    placement = PlacementConfig(
        vad_device=env.device("MIST_VOICE_VAD_DEVICE", place.vad_device),
        eot_device=env.device("MIST_VOICE_EOT_DEVICE", place.eot_device),
        eot_threads=env.positive_integer("MIST_VOICE_EOT_THREADS", place.eot_threads),
        stt_device=env.placement("MIST_VOICE_STT_DEVICE", place.stt_device),
        tts_device=env.placement("MIST_VOICE_TTS_DEVICE", place.tts_device),
        vad_model=env.pattern(
            "MIST_VOICE_VAD_MODEL", place.vad_model, _ONNX_MODEL, "<repo>@<revision>:<file>"
        ),
        eot_model=env.pattern(
            "MIST_VOICE_EOT_MODEL", place.eot_model, _ONNX_MODEL, "<repo>@<revision>:<file>"
        ),
        stt_engine=env.engine_id("MIST_VOICE_STT_ENGINE", place.stt_engine),
        stt_model=env.pattern(
            "MIST_VOICE_STT_MODEL", place.stt_model, _REPO_MODEL, "<repo>@<revision>"
        ),
        tts_engine=env.engine_id("MIST_VOICE_TTS_ENGINE", place.tts_engine),
        tts_model=env.pattern(
            "MIST_VOICE_TTS_MODEL", place.tts_model, _REPO_MODEL, "<repo>@<revision>"
        ),
        tts_voice=env.text("MIST_VOICE_TTS_VOICE", place.tts_voice),
        vram_budget_mib=env.integer("MIST_VOICE_VRAM_BUDGET_MIB", place.vram_budget_mib),
        release_timeout_s=env.positive_number(
            "MIST_VOICE_RELEASE_TIMEOUT_S", place.release_timeout_s
        ),
        models_dir=env.absolute_path("MIST_VOICE_MODELS_DIR", place.models_dir),
    )
    turn = TurnConfig(
        tool_cancel_grace_ms=env.integer(
            "MIST_VOICE_TOOL_CANCEL_GRACE_MS", TurnConfig().tool_cancel_grace_ms
        ),
    )
    return VoiceSettings(
        budget=env.budget("MIST_VOICE_BUDGET", DEFAULT_BUDGET),
        endpointing=endpointing,
        barge_in=barge_in,
        playback=playback,
        segmenter=segmenter,
        placement=placement,
        turn=turn,
        profile=env.text("VOICE_PROFILE", VoiceSettings().profile),
    )
