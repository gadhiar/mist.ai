"""Voice latency budget, voice configuration and the loader that reads them from the environment.

The budget is a set of per-stage targets that changes no behaviour: `turn_metrics` reports each
measured stage against it, and the latency bench compares against it. Behaviour lives in the
configuration dataclasses (`EndpointingConfig`, `BargeInConfig` and the rest). Both are read
once, at startup, by `load_voice_settings`, which raises `ValueError` naming the variable for any
malformed value, so a typo in `.env` stops the backend with its name instead of surfacing later
as an engine failure.
"""

import dataclasses
import json
import math
import posixpath
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TypeVar

from src.multimodal.voice_profile import DEFAULT_VOICE_PROFILE

from .protocol import MAX_WAKE_BACKLOG_MS

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class VoiceLatencyBudget:
    """Per-stage latency targets in milliseconds, with the name of the preset they came from.

    `total_p50_ms` is the sum of the seven stages from `end_of_turn_ms` to
    `transport_playback_ms` (`STAGE_FIELDS`). Every row after it is a separate target outside
    that sum: `total_p90_ms`, `tool_call_ms` and `barge_in_reaction_ms`, then the five screen
    and caption targets from `wake_screen_ms` to `event_to_pixel_ms`, which time what the user
    sees rather than a step of the reply. `wake_screen_ms` is wake to screen, server side: the
    name's end, estimated, to `attend` or `listen` sent (G6). `preset` travels with the figures
    so `turn_metrics.budget.preset` can say which preset a turn was judged against.
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

# The screen and caption targets, the same in every preset: wake to screen, server side (the
# name's end, estimated, to `attend` or `listen` sent, G6), caption lag behind speech,
# speech-to-screen skew, a spoken screen command, and a client event to its pixels.
_SCREEN_TARGETS = (205, 500, 150, 600, 50)


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
class PeakingBand:
    """One peaking biquad of the character stage's EQ (RBJ Audio EQ Cookbook, `peakingEQ`).

    `q` is the cookbook's Q in `alpha = sin(w0) / (2 * Q)`, not a bandwidth in octaves.
    """

    freq_hz: float
    gain_db: float
    q: float


@dataclass(frozen=True, slots=True)
class DetunedCopy:
    """One copy of the detuned double: its pitch offset and the centre of its read delay."""

    cents: float
    delay_ms: float


@dataclass(frozen=True, slots=True)
class LimiterConfig:
    """The peak limiter that ends the character stage (`character.PeakLimiter`).

    The ear test had no limiter; these values were chosen with it, after the stage could clip a
    loud onset, and each keeps the stage's sound below the peaks.

    - `knee_dbfs`, -1 dBFS: samples at or below it pass unchanged. The stage's level tests hold
      settled speech at the reference level at or below -1 dBFS, and its settled peaks on their
      signals measure -12.3 to -9.8 dBFS, so the knee leaves every settled sample alone; the
      1 dB above it is room for the soft curve.
    - `hold_ms`, 10 ms: one period of a 100 Hz voice, so a crest no deeper than one before it
      within the hold is scaled by the same gain rather than bent.
    - `release_ms`, 10 ms, linear in dB: short enough that hold plus release, 20 ms, ends inside
      each segment's tail (`CharacterConfig.__post_init__` checks that), and no longer than the
      hold, so the release starts without a jump. A linear release ends; an exponential one
      would never let the gain back to exactly 0 dB.
    """

    knee_dbfs: float = -1.0
    hold_ms: float = 10.0
    release_ms: float = 10.0

    def __post_init__(self) -> None:
        ceiling_dbfs = 20.0 * math.log10(32767.0 / 32768.0)
        if not self.knee_dbfs < ceiling_dbfs:
            raise ValueError(
                f"limiter knee {self.knee_dbfs} dBFS must lie below the PCM16 ceiling, "
                f"{ceiling_dbfs:.6f} dBFS"
            )
        if not (self.hold_ms > 0.0 and self.release_ms > 0.0):
            raise ValueError("limiter hold_ms and release_ms must be positive")
        if self.release_ms > self.hold_ms:
            raise ValueError(
                f"limiter release_ms {self.release_ms} must not exceed hold_ms {self.hold_ms}"
            )


@dataclass(frozen=True, slots=True)
class CharacterConfig:
    """Her character stage (`backend/realtime/character.py`), set from Raj's ear tests.

    The values are the ear test's (H20): a 2nd-order Butterworth high-pass, three peaking EQ
    bands, a 4:1 compressor on the 8 ms RMS of the EQ output smoothed over 15 ms of linear gain,
    and a detuned double of two copies, +15 cents centred 9 ms behind the dry voice and -15 cents
    centred 14 ms behind it, each mixed at `double_gain`. Each copy's read delay sweeps within
    `delay_ms +- sweep_ms / 2`, and its splices crossfade over `splice_ms`. The exception is
    `limiter`: the ear test had none, and `LimiterConfig` says why its values were chosen.
    `limiter=None` turns it off, so the output is rounded and hard-clipped, with the clipped
    samples counted, as before the limiter existed.

    `threshold_dbfs` is absolute because a stream cannot see a whole clip: the ear test's
    threshold was 26 dB below the maximum of the clip's own 8 ms RMS envelope, and the median of
    that maximum on the clips Raj heard was -14.47 dBFS, so -40.47 dBFS, rounded to -40.5.
    """

    highpass_hz: float = 100.0
    bands: tuple[PeakingBand, ...] = (
        PeakingBand(300.0, -3.0, 1.0),
        PeakingBand(4000.0, 4.0, 1.0),
        PeakingBand(9000.0, 2.0, 0.8),
    )
    rms_window_ms: float = 8.0
    threshold_dbfs: float = -40.5
    ratio: float = 4.0
    gain_smoothing_ms: float = 15.0
    copies: tuple[DetunedCopy, ...] = (DetunedCopy(15.0, 9.0), DetunedCopy(-15.0, 14.0))
    double_gain: float = 0.35  # per copy; Raj's pick
    sweep_ms: float = 12.0
    splice_ms: float = 4.0
    # The ear test matched each clip's whole-clip RMS to its input's; a stream cannot, so a fixed
    # gain does it for typical TTS output. Derived on a proxy for it (no clip may be read): the
    # 200 Hz test vowel (harmonics to 11 kHz, -12 dB per octave), 3 s, under 12 syllables of a
    # 200 ms Hann window and 50 ms of silence with peaks cycling 0, -6 and -12 dB, scaled so its
    # EQ output's 8 ms RMS envelope peaks at -14.47 dBFS. The value is 20 * log10 of the input's
    # RMS over the stage output's RMS with this gain at 0 dB, over the input's length, rounded to
    # 0.1 dB: 14.853, so 14.9. `test_default_output_gain_keeps_the_input_loudness` recomputes it.
    output_gain_db: float = 14.9
    limiter: LimiterConfig | None = LimiterConfig()

    def __post_init__(self) -> None:
        # Every frequency lies below the Nyquist frequency of the 24 kHz downlink, and each
        # copy's shortest read stays at least 1 ms behind the input, beyond the reach of the
        # interpolator's forward taps.
        frequencies = (self.highpass_hz, *(band.freq_hz for band in self.bands))
        for freq in frequencies:
            if not 0.0 < freq < 12000.0:
                raise ValueError(f"character stage frequency {freq} Hz must lie in (0, 12000) Hz")
        for band in self.bands:
            if not band.q > 0.0:
                raise ValueError(
                    f"peaking band at {band.freq_hz} Hz has q {band.q}; it must be > 0"
                )
        if not self.ratio >= 1.0:
            raise ValueError(f"compressor ratio {self.ratio} must be at least 1")
        if not 0.0 <= self.double_gain <= 1.0:
            raise ValueError(f"double_gain {self.double_gain} must lie in [0, 1]")
        if not (self.rms_window_ms > 0.0 and self.gain_smoothing_ms > 0.0 and self.splice_ms > 0.0):
            raise ValueError("rms_window_ms, gain_smoothing_ms and splice_ms must be positive")
        for copy in self.copies:
            if copy.delay_ms - self.sweep_ms / 2 < 1.0:
                raise ValueError(
                    f"copy at {copy.cents} cents reads as little as "
                    f"{copy.delay_ms - self.sweep_ms / 2} ms behind; the least is 1 ms"
                )
        if not self.splice_ms < self.sweep_ms / 2:
            raise ValueError(
                f"splice_ms {self.splice_ms} must be below half of sweep_ms {self.sweep_ms}"
            )
        # The limiter's memory must end inside each segment's tail, which lasts as long as the
        # copies' longest read, so no segment's limiting reaches the next.
        if self.limiter is not None and self.double_gain > 0.0 and self.copies:
            memory_ms = self.limiter.hold_ms + self.limiter.release_ms
            longest_ms = max(copy.delay_ms for copy in self.copies) + self.sweep_ms / 2
            if memory_ms > longest_ms:
                raise ValueError(
                    f"limiter hold_ms plus release_ms, {memory_ms} ms, outlasts the segment "
                    f"tail's {longest_ms} ms"
                )


@dataclass(frozen=True, slots=True)
class PlaybackConfig:
    """Downlink pacing, the client's playback buffer, and the character stage before the pacer.

    `character` configures the stage every spoken reply passes through before it is framed.
    None, the default here, leaves the engine's audio as it is. `load_voice_settings` sets None
    when `MIST_VOICE_CHARACTER` is false; otherwise it sets a `CharacterConfig` whose
    `double_gain` comes from `MIST_VOICE_CHARACTER_DOUBLE_GAIN` when that is set, with every
    other field at its default.
    """

    client_prebuffer_ms: int = 80
    max_lead_ms: int = 500
    fade_ms: int = 30
    progress_interval_ms: int = 100
    frame_ms: int = 40
    character: CharacterConfig | None = None


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
class EngineOption:
    """One place a speech slot may load: an engine id on a device.

    `engine` is an engine id, `[a-z0-9][a-z0-9-]*`. `device` is `cpu` or `cuda`: the host has
    one GPU, so an option never names a GPU index.
    """

    engine: str
    device: str


@dataclass(frozen=True, slots=True)
class PlacementConfig:
    """Where each engine runs, which voice TTS speaks, and how the GPU is released.

    STT and TTS each have a chain: an ordered tuple of `EngineOption`s, of which the first that
    loads wins, so "Nemotron on the GPU, else Parakeet on the CPU" is a setting rather than
    code. An empty chain turns the slot off. Each engine's adapter pins its own weights, so the
    chains carry no model setting. VAD and end of turn run on one device each, `cpu`, `cuda` or
    `cuda:<n>`, and their ONNX weights are `<repo>@<revision>:<file>`, read under `models_dir`.

    `tts_voice` is the id of the voice identity TTS speaks, `[a-z0-9][a-z0-9-]*`, read from
    `<voice_root>/<id>/`; `None` names no identity. `voice_cache_dir` is the directory for the
    per-engine assets built from an identity. Both paths are absolute. `release_timeout_s`
    bounds how long a GPU release waits for running jobs to stop. `kws_model` is the name
    spotter's model id, the same id form, read from `<models_dir>/speech/wake/<id>/`; `None`
    turns the server spotter off. The spotter runs on the CPU only, with `kws_threads` threads,
    so it has no device setting.
    """

    vad_device: str = "cpu"
    eot_device: str = "cpu"
    eot_threads: int = 4
    stt_chain: tuple[EngineOption, ...] = (
        EngineOption("nemotron-160", "cuda"),
        EngineOption("parakeet-cpu", "cpu"),
    )
    tts_chain: tuple[EngineOption, ...] = (EngineOption("pocket-clone", "cpu"),)
    vad_model: str = (
        "onnx-community/silero-vad@e71cae966052b992a7eca6b17738916ce0eca4ec:onnx/model.onnx"
    )
    eot_model: str = (
        "pipecat-ai/smart-turn-v3@f766f81d3cfdf7737ac64aad813d91bbfd56bf93"
        ":smart-turn-v3.2-cpu.onnx"
    )
    tts_voice: str | None = None
    voice_root: str = "/voices"
    voice_cache_dir: str = "/voice-cache"
    vram_budget_mib: int = 4500
    release_timeout_s: float = 10.0
    models_dir: str = "/models"
    kws_model: str | None = None
    kws_threads: int = 1


@dataclass(frozen=True, slots=True)
class TurnConfig:
    """Turn teardown limits: how long a dispatched tool may finish after a cancel."""

    tool_cancel_grace_ms: int = 2000


DEFAULT_WAKE_PHRASES: tuple[str, ...] = (
    "hey {name}",
    "{name}, how's it going",
    "{name}, what's up",
    "are you there {name}",
)
"""Templates that address MIST when they open or end an utterance; `{name}` is any name."""

DEFAULT_ADDRESS_STARTS: tuple[str, ...] = (
    "what",
    "what's",
    "how",
    "how's",
    "are",
    "can",
    "could",
    "would",
    "will",
    "do",
    "did",
    "set",
    "play",
    "turn",
    "tell",
    "show",
    "open",
    "close",
    "stop",
    "remind",
    "start",
    "put",
    "add",
    "call",
)
"""Words that start a request after a leading name with no comma.

"is" is not among them, so "mist is a good name" talks about the name rather than to her.
"""

DEFAULT_STOP_BEFORE: tuple[str, ...] = (
    "the",
    "a",
    "an",
    "this",
    "that",
    "some",
    "of",
    "in",
    "into",
    "through",
    "morning",
    "thick",
    "light",
    "sea",
    "mountain",
)
"""Words after which a name is a noun: "the morning mist", "we drove into mist".

A greeting that opens an utterance is read first, so "morning mist" alone still greets her.
"""

DEFAULT_GREETINGS: tuple[str, ...] = ("hey", "hi", "hello", "morning", "evening", "yo")
"""Greetings that make a following name a vocative at the start of an utterance."""

DEFAULT_NAME_ALIASES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("mist", ("missed", "miss", "myst", "midst", "mists", "mist's", "must", "mest")),
)
"""Each name with the near spellings a recogniser writes for it, which the gate reads as the
name only where the spotter heard it (`WakeConfig`)."""

# The phrases file's keys: `aliases` is an object from name to a list of words, and each other
# key a list of strings replacing that field's default.
_PHRASE_KEYS = ("phrases", "address_starts", "stop_before", "greetings", "aliases")
# One gate token, the form of `addressee._WORD`, which budget cannot import: addressee imports
# budget.
_ALIAS = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)*")

# `lookback_ms` plus this must stay within the wire's `MAX_WAKE_BACKLOG_MS`.
_LOOKBACK_HEADROOM_MS = 1000
# The least `endpoint_min_backlog_ms` may be; its most is `MAX_WAKE_BACKLOG_MS`.
_MIN_ENDPOINT_BACKLOG_MS = 500


@dataclass(frozen=True, slots=True)
class WakeConfig:
    """How MIST hears her name and decides she was addressed (Design doc, Wake by name).

    `names` are lower-case single words the spotter scores. `phrases`, `address_starts`,
    `stop_before` and `greetings` are the addressee gate's word sets, which
    `MIST_VOICE_WAKE_PHRASES_FILE` changes without a release; each template holds `{name}`
    once. A name hit at or above `kws_threshold` (an endpoint wake at or above
    `endpoint_threshold`) opens a candidate reaching back at most `lookback_ms`, across speech
    gaps of at most `join_gap_ms`. The gate decodes `gate_tail_ms` past the hit, and the
    reference may lead the hit's window by `reference_tolerance_ms`. An endpoint wake with
    less than `endpoint_min_backlog_ms` of audio before it is rejected. `followup_window_ms`
    above 0 keeps listening without the name after a turn; `attend_cue` sends `attend` at a
    hit, before the verdict.

    `aliases` maps a name to its near spellings ("missed" for "mist"), which the phrases file's
    `aliases` key replaces and `"aliases": {}` turns off; an entry for a name not in `names` has
    no effect. For a server hit the gate knows where the spotter heard the name, and reads it
    there (`addressee.vocative_match`):
    - An alias counts as the name only in a word overlapping the alias window, the
      `alias_window_ms` before the hit through the hit, because it sounds like the name and
      elsewhere is an ordinary word ("I missed the bus").
    - Even there, an alias needs a vocative's punctuation: as the first word, "," after it, or
      "." or "?" and then a word in `address_starts`; as the last word, "," before it, unless it
      is the only word ("Missed."). A configured name keeps its rules.
    - The trailing-speech guard: a rule anchored at the utterance's end (the last-word rule, or
      a phrase template matched at the end, or covering the whole decode and ending at the
      name, as "are you there {name}" can) counts only when the VAD heard no speech more than
      `trailing_speech_ms` past the hit, so a decode the gate tail cut short ("a cold mist" of
      "a cold mist hung over the lake") does not pass as said to her.
    - The full-stop form: a name or alias first, then "." or "?" and a word in `address_starts`
      ("Mist. What's the time?"), counts as the first-word rule's comma does.
    An endpoint wake has no hit position on the server, so none of this applies to it.
    """

    names: tuple[str, ...] = ("mist",)
    phrases: tuple[str, ...] = DEFAULT_WAKE_PHRASES
    address_starts: tuple[str, ...] = DEFAULT_ADDRESS_STARTS
    stop_before: tuple[str, ...] = DEFAULT_STOP_BEFORE
    greetings: tuple[str, ...] = DEFAULT_GREETINGS
    kws_threshold: float = 0.5
    endpoint_threshold: float = 0.5
    lookback_ms: int = 6000
    join_gap_ms: int = 1000
    gate_tail_ms: int = 300
    reference_tolerance_ms: int = 400
    endpoint_min_backlog_ms: int = 1500
    followup_window_ms: int = 0
    attend_cue: bool = True
    aliases: tuple[tuple[str, tuple[str, ...]], ...] = DEFAULT_NAME_ALIASES
    alias_window_ms: int = 400
    trailing_speech_ms: int = 100


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
    wake: WakeConfig = field(default_factory=WakeConfig)
    profile: str = DEFAULT_VOICE_PROFILE


_DEVICE = re.compile(r"cpu|cuda(?::\d+)?")
# Engine, identity and spotter model ids share one form.
_ID = re.compile(r"[a-z0-9][a-z0-9-]*")
_ID_FORM = "[a-z0-9][a-z0-9-]*"
_CHAIN_DEVICES = ("cpu", "cuda")
_NAME = re.compile(r"[a-z0-9]+(?:['-][a-z0-9]+)*")
_ONNX_MODEL = re.compile(r"[^@:\s]+@[^@:\s]+:[^@:\s]+")
# The per-slot variables the chains replaced, each with the chain variable that took its place.
# Set at all, one raises, because ignoring it would leave an engine where the variable no longer
# puts it.
_REMOVED_PER_SLOT = {
    "MIST_VOICE_STT_ENGINE": "MIST_VOICE_STT_CHAIN",
    "MIST_VOICE_STT_DEVICE": "MIST_VOICE_STT_CHAIN",
    "MIST_VOICE_STT_MODEL": "MIST_VOICE_STT_CHAIN",
    "MIST_VOICE_TTS_ENGINE": "MIST_VOICE_TTS_CHAIN",
    "MIST_VOICE_TTS_DEVICE": "MIST_VOICE_TTS_CHAIN",
    "MIST_VOICE_TTS_MODEL": "MIST_VOICE_TTS_CHAIN",
}
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

    def chain(self, name: str, default: tuple[EngineOption, ...]) -> tuple[EngineOption, ...]:
        return self._read(name, default, _parse_chain)

    def refuse_removed(self, removed: Mapping[str, str]) -> None:
        for name, replacement in removed.items():
            if name in self._environ:
                raise ValueError(f"{name}: removed; set {replacement} instead")

    def device(self, name: str, default: str) -> str:
        return self._read(name, default, _parse_device)

    def pattern(self, name: str, default: _T, pattern: re.Pattern[str], form: str) -> str | _T:
        def parse(var: str, raw: str) -> str:
            value = raw.strip()
            if not pattern.fullmatch(value):
                raise ValueError(f"{var}: expected {form}, got {raw!r}")
            return value

        return self._read(name, default, parse)

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

    def names(self, name: str, default: tuple[str, ...]) -> tuple[str, ...]:
        return self._read(name, default, _parse_names)

    def bounded_integer(self, name: str, default: int, low: int, high: int) -> int:
        def parse(var: str, raw: str) -> int:
            value = _parse_non_negative_int(var, raw)
            if not low <= value <= high:
                raise ValueError(f"{var}: expected an integer from {low} to {high}, got {raw!r}")
            return value

        return self._read(name, default, parse)

    def phrases_file(self, name: str) -> Mapping[str, tuple[object, ...]]:
        return self._read(name, {}, _parse_phrases_file)


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


def _parse_chain(name: str, raw: str) -> tuple[EngineOption, ...]:
    # Every item is checked, an empty one included, so a stray comma is an error rather than a
    # silently shorter chain.
    items = [item.strip() for item in raw.split(",")]
    if items == ["off"]:
        return ()
    form = "comma-separated <engine>@<device> options, or off alone"
    options: list[EngineOption] = []
    for item in items:
        if not item:
            raise ValueError(f"{name}: expected {form}, got {raw!r}")
        if item == "off":
            raise ValueError(f"{name}: off must stand alone, got {raw!r}")
        engine, at, device = item.partition("@")
        if not at:
            raise ValueError(f"{name}: option {item!r} has no device; expected {form}")
        if not _ID.fullmatch(engine):
            raise ValueError(f"{name}: engine id {engine!r} does not match {_ID_FORM}")
        if device not in _CHAIN_DEVICES:
            raise ValueError(f"{name}: device {device!r} in {item!r}; expected cpu or cuda")
        option = EngineOption(engine, device)
        if option in options:
            raise ValueError(f"{name}: option {item!r} is repeated in {raw!r}")
        options.append(option)
    return tuple(options)


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


def _parse_names(name: str, raw: str) -> tuple[str, ...]:
    names = tuple(_split_list(raw))
    if not names:
        raise ValueError(f"{name}: expected at least one comma-separated name, got {raw!r}")
    for item in names:
        if not _NAME.fullmatch(item):
            raise ValueError(f"{name}: {item!r} is not one lower-case word")
    return names


def _parse_phrases_file(name: str, raw: str) -> Mapping[str, tuple[object, ...]]:
    """The file's word sets by `WakeConfig` field, `aliases` as `DEFAULT_NAME_ALIASES` holds them.

    Each alias must be one gate token; whether its name is configured, and that it is no name
    itself, `_wake_config` checks, since the names come from another variable.
    """
    path = _parse_absolute_path(name, raw)
    try:
        with open(path, encoding="utf-8") as handle:
            content = json.load(handle)
    except OSError as exc:
        raise ValueError(f"{name}: cannot read {path}: {exc.strerror}") from exc
    except ValueError as exc:  # json.JSONDecodeError and UnicodeDecodeError
        raise ValueError(f"{name}: {path} is not valid JSON: {exc}") from exc
    if not isinstance(content, dict):
        raise ValueError(f"{name}: {path} must hold a JSON object")
    unknown = sorted(set(content) - set(_PHRASE_KEYS))
    if unknown:
        raise ValueError(
            f"{name}: {path} has unknown keys {', '.join(unknown)}; "
            f"expected any of {', '.join(_PHRASE_KEYS)}"
        )
    sets: dict[str, tuple[object, ...]] = {}
    for key, items in content.items():
        if key == "aliases":
            if not isinstance(items, dict):
                raise ValueError(f"{name}: {path} key 'aliases' must be an object")
            aliases: list[tuple[str, tuple[str, ...]]] = []
            for owner, words in items.items():
                if not isinstance(words, list) or not all(isinstance(w, str) for w in words):
                    raise ValueError(
                        f"{name}: {path} aliases of {owner!r} must be a list of strings"
                    )
                for word in words:
                    if not _ALIAS.fullmatch(word):
                        raise ValueError(
                            f"{name}: {path} alias {word!r} of {owner!r} is not one lower-case "
                            "word of letters and digits, with apostrophes only inside"
                        )
                aliases.append((owner, tuple(words)))
            sets[key] = tuple(aliases)
            continue
        if not isinstance(items, list) or not all(isinstance(item, str) for item in items):
            raise ValueError(f"{name}: {path} key {key!r} must be a list of strings")
        stripped = tuple(item.strip() for item in items)
        if "" in stripped:
            raise ValueError(f"{name}: {path} key {key!r} holds an empty string")
        sets[key] = stripped
    for template in sets.get("phrases", ()):
        if template.count("{name}") != 1:
            raise ValueError(f"{name}: template {template!r} must hold {{name}} exactly once")
    return sets


def _wake_config(env: _Env, barge_in: BargeInConfig) -> WakeConfig:
    defaults = WakeConfig()
    lookback_limit = MAX_WAKE_BACKLOG_MS - _LOOKBACK_HEADROOM_MS
    wake = WakeConfig(
        names=env.names("MIST_VOICE_WAKE_NAMES", defaults.names),
        kws_threshold=env.probability("MIST_VOICE_KWS_THRESHOLD", defaults.kws_threshold),
        endpoint_threshold=env.probability(
            "MIST_VOICE_WAKE_ENDPOINT_THRESHOLD", defaults.endpoint_threshold
        ),
        lookback_ms=env.bounded_integer(
            "MIST_VOICE_WAKE_LOOKBACK_MS", defaults.lookback_ms, 1, lookback_limit
        ),
        join_gap_ms=env.positive_integer("MIST_VOICE_WAKE_JOIN_GAP_MS", defaults.join_gap_ms),
        gate_tail_ms=env.positive_integer("MIST_VOICE_WAKE_GATE_TAIL_MS", defaults.gate_tail_ms),
        reference_tolerance_ms=env.positive_integer(
            "MIST_VOICE_WAKE_REFERENCE_TOLERANCE_MS", defaults.reference_tolerance_ms
        ),
        endpoint_min_backlog_ms=env.bounded_integer(
            "MIST_VOICE_WAKE_ENDPOINT_MIN_BACKLOG_MS",
            defaults.endpoint_min_backlog_ms,
            _MIN_ENDPOINT_BACKLOG_MS,
            MAX_WAKE_BACKLOG_MS,
        ),
        followup_window_ms=env.integer(
            "MIST_VOICE_FOLLOWUP_WINDOW_MS", defaults.followup_window_ms
        ),
        attend_cue=env.boolean("MIST_VOICE_WAKE_ATTEND_CUE", defaults.attend_cue),
        alias_window_ms=env.positive_integer(
            "MIST_VOICE_WAKE_ALIAS_WINDOW_MS", defaults.alias_window_ms
        ),
        trailing_speech_ms=env.positive_integer(
            "MIST_VOICE_WAKE_TRAILING_SPEECH_MS", defaults.trailing_speech_ms
        ),
    )
    # Arbitration runs during the gate tail, so it must end by the time the tail does.
    if barge_in.wake_arbitration_ms > wake.gate_tail_ms:
        raise ValueError(
            f"MIST_VOICE_WAKE_ARBITRATION_MS: {barge_in.wake_arbitration_ms} ms is longer than "
            f"MIST_VOICE_WAKE_GATE_TAIL_MS, {wake.gate_tail_ms} ms; arbitration must end within "
            "the gate tail"
        )
    sets = env.phrases_file("MIST_VOICE_WAKE_PHRASES_FILE")
    # An alias of a name the spotter is not listening for could never be read, and a name
    # listed as an alias would lose its own rules at the hit; either is a mistake in the file.
    for owner, words in sets.get("aliases", ()):  # type: ignore[misc]
        if owner not in wake.names:
            raise ValueError(
                f"MIST_VOICE_WAKE_PHRASES_FILE: aliases name {owner!r}, which is not one of "
                f"the names, {', '.join(wake.names)}"
            )
        for word in words:
            if word in wake.names:
                raise ValueError(
                    f"MIST_VOICE_WAKE_PHRASES_FILE: alias {word!r} of {owner!r} is a configured "
                    "name"
                )
    return dataclasses.replace(wake, **sets)


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
    the GPU yield, the remote speech service and the earlier profile-based cloned voice used are
    not read either, so a stale `.env` entry for one of them changes nothing.

    `MIST_VOICE_STT_CHAIN` and `MIST_VOICE_TTS_CHAIN` are comma-separated `<engine>@<device>`
    options in the order they are tried, with `cpu` or `cuda` as the device, or `off` alone for
    an empty chain. The per-slot variables they replaced (`MIST_VOICE_STT_ENGINE`,
    `MIST_VOICE_STT_DEVICE`, `MIST_VOICE_STT_MODEL` and the three `MIST_VOICE_TTS_` ones) raise,
    naming their chain, whatever their value: ignored, a stale one would leave an engine where
    the operator no longer expects it. `MIST_VOICE_TTS_VOICE` is a voice identity id, read
    under `MIST_VOICE_IDENTITY_ROOT`; `MIST_VOICE_CACHE_DIR` is where assets built from it go.

    `MIST_VOICE_WAKE_PHRASES_FILE` is the one variable that names a file: a JSON object with
    any of `phrases`, `address_starts`, `stop_before` and `greetings`, each a list of strings
    that replaces that set; a key it omits keeps its default.

    `MIST_VOICE_CHARACTER` (default true) runs every spoken reply through the character stage, a
    `CharacterConfig` with every field at its default except `double_gain`; false turns off the
    whole stage, its limiter included, which has no variable of its own.
    `MIST_VOICE_CHARACTER_DOUBLE_GAIN` (default 0.35) is that `double_gain`, the detuned double's
    gain per copy, from 0 to 1. It is checked even with the stage off, so a bad value
    fails at startup rather than when the stage is next turned on.

    Raises:
        ValueError: A set variable is malformed or removed; the message starts with the
            variable's name.
    """
    env = _Env(environ)
    env.refuse_removed(_REMOVED_PER_SLOT)
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
    double_gain = env.probability("MIST_VOICE_CHARACTER_DOUBLE_GAIN", CharacterConfig().double_gain)
    character_on = env.boolean("MIST_VOICE_CHARACTER", True)
    playback = PlaybackConfig(
        client_prebuffer_ms=env.integer(
            "MIST_VOICE_CLIENT_PREBUFFER_MS", PlaybackConfig().client_prebuffer_ms
        ),
        character=CharacterConfig(double_gain=double_gain) if character_on else None,
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
        stt_chain=env.chain("MIST_VOICE_STT_CHAIN", place.stt_chain),
        tts_chain=env.chain("MIST_VOICE_TTS_CHAIN", place.tts_chain),
        vad_model=env.pattern(
            "MIST_VOICE_VAD_MODEL", place.vad_model, _ONNX_MODEL, "<repo>@<revision>:<file>"
        ),
        eot_model=env.pattern(
            "MIST_VOICE_EOT_MODEL", place.eot_model, _ONNX_MODEL, "<repo>@<revision>:<file>"
        ),
        tts_voice=env.pattern(
            "MIST_VOICE_TTS_VOICE", place.tts_voice, _ID, f"an identity id matching {_ID_FORM}"
        ),
        voice_root=env.absolute_path("MIST_VOICE_IDENTITY_ROOT", place.voice_root),
        voice_cache_dir=env.absolute_path("MIST_VOICE_CACHE_DIR", place.voice_cache_dir),
        vram_budget_mib=env.integer("MIST_VOICE_VRAM_BUDGET_MIB", place.vram_budget_mib),
        release_timeout_s=env.positive_number(
            "MIST_VOICE_RELEASE_TIMEOUT_S", place.release_timeout_s
        ),
        models_dir=env.absolute_path("MIST_VOICE_MODELS_DIR", place.models_dir),
        kws_model=env.pattern(
            "MIST_VOICE_KWS_MODEL",
            place.kws_model,
            _ID,
            f"a model id matching {_ID_FORM}",
        ),
        kws_threads=env.positive_integer("MIST_VOICE_KWS_THREADS", place.kws_threads),
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
        wake=_wake_config(env, barge_in),
        profile=env.text("VOICE_PROFILE", VoiceSettings().profile),
    )
