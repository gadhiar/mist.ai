"""Generate the host tier's speech fixtures: invented phrases spoken by MIST's TTS, composed into
scenarios, with a manifest of the expected text and boundaries.

The host tier (`tests/speech_models`) and the latency bench (`scripts/voice_bench`) read what this
writes. Each phrase in `phrases.json` is invented and holds no personal data. Every phrase is
spoken through the first option of the TTS chain (`MIST_VOICE_TTS_CHAIN`), with the voice identity
named by `--voice-root` and `--voice-id`, then trimmed of the engine's own leading and trailing
silence and composed into one of six scenarios:

- `complete`: a whole request, which end of turn must call finished;
- `pause`: a request broken mid-sentence by `PAUSE_MS` of silence, unfinished at the pause and
  finished at the end;
- `hesitation`: a request that trails off, unfinished at its end;
- `keyword_echo`: a barge-in keyword over MIST's own speech as a microphone hears it, her line
  attenuated `ECHO_ATTENUATION_DB` and delayed `ECHO_DELAY_MS` behind its playback;
- `noise`: a burst of white noise with no speech, which STT should not turn into words;
- `wake`: her name, then a request, with the name ending late enough that the audio before it
  is a backlog an endpoint wake may send (`WakeConfig.endpoint_min_backlog_ms`).

The output goes to `tests/fixtures/voice/generated/` only, which `.gitignore` excludes, because
the audio is generated from a private voice identity and this repository is public. Any other
`--out` is refused before the engine loads. Run it in the backend container, from the repository
root:

    python scripts/voice_fixtures/generate_fixtures.py --voice-root /voices --voice-id <id>
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import sys
import wave
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    # Run as a file (`python scripts/voice_fixtures/generate_fixtures.py`), the repository root
    # is not on the path, and `backend` would not import.
    sys.path.insert(0, str(REPO_ROOT))

from backend.realtime.budget import (
    BargeInConfig,
    EndpointingConfig,
    EngineOption,
    VoiceSettings,
    WakeConfig,
    load_voice_settings,
)
from backend.realtime.engines.audio import OUTPUT_RATE, resample
from backend.realtime.speech import CancelToken, TextSegment, VoiceInfo

RATE = 16000
"""The uplink rate: fixtures are what a microphone sends."""
FRAME_MS = 20
"""One uplink frame; the wake backlog is sent in whole frames."""

GENERATED_DIR = Path("tests", "fixtures", "voice", "generated")
"""The one output directory, relative to the repository root; `.gitignore` excludes it."""
MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1
PHRASES_PATH = Path(__file__).resolve().with_name("phrases.json")

LEAD_MS = 500
TAIL_MS = 1500
"""Long enough for the silence cap, `EndpointingConfig.eot_max_silence_ms`, to end the turn."""
PAUSE_MS = 800
"""Past `endpoint_silence_ms`, so end of turn is asked, and short of `eot_max_silence_ms`."""
WAKE_GAP_MS = 240
ECHO_ATTENUATION_DB = 20.0
ECHO_DELAY_MS = 50
KEYWORD_OFFSET_MS = 600
"""How far into her echoed line the keyword starts."""
EOT_PROBE_MS = EndpointingConfig().endpoint_silence_ms
"""Where end of turn is first asked: this much silence after the speech."""
BACKLOG_MARGIN_MS = 100
WAKE_BACKLOG_MS = WakeConfig().endpoint_min_backlog_ms + BACKLOG_MARGIN_MS
"""The least audio up to the name's end, above the endpoint minimum by a margin."""

TRIM_THRESHOLD_DB = -40.0
TRIM_FRAME_MS = 10

Scenario = Literal["complete", "pause", "hesitation", "keyword_echo", "noise", "wake"]


class FixtureError(ValueError):
    """A phrase, an argument or the voice cannot make a fixture; the message names which."""


# -- The manifest's records -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Span:
    """A stretch of a fixture, in milliseconds from its first sample."""

    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class EotProbe:
    """Where end of turn is asked, and whether the speaker has finished there."""

    at_ms: int
    complete: bool


@dataclass(frozen=True, slots=True)
class WakeInfo:
    """A wake fixture's name end, which is the backlog an endpoint `wake` sends.

    `kind` is the turn kind the request is expected to make, `plain` or `tool`; the bench
    reports by the kind the server measured, so this is a label, not a check.
    """

    name_end_ms: int
    kind: str


@dataclass(frozen=True, slots=True)
class EchoInfo:
    """Where the keyword and her echoed line sit; the echo starts `delay_ms` after playback."""

    keyword: str
    keyword_span: Span
    echo_text: str
    echo_span: Span
    playback_start_ms: int
    attenuation_db: float
    delay_ms: int


@dataclass(frozen=True, slots=True)
class NoiseInfo:
    """A noise burst's place, RMS level and generator seed."""

    span: Span
    level_dbfs: float
    seed: int


@dataclass(frozen=True, slots=True)
class FixtureEntry:
    """One fixture as the manifest records it.

    `text` is what STT should transcribe: the request, the keyword alone over the echo, or ""
    for noise. `onset_ms` is the first speech after silence, None where something else (an echo
    or noise) comes first. `speech` holds the spoken stretches in order.
    """

    id: str
    scenario: Scenario
    file: str
    text: str
    duration_ms: int
    onset_ms: int | None
    speech: tuple[Span, ...]
    eot_probes: tuple[EotProbe, ...]
    wake: WakeInfo | None = None
    echo: EchoInfo | None = None
    noise: NoiseInfo | None = None


@dataclass(frozen=True, slots=True)
class Fixture:
    """A manifest entry and its audio, mono float32 at `RATE`."""

    entry: FixtureEntry
    audio: np.ndarray


@dataclass(frozen=True, slots=True)
class Manifest:
    """What `write_fixtures` wrote: the voice that spoke and every entry."""

    version: int
    sample_rate: int
    voice: VoiceInfo
    fixtures: tuple[FixtureEntry, ...]


# -- Signal helpers ---------------------------------------------------------------------------


def samples(ms: int) -> int:
    """Milliseconds as a sample count at `RATE`."""
    return ms * RATE // 1000


def duration_ms(audio: np.ndarray) -> int:
    """A clip's length in whole milliseconds, rounded."""
    return round(len(audio) * 1000 / RATE)


def silence(ms: int) -> np.ndarray:
    """`ms` of zeros."""
    return np.zeros(samples(ms), dtype=np.float32)


def trim_silence(audio: np.ndarray, threshold_db: float = TRIM_THRESHOLD_DB) -> np.ndarray:
    """The clip without its leading and trailing silence.

    A 10 ms frame is silent when its RMS is more than `threshold_db` below the loudest frame's,
    so the cut follows the clip's own level rather than an absolute one.

    Raises:
        FixtureError: The clip is silent throughout.
    """
    clip = np.asarray(audio, dtype=np.float32).reshape(-1)
    step = samples(TRIM_FRAME_MS)
    count = math.ceil(len(clip) / step)
    if count == 0:
        raise FixtureError("the engine returned a silent clip")
    framed = np.zeros(count * step, dtype=np.float64)
    framed[: len(clip)] = clip
    rms = np.sqrt(np.mean(np.square(framed.reshape(count, step)), axis=1))
    peak = float(rms.max())
    if peak <= 0.0:
        raise FixtureError("the engine returned a silent clip")
    loud = np.flatnonzero(rms >= peak * 10 ** (threshold_db / 20))
    return clip[loud[0] * step : min((loud[-1] + 1) * step, len(clip))]


def noise_burst(burst_ms: int, level_dbfs: float, seed: int) -> np.ndarray:
    """White noise of `burst_ms` whose RMS is `level_dbfs`, the same for the same seed."""
    noise = np.random.default_rng(seed).standard_normal(samples(burst_ms))
    noise *= 10 ** (level_dbfs / 20) / np.sqrt(np.mean(np.square(noise)))
    return noise.astype(np.float32)


def _place(length: int, tracks: Sequence[tuple[int, np.ndarray]]) -> np.ndarray:
    """A clip of `length` samples with each `(offset, samples)` track added at its offset."""
    out = np.zeros(length, dtype=np.float32)
    for offset, track in tracks:
        out[offset : offset + len(track)] += track
    return out


def _gain(db: float) -> float:
    return 10 ** (-db / 20)


# -- Scenarios --------------------------------------------------------------------------------


def complete_fixture(fixture_id: str, text: str, clip: np.ndarray) -> Fixture:
    """A whole request between silences, finished at its end."""
    end = LEAD_MS + duration_ms(clip)
    audio = np.concatenate([silence(LEAD_MS), clip, silence(TAIL_MS)])
    entry = FixtureEntry(
        id=fixture_id,
        scenario="complete",
        file=f"{fixture_id}.wav",
        text=text,
        duration_ms=duration_ms(audio),
        onset_ms=LEAD_MS,
        speech=(Span(LEAD_MS, end),),
        eot_probes=(EotProbe(end + EOT_PROBE_MS, True),),
    )
    return Fixture(entry, audio)


def pause_fixture(
    fixture_id: str, before: str, after: str, before_clip: np.ndarray, after_clip: np.ndarray
) -> Fixture:
    """A request broken by `PAUSE_MS` of silence: unfinished at the pause, finished at the end."""
    first_end = LEAD_MS + duration_ms(before_clip)
    second_start = first_end + PAUSE_MS
    second_end = second_start + duration_ms(after_clip)
    audio = np.concatenate(
        [silence(LEAD_MS), before_clip, silence(PAUSE_MS), after_clip, silence(TAIL_MS)]
    )
    entry = FixtureEntry(
        id=fixture_id,
        scenario="pause",
        file=f"{fixture_id}.wav",
        text=f"{before} {after}",
        duration_ms=duration_ms(audio),
        onset_ms=LEAD_MS,
        speech=(Span(LEAD_MS, first_end), Span(second_start, second_end)),
        eot_probes=(
            EotProbe(first_end + EOT_PROBE_MS, False),
            EotProbe(second_end + EOT_PROBE_MS, True),
        ),
    )
    return Fixture(entry, audio)


def hesitation_fixture(fixture_id: str, text: str, clip: np.ndarray) -> Fixture:
    """A request that trails off, unfinished at its end."""
    fixture = complete_fixture(fixture_id, text, clip)
    entry = dataclasses.replace(
        fixture.entry,
        scenario="hesitation",
        eot_probes=(EotProbe(fixture.entry.speech[-1].end_ms + EOT_PROBE_MS, False),),
    )
    return Fixture(entry, fixture.audio)


def keyword_echo_fixture(
    fixture_id: str, keyword: str, keyword_clip: np.ndarray, echo_text: str, echo_clip: np.ndarray
) -> Fixture:
    """A keyword said over her own line, as the microphone hears it.

    Her line plays from `LEAD_MS` and reaches the microphone `ECHO_DELAY_MS` later and
    `ECHO_ATTENUATION_DB` quieter. The keyword starts `KEYWORD_OFFSET_MS` into the echo.
    """
    echo_start = LEAD_MS + ECHO_DELAY_MS
    echo_end = echo_start + duration_ms(echo_clip)
    keyword_start = echo_start + KEYWORD_OFFSET_MS
    keyword_end = keyword_start + duration_ms(keyword_clip)
    length = samples(max(echo_end, keyword_end) + TAIL_MS)
    audio = _place(
        length,
        [
            (samples(echo_start), echo_clip * _gain(ECHO_ATTENUATION_DB)),
            (samples(keyword_start), keyword_clip),
        ],
    )
    entry = FixtureEntry(
        id=fixture_id,
        scenario="keyword_echo",
        file=f"{fixture_id}.wav",
        text=keyword,
        duration_ms=duration_ms(audio),
        onset_ms=None,
        speech=(Span(keyword_start, keyword_end),),
        eot_probes=(),
        echo=EchoInfo(
            keyword=keyword,
            keyword_span=Span(keyword_start, keyword_end),
            echo_text=echo_text,
            echo_span=Span(echo_start, echo_end),
            playback_start_ms=LEAD_MS,
            attenuation_db=ECHO_ATTENUATION_DB,
            delay_ms=ECHO_DELAY_MS,
        ),
    )
    return Fixture(entry, audio)


def noise_fixture(fixture_id: str, *, burst_ms: int, level_dbfs: float, seed: int) -> Fixture:
    """A noise burst between silences, with no speech and no expected text."""
    audio = np.concatenate(
        [silence(LEAD_MS), noise_burst(burst_ms, level_dbfs, seed), silence(TAIL_MS)]
    )
    entry = FixtureEntry(
        id=fixture_id,
        scenario="noise",
        file=f"{fixture_id}.wav",
        text="",
        duration_ms=duration_ms(audio),
        onset_ms=None,
        speech=(),
        eot_probes=(),
        noise=NoiseInfo(Span(LEAD_MS, LEAD_MS + burst_ms), level_dbfs, seed),
    )
    return Fixture(entry, audio)


def wake_fixture(
    fixture_id: str,
    name: str,
    request: str,
    kind: str,
    name_clip: np.ndarray,
    request_clip: np.ndarray,
) -> Fixture:
    """Her name, then a request; the audio through the name's end is the endpoint's backlog.

    The lead is stretched so the name ends at least `WAKE_BACKLOG_MS` in, on a frame boundary,
    and never shorter than `LEAD_MS`.
    """
    name_ms = duration_ms(name_clip)
    name_end = max(LEAD_MS + name_ms, WAKE_BACKLOG_MS)
    name_end = math.ceil(name_end / FRAME_MS) * FRAME_MS
    lead = name_end - name_ms
    request_start = name_end + WAKE_GAP_MS
    request_end = request_start + duration_ms(request_clip)
    length = samples(request_end + TAIL_MS)
    audio = _place(length, [(samples(lead), name_clip), (samples(request_start), request_clip)])
    entry = FixtureEntry(
        id=fixture_id,
        scenario="wake",
        file=f"{fixture_id}.wav",
        text=f"{name} {request}",
        duration_ms=duration_ms(audio),
        onset_ms=lead,
        speech=(Span(lead, name_end), Span(request_start, request_end)),
        eot_probes=(EotProbe(request_end + EOT_PROBE_MS, True),),
        wake=WakeInfo(name_end_ms=name_end, kind=kind),
    )
    return Fixture(entry, audio)


# -- Phrases ----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PausePhrase:
    """A request in two halves, said either side of the pause."""

    before: str
    after: str


@dataclass(frozen=True, slots=True)
class EchoPhrase:
    """A barge-in keyword and the line of hers it is said over."""

    keyword: str
    echo: str


@dataclass(frozen=True, slots=True)
class WakePhrase:
    """Her name as said to her, the request after it, and the turn kind it should make."""

    name: str
    request: str
    kind: str


@dataclass(frozen=True, slots=True)
class NoiseSpec:
    """A noise burst's length, RMS level and seed."""

    burst_ms: int
    level_dbfs: float
    seed: int


@dataclass(frozen=True, slots=True)
class Phrases:
    """`phrases.json`, checked."""

    complete: tuple[str, ...]
    pause: tuple[PausePhrase, ...]
    hesitation: tuple[str, ...]
    keyword_echo: tuple[EchoPhrase, ...]
    wake: tuple[WakePhrase, ...]
    noise: tuple[NoiseSpec, ...]


def _strings(raw: Mapping[str, object], key: str) -> tuple[str, ...]:
    items = raw.get(key)
    if not isinstance(items, list) or not all(isinstance(i, str) and i.strip() for i in items):
        raise FixtureError(f"phrases: {key!r} must be a list of non-empty strings")
    return tuple(items)


def _records(raw: Mapping[str, object], key: str, fields: tuple[str, ...]) -> list[dict]:
    items = raw.get(key)
    if not isinstance(items, list) or not all(
        isinstance(item, dict) and set(item) == set(fields) for item in items
    ):
        raise FixtureError(f"phrases: {key!r} must be a list of objects with {', '.join(fields)}")
    return items


def load_phrases(path: Path) -> Phrases:
    """Read and check `phrases.json`.

    Each keyword must be a default barge-in keyword, so the fixture tests the words the policy
    listens for, and each wake name must say one of her default names.

    Raises:
        FixtureError: The file is malformed, or a keyword or name is outside those sets; the
            message names the key or the value.
    """
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise FixtureError(f"phrases: cannot read {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise FixtureError(f"phrases: {path} must hold a JSON object")
    keywords = set(BargeInConfig().keywords)
    names = set(WakeConfig().names)
    echoes = tuple(
        EchoPhrase(item["keyword"], item["echo"])
        for item in _records(raw, "keyword_echo", ("keyword", "echo"))
    )
    for echo in echoes:
        if echo.keyword not in keywords:
            raise FixtureError(f"phrases: keyword {echo.keyword!r} is not a barge-in keyword")
    wakes = tuple(
        WakePhrase(item["name"], item["request"], item["kind"])
        for item in _records(raw, "wake", ("name", "request", "kind"))
    )
    for wake in wakes:
        words = {word.strip(",.?!").lower() for word in wake.name.split()}
        if not words & names:
            raise FixtureError(f"phrases: wake name {wake.name!r} says none of {sorted(names)}")
    return Phrases(
        complete=_strings(raw, "complete"),
        pause=tuple(
            PausePhrase(item["before"], item["after"])
            for item in _records(raw, "pause", ("before", "after"))
        ),
        hesitation=_strings(raw, "hesitation"),
        keyword_echo=echoes,
        wake=wakes,
        noise=tuple(
            NoiseSpec(int(item["burst_ms"]), float(item["level_dbfs"]), int(item["seed"]))
            for item in _records(raw, "noise", ("burst_ms", "level_dbfs", "seed"))
        ),
    )


# -- The voice --------------------------------------------------------------------------------


class ClipSource(Protocol):
    """Speaks a phrase: mono float32 audio at `RATE`, silence at the ends allowed."""

    voice: VoiceInfo

    def clip(self, text: str) -> np.ndarray:
        """The phrase spoken once."""
        ...


class SynthesizerClips:
    """A `SpeechSynthesizer`'s output for a whole phrase, resampled to the uplink rate."""

    def __init__(self, synthesizer: object) -> None:
        self._synthesizer = synthesizer
        self.voice: VoiceInfo = synthesizer.voice  # type: ignore[attr-defined]

    def clip(self, text: str) -> np.ndarray:
        """One `synthesize` call over the whole phrase, as a single segment."""
        segment = TextSegment(text=text, start_char=0, end_char=len(text))
        chunks = self._synthesizer.synthesize(segment, CancelToken())  # type: ignore[attr-defined]
        pcm = b"".join(chunk.pcm16 for chunk in chunks)
        audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        return resample(audio, OUTPUT_RATE, RATE)


def first_tts_option(
    environ: Mapping[str, str], voice_root: str, voice_id: str
) -> tuple[VoiceSettings, EngineOption]:
    """The voice settings narrowed to the TTS chain's first option, speaking `voice_id`.

    The identity comes from the arguments, which replace `MIST_VOICE_IDENTITY_ROOT` and
    `MIST_VOICE_TTS_VOICE`; the STT chain is emptied, so loading these settings loads TTS alone.

    Raises:
        FixtureError: A voice setting is malformed (the message names the variable), or the TTS
            chain is off.
    """
    merged = {
        **environ,
        "MIST_VOICE_IDENTITY_ROOT": voice_root,
        "MIST_VOICE_TTS_VOICE": voice_id,
    }
    try:
        settings = load_voice_settings(merged)
    except ValueError as exc:
        raise FixtureError(str(exc)) from exc
    chain = settings.placement.tts_chain
    if not chain:
        raise FixtureError("MIST_VOICE_TTS_CHAIN is off: there is no TTS engine to speak with")
    placement = dataclasses.replace(settings.placement, stt_chain=(), tts_chain=(chain[0],))
    return dataclasses.replace(settings, placement=placement), chain[0]


def load_clip_source(voice_root: str, voice_id: str, environ: Mapping[str, str]) -> ClipSource:
    """The TTS chain's first option, loaded the way the backend loads it, as a clip source.

    Raises:
        FixtureError: The settings are malformed, the chain is off, or the engine did not load;
            the backend's WARNING above the error says why.
    """
    from backend.factories import build_speech_engines

    settings, option = first_tts_option(environ, voice_root, voice_id)
    engines = build_speech_engines(settings)
    if engines.synthesizer is None:
        raise FixtureError(
            f"{option.engine}@{option.device} did not load; the log line above names why"
        )
    return SynthesizerClips(engines.synthesizer)


def build_fixtures(phrases: Phrases, source: ClipSource) -> list[Fixture]:
    """Speak every phrase once, trim each clip, and compose every scenario."""

    def say(text: str) -> np.ndarray:
        return trim_silence(source.clip(text))

    fixtures: list[Fixture] = []
    for n, text in enumerate(phrases.complete, 1):
        fixtures.append(complete_fixture(f"complete-{n:02d}", text, say(text)))
    for n, pause in enumerate(phrases.pause, 1):
        fixtures.append(
            pause_fixture(
                f"pause-{n:02d}", pause.before, pause.after, say(pause.before), say(pause.after)
            )
        )
    for n, text in enumerate(phrases.hesitation, 1):
        fixtures.append(hesitation_fixture(f"hesitation-{n:02d}", text, say(text)))
    for n, echo in enumerate(phrases.keyword_echo, 1):
        fixtures.append(
            keyword_echo_fixture(
                f"keyword-echo-{n:02d}", echo.keyword, say(echo.keyword), echo.echo, say(echo.echo)
            )
        )
    for n, spec in enumerate(phrases.noise, 1):
        fixtures.append(
            noise_fixture(
                f"noise-{n:02d}", burst_ms=spec.burst_ms, level_dbfs=spec.level_dbfs, seed=spec.seed
            )
        )
    for n, wake in enumerate(phrases.wake, 1):
        fixtures.append(
            wake_fixture(
                f"wake-{n:02d}",
                wake.name,
                wake.request,
                wake.kind,
                say(wake.name),
                say(wake.request),
            )
        )
    return fixtures


# -- Files ------------------------------------------------------------------------------------


def write_wav(path: Path, audio: np.ndarray) -> None:
    """Mono PCM16 at `RATE`, clipped to [-1, 1]."""
    pcm = np.rint(np.clip(audio, -1.0, 1.0) * 32767).astype("<i2").tobytes()
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(RATE)
        handle.writeframes(pcm)


def read_wav(path: Path) -> np.ndarray:
    """A fixture's audio as float32, checked to be mono PCM16 at `RATE`.

    Raises:
        FixtureError: The file is not mono 16-bit audio at `RATE`.
    """
    with wave.open(str(path), "rb") as handle:
        shape = (handle.getnchannels(), handle.getsampwidth(), handle.getframerate())
        if shape != (1, 2, RATE):
            raise FixtureError(f"{path.name}: expected mono PCM16 at {RATE} Hz, got {shape}")
        pcm = handle.readframes(handle.getnframes())
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32767


def write_fixtures(out_dir: Path, fixtures: Sequence[Fixture], voice: VoiceInfo) -> None:
    """Write each fixture's WAV, then the manifest, which is replaced last and whole."""
    out_dir.mkdir(parents=True, exist_ok=True)
    for fixture in fixtures:
        write_wav(out_dir / fixture.entry.file, fixture.audio)
    manifest = Manifest(
        version=MANIFEST_VERSION,
        sample_rate=RATE,
        voice=voice,
        fixtures=tuple(fixture.entry for fixture in fixtures),
    )
    staging = out_dir / f"{MANIFEST_NAME}.tmp"
    staging.write_text(json.dumps(dataclasses.asdict(manifest), indent=2) + "\n", encoding="utf-8")
    staging.replace(out_dir / MANIFEST_NAME)


def _span(raw: Mapping[str, int]) -> Span:
    return Span(raw["start_ms"], raw["end_ms"])


def _entry(raw: Mapping) -> FixtureEntry:
    wake, echo, noise = raw.get("wake"), raw.get("echo"), raw.get("noise")
    return FixtureEntry(
        id=raw["id"],
        scenario=raw["scenario"],
        file=raw["file"],
        text=raw["text"],
        duration_ms=raw["duration_ms"],
        onset_ms=raw["onset_ms"],
        speech=tuple(_span(span) for span in raw["speech"]),
        eot_probes=tuple(EotProbe(p["at_ms"], p["complete"]) for p in raw["eot_probes"]),
        wake=None if wake is None else WakeInfo(wake["name_end_ms"], wake["kind"]),
        echo=(
            None
            if echo is None
            else EchoInfo(
                keyword=echo["keyword"],
                keyword_span=_span(echo["keyword_span"]),
                echo_text=echo["echo_text"],
                echo_span=_span(echo["echo_span"]),
                playback_start_ms=echo["playback_start_ms"],
                attenuation_db=echo["attenuation_db"],
                delay_ms=echo["delay_ms"],
            )
        ),
        noise=(
            None
            if noise is None
            else NoiseInfo(_span(noise["span"]), noise["level_dbfs"], noise["seed"])
        ),
    )


def read_manifest(out_dir: Path) -> Manifest:
    """The manifest `write_fixtures` wrote in `out_dir`.

    Raises:
        FixtureError: It is missing, malformed, or of another version.
    """
    path = out_dir / MANIFEST_NAME
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw["version"] != MANIFEST_VERSION:
            raise FixtureError(f"{path}: version {raw['version']}, expected {MANIFEST_VERSION}")
        return Manifest(
            version=raw["version"],
            sample_rate=raw["sample_rate"],
            voice=VoiceInfo(**raw["voice"]),
            fixtures=tuple(_entry(item) for item in raw["fixtures"]),
        )
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, FixtureError):
            raise
        raise FixtureError(f"{path}: cannot read the manifest: {exc!r}") from exc


def resolve_out(out: Path, repo_root: Path) -> Path:
    """`out`, resolved, if it is the generated directory of `repo_root`.

    Both are compared after resolving symlinks, except that the expected path is built on the
    resolved root, so a generated directory that is itself a link to somewhere else is refused.

    Raises:
        FixtureError: `out` is any other directory.
    """
    expected = repo_root.resolve() / GENERATED_DIR
    actual = out.resolve()
    if actual != expected:
        raise FixtureError(
            f"--out must be {GENERATED_DIR.as_posix()} under the repository root, which "
            f".gitignore excludes; {out} resolves to {actual}"
        )
    return actual


SourceLoader = Callable[[str, str, Mapping[str, str]], ClipSource]


def main(
    argv: Sequence[str] | None = None,
    *,
    repo_root: Path = REPO_ROOT,
    environ: Mapping[str, str] | None = None,
    load_source: SourceLoader = load_clip_source,
) -> int:
    """Generate the fixtures; 0 on success, 2 for a refused `--out`, 1 when the voice fails."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--voice-root", required=True, help="the voice identity root, absolute")
    parser.add_argument("--voice-id", required=True, help="the voice identity id to speak with")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help=f"must be {GENERATED_DIR.as_posix()} under the repository root (the default)",
    )
    parser.add_argument("--phrases", type=Path, default=PHRASES_PATH)
    args = parser.parse_args(argv)
    try:
        out = resolve_out(
            args.out if args.out is not None else repo_root / GENERATED_DIR, repo_root
        )
    except FixtureError as exc:
        print(f"generate_fixtures: {exc}", file=sys.stderr)
        return 2
    try:
        phrases = load_phrases(args.phrases)
        source = load_source(
            args.voice_root, args.voice_id, os.environ if environ is None else environ
        )
        fixtures = build_fixtures(phrases, source)
    except FixtureError as exc:
        print(f"generate_fixtures: {exc}", file=sys.stderr)
        return 1
    write_fixtures(out, fixtures, source.voice)
    print(f"generate_fixtures: wrote {len(fixtures)} fixtures and {MANIFEST_NAME} to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
