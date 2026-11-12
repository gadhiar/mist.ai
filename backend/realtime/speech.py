"""Speech engine contracts: the protocols model adapters implement and the values they exchange.

The realtime layer never imports a model library. Ingress, the turn controller and the session
talk to these protocols, and the adapters under `backend/realtime/engines/` (Silero VAD, Smart
Turn, and the STT and TTS engines step 0 picks) implement them, so an engine can move between
the GPU and the CPU by configuration alone. Every engine call is blocking and runs on an
`InferenceLane` worker thread (`lanes.py`); a `CancelToken` is how a barge-in reaches a call
already running there.
"""

import bisect
import math
import threading
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

from .protocol import Spotter

Phase = Literal["idle", "held", "thinking", "speaking", "text"]
"""The turn controller's phase, which the barge-in policy reads to judge user speech."""


class CancelToken:
    """A one-way flag a turn sets to stop the engine work it started.

    Engines run on lane threads, so the flag is a `threading.Event`: an engine checks it between
    chunks or before a pending decode, and the lane checks it before a queued job starts and
    after a running one returns. Once set it stays set; a turn makes a new token per job group.
    """

    __slots__ = ("_event",)

    def __init__(self) -> None:
        self._event = threading.Event()

    def set(self) -> None:
        """Mark the work cancelled. Safe to call from any thread, and more than once."""
        self._event.set()

    def is_set(self) -> bool:
        """Whether the work has been cancelled."""
        return self._event.is_set()


@dataclass(frozen=True, slots=True)
class VoiceInfo:
    """The voice a synthesizer speaks with, as reported to clients and logs.

    MIST speaks one of the engine's stock voices, not a cloned one, so the report names the
    engine (its id as `speech_arms.json` names it), the stock voice and the voice profile
    (`VoiceSettings.profile`) the voice was chosen for.
    """

    engine: str
    voice: str
    profile: str


@dataclass(frozen=True, slots=True)
class TextSegment:
    """A span of the reply text sent to TTS as one synthesis call.

    The character offsets locate the segment in the turn's generated text, which is how the
    playback ledger maps played audio back to the spoken prefix.
    """

    text: str
    start_char: int
    end_char: int


@dataclass(frozen=True, slots=True)
class SynthChunk:
    """A piece of synthesized audio and the segment it voices."""

    pcm16: bytes
    samples: int
    segment: TextSegment


@dataclass(frozen=True, slots=True)
class TranscriptUpdate:
    """A partial or final transcript of the utterance so far.

    `stable_chars` counts the leading characters not expected to change in later partials.
    `no_speech_prob` and `avg_logprob` are Whisper's own non-speech signals, returned rather than
    acted on so that ingress applies the drop rule once, with configurable thresholds, and
    publishes what it dropped; engines without them leave both `None`.
    """

    text: str
    stable_chars: int
    is_final: bool
    audio_ms: int
    no_speech_prob: float | None = None
    avg_logprob: float | None = None


class VadStream(Protocol):
    """Per-utterance voice activity state; recurrent models carry state across windows."""

    def score(self, window: np.ndarray) -> float:
        """Score one window of 512 float32 samples at 16 kHz as a speech probability."""
        ...

    def reset(self) -> None:
        """Clear the carried state at an utterance boundary."""
        ...


class VoiceActivityDetector(Protocol):
    """A VAD engine, shared by sessions, that opens one stream per audio source."""

    def open_stream(self) -> VadStream:
        """Open a stream with fresh state."""
        ...


class TurnEndDetector(Protocol):
    """A semantic end-of-turn model that judges whether the speaker has finished."""

    def probability_complete(self, audio: np.ndarray) -> float:
        """Score up to the last 8 s of 16 kHz float32 audio as the probability the turn ended."""
        ...


class TranscriptStream(Protocol):
    """One utterance's transcription: audio is pushed in, decodes run on the STT lane."""

    def push(self, pcm: np.ndarray) -> None:
        """Append 16 kHz float32 audio to the utterance."""
        ...

    def decode_partial(self, token: CancelToken) -> TranscriptUpdate:
        """Transcribe the utterance so far; a set token skips a decode not yet started."""
        ...

    def decode_final(self, token: CancelToken) -> TranscriptUpdate:
        """Transcribe the whole utterance as its final transcript."""
        ...

    def close(self) -> None:
        """End the stream; a second call does nothing.

        A GPU release or a cut can close a stream while one of its decodes is still running on
        the STT lane, so that decode's result is discarded: the decode raises `LaneJobCancelled`
        (`lanes.py`) instead of returning a transcript of an utterance nobody is listening to,
        and so does a decode that starts after `close`. A push after `close` raises
        `RealtimeError`. Each owner closes its stream exactly once.
        """
        ...


class StreamingTranscriber(Protocol):
    """An STT engine that opens one stream per utterance."""

    model_id: str

    def open_stream(self) -> TranscriptStream:
        """Open a stream for a new utterance."""
        ...


class SpeechSynthesizer(Protocol):
    """A TTS engine speaking one stock voice, chosen when the adapter is built.

    `engine_id` is the engine's id (`PlacementConfig.tts_engine`) and `voice` the stock voice it
    speaks. `sample_rate` is the rate of the audio the adapter yields: 24000 whatever the
    engine's own rate, which the adapter converts from.
    """

    model_id: str
    engine_id: str
    sample_rate: int
    voice: VoiceInfo

    def synthesize(self, segment: TextSegment, token: CancelToken) -> Iterator[SynthChunk]:
        """Synthesize one segment, checking the token before each chunk."""
        ...


@dataclass(frozen=True, slots=True)
class NameHit:
    """A name spotter's score crossed its threshold on a rising edge.

    `end_pos` is an uplink position, a count of 20 ms frames: the position of the last frame of
    the hop whose score crossed, so ingress can find where the name was said from it. `keyword`
    is the name the model scored and `score` the crossing score, on that model's own scale.
    """

    end_pos: int
    keyword: str
    score: float


class NameSpotterStream(Protocol):
    """One audio source's spotting state: the feature window and the refractory period."""

    def push(self, pcm: bytes, pos: int) -> list[NameHit]:
        """Score one 20 ms frame of PCM16 mono 16 kHz audio at uplink position `pos`.

        Returns the hits whose hop ended on this frame, usually none.
        """
        ...

    def reset(self) -> None:
        """Forget the partial context, so the next push starts a new one."""
        ...


@dataclass(frozen=True, slots=True)
class SpotterCalibration:
    """One spotter's score scale, mapped onto two scales every spotter shares.

    Scores from different models are not comparable, so arbitration between a server hit and an
    endpoint wake ranks them on these instead. Both tables are `(score, value)` points with
    ascending scores, measured on the tuning split. `false_accepts` maps a score to the
    estimated false accepts per hour of tuning media at that score, made monotone, so its rates
    do not increase.
    `positive_quantile` maps a score to the share, from 0 to 1, of the tuning positives that
    scored at or below it, so its shares do not decrease.
    """

    false_accepts: tuple[tuple[float, float], ...]
    positive_quantile: tuple[tuple[float, float], ...]


class NameSpotter(Protocol):
    """A model trained to spot MIST's names, shared by sessions; CPU-only.

    `window_ms` is the audio one score looks at. `hit_lag_ms` is the typical delay from the
    name's end to the hit, from the model's manifest, which is how `wake_screen_ms` estimates
    when the name ended. `calibration` holds each spotter's tables from the tuning pass; a
    spotter with no entry is uncalibrated, and its events rank by margin below every calibrated
    one (`arbitration_key`).
    """

    model_id: str
    names: tuple[str, ...]
    window_ms: int
    hit_lag_ms: int
    calibration: Mapping[Spotter, SpotterCalibration]

    def open_stream(self, threshold: float) -> NameSpotterStream:
        """Open a stream with fresh context that reports scores at or above `threshold`."""
        ...

    def score_window(self, pcm: bytes) -> float:
        """The highest score over a finished buffer of PCM16 mono 16 kHz audio."""
        ...


def _interpolate(score: float, table: tuple[tuple[float, float], ...]) -> float | None:
    if not table:
        return None
    if score <= table[0][0]:
        return table[0][1]
    if score >= table[-1][0]:
        return table[-1][1]
    # The first point at or above the score; the one before it lies strictly below, so the two
    # scores differ and the division is safe.
    upper = bisect.bisect_left(table, score, key=lambda point: point[0])
    (x0, y0), (x1, y1) = table[upper - 1], table[upper]
    return y0 + (y1 - y0) * (score - x0) / (x1 - x0)


def false_accepts_per_hour(score: float, table: tuple[tuple[float, float], ...]) -> float | None:
    """Estimated false accepts per hour at `score`, read from a `false_accepts` table.

    Interpolates linearly between the two points around the score and clamps to the table's
    first or last value outside it, so a score above every tuning false accept reads the
    table's floor. None for an empty table.
    """
    return _interpolate(score, table)


def positive_quantile(score: float, table: tuple[tuple[float, float], ...]) -> float | None:
    """The share, from 0 to 1, of tuning positives scoring at or below `score`.

    Interpolates and clamps as `false_accepts_per_hour` does. None for an empty table.
    """
    return _interpolate(score, table)


def arbitration_key(
    score: float, threshold: float, cal: SpotterCalibration | None
) -> tuple[float, float]:
    """The key that ranks wake candidates from different spotters; the lowest key wins.

    Calibrated (both tables present): `(false accepts per hour, 1 - positive quantile)`. A real
    wake usually scores above every false accept in the tuning media, so two candidates often
    tie at the false-accept floor, and the quantile then separates a clear wake from a marginal
    one. Uncalibrated: `(inf, -(score - threshold))`, so every calibrated candidate outranks it
    and two uncalibrated ones fall back to their margin. Ties after the key go to arrival order,
    which the caller applies.
    """
    if cal is not None:
        rate = false_accepts_per_hour(score, cal.false_accepts)
        share = positive_quantile(score, cal.positive_quantile)
        if rate is not None and share is not None:
            return (rate, 1.0 - share)
    return (math.inf, -(score - threshold))
