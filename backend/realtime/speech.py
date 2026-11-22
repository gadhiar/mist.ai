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

NEUTRAL_STATE = "neutral"
"""The speaking state every segment has by default and every engine voices."""

SPEAKING_STATES: frozenset[str] = frozenset({"neutral", "calm", "awake", "sleepy", "excited"})
"""The speaking-state vocabulary every chooser, identity and adapter shares (Raj, 12 Nov).

It lives in the realtime contract so that the chooser upstream, the identity manifest
(`voice_assets.py`) and every adapter use one set of names. Adding a state is one entry here
plus, for an engine that voices it from a clip, one clip in the identity.
"""


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

    MIST speaks her own voice identity (`voice_assets.VoiceIdentity`) through whichever engine
    the TTS chain loaded, so the report names the engine (its id as the TTS chain names it, for
    example `pocket-clone`), the voice identity id (`VoiceIdentity.voice_id`, for example
    `mist-v1`) and the voice profile (`VoiceSettings.profile`, `mist` by default).
    """

    engine: str
    voice: str
    profile: str


@dataclass(frozen=True, slots=True)
class TextSegment:
    """A span of the reply text sent to TTS as one synthesis call.

    The character offsets locate the segment in the turn's generated text, which is how the
    playback ledger maps played audio back to the spoken prefix.

    `state` is the speaking state requested for this segment, a member of `SPEAKING_STATES`. A
    segment built without one is `neutral` (`NEUTRAL_STATE`), so every `TextSegment(text, start,
    end)` keeps its meaning. The request is not a promise: an engine that does not voice the
    state voices `neutral` instead (the rules are on `SpeechSynthesizer`), and the chunk's own
    `state` says which was voiced.
    """

    text: str
    start_char: int
    end_char: int
    state: str = NEUTRAL_STATE


@dataclass(frozen=True, slots=True)
class SynthChunk:
    """A piece of synthesized audio and the segment it voices.

    `state` is the speaking state the audio was voiced in, after any fallback to `neutral`, so
    it can differ from `segment.state`.
    """

    pcm16: bytes
    samples: int
    segment: TextSegment
    state: str = NEUTRAL_STATE


@dataclass(frozen=True, slots=True)
class WordTiming:
    """One word of a transcript and where the engine heard it.

    `start_ms` and `end_ms` are milliseconds from the stream's first pushed sample. `text` is the
    word as the transcript writes it, punctuation included ("Mist,").
    """

    text: str
    start_ms: int
    end_ms: int


@dataclass(frozen=True, slots=True)
class TranscriptUpdate:
    """A partial or final transcript of the utterance so far.

    `stable_chars` counts the leading characters not expected to change in later partials.
    `no_speech_prob` and `avg_logprob` are Whisper's own non-speech signals, returned rather than
    acted on so that ingress applies the drop rule once, with configurable thresholds, and
    publishes what it dropped; engines without them leave both `None`.

    `words` are the transcript's words with their timings, so a reader can tell which word was
    heard at a given moment. When given, they join with single spaces to the text with its whitespace
    collapsed, and no word is empty or holds whitespace; construction refuses anything else, so a
    timing always belongs to the word the text shows. An engine that cannot meet that for a
    decode, or has no timings, leaves `words` `None` for it.
    """

    text: str
    stable_chars: int
    is_final: bool
    audio_ms: int
    no_speech_prob: float | None = None
    avg_logprob: float | None = None
    words: tuple[WordTiming, ...] | None = None

    def __post_init__(self) -> None:
        if self.words is not None and not TranscriptUpdate.words_match(self.text, self.words):
            # The message names no word: a transcript never reaches a log or an error.
            raise ValueError(
                f"transcript words do not join to its text ({len(self.words)} words given)"
            )

    @staticmethod
    def words_match(text: str, words: tuple[WordTiming, ...]) -> bool:
        """Whether `words` meet the invariant above for `text`."""
        if any(word.text.split() != [word.text] for word in words):
            return False
        return " ".join(word.text for word in words) == " ".join(text.split())


class VadStream(Protocol):
    """Per-utterance voice activity state; recurrent models carry state across windows."""

    def score(self, window: np.ndarray) -> float:
        """Score one window of 512 float32 samples at 16 kHz as a speech probability."""
        ...

    def reset(self) -> None:
        """Clear the carried state at an utterance boundary."""
        ...


class VoiceActivityDetector(Protocol):
    """A VAD engine, shared by sessions, that opens one stream per audio source.

    `model_id` names the model it loaded, which the engine walk reports in the VAD's status.
    """

    model_id: str

    def open_stream(self) -> VadStream:
        """Open a stream with fresh state."""
        ...


class TurnEndDetector(Protocol):
    """A semantic end-of-turn model that judges whether the speaker has finished.

    `model_id` names the model it loaded, which the engine walk reports in the end-of-turn
    status.
    """

    model_id: str

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

    def release(self) -> None:
        """Free what the engine holds; a second call does nothing.

        This is the one hook a GPU release needs from an engine. The runtime calls it on the
        lane thread after the engine's jobs have stopped, and drops the engine after it returns.
        An in-process adapter drops its model references here, and the runtime then empties the
        CUDA cache; an adapter that proxies to a sidecar asks the sidecar to unload, so an
        engine in its own container frees its memory too. A `release` that raises is logged at
        ERROR with the engine id and no text, and the drop goes on. After `release`, any call
        but `release` raises `SpeechEngineError`.
        """
        ...


class SpeechSynthesizer(Protocol):
    """A TTS engine speaking MIST's voice identity, chosen when the adapter is built.

    `engine_id` is the `engine` of the `PlacementConfig.tts_chain` option the adapter was built
    from, and `voice` reports the voice identity it speaks. `sample_rate` is the rate of the audio
    the adapter yields: 24000 whatever the engine's own rate, which the adapter converts from.

    Speaking states. `states` holds the states the engine voices distinctly: always
    `NEUTRAL_STATE`, and nothing outside `SPEAKING_STATES`; an engine with no per-state control
    declares `frozenset({"neutral"})`. `synthesize` reads `segment.state`:

    - A state in `states` is voiced with the adapter's own control for it (Pocket: the voice
      state computed from the identity's clip for that state), and every chunk carries it.
    - Any other state is voiced as `neutral`, and every chunk carries `neutral`. It never
      raises. The first such request per adapter and state logs one line naming the engine,
      the identity and the state: INFO for a state in `SPEAKING_STATES` the engine does not
      voice, WARNING for a state outside the vocabulary.
    - No log line from an adapter holds segment text, a transcript or audio.
    """

    model_id: str
    engine_id: str
    sample_rate: int
    voice: VoiceInfo
    states: frozenset[str]

    def synthesize(self, segment: TextSegment, token: CancelToken) -> Iterator[SynthChunk]:
        """Synthesize one segment in its speaking state, checking the token before each chunk."""
        ...

    def release(self) -> None:
        """Free what the engine holds; a second call does nothing.

        This is the one hook a GPU release needs from an engine. The runtime calls it on the
        lane thread after the engine's jobs have stopped, and drops the engine after it returns.
        An in-process adapter drops its model references here, and the runtime then empties the
        CUDA cache; an adapter that proxies to a sidecar asks the sidecar to unload, so an
        engine in its own container frees its memory too. A `release` that raises is logged at
        ERROR with the engine id and no text, and the drop goes on. After `release`, any call
        but `release` raises `SpeechEngineError`.
        """
        ...


@dataclass(frozen=True, slots=True)
class NameHit:
    """A name spotter's score reached its threshold outside the cooldown of an earlier hit.

    `end_pos` is an uplink position, a count of 20 ms frames: the position of the frame whose
    push completed the scoring step that hit, so ingress can find where the name was said from
    it. `keyword` is the name the model scored and `score` the hit's score, on that model's own
    scale.
    """

    end_pos: int
    keyword: str
    score: float


class NameSpotterStream(Protocol):
    """One audio source's spotting state: the model's streaming state, its window, the cooldown.

    A hit is a scoring step at or above the threshold with no hit in the `window_ms` before it.
    """

    def push(self, pcm: bytes, pos: int) -> list[NameHit]:
        """Score one 20 ms frame of PCM16 mono 16 kHz audio at uplink position `pos`.

        Returns the hits of the steps this frame completed, usually none.
        """
        ...

    def reset(self, *, warm_up: bool = True) -> None:
        """Forget the context, so the next frame starts a new one.

        With `warm_up` the new context is settled on silence first. Without it, which a reset
        while the spotter is behind uses, there is no silence and no hit until `warmup_ms` of
        audio has been fed.
        """
        ...


@dataclass(frozen=True, slots=True)
class SpotterCalibration:
    """One spotter's score scale, mapped onto two scales every spotter shares.

    Scores from different models are not comparable, so arbitration between a server hit and an
    endpoint wake ranks them on these instead. Both tables are `(score, value)` points with
    ascending scores. `false_accepts` maps a score to the estimated false accepts per hour of
    the negative audio the table was measured on (the tuning media for the device's spotter, an
    ambient set for the server's) at that score, made monotone, so its rates do not increase.
    `positive_quantile` maps a score to the share, from 0 to 1, of the tuning positives that
    scored at or below it, so its shares do not decrease.
    """

    false_accepts: tuple[tuple[float, float], ...]
    positive_quantile: tuple[tuple[float, float], ...]


class NameSpotter(Protocol):
    """A model trained to spot MIST's names, shared by sessions; CPU-only.

    `window_ms` is the audio one name spans, and the least time between two hits of a stream.
    `hit_lag_ms` is the typical delay from the name's end to the hit, from the model's manifest,
    which is how `wake_screen_ms` estimates when the name ended. `calibration` holds each
    spotter's tables; a spotter with no entry is uncalibrated, and its events rank by margin
    below every calibrated one (`arbitration_key`).
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
        """The highest score over a finished buffer of PCM16 mono 16 kHz audio.

        It is scored from a fresh context, so it reaches a threshold when a fresh stream pushed
        the same buffer would report a hit.
        """
        ...


class WakeVerifier(Protocol):
    """A second-stage acoustic model that judges whether a name hit was said to MIST.

    The spotter only hears the name; the verifier scores the audio around the hit, from
    `before_ms` before it to `after_ms` after it, and a window shorter than `min_window_ms` is
    not scored. It was trained against one spotter, `spotter_model_id`, whose hits its threshold
    table was calibrated on, so a score means nothing for another spotter's hits. CPU-only.
    """

    model_id: str
    spotter_model_id: str
    before_ms: int
    after_ms: int
    min_window_ms: int

    def threshold_at(self, cutoff: float) -> float | None:
        """The score a hit at spotter cutoff `cutoff` must reach, or None below the table."""
        ...

    def score(self, pcm: bytes) -> float:
        """Score one window of PCM16 mono 16 kHz audio, as `NameSpotter.score_window` takes it.

        Raises:
            SpeechEngineError: The window cannot be scored, or the model's output is not one
                finite score.
        """
        ...

    def close(self) -> None:
        """Release the verifier's session, when its owner is done with it.

        The protocol does not require `score` to raise after `close`; an adapter that frees a
        model session raises `SpeechEngineError` from a later `score`.
        """
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
