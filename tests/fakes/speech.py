"""Deterministic speech engines implementing the protocols in `backend/realtime/speech.py`.

Every realtime test runs against these instead of real models: `EnergyVad` scores loudness,
`ScriptedTranscriber` returns scripted text, `ScriptedTurnEnd` returns scripted probabilities,
`FakeSynthesizer` emits tones and `ScriptedNameSpotter` hears the names where a test scripts
them. Each records what it was given, so a test asserts on what reached the engine, and none
depends on time or randomness. The classes are plain and unslotted so a test can subclass or
wrap one to observe or delay a call. The synthesizer and the transcriber follow the engine
release rules (`speech.py`): they count `release()` calls, can be told to raise from one, and
refuse every other call after it with `SpeechEngineError`.
"""

import logging
import math
import time
import zlib
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from backend.realtime.errors import RealtimeError, SpeechEngineError
from backend.realtime.lanes import LaneJobCancelled
from backend.realtime.protocol import Spotter
from backend.realtime.speech import (
    NEUTRAL_STATE,
    SPEAKING_STATES,
    CancelToken,
    NameHit,
    SpotterCalibration,
    SynthChunk,
    TextSegment,
    TranscriptUpdate,
    VoiceInfo,
)

logger = logging.getLogger(__name__)

STT_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000

FAKE_VOICE = VoiceInfo(engine="fake", voice="fake-voice", profile="mist")


class EnergyVadStream:
    """Scores a window by its RMS: `rms / (rms + reference_rms)`, 0.5 at the reference level."""

    def __init__(self, reference_rms: float) -> None:
        self.reference_rms = reference_rms
        self.scores: list[float] = []
        self.resets = 0

    def score(self, window: np.ndarray) -> float:
        rms = float(np.sqrt(np.mean(np.square(window, dtype=np.float64))))
        result = rms / (rms + self.reference_rms)
        self.scores.append(result)
        return result

    def reset(self) -> None:
        self.resets += 1


class EnergyVad:
    """A VAD that calls loud windows speech; `scripted_audio` bursts score near 0.9."""

    def __init__(self, reference_rms: float = 0.01) -> None:
        self.reference_rms = reference_rms
        self.streams: list[EnergyVadStream] = []

    def open_stream(self) -> EnergyVadStream:
        stream = EnergyVadStream(self.reference_rms)
        self.streams.append(stream)
        return stream


@dataclass(frozen=True, slots=True)
class ScriptedUtterance:
    """What the transcriber hears for one utterance, with Whisper's non-speech scores."""

    text: str
    no_speech_prob: float | None = None
    avg_logprob: float | None = None


def _common_word_prefix_chars(previous: str, current: str) -> int:
    shared = 0
    for before, now in zip(previous.split(), current.split()):
        if before != now:
            break
        shared += 1
    if shared == 0:
        return 0
    return len(" ".join(current.split()[:shared]))


class ScriptedTranscriptStream:
    """Reveals the utterance's words as audio arrives: one word per `ms_per_word` pushed.

    `stable_chars` follows the Whisper adapter's rule, the longest common word prefix of the
    last two partials; the final is the whole text, all of it stable. `on_decode` runs inside
    each decode, before its result is recorded, so a test can hold a decode in flight or close
    the stream under it.
    """

    def __init__(
        self,
        utterance: ScriptedUtterance,
        *,
        ms_per_word: int,
        on_decode: Callable[[], object] | None = None,
    ) -> None:
        self.utterance = utterance
        self.ms_per_word = ms_per_word
        self.on_decode = on_decode
        self.pushed: list[int] = []
        self.partials: list[TranscriptUpdate] = []
        self.finals: list[TranscriptUpdate] = []
        self.closed = False

    @property
    def audio_ms(self) -> int:
        return sum(self.pushed) * 1000 // STT_SAMPLE_RATE

    def push(self, pcm: np.ndarray) -> None:
        self._check_open("push")
        self.pushed.append(len(pcm))

    def close(self) -> None:
        self.closed = True

    def _check_open(self, action: str) -> None:
        if self.closed:
            raise RealtimeError(f"transcript stream is closed: cannot {action}")

    def _check_decode_open(self, action: str) -> None:
        # A decode runs as a lane job, so a close cuts it as another holder's cancel would.
        if self.closed:
            raise LaneJobCancelled(f"transcript stream is closed: cannot {action}")

    def _decoding(self) -> None:
        """Stand in for the decode's model time; a close landing here discards the result."""
        self._check_decode_open("decode")
        if self.on_decode is not None:
            self.on_decode()

    def decode_partial(self, token: CancelToken) -> TranscriptUpdate:
        self._decoding()
        self._check_decode_open("return a partial")
        words = self.utterance.text.split()
        text = " ".join(words[: self.audio_ms // self.ms_per_word])
        previous = self.partials[-1].text if self.partials else ""
        update = TranscriptUpdate(
            text,
            _common_word_prefix_chars(previous, text),
            False,
            self.audio_ms,
            self.utterance.no_speech_prob,
            self.utterance.avg_logprob,
        )
        self.partials.append(update)
        return update

    def decode_final(self, token: CancelToken) -> TranscriptUpdate:
        self._decoding()
        self._check_decode_open("return a final")
        text = self.utterance.text
        update = TranscriptUpdate(
            text,
            len(text),
            True,
            self.audio_ms,
            self.utterance.no_speech_prob,
            self.utterance.avg_logprob,
        )
        self.finals.append(update)
        return update


class ScriptedTranscriber:
    """Each `open_stream` takes the next scripted utterance; past the script, empty text."""

    def __init__(
        self,
        script: Sequence[ScriptedUtterance | str],
        *,
        model_id: str = "scripted-stt",
        ms_per_word: int = 200,
        on_decode: Callable[[], object] | None = None,
        release_error: BaseException | None = None,
    ) -> None:
        self.model_id = model_id
        self.ms_per_word = ms_per_word
        self.on_decode = on_decode
        self.release_error = release_error
        self.script = [
            item if isinstance(item, ScriptedUtterance) else ScriptedUtterance(item)
            for item in script
        ]
        self.streams: list[ScriptedTranscriptStream] = []
        self.releases = 0

    def release(self) -> None:
        """Count the call; the transcriber is released even when told to raise."""
        self.releases += 1
        if self.release_error is not None:
            raise self.release_error

    def open_stream(self) -> ScriptedTranscriptStream:
        if self.releases:
            raise SpeechEngineError(f"transcriber {self.model_id!r} is released")
        index = len(self.streams)
        utterance = self.script[index] if index < len(self.script) else ScriptedUtterance("")
        stream = ScriptedTranscriptStream(
            utterance, ms_per_word=self.ms_per_word, on_decode=self.on_decode
        )
        self.streams.append(stream)
        return stream


class ScriptedTurnEnd:
    """Returns scripted end-of-turn probabilities in order, repeating the last one after."""

    def __init__(self, probabilities: Sequence[float]) -> None:
        if not probabilities:
            raise ValueError("ScriptedTurnEnd needs at least one probability")
        self.probabilities = list(probabilities)
        self.calls: list[int] = []

    def probability_complete(self, audio: np.ndarray) -> float:
        index = min(len(self.calls), len(self.probabilities) - 1)
        self.calls.append(len(audio))
        return self.probabilities[index]


class FakeSynthesizer:
    """Synthesizes a segment as a tone: `ms_per_char` of audio per character.

    The tone's pitch comes from a CRC of the text, so two segments sound different and the same
    segment is identical on every run. Audio is split into chunks of at most `chunk_ms`; before
    each chunk the token is checked, `sleep(chunk_delay_s)` stands in for inference time, and the
    token is checked again, so a cancel lands between chunks as it does with a real engine.

    It voices by the speaking-state rules every adapter follows (`SpeechSynthesizer`): a state
    in `states` is carried on every chunk; any other is voiced `neutral`, with one log record
    per state, INFO inside the vocabulary and WARNING outside it. The tone does not change with
    the state, since only the chunk's `state` is under test. A call is recorded when it is made,
    not when its chunks are first read.
    """

    def __init__(
        self,
        *,
        model_id: str = "fake-tts",
        engine_id: str = "fake",
        voice: VoiceInfo = FAKE_VOICE,
        states: frozenset[str] = frozenset({NEUTRAL_STATE}),
        sample_rate: int = TTS_SAMPLE_RATE,
        ms_per_char: int = 50,
        chunk_ms: int = 200,
        chunk_delay_s: float = 0.0,
        sleep: Callable[[float], object] = time.sleep,
        release_error: BaseException | None = None,
    ) -> None:
        if NEUTRAL_STATE not in states or not states <= SPEAKING_STATES:
            raise ValueError("FakeSynthesizer: states must hold neutral and only known states")
        self.model_id = model_id
        self.engine_id = engine_id
        self.voice = voice
        self.states = states
        self.sample_rate = sample_rate
        self.ms_per_char = ms_per_char
        self.chunk_ms = chunk_ms
        self.chunk_delay_s = chunk_delay_s
        self.sleep = sleep
        self.release_error = release_error
        self.segments: list[TextSegment] = []
        self.releases = 0
        self._fallbacks_logged: set[str] = set()

    def release(self) -> None:
        """Count the call; the synthesizer is released even when told to raise."""
        self.releases += 1
        if self.release_error is not None:
            raise self.release_error

    def synthesize(self, segment: TextSegment, token: CancelToken) -> Iterator[SynthChunk]:
        if self.releases:
            raise SpeechEngineError(f"synthesizer {self.engine_id!r} is released")
        self.segments.append(segment)
        return self._chunks(segment, self._voiced_state(segment.state), token)

    def _voiced_state(self, requested: str) -> str:
        if requested in self.states:
            return requested
        if requested not in self._fallbacks_logged:
            self._fallbacks_logged.add(requested)
            known = requested in SPEAKING_STATES
            logger.log(
                logging.INFO if known else logging.WARNING,
                "TTS engine %r (voice %r) does not voice %s state %r; voicing %r",
                self.engine_id,
                self.voice.voice,
                "the" if known else "the unknown",
                requested,
                NEUTRAL_STATE,
            )
        return NEUTRAL_STATE

    def _chunks(self, segment: TextSegment, state: str, token: CancelToken) -> Iterator[SynthChunk]:
        total = len(segment.text) * self.ms_per_char * self.sample_rate // 1000
        per_chunk = self.chunk_ms * self.sample_rate // 1000
        frequency = 200 + zlib.crc32(segment.text.encode("utf-8")) % 400
        for start in range(0, total, per_chunk):
            if token.is_set():
                return
            if self.chunk_delay_s:
                self.sleep(self.chunk_delay_s)
            if token.is_set():
                return
            count = min(per_chunk, total - start)
            t = np.arange(start, start + count) / self.sample_rate
            tone = 0.3 * np.sin(2 * math.pi * frequency * t)
            pcm16 = (tone * 32767).astype("<i2").tobytes()
            yield SynthChunk(pcm16=pcm16, samples=count, segment=segment, state=state)


class ScriptedNameSpotterStream:
    """Returns each scripted hit at its `end_pos`, when its score reaches the stream's threshold.

    The partial context is the positions pushed since the last `reset`. A hit is reported once
    per context, as a spotter's cooldown reports a name once, so a frame pushed again in the
    same context gives no second hit, and after a `reset` it does. `warm` records its position
    in `warmed` and reports nothing; `reset` records its `warm_up` in `warm_ups`.
    """

    def __init__(self, hits: Sequence[NameHit], threshold: float) -> None:
        self.hits = hits
        self.threshold = threshold
        self.pushed: list[int] = []
        self.warmed: list[int] = []
        self.context: list[int] = []
        self.resets = 0
        self.warm_ups: list[bool] = []
        self._reported: set[int] = set()

    def push(self, pcm: bytes, pos: int) -> list[NameHit]:
        self.pushed.append(pos)
        self.context.append(pos)
        found: list[NameHit] = []
        for index, hit in enumerate(self.hits):
            if hit.end_pos == pos and hit.score >= self.threshold and index not in self._reported:
                self._reported.add(index)
                found.append(hit)
        return found

    def warm(self, pcm: bytes, pos: int) -> None:
        self.warmed.append(pos)

    def reset(self, *, warm_up: bool = True) -> None:
        self.context = []
        self._reported = set()
        self.resets += 1
        self.warm_ups.append(warm_up)


class ScriptedNameSpotter:
    """A name spotter that hears the names where the test says, and nowhere else.

    `hits` script what streams report, by uplink position. `markers` script `score_window`: a
    buffer holding a marker byte pattern scores that marker's score, the highest when it holds
    several, and 0.0 when it holds none, which is how a test scripts media that says the name.
    """

    def __init__(
        self,
        hits: Sequence[NameHit] = (),
        *,
        markers: Mapping[bytes, float] | None = None,
        model_id: str = "scripted-kws",
        names: tuple[str, ...] = ("mist",),
        window_ms: int = 1280,
        hit_lag_ms: int = 0,
        calibration: Mapping[Spotter, SpotterCalibration] | None = None,
    ) -> None:
        scripted = dict(markers or {})
        if b"" in scripted:
            raise ValueError("ScriptedNameSpotter: an empty marker would match every buffer")
        self.hits = list(hits)
        self.markers = scripted
        self.model_id = model_id
        self.names = names
        self.window_ms = window_ms
        self.hit_lag_ms = hit_lag_ms
        self.calibration: Mapping[Spotter, SpotterCalibration] = dict(calibration or {})
        self.streams: list[ScriptedNameSpotterStream] = []
        self.scored: list[int] = []

    def open_stream(self, threshold: float) -> ScriptedNameSpotterStream:
        stream = ScriptedNameSpotterStream(self.hits, threshold)
        self.streams.append(stream)
        return stream

    def score_window(self, pcm: bytes) -> float:
        self.scored.append(len(pcm))
        return max((score for marker, score in self.markers.items() if marker in pcm), default=0.0)
