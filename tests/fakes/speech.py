"""Deterministic speech engines implementing the protocols in `backend/realtime/speech.py`.

Every realtime test runs against these instead of real models: `EnergyVad` scores loudness,
`ScriptedTranscriber` returns scripted text, `ScriptedTurnEnd` returns scripted probabilities and
`FakeSynthesizer` emits tones. Each records what it was given, so a test asserts on what reached
the engine, and none depends on time or randomness. The classes are plain and unslotted so a
test can subclass or wrap one to observe or delay a call.
"""

import math
import time
import zlib
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np

from backend.realtime.speech import (
    CancelToken,
    SynthChunk,
    TextSegment,
    TranscriptUpdate,
    VoiceInfo,
)

STT_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000

BUILTIN_VOICE = VoiceInfo(profile=None, source="builtin", seconds=None, sha256_prefix=None)


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
    last two partials; the final is the whole text, all of it stable.
    """

    def __init__(self, utterance: ScriptedUtterance, *, ms_per_word: int) -> None:
        self.utterance = utterance
        self.ms_per_word = ms_per_word
        self.pushed: list[int] = []
        self.partials: list[TranscriptUpdate] = []
        self.finals: list[TranscriptUpdate] = []

    @property
    def audio_ms(self) -> int:
        return sum(self.pushed) * 1000 // STT_SAMPLE_RATE

    def push(self, pcm: np.ndarray) -> None:
        self.pushed.append(len(pcm))

    def decode_partial(self, token: CancelToken) -> TranscriptUpdate:
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
    ) -> None:
        self.model_id = model_id
        self.ms_per_word = ms_per_word
        self.script = [
            item if isinstance(item, ScriptedUtterance) else ScriptedUtterance(item)
            for item in script
        ]
        self.streams: list[ScriptedTranscriptStream] = []

    def open_stream(self) -> ScriptedTranscriptStream:
        index = len(self.streams)
        utterance = self.script[index] if index < len(self.script) else ScriptedUtterance("")
        stream = ScriptedTranscriptStream(utterance, ms_per_word=self.ms_per_word)
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
    """

    def __init__(
        self,
        *,
        model_id: str = "fake-tts",
        variant: str = "turbo",
        voice: VoiceInfo = BUILTIN_VOICE,
        sample_rate: int = TTS_SAMPLE_RATE,
        ms_per_char: int = 50,
        chunk_ms: int = 200,
        chunk_delay_s: float = 0.0,
        sleep: Callable[[float], object] = time.sleep,
    ) -> None:
        self.model_id = model_id
        self.variant = variant
        self.voice = voice
        self.sample_rate = sample_rate
        self.ms_per_char = ms_per_char
        self.chunk_ms = chunk_ms
        self.chunk_delay_s = chunk_delay_s
        self.sleep = sleep
        self.segments: list[TextSegment] = []

    def synthesize(self, segment: TextSegment, token: CancelToken) -> Iterator[SynthChunk]:
        self.segments.append(segment)
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
            yield SynthChunk(pcm16=pcm16, samples=count, segment=segment)
