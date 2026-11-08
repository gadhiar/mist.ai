"""Speech engine contracts: the protocols model adapters implement and the values they exchange.

The realtime layer never imports a model library. Ingress, the turn controller and the session
talk to these protocols, and the adapters under `backend/realtime/engines/` (Silero VAD, Smart
Turn, and the STT and TTS engines step 0 picks) implement them, so an engine can move between
the GPU and the CPU by configuration alone. Every engine call is blocking and runs on an
`InferenceLane` worker thread (`lanes.py`); a `CancelToken` is how a barge-in reaches a call
already running there.
"""

import threading
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np

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
        the STT lane, so that decode's result is discarded: the decode raises `RealtimeError`
        instead of returning a transcript of an utterance nobody is listening to. A push or a
        decode after `close` raises `RealtimeError` too. Each owner closes its stream exactly
        once.
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
