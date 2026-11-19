"""NVIDIA's Nemotron streaming English model behind `StreamingTranscriber`, on the GPU.

Engine id `nemotron-160`, the arm of the same name in `scripts/model_bench/speech/speech_arms.json`
(the parity test pins both to it): `nvidia/nemotron-speech-streaming-en-0.6b` through NeMo 2.7.3,
on `cuda` only, the one device the arm records, fed 160 ms chunks.

Weights. The `.nemo` file is read from `<models_dir>/speech/<org>/<repo>/`, where the bench probe
reads it, and restored from that local path with the probe's call, `ASRModel.restore_from(<path>,
map_location=<device>)`; the adapter makes no hub call. The arm records no revision, so the file is
pinned by the size and sha256 host step H13 recorded; another file raises naming its path. The
2.4 GB hash runs once per file per process, so a reload after a GPU release does not repeat it.

Streaming. The model is set up as the probe sets it up (`set_default_att_context_size([70, 1])`,
then `setup_streaming_params`), and each stream steps it as the probe's `NemoStreamingTranscriber`
does: features recomputed over the audio fed so far, an encoder step for each chunk of feature
frames lying wholly inside it, and the encoder caches and hypotheses carried between steps.
`push` only buffers, since it runs on the caller's thread; every model call runs in a decode, on
the STT lane. A decode feeds the buffered audio in whole 160 ms chunks, running the steps each
chunk completes, so a partial costs only the new audio's steps and returns the hypothesis so far.
`decode_final` flushes a copy of the stream's state (the last short chunk, then the remaining
frames with `keep_all_outputs`), because ingress may resume an utterance after a final, and the
live stream must then go on from where its whole chunks left off.

Word timings. Each step covers known feature frames, so the stream stamps the words a step's text
adds or changes with that step's span: a word starts with the first step that emitted it and ends
with the last step that changed it. The span comes from the frames at the preprocessor config's
`window_stride`, and no NeMo hypothesis timestamp is read. A config without a positive stride is
logged once at load, and its transcripts carry no words. The bookkeeping lives in the stream's
state, so the final's flush stamps its copy and the live stream keeps its own.

This module also holds what both speech-to-text adapters share (the stream's buffering, closing
and partial history, release, and the library error rule); `parakeet_stt.py` imports it.

NeMo and torch are imported only by `import_nemo`, which `load` is handed, so importing this
module loads neither and tests pass a fake library.
"""

import copy
import hashlib
import importlib
import logging
import re
import stat
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Protocol, TypeVar

import numpy as np

from ..budget import PlacementConfig
from ..errors import RealtimeError, SpeechEngineError
from ..lanes import LaneJobCancelled
from ..speech import CancelToken, TranscriptUpdate, WordTiming

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# Shared by the speech-to-text adapters
# ---------------------------------------------------------------------------

# What the engine libraries and torch raise for a bad input, file or state. Converted to
# `SpeechEngineError`, so no library error crosses the lane (`lanes.py`) and none is bare
# `Exception`. AssertionError is here because the libraries check shapes with `assert`.
LIBRARY_ERRORS: tuple[type[BaseException], ...] = (
    RuntimeError,
    ValueError,
    TypeError,
    KeyError,
    IndexError,
    OSError,
    AssertionError,
)

_WORD = re.compile(r"\S+")


def stable_prefix_chars(previous: str, current: str) -> int:
    """The characters of `current` its words share with `previous`, from the start.

    The longest common word prefix of the last two partials: the words a streaming or
    re-decoding engine has given twice in a row are the ones not expected to change.
    """
    before = previous.split()
    end = 0
    for index, word in enumerate(_WORD.finditer(current)):
        if index >= len(before) or word.group() != before[index]:
            break
        end = word.end()
    return end


def audio_ms(samples: int) -> int:
    """Milliseconds of 16 kHz audio."""
    return samples * 1000 // SAMPLE_RATE


def checked_words(text: str, words: tuple[WordTiming, ...] | None) -> tuple[WordTiming, ...] | None:
    """`words` when they meet `TranscriptUpdate`'s words invariant for `text`, else None.

    A decode whose timings cannot be tied to its text still returns its text; only its words
    are dropped.
    """
    if words is None or not TranscriptUpdate.words_match(text, words):
        return None
    return words


def library_error(engine_id: str, action: str, exc: BaseException) -> SpeechEngineError:
    """A `SpeechEngineError` naming the action and the library error's type, never its text.

    A library message can echo its input, which for a decode is the utterance.
    """
    return SpeechEngineError(f"{engine_id}: {action} failed ({type(exc).__name__})")


def log_scores_absent(log: logging.Logger, engine_id: str) -> None:
    """The one INFO line at load for an engine without Whisper's non-speech scores."""
    log.info(
        "%s reports no no_speech_prob or avg_logprob, so the ingress non-speech filter is off "
        "for it",
        engine_id,
    )


class BufferedTranscriptStream:
    """What both adapters' streams share: buffered audio, closing, and the partial history.

    `push` runs on the caller's thread (ingress pushes from the event loop) and only appends a
    copy of the frame; the model runs in the decodes, one at a time on the STT lane. A lock
    guards the buffered audio and the closed state, since `push` and `close` can land while a
    decode runs. A decode takes the audio pushed since the last one when it starts, and checks
    again when the model returns: a stream closed in between discards the result with
    `LaneJobCancelled`, as the `TranscriptStream` contract says.
    """

    __slots__ = ("_closed", "_engine_id", "_last_partial", "_lock", "_on_close", "_pending")

    def __init__(
        self, engine_id: str, on_close: Callable[["BufferedTranscriptStream"], None]
    ) -> None:
        self._engine_id = engine_id
        self._on_close = on_close
        self._lock = threading.Lock()
        self._pending: list[np.ndarray] = []
        self._closed: str | None = None
        self._last_partial = ""

    def push(self, pcm: np.ndarray) -> None:
        """Append 16 kHz float32 audio to the utterance.

        Raises:
            SpeechEngineError: The engine was released (a `RealtimeError`).
            RealtimeError: The stream was closed.
        """
        frame = np.array(pcm, dtype=np.float32, copy=True).reshape(-1)
        with self._lock:
            if self._closed == "released":
                raise SpeechEngineError(f"{self._engine_id} was released")
            if self._closed is not None:
                raise RealtimeError(f"{self._engine_id} transcript stream is closed")
            self._pending.append(frame)

    def close(self) -> None:
        """End the stream and drop its audio and state; a second call does nothing."""
        if self._shut("closed"):
            self._on_close(self)

    def close_for_release(self) -> None:
        """Close the stream because its engine was released, so a later push says so."""
        self._shut("released")

    def _shut(self, reason: str) -> bool:
        with self._lock:
            if self._closed is not None:
                return False
            self._closed = reason
            self._pending = []
        self._drop_state()
        return True

    def _drop_state(self) -> None:
        """Drop what the subclass holds for the utterance; runs once, on close."""

    def _start(self, token: CancelToken, action: str) -> list[np.ndarray]:
        """Take the audio pushed since the last decode, unless the decode is not wanted.

        Raises:
            LaneJobCancelled: The stream is closed, or the token is set.
        """
        with self._lock:
            self._check_open(action)
            if token.is_set():
                raise LaneJobCancelled(
                    f"{self._engine_id}: the token was set before the {action} started"
                )
            pending, self._pending = self._pending, []
        return pending

    def _check_open(self, action: str) -> None:
        if self._closed is not None:
            raise LaneJobCancelled(
                f"{self._engine_id} transcript stream is {self._closed}: {action} discarded"
            )

    def _still_open(self, action: str) -> None:
        """Raise `LaneJobCancelled` if the stream was closed while the model ran."""
        with self._lock:
            self._check_open(action)

    def _partial(
        self, text: str, samples: int, words: tuple[WordTiming, ...] | None
    ) -> TranscriptUpdate:
        stable = stable_prefix_chars(self._last_partial, text)
        self._last_partial = text
        return TranscriptUpdate(
            text, stable, False, audio_ms(samples), words=checked_words(text, words)
        )

    @staticmethod
    def _final(text: str, samples: int, words: tuple[WordTiming, ...] | None) -> TranscriptUpdate:
        return TranscriptUpdate(
            text, len(text), True, audio_ms(samples), words=checked_words(text, words)
        )


S = TypeVar("S", bound=BufferedTranscriptStream)


class TranscriberBase:
    """Open streams and release, as both adapters do them.

    `release` closes every open stream (a later push raises `SpeechEngineError`, a later
    decode `LaneJobCancelled`), drops the model, and makes `open_stream` raise.
    """

    engine_id: str
    model_id: str
    _log: logging.Logger

    __slots__ = ("_released", "_streams", "_streams_lock")

    def __init__(self) -> None:
        self._streams_lock = threading.Lock()
        self._streams: set[BufferedTranscriptStream] = set()
        self._released = False

    def _register(self, stream: S) -> S:
        """Track a new stream so `release` can close it.

        Raises:
            SpeechEngineError: The engine was released.
        """
        with self._streams_lock:
            if self._released:
                raise SpeechEngineError(f"{self.engine_id} was released")
            self._streams.add(stream)
        return stream

    def release(self) -> None:
        """Close the open streams and drop the model; a second call does nothing."""
        with self._streams_lock:
            if self._released:
                return
            self._released = True
            streams = list(self._streams)
            self._streams.clear()
        for stream in streams:
            stream.close_for_release()
        self._drop_model()
        self._log.info("%s released", self.engine_id)

    def _forget(self, stream: BufferedTranscriptStream) -> None:
        with self._streams_lock:
            self._streams.discard(stream)

    def _drop_model(self) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Nemotron
# ---------------------------------------------------------------------------

ENGINE_ID = "nemotron-160"
REPOSITORY = "nvidia/nemotron-speech-streaming-en-0.6b"
MODEL_FILE = "nemotron-speech-streaming-en-0.6b.nemo"
# Relative to `<models_dir>/speech/`, the directory the bench mounts as its weights directory.
WEIGHT_PATH = f"{REPOSITORY}/{MODEL_FILE}"
# The `.nemo` file host step H13 recorded, since neither the arm nor step 0 records a revision.
MODEL_BYTES = 2473041920
MODEL_SHA256 = "283638054c44f6794e74fe9af9048d78a6d9d6c058c12131856c7859a62ac9cd"
MODEL_ID = f"{REPOSITORY}@sha256:{MODEL_SHA256}"
DEVICE = "cuda"
CHUNK_MS = 160
CHUNK_SAMPLES = SAMPLE_RATE * CHUNK_MS // 1000
# The attention context for 160 ms chunks, [left, right] in 80 ms encoder frames, as the probe's
# `NEMOTRON_ATT_CONTEXT`.
ATT_CONTEXT = (70, 1)
# Feature frames short of the audio's end where a step may not start yet, because the STFT window
# there is still open; the probe's `guard_frames`.
GUARD_FRAMES = 2

# Loading also meets the library's attribute errors: a NeMo release without one of the streaming
# calls, or an OmegaConf config missing a key.
_LOAD_ERRORS = (*LIBRARY_ERRORS, AttributeError)


class NemoModel(Protocol):
    """The calls the adapter makes on a restored NeMo `ASRModel` (NeMo 2.7.3).

    Also `encoder.set_default_att_context_size`, `encoder.setup_streaming_params`,
    `encoder.get_initial_cache_state`, `encoder.streaming_cfg`, `_cfg.preprocessor`,
    `from_config_dict` and `parameters`, as the probe calls them, and the preprocessor config's
    `window_stride` for word timings.
    """

    def eval(self) -> object:
        """Put the model in inference mode."""
        ...

    def conformer_stream_step(self, **kwargs: object) -> tuple:
        """One cache-aware encoder and decoder step."""
        ...


@dataclass(frozen=True, slots=True)
class NemoLibrary:
    """What the adapter needs from NeMo and torch, bound once by `import_nemo`.

    `restore_from` is `ASRModel.restore_from(restore_path, map_location=...)`. `torch` is the
    module, for the tensor slicing between steps.
    """

    restore_from: Callable[..., NemoModel]
    torch: ModuleType


def import_nemo() -> NemoLibrary:
    """Import NeMo's ASR models and torch and bind what the adapter calls.

    Raises:
        SpeechEngineError: nemo_toolkit or torch is not installed.
    """
    try:
        torch = importlib.import_module("torch")
        models = importlib.import_module("nemo.collections.asr.models")
    except ImportError as exc:
        raise SpeechEngineError(
            f"nemo_toolkit is not installed ({exc.name}); requirements-voice.txt pins it"
        ) from exc
    return NemoLibrary(restore_from=models.ASRModel.restore_from, torch=torch)


@dataclass(frozen=True, slots=True)
class FilePin:
    """A file's expected size in bytes and sha256, hex."""

    size: int
    sha256: str


MODEL_PIN = FilePin(size=MODEL_BYTES, sha256=MODEL_SHA256)

# Files that matched a pin in this process: path, size, times, inode and the pinned digest.
_verified: set[tuple[str, int, int, int, int, str]] = set()
_verified_lock = threading.Lock()


def _sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verify_file(path: Path, pin: FilePin) -> None:
    """Check `path` against `pin`: the size first, then the sha256, once per file per process.

    Raises:
        SpeechEngineError: The file is missing, unreadable, or of another size or sha256; the
            message names the path.
    """
    try:
        info = path.stat()
    except OSError as exc:
        raise SpeechEngineError(f"{ENGINE_ID}: model file {path} is missing or unreadable") from exc
    if not stat.S_ISREG(info.st_mode):
        raise SpeechEngineError(f"{ENGINE_ID}: model file {path} is not a file")
    if info.st_size != pin.size:
        raise SpeechEngineError(
            f"{ENGINE_ID}: model file {path} is {info.st_size} bytes, not the pinned {pin.size}"
        )
    key = (str(path), info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_ino, pin.sha256)
    with _verified_lock:
        if key in _verified:
            return
    try:
        digest = _sha256(path)
    except OSError as exc:
        raise SpeechEngineError(f"{ENGINE_ID}: cannot read model file {path}") from exc
    if digest != pin.sha256:
        raise SpeechEngineError(
            f"{ENGINE_ID}: model file {path} has sha256 {digest}, not the pinned {pin.sha256}"
        )
    with _verified_lock:
        _verified.add(key)


def _pick(value: object, first: bool) -> int:
    """A streaming setting that may differ for the first step: `[first, rest]` or one value."""
    if isinstance(value, list | tuple):
        return int(value[0] if first else value[1])
    return int(value)


def _hypothesis_text(reply: object) -> str:
    """The text of NeMo's first hypothesis, whichever shape the step returned (the probe's)."""
    if isinstance(reply, tuple):
        reply = reply[0]
    first = reply[0] if isinstance(reply, list | tuple) else reply
    if isinstance(first, list | tuple):
        first = first[0]
    text = getattr(first, "text", first)
    return str(text).strip()


@dataclass(slots=True)
class _StreamState:
    """One utterance's streaming state: the audio fed, the step position and what it carries.

    `words` is the per-step word bookkeeping: each word of `text` with the span of the steps
    that made it. `stamp` keeps it after every step that returns text.
    """

    audio: np.ndarray
    cache_channel: object
    cache_time: object
    cache_channel_len: object
    frame_idx: int = 0
    step: int = 0
    text: str = ""
    hypotheses: object = None
    pred_out: object = None
    words: list[WordTiming] = field(default_factory=list)

    def stamp(self, start_ms: int, end_ms: int) -> None:
        """Stamp the words this step's text added or changed with the step's audio span.

        Words are matched by position, since a greedy streaming hypothesis only grows at its
        end. A word the step left as it was keeps its span. A changed word keeps the start of
        the first step that emitted it and ends with this step; a new word spans this step.
        Words the text no longer holds are dropped.
        """
        stamped = []
        for index, word in enumerate(self.text.split()):
            if index >= len(self.words):
                stamped.append(WordTiming(word, start_ms, end_ms))
            elif self.words[index].text == word:
                stamped.append(self.words[index])
            else:
                stamped.append(WordTiming(word, self.words[index].start_ms, end_ms))
        self.words = stamped


class _Streamer:
    """The restored model and its streaming preprocessor, stepped as the probe steps them.

    `frame_ms` is the preprocessor's feature frame stride in milliseconds, which places each
    step's frames in the audio; None when the config gives none, and then no step is stamped.
    """

    __slots__ = ("_device", "_model", "_preprocessor", "_torch", "frame_ms")

    def __init__(
        self,
        model: NemoModel,
        preprocessor: object,
        torch: ModuleType,
        frame_ms: float | None = None,
    ) -> None:
        self._model = model
        self._preprocessor = preprocessor
        self._torch = torch
        self._device = next(model.parameters()).device
        self.frame_ms = frame_ms

    def words(self, state: _StreamState | None) -> tuple[WordTiming, ...] | None:
        """The state's timed words, or None when the stream keeps no timings."""
        if self.frame_ms is None:
            return None
        return () if state is None else tuple(state.words)

    def new_state(self) -> _StreamState:
        channel, time, channel_len = self._model.encoder.get_initial_cache_state(batch_size=1)
        return _StreamState(np.zeros(0, dtype=np.float32), channel, time, channel_len)

    def feed(self, state: _StreamState, chunk: np.ndarray) -> None:
        """Append `chunk` and run the encoder steps it completes (the probe's `feed`)."""
        state.audio = np.concatenate([state.audio, chunk])
        self.run_steps(state, final=False)

    def flush(self, state: _StreamState, tail: np.ndarray) -> str:
        """Feed the last short chunk and run the remaining frames (the probe's `finish`)."""
        if tail.size:
            self.feed(state, tail)
        if state.audio.size:
            self.run_steps(state, final=True)
        return state.text

    def _features(self, audio: np.ndarray) -> object:
        torch = self._torch
        signal = torch.from_numpy(audio).unsqueeze(0).to(self._device)
        length = torch.tensor([signal.shape[1]], device=self._device)
        features, _ = self._preprocessor(input_signal=signal, length=length)
        return features

    def run_steps(self, state: _StreamState, *, final: bool) -> None:
        """Run every step whose frames are complete; with `final`, every remaining frame.

        The probe's `_run_steps` on `state`. Each field changes only after the model's step
        returns, so a step that raises leaves the state where it was and the next decode runs
        that step again.
        """
        torch = self._torch
        cfg = self._model.encoder.streaming_cfg
        with torch.inference_mode():
            features = self._features(state.audio)
            total = features.shape[-1]
            usable = total if final else total - GUARD_FRAMES
            while state.frame_idx < total:
                first = state.step == 0
                chunk = _pick(cfg.chunk_size, first)
                shift = _pick(cfg.shift_size, first)
                end = state.frame_idx + chunk
                last = final and end >= total
                if not final and end > usable:
                    return
                cache_size = _pick(cfg.pre_encode_cache_size, first)
                start = max(0, state.frame_idx - cache_size)
                pre = features[:, :, start : state.frame_idx]
                pad = cache_size - pre.shape[-1]
                body = features[:, :, state.frame_idx : end]
                parts = [pre, body]
                if pad > 0:
                    parts.insert(0, torch.zeros_like(features[:, :, :pad]))
                signal = torch.cat(parts, dim=-1)
                length = torch.tensor([signal.shape[-1]], device=self._device)
                (
                    state.pred_out,
                    texts,
                    state.cache_channel,
                    state.cache_time,
                    state.cache_channel_len,
                    state.hypotheses,
                ) = self._model.conformer_stream_step(
                    processed_signal=signal,
                    processed_signal_length=length,
                    cache_last_channel=state.cache_channel,
                    cache_last_time=state.cache_time,
                    cache_last_channel_len=state.cache_channel_len,
                    keep_all_outputs=last,
                    previous_hypotheses=state.hypotheses,
                    previous_pred_out=state.pred_out,
                    drop_extra_pre_encoded=0 if first else cfg.drop_extra_pre_encoded,
                    return_transcription=True,
                )
                if texts:
                    state.text = _hypothesis_text(list(texts))
                    if self.frame_ms is not None:
                        # The step's new frames, ending no later than the audio fed.
                        start_ms = round(state.frame_idx * self.frame_ms)
                        end_ms = min(
                            round(min(end, total) * self.frame_ms), audio_ms(state.audio.size)
                        )
                        state.stamp(start_ms, max(start_ms, end_ms))
                state.frame_idx += shift
                state.step += 1
                if last:
                    return


def _streaming_preprocessor(model: NemoModel) -> object:
    """The model's preprocessor without dither or padding, as the probe builds it."""
    cfg = copy.deepcopy(model._cfg.preprocessor)
    cfg.dither = 0.0
    cfg.pad_to = 0
    return model.from_config_dict(cfg).to(next(model.parameters()).device)


def _frame_ms(model: NemoModel) -> float | None:
    """The preprocessor's feature frame stride in milliseconds, or None if it gives none.

    NeMo's mel preprocessor config holds the stride as `window_stride`, in seconds. Whether this
    checkpoint's does is unverified here, so anything but a positive number reads as absent.
    """
    stride = getattr(model._cfg.preprocessor, "window_stride", None)
    if isinstance(stride, bool) or not isinstance(stride, int | float) or stride <= 0:
        return None
    return float(stride) * 1000


class NemotronStream(BufferedTranscriptStream):
    """One utterance on Nemotron: audio fed in 160 ms chunks, caches carried between decodes."""

    __slots__ = ("_owner", "_state", "_unfed")

    def __init__(self, owner: "NemotronStreamingTranscriber") -> None:
        super().__init__(ENGINE_ID, owner._forget)
        self._owner = owner
        self._unfed = np.zeros(0, dtype=np.float32)
        self._state: _StreamState | None = None

    def decode_partial(self, token: CancelToken) -> TranscriptUpdate:
        """Feed the whole chunks pushed since the last decode; return the hypothesis so far.

        Raises:
            LaneJobCancelled: The token was set, or the stream was closed.
            SpeechEngineError: The engine failed; the message holds no text.
        """
        action = "partial decode"
        streamer = self._prepare(token, action)
        try:
            state = self._feed_chunks(streamer, token, action)
        except LIBRARY_ERRORS as exc:
            # `from None`: the library's message may echo the utterance.
            raise library_error(ENGINE_ID, action, exc) from None
        self._still_open(action)
        words = streamer.words(state)
        if state is None:
            return self._partial("", 0, words)
        return self._partial(state.text, state.audio.size, words)

    def decode_final(self, token: CancelToken) -> TranscriptUpdate:
        """Flush a copy of the stream: the final transcript, with the live stream left as it was.

        Raises:
            LaneJobCancelled: The token was set, or the stream was closed.
            SpeechEngineError: The engine failed; the message holds no text.
        """
        action = "final decode"
        streamer = self._prepare(token, action)
        try:
            state = self._feed_chunks(streamer, token, action)
            tail = self._unfed
            if state is None and not tail.size:
                text, samples, words = "", 0, streamer.words(None)
            else:
                live = state if state is not None else self._new_state(streamer)
                # The copy carries the live stream's word bookkeeping, and the flush's steps
                # stamp the copy alone.
                flushed = copy.deepcopy(live)
                samples = flushed.audio.size + tail.size
                text = streamer.flush(flushed, tail)
                words = streamer.words(flushed)
        except LIBRARY_ERRORS as exc:
            raise library_error(ENGINE_ID, action, exc) from None
        self._still_open(action)
        return self._final(text, samples, words)

    def _prepare(self, token: CancelToken, action: str) -> _Streamer:
        pending = self._start(token, action)
        streamer = self._owner._streamer
        if streamer is None:
            raise LaneJobCancelled(f"{ENGINE_ID} was released: {action} discarded")
        if pending:
            self._unfed = np.concatenate([self._unfed, *pending])
        return streamer

    def _new_state(self, streamer: _Streamer) -> _StreamState:
        state = streamer.new_state()
        self._state = state
        return state

    def _feed_chunks(
        self, streamer: _Streamer, token: CancelToken, action: str
    ) -> _StreamState | None:
        """Feed every whole chunk buffered, checking the token and the stream between them."""
        state = self._state
        while self._unfed.size >= CHUNK_SAMPLES:
            if state is None:
                state = self._new_state(streamer)
            chunk, self._unfed = self._unfed[:CHUNK_SAMPLES], self._unfed[CHUNK_SAMPLES:]
            state.audio = np.concatenate([state.audio, chunk])
            streamer.run_steps(state, final=False)
            self._still_open(action)
            if token.is_set():
                raise LaneJobCancelled(f"{ENGINE_ID}: the token was set during the {action}")
        return state

    def _drop_state(self) -> None:
        # Frees the stream's encoder caches, which live on the GPU.
        self._state = None
        self._unfed = np.zeros(0, dtype=np.float32)


class NemotronStreamingTranscriber(TranscriberBase):
    """Nemotron's cache-aware streaming on the GPU, one `NemotronStream` per utterance."""

    engine_id = ENGINE_ID
    model_id = MODEL_ID
    _log = logger

    __slots__ = ("_streamer",)

    def __init__(
        self,
        model: NemoModel,
        preprocessor: object,
        torch: ModuleType,
        frame_ms: float | None = None,
    ) -> None:
        super().__init__()
        self._streamer: _Streamer | None = _Streamer(model, preprocessor, torch, frame_ms)

    @classmethod
    def load(
        cls,
        placement: PlacementConfig,
        *,
        library_loader: Callable[[], NemoLibrary],
        device: str = DEVICE,
        model_pin: FilePin = MODEL_PIN,
    ) -> "NemotronStreamingTranscriber":
        """Check the pinned `.nemo` file, restore it on `device` and set up 160 ms streaming.

        Raises:
            SpeechEngineError: The device is not `cuda`, the file is missing or does not match
                `model_pin` (the message names its path), NeMo is not installed, or the model
                failed to load.
        """
        if device != DEVICE:
            raise SpeechEngineError(
                f"{ENGINE_ID} runs on {DEVICE} only, the one device its arm was measured on, "
                f"not {device!r}"
            )
        path = Path(placement.models_dir).joinpath("speech", *WEIGHT_PATH.split("/"))
        verify_file(path, model_pin)
        library = library_loader()
        try:
            model = library.restore_from(str(path), map_location=device)
            model.eval()
            model.encoder.set_default_att_context_size(list(ATT_CONTEXT))
            model.encoder.setup_streaming_params()
            preprocessor = _streaming_preprocessor(model)
            frame_ms = _frame_ms(model)
            transcriber = cls(model, preprocessor, library.torch, frame_ms)
        except _LOAD_ERRORS as exc:
            raise SpeechEngineError(
                f"{ENGINE_ID}: loading {path} failed ({type(exc).__name__})"
            ) from exc
        logger.info("%s loaded on %s, streaming %d ms chunks", ENGINE_ID, device, CHUNK_MS)
        log_scores_absent(logger, ENGINE_ID)
        if frame_ms is None:
            logger.warning(
                "%s: the preprocessor config gives no positive window_stride, so its transcripts "
                "carry no word timings",
                ENGINE_ID,
            )
        return transcriber

    def open_stream(self) -> NemotronStream:
        """Open a stream for a new utterance; its encoder caches are made at its first decode.

        Raises:
            SpeechEngineError: The engine was released.
        """
        return self._register(NemotronStream(self))

    def _drop_model(self) -> None:
        self._streamer = None
