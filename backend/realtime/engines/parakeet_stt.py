"""NVIDIA's Parakeet TDT 0.6B v2 as ONNX behind `StreamingTranscriber`, on the CPU.

Engine id `parakeet-cpu`, the arm of the same name in `scripts/model_bench/speech/speech_arms.json`
(the parity test pins both to it): onnx-asr's `nemo-parakeet-tdt-0.6b-v2` on onnxruntime's CPU
provider, 4 threads. Parakeet is a batch model, so each partial decodes the whole utterance so
far and the final decodes all of it; the stream keeps every pushed sample until it closes.

Weights. The ONNX conversion is read from `<models_dir>/speech/istupakov/parakeet-tdt-0.6b-v2-onnx/`,
the directory the bench probe reads, and loaded with the probe's call,
`onnx_asr.load_model(<id>, path=<dir>, sess_options=<options>,
providers=["CPUExecutionProvider"])`, from that local path; the adapter makes no hub call. The arm
records no revision. The loaded model is wrapped with `.with_timestamps()`, so each decode also
returns its tokens and their times.

Word timings. Each character of the text is traced to the token that produced it, through the
library's own spacing rule, and the words are the result split at spaces. A word starts at its
first character's token and ends where the next word starts; the last ends with the audio
decoded. The text itself is the library's, stripped, as without timings.

Threads. The 4 threads are set on the session's own `SessionOptions` (`intra_op_num_threads`, with
one inter-op thread), as the probe sets them, so they bound this engine's sessions alone. No
process-wide setter runs, so nothing here competes with the Pocket TTS adapter's
`torch.set_num_threads`.

onnx-asr and onnxruntime are imported only by `import_onnx_asr`, which `load` is handed, so
importing this module loads neither and tests pass a fake library.
"""

import importlib
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ..budget import PlacementConfig
from ..errors import SpeechEngineError
from ..lanes import LaneJobCancelled
from ..speech import CancelToken, TranscriptUpdate, WordTiming
from .nemotron_stt import (
    LIBRARY_ERRORS,
    SAMPLE_RATE,
    BufferedTranscriptStream,
    TranscriberBase,
    audio_ms,
    checked_words,
    library_error,
    log_scores_absent,
)

logger = logging.getLogger(__name__)

ENGINE_ID = "parakeet-cpu"
ONNX_ASR_ID = "nemo-parakeet-tdt-0.6b-v2"
MODEL_ID = ONNX_ASR_ID
# Relative to `<models_dir>/speech/`, the directory the bench mounts as its weights directory.
WEIGHT_PATH = "istupakov/parakeet-tdt-0.6b-v2-onnx"
DEVICE = "cpu"
THREADS = 4
INTER_OP_THREADS = 1
PROVIDERS = ("CPUExecutionProvider",)

# onnxruntime's own errors, from its pybind module. They derive from `Exception` alone, so they
# are named one by one rather than caught as a base class.
_ORT_ERRORS = (
    "EPFail",
    "EngineError",
    "Fail",
    "InvalidArgument",
    "InvalidGraph",
    "InvalidProtobuf",
    "ModelLoaded",
    "NoModel",
    "NoSuchFile",
    "NotImplemented",
    "RuntimeException",
)


class SessionOptions(Protocol):
    """The `onnxruntime.SessionOptions` fields the adapter sets."""

    intra_op_num_threads: int
    inter_op_num_threads: int


class TimestampedText(Protocol):
    """What the adapter reads from onnx-asr's `TimestampedResult` (0.12.0, `asr.py:20-31`).

    `tokens` are vocabulary strings with "▁" read as a space; `timestamps` are each token's
    emitting encoder frame in seconds, 80 ms apart for Parakeet.
    """

    text: str
    tokens: list[str] | None
    timestamps: list[float] | None


class ParakeetModel(Protocol):
    """The call the adapter makes on onnx-asr's timestamped adapter (0.12.0)."""

    def recognize(self, waveform: np.ndarray, *, sample_rate: int) -> TimestampedText:
        """The transcript of one 16 kHz float32 utterance, with its tokens and their times."""
        ...


class OnnxAsrModel(Protocol):
    """What `onnx_asr.load_model` returns, as the adapter uses it: the text adapter."""

    def with_timestamps(self) -> ParakeetModel:
        """The same model behind an adapter whose results carry tokens and timestamps."""
        ...


@dataclass(frozen=True, slots=True)
class OnnxAsrLibrary:
    """What the adapter needs from onnx-asr and onnxruntime, bound once by `import_onnx_asr`.

    `load_model` is `onnx_asr.load_model`; `session_options` is `onnxruntime.SessionOptions`;
    `errors` are onnxruntime's own exception classes, which a load or decode may raise.
    """

    load_model: Callable[..., OnnxAsrModel]
    session_options: Callable[[], SessionOptions]
    errors: tuple[type[BaseException], ...]


# onnx-asr's spacing rule, `_AsrWithDecoding.DECODE_SPACE_PATTERN` (0.12.0, `asr.py:109`, applied
# at `:147`): a leading space and a space before anything but a word character go, and a space
# before a word character stays.
DECODE_SPACE_PATTERN = re.compile(r"\A\s|\s\B|(\s)\b")
# SentencePiece's word mark, U+2581, which the library reads as a space (`asr.py:123`).
WORD_MARK = "▁"


def token_words(
    tokens: list[str] | None, timestamps: list[float] | None, text: str, end_ms: int
) -> tuple[WordTiming, ...] | None:
    """The words of a decode, each timed from the token that produced its first character.

    The tokens are joined with each character's token kept, and the library's spacing rule is
    applied to that, so every character of the text is traced to its token, and a token that is
    only a space separates two words. A word starts at its first character's token and ends
    where the next word starts; the last ends at `end_ms`, the end of the audio decoded, because
    a token's time marks only the frame that emitted it. None when the tokens or times are
    missing, do not pair up, run backwards, or do not give `text`'s words.
    """
    if tokens is None or timestamps is None or len(tokens) != len(timestamps):
        return None
    chars: list[str] = []
    owners: list[int] = []
    for index, token in enumerate(tokens):
        for char in token.replace(WORD_MARK, " "):
            chars.append(char)
            owners.append(index)
    joined = "".join(chars)
    spaced: list[tuple[str, int]] = []
    kept = 0
    for match in DECODE_SPACE_PATTERN.finditer(joined):
        spaced.extend(zip(joined[kept : match.start()], owners[kept : match.start()]))
        if match.group(1):
            spaced.append((" ", owners[match.start(1)]))
        kept = match.end()
    spaced.extend(zip(joined[kept:], owners[kept:]))

    # Each word with the start of the token that produced its first character.
    found: list[tuple[str, int]] = []
    letters: list[str] = []
    first = 0
    for char, index in [*spaced, (" ", 0)]:
        if not char.isspace():
            if not letters:
                first = index
            letters.append(char)
        elif letters:
            found.append(("".join(letters), round(timestamps[first] * 1000)))
            letters = []
    starts = [start for _, start in found]
    if starts != sorted(starts):
        return None
    ends = [*starts[1:], max(end_ms, starts[-1])] if starts else []
    words = tuple(WordTiming(word, start, end) for (word, start), end in zip(found, ends))
    return checked_words(text, words)


def import_onnx_asr() -> OnnxAsrLibrary:
    """Import onnx-asr and onnxruntime and bind what the adapter calls.

    Raises:
        SpeechEngineError: onnx-asr or onnxruntime is not installed.
    """
    try:
        ort = importlib.import_module("onnxruntime")
        state = importlib.import_module("onnxruntime.capi.onnxruntime_pybind11_state")
        onnx_asr = importlib.import_module("onnx_asr")
    except ImportError as exc:
        raise SpeechEngineError(
            f"onnx-asr or onnxruntime is not installed ({exc.name}); requirements-voice.txt "
            "pins onnx-asr and requirements.txt onnxruntime"
        ) from exc
    errors = tuple(
        error
        for error in (getattr(state, name, None) for name in _ORT_ERRORS)
        if isinstance(error, type) and issubclass(error, BaseException)
    )
    return OnnxAsrLibrary(
        load_model=onnx_asr.load_model, session_options=ort.SessionOptions, errors=errors
    )


class ParakeetStream(BufferedTranscriptStream):
    """One utterance on Parakeet: every decode reads all the audio pushed so far."""

    __slots__ = ("_audio", "_owner")

    def __init__(self, owner: "ParakeetTranscriber") -> None:
        super().__init__(ENGINE_ID, owner._forget)
        self._owner = owner
        self._audio = np.zeros(0, dtype=np.float32)

    def decode_partial(self, token: CancelToken) -> TranscriptUpdate:
        """Decode the utterance so far.

        Raises:
            LaneJobCancelled: The token was set, or the stream was closed.
            SpeechEngineError: The engine failed; the message holds no text.
        """
        text, samples, words = self._decode(token, "partial decode")
        return self._partial(text, samples, words)

    def decode_final(self, token: CancelToken) -> TranscriptUpdate:
        """Decode the whole utterance; the stream goes on if more audio follows.

        Raises:
            LaneJobCancelled: The token was set, or the stream was closed.
            SpeechEngineError: The engine failed; the message holds no text.
        """
        text, samples, words = self._decode(token, "final decode")
        return self._final(text, samples, words)

    def _decode(
        self, token: CancelToken, action: str
    ) -> tuple[str, int, tuple[WordTiming, ...] | None]:
        pending = self._start(token, action)
        engine = self._owner._engine
        if engine is None:
            raise LaneJobCancelled(f"{ENGINE_ID} was released: {action} discarded")
        model, errors = engine
        if pending:
            self._audio = np.concatenate([self._audio, *pending])
        audio = self._audio
        text = ""
        words: tuple[WordTiming, ...] | None = ()
        if audio.size:
            try:
                result = model.recognize(audio, sample_rate=SAMPLE_RATE)
                text = str(result.text).strip()
                words = token_words(result.tokens, result.timestamps, text, audio_ms(audio.size))
            except errors as exc:
                # `from None`: the library's message may echo the utterance.
                raise library_error(ENGINE_ID, action, exc) from None
        self._still_open(action)
        return text, audio.size, words

    def _drop_state(self) -> None:
        self._audio = np.zeros(0, dtype=np.float32)


class ParakeetTranscriber(TranscriberBase):
    """Parakeet TDT 0.6B v2 on onnxruntime's CPU provider, one `ParakeetStream` per utterance."""

    engine_id = ENGINE_ID
    model_id = MODEL_ID
    _log = logger

    __slots__ = ("_engine",)

    def __init__(self, model: ParakeetModel, errors: tuple[type[BaseException], ...]) -> None:
        super().__init__()
        # The model and the errors a decode converts: the library's own and the common ones.
        self._engine: tuple[ParakeetModel, tuple[type[BaseException], ...]] | None = (
            model,
            (*LIBRARY_ERRORS, *errors),
        )

    @classmethod
    def load(
        cls,
        placement: PlacementConfig,
        *,
        library_loader: Callable[[], OnnxAsrLibrary],
        device: str = DEVICE,
    ) -> "ParakeetTranscriber":
        """Load the ONNX model from under `models_dir` on the CPU, 4 threads per session.

        Raises:
            SpeechEngineError: The device is not `cpu`, the weights directory is missing or
                empty (the message names its path), onnx-asr is not installed, or the model
                failed to load.
        """
        if device != DEVICE:
            raise SpeechEngineError(
                f"{ENGINE_ID} runs on {DEVICE} only, the one device its arm was measured on, "
                f"not {device!r}"
            )
        path = Path(placement.models_dir).joinpath("speech", *WEIGHT_PATH.split("/"))
        if not path.is_dir() or not any(path.iterdir()):
            raise SpeechEngineError(f"{ENGINE_ID}: weights directory {path} is missing or empty")
        library = library_loader()
        errors = (*LIBRARY_ERRORS, *library.errors)
        try:
            options = library.session_options()
            options.intra_op_num_threads = THREADS
            options.inter_op_num_threads = INTER_OP_THREADS
            model = library.load_model(
                ONNX_ASR_ID, path=str(path), sess_options=options, providers=list(PROVIDERS)
            ).with_timestamps()
        except errors as exc:
            raise SpeechEngineError(
                f"{ENGINE_ID}: loading {path} failed ({type(exc).__name__})"
            ) from exc
        transcriber = cls(model, library.errors)
        logger.info(
            "%s loaded %s on %s with %d threads per session", ENGINE_ID, MODEL_ID, DEVICE, THREADS
        )
        log_scores_absent(logger, ENGINE_ID)
        return transcriber

    def open_stream(self) -> ParakeetStream:
        """Open a stream for a new utterance.

        Raises:
            SpeechEngineError: The engine was released.
        """
        return self._register(ParakeetStream(self))

    def _drop_model(self) -> None:
        self._engine = None
