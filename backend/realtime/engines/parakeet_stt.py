"""NVIDIA's Parakeet TDT 0.6B v2 as ONNX behind `StreamingTranscriber`, on the CPU.

Engine id `parakeet-cpu`, the arm of the same name in `scripts/model_bench/speech/speech_arms.json`
(the parity test pins both to it): onnx-asr's `nemo-parakeet-tdt-0.6b-v2` on onnxruntime's CPU
provider, 4 threads. Parakeet is a batch model, so each partial decodes the whole utterance so
far and the final decodes all of it; the stream keeps every pushed sample until it closes.

Weights. The ONNX conversion is read from `<models_dir>/speech/istupakov/parakeet-tdt-0.6b-v2-onnx/`,
the directory the bench probe reads, and loaded with the probe's call,
`onnx_asr.load_model(<id>, path=<dir>, sess_options=<options>,
providers=["CPUExecutionProvider"])`, from that local path; the adapter makes no hub call. The arm
records no revision.

Threads. The 4 threads are set on the session's own `SessionOptions` (`intra_op_num_threads`, with
one inter-op thread), as the probe sets them, so they bound this engine's sessions alone. No
process-wide setter runs, so nothing here competes with the Pocket TTS adapter's
`torch.set_num_threads`.

onnx-asr and onnxruntime are imported only by `import_onnx_asr`, which `load` is handed, so
importing this module loads neither and tests pass a fake library.
"""

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from ..budget import PlacementConfig
from ..errors import SpeechEngineError
from ..lanes import LaneJobCancelled
from ..speech import CancelToken, TranscriptUpdate
from .nemotron_stt import (
    LIBRARY_ERRORS,
    SAMPLE_RATE,
    BufferedTranscriptStream,
    TranscriberBase,
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


class ParakeetModel(Protocol):
    """The call the adapter makes on a loaded onnx-asr model (0.12.0)."""

    def recognize(self, waveform: np.ndarray, *, sample_rate: int) -> str:
        """The transcript of one 16 kHz float32 utterance."""
        ...


@dataclass(frozen=True, slots=True)
class OnnxAsrLibrary:
    """What the adapter needs from onnx-asr and onnxruntime, bound once by `import_onnx_asr`.

    `load_model` is `onnx_asr.load_model`; `session_options` is `onnxruntime.SessionOptions`;
    `errors` are onnxruntime's own exception classes, which a load or decode may raise.
    """

    load_model: Callable[..., ParakeetModel]
    session_options: Callable[[], SessionOptions]
    errors: tuple[type[BaseException], ...]


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
        text, samples = self._decode(token, "partial decode")
        return self._partial(text, samples)

    def decode_final(self, token: CancelToken) -> TranscriptUpdate:
        """Decode the whole utterance; the stream goes on if more audio follows.

        Raises:
            LaneJobCancelled: The token was set, or the stream was closed.
            SpeechEngineError: The engine failed; the message holds no text.
        """
        text, samples = self._decode(token, "final decode")
        return self._final(text, samples)

    def _decode(self, token: CancelToken, action: str) -> tuple[str, int]:
        pending = self._start(token, action)
        engine = self._owner._engine
        if engine is None:
            raise LaneJobCancelled(f"{ENGINE_ID} was released: {action} discarded")
        model, errors = engine
        if pending:
            self._audio = np.concatenate([self._audio, *pending])
        audio = self._audio
        text = ""
        if audio.size:
            try:
                text = str(model.recognize(audio, sample_rate=SAMPLE_RATE)).strip()
            except errors as exc:
                # `from None`: the library's message may echo the utterance.
                raise library_error(ENGINE_ID, action, exc) from None
        self._still_open(action)
        return text, audio.size

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
            )
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
