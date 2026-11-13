"""Silero VAD (the ONNX v5 build) behind `VoiceActivityDetector`, and the ONNX loading both VAD
and end of turn share.

Ingress scores every 512-sample window of every uplink, so the model runs on onnxruntime rather
than torch: one small single-threaded session, shared by every stream, with each stream carrying
its own recurrent state. The loading helpers (`SessionSpec`, `resolve_model_path`,
`select_providers`, `open_onnx_session`) live here because Smart Turn loads the same way and the
engines package keeps one module per adapter; `smart_turn.py` imports them from this module.

Weights are never downloaded. ADR-025 keeps the stack air-gapped capable, so a model setting
`<repo>@<revision>:<file>` names a file the host staged at `<models_dir>/speech/<repo>/<file>`;
the revision is the pin that staging fetched, and a missing file is a load error naming the path.

onnxruntime is imported only inside `open_onnx_session` and `onnxruntime_providers`, so importing
this module loads no engine package and the unit tier, which has no onnxruntime, can inject a
fake session factory.
"""

import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Protocol

import numpy as np

from ..budget import PlacementConfig
from ..errors import SpeechEngineError

logger = logging.getLogger(__name__)

CPU_PROVIDER = "CPUExecutionProvider"
CUDA_PROVIDER = "CUDAExecutionProvider"

SAMPLE_RATE = 16000
WINDOW_SAMPLES = 512
CONTEXT_SAMPLES = 64
"""Silero v5 at 16 kHz sees each window behind the last 64 samples of the one before it.

silero-vad's own `OnnxWrapper` prepends this context; without it the model scores each window
blind to its left edge and its probabilities drift from the reference.
"""
STATE_SHAPE = (2, 1, 128)
VAD_INPUTS = frozenset({"input", "state", "sr"})


class OnnxInput(Protocol):
    """A model input as an onnxruntime session describes it."""

    @property
    def name(self) -> str:
        """The input's name, which the load check compares with the expected inputs."""
        ...


class OnnxSession(Protocol):
    """The part of `onnxruntime.InferenceSession` the adapters use, so tests can fake it."""

    def get_inputs(self) -> Sequence[OnnxInput]:
        """The model's inputs, read once at load to catch a wrong file early."""
        ...

    def get_providers(self) -> Sequence[str]:
        """The providers the session actually loaded, which can drop a requested CUDA one."""
        ...

    def run(
        self, output_names: Sequence[str] | None, input_feed: Mapping[str, np.ndarray]
    ) -> Sequence[np.ndarray]:
        """Run the model; `None` for `output_names` returns every output in model order."""
        ...


@dataclass(frozen=True, slots=True)
class SessionSpec:
    """Everything a session factory needs to open one model, decided before onnxruntime loads.

    `providers` is in priority order, as `InferenceSession(providers=...)` takes it, and
    `device_id` is the CUDA device index, read only when the CUDA provider is listed. A zero
    thread count lets onnxruntime choose.
    """

    path: str
    providers: tuple[str, ...]
    device_id: int
    intra_op_num_threads: int
    inter_op_num_threads: int


SessionFactory = Callable[[SessionSpec], OnnxSession]
ProviderLister = Callable[[], Sequence[str]]


def resolve_model_path(models_dir: str, setting: str, setting_name: str) -> Path:
    """The staged file a `<repo>@<revision>:<file>` setting names, under `models_dir/speech/`.

    A setting is checked again here although `budget.py` validated its form, because a value
    built in code skips the environment parser, and a `..` segment would read outside the
    read-only weights root.

    Raises:
        SpeechEngineError: The setting is malformed, leaves the weights root, or names a file
            that is not there. The message names the setting or the path.
    """
    repo, at, rest = setting.partition("@")
    revision, colon, file = rest.partition(":")
    if not (at and colon and repo and revision and file):
        raise SpeechEngineError(f"{setting_name}={setting!r}: expected <repo>@<revision>:<file>")
    parts = [*PurePosixPath(repo).parts, *PurePosixPath(file).parts]
    if PurePosixPath(file).is_absolute() or PurePosixPath(repo).is_absolute():
        raise SpeechEngineError(f"{setting_name}={setting!r}: repo and file must be relative")
    if any(part in ("", ".", "..") for part in parts):
        raise SpeechEngineError(f"{setting_name}={setting!r}: a path segment leaves models_dir")
    path = Path(models_dir, "speech", *parts)
    if not path.is_file():
        raise SpeechEngineError(
            f"{setting_name}: model file {path} not found; stage {repo}@{revision}:{file} "
            "under the weights root, since the adapters never download"
        )
    return path


def select_providers(
    device: str, setting_name: str, available: ProviderLister
) -> tuple[tuple[str, ...], int]:
    """The execution providers and CUDA device index for a `cpu`, `cuda` or `cuda:<n>` setting.

    A CUDA request on an onnxruntime build without the CUDA provider still loads, on the CPU,
    because the realtime path must come up on the CPU-only image; the WARNING names the setting
    so that the fallback is visible rather than a silent slowdown.
    """
    if device == "cpu":
        return (CPU_PROVIDER,), 0
    kind, _, index = device.partition(":")
    if kind != "cuda" or (index and not index.isdigit()):
        raise SpeechEngineError(f"{setting_name}={device!r}: expected cpu, cuda or cuda:<n>")
    if CUDA_PROVIDER not in available():
        logger.warning(
            "%s=%s asks for %s, which this onnxruntime build does not offer; running on the CPU",
            setting_name,
            device,
            CUDA_PROVIDER,
        )
        return (CPU_PROVIDER,), 0
    return (CUDA_PROVIDER, CPU_PROVIDER), int(index or 0)


def open_checked_session(
    spec: SessionSpec,
    factory: SessionFactory,
    expected_inputs: frozenset[str],
    device_setting: str,
) -> OnnxSession:
    """Open a session and check its inputs, so a wrong file fails at load and not mid-turn.

    onnxruntime can also drop a requested CUDA provider at session creation, for example when
    the CUDA libraries do not load; that is logged with the device setting, like the fallback in
    `select_providers`.

    Raises:
        SpeechEngineError: The model's input names differ from `expected_inputs`; the message
            names each missing and unexpected input and the model path.
    """
    session = factory(spec)
    names = {model_input.name for model_input in session.get_inputs()}
    missing = sorted(expected_inputs - names)
    unexpected = sorted(names - expected_inputs)
    if missing or unexpected:
        raise SpeechEngineError(
            f"{spec.path}: model inputs do not match; missing {missing}, unexpected {unexpected}"
        )
    if CUDA_PROVIDER in spec.providers and CUDA_PROVIDER not in session.get_providers():
        logger.warning(
            "%s asks for %s, but the session for %s did not load it; running on the CPU",
            device_setting,
            CUDA_PROVIDER,
            spec.path,
        )
    return session


def onnxruntime_providers() -> list[str]:
    """`onnxruntime.get_available_providers()`, imported only when an engine is built.

    Raises:
        SpeechEngineError: onnxruntime is not installed.
    """
    try:
        import onnxruntime
    except ImportError as exc:
        raise SpeechEngineError(
            "onnxruntime is not installed; it runs VAD and end of turn"
        ) from exc
    return list(onnxruntime.get_available_providers())


def open_onnx_session(spec: SessionSpec) -> OnnxSession:
    """The real session factory: an `onnxruntime.InferenceSession` built from `spec`.

    Execution is sequential because both models are small single-path graphs, where parallel
    node execution only adds thread hand-offs.

    Raises:
        SpeechEngineError: onnxruntime is not installed, or it cannot load the file.
    """
    try:
        import onnxruntime
        from onnxruntime.capi.onnxruntime_pybind11_state import (
            Fail,
            InvalidArgument,
            InvalidProtobuf,
            NoSuchFile,
        )
    except ImportError as exc:
        raise SpeechEngineError(
            "onnxruntime is not installed; it runs VAD and end of turn"
        ) from exc
    options = onnxruntime.SessionOptions()
    options.intra_op_num_threads = spec.intra_op_num_threads
    options.inter_op_num_threads = spec.inter_op_num_threads
    options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
    provider_options = [
        {"device_id": str(spec.device_id)} if provider == CUDA_PROVIDER else {}
        for provider in spec.providers
    ]
    try:
        return onnxruntime.InferenceSession(
            spec.path,
            sess_options=options,
            providers=list(spec.providers),
            provider_options=provider_options,
        )
    except (Fail, InvalidArgument, InvalidProtobuf, NoSuchFile, RuntimeError) as exc:
        raise SpeechEngineError(
            f"{spec.path}: onnxruntime could not load the model: {exc}"
        ) from exc


class SileroVadStream:
    """One audio source's Silero state: the recurrent state and the last window's tail.

    Both are per stream because the session is shared; resetting at an utterance boundary keeps
    one utterance's trailing speech from biasing the next one's onset.
    """

    __slots__ = ("_context", "_session", "_state")

    def __init__(self, session: OnnxSession) -> None:
        self._session = session
        self._state = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)

    def score(self, window: np.ndarray) -> float:
        """Score 512 samples at 16 kHz; the model's state and context move on by one window.

        Raises:
            SpeechEngineError: The window is not 512 mono samples.
        """
        samples = np.asarray(window, dtype=np.float32)
        if samples.shape != (WINDOW_SAMPLES,):
            raise SpeechEngineError(
                f"Silero VAD scores windows of {WINDOW_SAMPLES} samples, got shape {samples.shape}"
            )
        framed = np.concatenate([self._context, samples[np.newaxis, :]], axis=1)
        output, state = self._session.run(
            None,
            {
                "input": framed,
                "state": self._state,
                "sr": np.array(SAMPLE_RATE, dtype=np.int64),
            },
        )
        self._state = np.asarray(state, dtype=np.float32)
        self._context = framed[:, -CONTEXT_SAMPLES:]
        return float(np.asarray(output).reshape(-1)[0])

    def reset(self) -> None:
        """Clear the state and context at an utterance boundary."""
        self._state = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._context = np.zeros((1, CONTEXT_SAMPLES), dtype=np.float32)


class SileroVad:
    """Silero VAD v5 over one onnxruntime session that every stream shares.

    A session's `run` is safe to call from several threads, and each stream keeps its own
    state, so one loaded model serves every audio source.
    """

    __slots__ = ("_session",)

    def __init__(self, session: OnnxSession) -> None:
        self._session = session

    @classmethod
    def load(
        cls,
        placement: PlacementConfig,
        *,
        session_factory: SessionFactory,
        available_providers: ProviderLister,
    ) -> "SileroVad":
        """Open `placement.vad_model` under `placement.models_dir` on `placement.vad_device`.

        The session is single threaded: one window is about a millisecond of work, and the
        threads are better left to STT and end of turn.

        Raises:
            SpeechEngineError: The setting is malformed, the file is missing, or the model's
                inputs are not Silero v5's `input`, `state` and `sr`.
        """
        path = resolve_model_path(placement.models_dir, placement.vad_model, "MIST_VOICE_VAD_MODEL")
        providers, device_id = select_providers(
            placement.vad_device, "MIST_VOICE_VAD_DEVICE", available_providers
        )
        spec = SessionSpec(
            path=str(path),
            providers=providers,
            device_id=device_id,
            intra_op_num_threads=1,
            inter_op_num_threads=1,
        )
        session = open_checked_session(spec, session_factory, VAD_INPUTS, "MIST_VOICE_VAD_DEVICE")
        logger.info("Silero VAD loaded from %s on %s", path, providers[0])
        return cls(session)

    def open_stream(self) -> SileroVadStream:
        """Open a stream with zero state, for a new audio source."""
        return SileroVadStream(self._session)
