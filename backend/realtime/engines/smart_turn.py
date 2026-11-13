"""Smart Turn v3.2 (ONNX) behind `TurnEndDetector`: does the speaker sound finished?

Silence alone cannot tell a pause for thought from the end of a request, so after
`EndpointingConfig.endpoint_silence_ms` of silence ingress asks this model, which reads the
prosody of the last 8 s. The input path follows `pipecat-ai/smart-turn`'s `inference.py`, the
one the model was trained and published with: the last 8 s at 16 kHz, zero-padded at the front
when shorter, through `WhisperFeatureExtractor(chunk_length=8)` into `input_features` of shape
(1, 80, 800), and the first output as the probability. Any other feature path would score audio
the model never saw.

The session runs with `PlacementConfig.eot_threads` intra-op threads, the thread count the speech
bench measured `smart-turn-v3.2-cpu.onnx` with. onnxruntime and transformers are imported only
when an engine is built, so importing this module loads neither.
"""

import logging
from collections.abc import Callable
from typing import Protocol

import numpy as np

from ..budget import EndpointingConfig, PlacementConfig
from ..errors import SpeechEngineError
from .silero_vad import (
    OnnxSession,
    ProviderLister,
    SessionFactory,
    SessionSpec,
    open_checked_session,
    resolve_model_path,
    select_providers,
)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
WINDOW_SECONDS = 8
WINDOW_SAMPLES = WINDOW_SECONDS * SAMPLE_RATE
EOT_INPUTS = frozenset({"input_features"})


class Features(Protocol):
    """What a Whisper feature extractor returns: a batch of log-mel frames."""

    @property
    def input_features(self) -> np.ndarray:
        """Log-mel frames of shape (1, 80, 800) for 8 s of audio."""
        ...


class FeatureExtractor(Protocol):
    """The call `transformers.WhisperFeatureExtractor` answers, so tests can record or fake it."""

    def __call__(
        self,
        raw_speech: np.ndarray,
        *,
        sampling_rate: int,
        return_tensors: str,
        padding: str,
        max_length: int,
        truncation: bool,
        do_normalize: bool,
    ) -> Features:
        """Turn 16 kHz float32 audio into the model's `input_features`."""
        ...


def whisper_features() -> FeatureExtractor:
    """The real extractor, `WhisperFeatureExtractor(chunk_length=8)`, imported when called.

    Raises:
        SpeechEngineError: transformers is not installed.
    """
    try:
        from transformers import WhisperFeatureExtractor
    except ImportError as exc:
        raise SpeechEngineError("transformers is not installed; Smart Turn needs it") from exc
    return WhisperFeatureExtractor(chunk_length=WINDOW_SECONDS)


def last_eight_seconds(audio: np.ndarray) -> np.ndarray:
    """The last 8 s of mono 16 kHz audio as float32, zero-padded at the front when shorter.

    Padding goes at the front so that the end of the speech, which the model judges, always
    sits at the end of the window.

    Raises:
        SpeechEngineError: The audio is not one-dimensional.
    """
    samples = np.asarray(audio, dtype=np.float32)
    if samples.ndim != 1:
        raise SpeechEngineError(f"Smart Turn takes mono audio, got shape {samples.shape}")
    tail = samples[-WINDOW_SAMPLES:]
    if len(tail) == WINDOW_SAMPLES:
        return tail
    return np.concatenate([np.zeros(WINDOW_SAMPLES - len(tail), dtype=np.float32), tail])


class SmartTurnDetector:
    """Smart Turn v3.2 over one onnxruntime session and a Whisper feature extractor."""

    __slots__ = ("_endpointing", "_features", "_session")

    def __init__(
        self,
        session: OnnxSession,
        feature_extractor: FeatureExtractor,
        endpointing: EndpointingConfig,
    ) -> None:
        self._session = session
        self._features = feature_extractor
        self._endpointing = endpointing

    @classmethod
    def load(
        cls,
        placement: PlacementConfig,
        endpointing: EndpointingConfig,
        *,
        session_factory: SessionFactory,
        available_providers: ProviderLister,
        feature_extractor_factory: Callable[[], FeatureExtractor],
    ) -> "SmartTurnDetector":
        """Open `placement.eot_model` under `placement.models_dir` on `placement.eot_device`.

        Inter-op threads stay at one, as `inference.py` sets them: the graph is one path, so
        only intra-op threads shorten a call.

        Raises:
            SpeechEngineError: The setting is malformed, the file is missing, the model has no
                `input_features` input, or a dependency is not installed.
        """
        path = resolve_model_path(placement.models_dir, placement.eot_model, "MIST_VOICE_EOT_MODEL")
        providers, device_id = select_providers(
            placement.eot_device, "MIST_VOICE_EOT_DEVICE", available_providers
        )
        spec = SessionSpec(
            path=str(path),
            providers=providers,
            device_id=device_id,
            intra_op_num_threads=placement.eot_threads,
            inter_op_num_threads=1,
        )
        session = open_checked_session(spec, session_factory, EOT_INPUTS, "MIST_VOICE_EOT_DEVICE")
        logger.info(
            "Smart Turn loaded from %s on %s with %d threads",
            path,
            providers[0],
            placement.eot_threads,
        )
        return cls(session, feature_extractor_factory(), endpointing)

    def probability_complete(self, audio: np.ndarray) -> float:
        """The probability that the turn has ended, from up to the last 8 s of audio.

        Raises:
            SpeechEngineError: The audio is not mono.
        """
        features = self._features(
            last_eight_seconds(audio),
            sampling_rate=SAMPLE_RATE,
            return_tensors="np",
            padding="max_length",
            max_length=WINDOW_SAMPLES,
            truncation=True,
            do_normalize=True,
        )
        batch = np.asarray(features.input_features, dtype=np.float32)
        outputs = self._session.run(None, {"input_features": batch})
        return float(np.asarray(outputs[0]).reshape(-1)[0])

    def is_complete(self, audio: np.ndarray) -> bool:
        """Whether the probability is above `EndpointingConfig.eot_threshold`.

        Strictly above, as `inference.py` and the endpointing contract both read it, so a
        probability equal to the threshold keeps the turn open.
        """
        return self.probability_complete(audio) > self._endpointing.eot_threshold
