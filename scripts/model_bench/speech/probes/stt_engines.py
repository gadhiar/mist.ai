"""Run 3's speech-to-text adapters, each meeting `engines.Transcriber`, and their factory.

Each adapter takes its loaded model in the constructor, so the unit tests drive it with a fake;
`make_transcriber` is the one place that imports an engine package and loads weights, and it does
so only when called, inside a bench container, from the speech weights directory mounted
read-only at `/weights` with `HF_HUB_OFFLINE=1`. Run 7's chain service wraps the same factory
(`make_transcriber("parakeet-cpu", weights_dir=..., threads=4)`), so the chained turn times the
adapter run 3 timed.

Engines (mist-model-host plan v2, engine facts):
- `whisper-base`: openai-whisper `base`, fp32 on the GPU, decoded with `fp16=False` and MIST's own
  `initial_prompt`, as `src/multimodal/stt.py` runs it.
- `parakeet-gpu`: Parakeet TDT 0.6B v2 through NeMo 2.7.3 on the GPU.
- `nemotron-160`: NVIDIA's streaming English model through NeMo 2.7.3, fed 160 ms chunks.
- `parakeet-cpu`: the onnx-asr conversion of Parakeet TDT 0.6B v2 on onnxruntime, 4 CPU threads.

No NeMo, onnx-asr or weights are reachable where this was written: the NeMo and onnx-asr calls
below follow their documented interfaces and are tested only through fakes. Host step H2 loads
`nemotron-160` and the onnx-asr Parakeet once in the asr image to confirm them; if
`nemotron-160` does not load (its model card names a NeMo and transformers newer than 2.7.3
pins), the probe reports it "not loaded" with the error, and any other route needs a plan
amendment.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from .. import core

if TYPE_CHECKING:
    import numpy

SAMPLE_RATE = 16000
MAX_CPU_THREADS = 4

# MIST's own Whisper prompt, copied from src/multimodal/stt.py:23-26 so the bench decodes as MIST
# does; tests/unit/model_bench/test_speech_stt.py parses that file and fails if the two differ.
WHISPER_INITIAL_PROMPT = (
    "MIST is an AI assistant with knowledge graph capabilities. "
    "The user is asking about MIST, the AI system, not missed or mist."
)
WHISPER_OPTIONS = {"fp16": False, "initial_prompt": WHISPER_INITIAL_PROMPT}

# Each engine's weights, relative to the mounted speech weights directory. Hub models follow H2's
# `<org>/<repo>/` download layout; openai-whisper's `base.pt` comes from the package's own
# download URL, not the hub, so it sits in a directory named for the package.
WEIGHT_PATHS = {
    "whisper-base": "openai-whisper/base.pt",
    "parakeet-gpu": "nvidia/parakeet-tdt-0.6b-v2/parakeet-tdt-0.6b-v2.nemo",
    "parakeet-cpu": "istupakov/parakeet-tdt-0.6b-v2-onnx",
    "nemotron-160": "nvidia/nemotron-speech-streaming-en-0.6b/nemotron-speech-streaming-en-0.6b.nemo",
}

# The onnx-asr model id for Parakeet TDT 0.6B v2, inferred from the documented v3 id; H2 confirms
# it (speech_arms.json carries it with `verified: false`).
ONNX_ASR_PARAKEET_ID = "nemo-parakeet-tdt-0.6b-v2"

# Nemotron's attention context for 160 ms chunks: [left, right] in encoder frames of 80 ms, so
# one frame of right context is a 160 ms chunk (model card's table of chunk sizes; H2 confirms).
NEMOTRON_ATT_CONTEXT = [70, 1]


def _require_16k(sample_rate: int) -> None:
    if sample_rate != SAMPLE_RATE:
        raise ValueError(f"the bench transcribes {SAMPLE_RATE} Hz audio, not {sample_rate}")


def _float32(audio: numpy.ndarray) -> numpy.ndarray:
    import numpy as np

    return np.ascontiguousarray(audio, dtype=np.float32).reshape(-1)


def _hypothesis_text(reply: object) -> str:
    """The text of NeMo's first hypothesis, whichever documented shape `transcribe` returned.

    NeMo 2.x returns a list of `Hypothesis` objects (with `.text`) or of strings, and for some
    decoders a `(best, all)` pair of such lists.
    """
    if isinstance(reply, tuple):
        reply = reply[0]
    first = reply[0] if isinstance(reply, list | tuple) else reply
    if isinstance(first, list | tuple):
        first = first[0]
    text = getattr(first, "text", first)
    return str(text).strip()


class WhisperTranscriber:
    """openai-whisper with MIST's decode options."""

    def __init__(self, model):
        self._model = model

    def transcribe(self, audio: numpy.ndarray, sample_rate: int) -> str:
        """Transcribe a whole utterance as `WhisperSTT.transcribe_audio` does."""
        _require_16k(sample_rate)
        result = self._model.transcribe(_float32(audio), **WHISPER_OPTIONS)
        return str(result["text"]).strip()


class NemoTranscriber:
    """A NeMo batch ASR model (Parakeet TDT on the GPU)."""

    def __init__(self, model):
        self._model = model

    def transcribe(self, audio: numpy.ndarray, sample_rate: int) -> str:
        """Transcribe one utterance passed as an in-memory 16 kHz array."""
        _require_16k(sample_rate)
        reply = self._model.transcribe([_float32(audio)], batch_size=1, verbose=False)
        return _hypothesis_text(reply)


class OnnxAsrTranscriber:
    """An onnx-asr model on onnxruntime's CPU provider (Parakeet as ONNX)."""

    def __init__(self, model):
        self._model = model

    def transcribe(self, audio: numpy.ndarray, sample_rate: int) -> str:
        """Recognize one utterance."""
        _require_16k(sample_rate)
        return str(self._model.recognize(_float32(audio), sample_rate=sample_rate)).strip()


class NemoStreamingTranscriber:
    """NeMo's cache-aware streaming over audio fed chunk by chunk (`nemotron-160`).

    `feed` appends a chunk and runs every encoder step whose feature frames are complete, as a
    live microphone would allow; `finish` runs the remaining frames with `keep_all_outputs` and
    returns the text. This follows NeMo's `CacheAwareStreamingAudioBuffer` and its cache-aware
    streaming example (`conformer_stream_step` with the encoder's `streaming_cfg`), with one
    difference: features are recomputed over the audio received so far, so steps start only once
    their frames lie wholly inside it (`guard_frames` short of the end, where the STFT window is
    still open). H2 confirms the calls against NeMo 2.7.3.
    """

    def __init__(self, model, *, guard_frames: int = 2):
        import torch

        self._torch = torch
        self._model = model
        self._guard = guard_frames
        self._device = next(model.parameters()).device
        self._preprocessor = self._streaming_preprocessor(model)
        self.reset()

    @staticmethod
    def _streaming_preprocessor(model):
        import copy

        cfg = copy.deepcopy(model._cfg.preprocessor)
        cfg.dither = 0.0
        cfg.pad_to = 0
        return model.from_config_dict(cfg).to(next(model.parameters()).device)

    def reset(self) -> None:
        """Start a new utterance: empty audio, fresh encoder caches, no hypotheses."""
        import numpy as np

        self._audio = np.zeros(0, dtype=np.float32)
        self._frame_idx = 0
        self._step = 0
        self._text = ""
        cache = self._model.encoder.get_initial_cache_state(batch_size=1)
        self._cache_channel, self._cache_time, self._cache_channel_len = cache
        self._hypotheses = None
        self._pred_out = None

    def _features(self):
        torch = self._torch
        signal = torch.from_numpy(self._audio).unsqueeze(0).to(self._device)
        length = torch.tensor([signal.shape[1]], device=self._device)
        features, _ = self._preprocessor(input_signal=signal, length=length)
        return features

    @staticmethod
    def _pick(value, first: bool) -> int:
        if isinstance(value, list | tuple):
            return int(value[0] if first else value[1])
        return int(value)

    def _run_steps(self, *, final: bool) -> None:
        torch = self._torch
        cfg = self._model.encoder.streaming_cfg
        features = self._features()
        total = features.shape[-1]
        usable = total if final else total - self._guard
        while self._frame_idx < total:
            first = self._step == 0
            chunk = self._pick(cfg.chunk_size, first)
            shift = self._pick(cfg.shift_size, first)
            end = self._frame_idx + chunk
            last = final and end >= total
            if not final and end > usable:
                return
            cache_size = self._pick(cfg.pre_encode_cache_size, first)
            start = max(0, self._frame_idx - cache_size)
            pre = features[:, :, start : self._frame_idx]
            pad = cache_size - pre.shape[-1]
            body = features[:, :, self._frame_idx : end]
            parts = [pre, body]
            if pad > 0:
                parts.insert(0, torch.zeros_like(features[:, :, :pad]))
            signal = torch.cat(parts, dim=-1)
            length = torch.tensor([signal.shape[-1]], device=self._device)
            with torch.inference_mode():
                (
                    self._pred_out,
                    texts,
                    self._cache_channel,
                    self._cache_time,
                    self._cache_channel_len,
                    self._hypotheses,
                ) = self._model.conformer_stream_step(
                    processed_signal=signal,
                    processed_signal_length=length,
                    cache_last_channel=self._cache_channel,
                    cache_last_time=self._cache_time,
                    cache_last_channel_len=self._cache_channel_len,
                    keep_all_outputs=last,
                    previous_hypotheses=self._hypotheses,
                    previous_pred_out=self._pred_out,
                    drop_extra_pre_encoded=0 if first else cfg.drop_extra_pre_encoded,
                    return_transcription=True,
                )
            if texts:
                self._text = _hypothesis_text(list(texts))
            self._frame_idx += shift
            self._step += 1
            if last:
                return

    def feed(self, chunk: numpy.ndarray) -> None:
        """Append `chunk` and run the encoder steps it completes."""
        import numpy as np

        self._audio = np.concatenate([self._audio, _float32(chunk)])
        self._run_steps(final=False)

    def finish(self) -> str:
        """Run the remaining frames, return the final text and reset for the next utterance."""
        self._run_steps(final=True)
        text = self._text
        self.reset()
        return text

    def transcribe(self, audio: numpy.ndarray, sample_rate: int) -> str:
        """Stream the whole utterance unpaced (used for warm-ups and run 5's memory loop)."""
        _require_16k(sample_rate)
        self.reset()
        self.feed(audio)
        return self.finish()


# ---------------------------------------------------------------------------
# The factory
# ---------------------------------------------------------------------------


def engine_config(engine_id: str) -> dict:
    """The engine's entry in speech_arms.json's `stt` section."""
    engines = core.load_json(core.SPEECH_ARMS_PATH)["stt"]
    if engine_id not in engines:
        raise KeyError(f"unknown stt engine {engine_id!r}; known: {list(engines)}")
    return engines[engine_id]


def weights_path(engine_id: str, weights_dir: Path) -> Path:
    if engine_id not in WEIGHT_PATHS:
        raise KeyError(f"unknown stt engine {engine_id!r}; known: {list(WEIGHT_PATHS)}")
    return weights_dir / WEIGHT_PATHS[engine_id]


def _load_whisper(path: Path, threads: int):
    import whisper

    # A checkpoint path rather than a model name, so nothing is downloaded; fp32 by default.
    return WhisperTranscriber(whisper.load_model(str(path), device="cuda"))


def _restore_nemo(path: Path):
    from nemo.collections.asr.models import ASRModel

    model = ASRModel.restore_from(str(path), map_location="cuda")
    model.eval()
    return model


def _load_parakeet_gpu(path: Path, threads: int):
    return NemoTranscriber(_restore_nemo(path))


def _load_nemotron(path: Path, threads: int):
    model = _restore_nemo(path)
    model.encoder.set_default_att_context_size(NEMOTRON_ATT_CONTEXT)
    model.encoder.setup_streaming_params()
    return NemoStreamingTranscriber(model)


def _load_parakeet_cpu(path: Path, threads: int):
    import onnx_asr
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    model = onnx_asr.load_model(
        ONNX_ASR_PARAKEET_ID,
        path=str(path),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    return OnnxAsrTranscriber(model)


_LOADERS: dict[str, Callable[[Path, int], object]] = {
    "whisper-base": _load_whisper,
    "parakeet-gpu": _load_parakeet_gpu,
    "parakeet-cpu": _load_parakeet_cpu,
    "nemotron-160": _load_nemotron,
}


def make_transcriber(engine_id: str, *, weights_dir: Path, threads: int = MAX_CPU_THREADS):
    """Load `engine_id` from `weights_dir` and return its adapter.

    `threads` bounds the CPU engines' intra-op threads, at most 4 as run 3 and the chain run them.
    """
    path = weights_path(engine_id, weights_dir)
    if not 1 <= threads <= MAX_CPU_THREADS:
        raise ValueError(f"CPU engines run on 1 to {MAX_CPU_THREADS} threads, not {threads}")
    return _LOADERS[engine_id](path, threads)


def is_streaming(engine) -> bool:
    """Whether the adapter takes chunks (`feed` and `finish`), as `nemotron-160`'s does."""
    return callable(getattr(engine, "feed", None)) and callable(getattr(engine, "finish", None))
