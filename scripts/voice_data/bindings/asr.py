"""The speech recognisers behind the `Asr` protocol: WhisperX, CrisperWhisper and Whisper turbo.

`load` returns the clean stage's pair, the timing ASR first, then the verbatim one:

- WhisperX's recipe, taken apart: faster-whisper's `WhisperModel` transcribes from the pinned
  snapshot, then `whisperx.alignment` aligns the words with the torchaudio wav2vec2 checkpoint,
  which torchaudio reads from `$TORCH_HOME/hub/checkpoints`, where `fetch` put it as a `url`
  model. `whisperx.load_model` is not used, because it also loads a VAD model whose weights the
  cache may not hold.
- CrisperWhisper, in its verbatim mode, keeps the disfluencies the training text needs.

`load_turbo` loads Whisper large-v3-turbo from the score family through transformers' ASR
pipeline. No seed stage calls it; the take gate's CER (T15, T16) will.

Each model takes 16 kHz audio, resampled here. Words whose times the aligner could not find
(numbers, often) take the previous word's end, so a speech rate still counts them.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..stages.clean import resample
from ..stages.fetch import (
    EXTRACT,
    SCORE,
    cache_file,
    library_errors,
    model_dir,
    prepare,
)
from .protocols import Transcript, Word

WHISPERX_MODEL = "whisperx-asr-large-v3"
ALIGN_MODEL = "whisperx-align-en"
CRISPER_MODEL = "crisperwhisper"
TURBO_MODEL = "whisper-large-v3-turbo"
LANGUAGE = "en"
RATE = 16000


def _clip(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    return np.ascontiguousarray(resample(audio, sample_rate, RATE), dtype=np.float32)


def aligned_words(segments: Sequence[Mapping[str, object]]) -> tuple[Word, ...]:
    """Return the aligned words in order; a word without times takes the previous word's end."""
    words: list[Word] = []
    for segment in segments:
        previous = float(segment.get("start") or 0.0)  # type: ignore[arg-type]
        for item in segment.get("words") or []:  # type: ignore[union-attr]
            start = item.get("start")
            end = item.get("end")
            start_s = previous if start is None else float(start)
            end_s = start_s if end is None else float(end)
            words.append(Word(str(item["word"]), start_s, end_s))
            previous = end_s
    return tuple(words)


class WhisperX:
    """faster-whisper's text with WhisperX's word alignment."""

    name = "whisperx"

    def __init__(
        self, model: object, align_model: object, metadata: object, align: object, device: str
    ) -> None:
        self._model = model
        self._align_model = align_model
        self._metadata = metadata
        self._align = align
        self._device = device

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        """Return the text and its aligned words."""
        clip = _clip(audio, sample_rate)
        with library_errors(WHISPERX_MODEL):
            segments, _info = self._model.transcribe(  # type: ignore[attr-defined]
                clip, language=LANGUAGE, condition_on_previous_text=False
            )
            pieces = [
                {"text": str(s.text), "start": float(s.start), "end": float(s.end)}
                for s in segments
            ]
            if not pieces:
                return Transcript(text="", words=())
            result = self._align(  # type: ignore[operator]
                pieces,
                self._align_model,
                self._metadata,
                clip,
                self._device,
                return_char_alignments=False,
                print_progress=False,
            )
        text = " ".join(p["text"].strip() for p in pieces).strip()  # type: ignore[attr-defined]
        return Transcript(text=text, words=aligned_words(result["segments"]))


class CrisperWhisper:
    """CrisperWhisper's verbatim text; its word times are not asked for."""

    name = "crisperwhisper"

    def __init__(self, model: object) -> None:
        self._model = model

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        """Return the verbatim text, disfluencies kept."""
        clip = _clip(audio, sample_rate)
        with library_errors(CRISPER_MODEL):
            result = self._model.transcribe(  # type: ignore[attr-defined]
                clip, language=LANGUAGE, mode="verbatim", sr=RATE
            )
        return Transcript(text=str(result.text).strip(), words=())


class WhisperTurbo:
    """Whisper large-v3-turbo's text, for CER."""

    name = TURBO_MODEL

    def __init__(self, pipe: object) -> None:
        self._pipe = pipe

    def transcribe(self, audio: np.ndarray, sample_rate: int) -> Transcript:
        """Return the text; this engine gives no word times here."""
        with library_errors(TURBO_MODEL):
            out = self._pipe(  # type: ignore[operator]
                {"raw": _clip(audio, sample_rate), "sampling_rate": RATE},
                generate_kwargs={"language": LANGUAGE, "task": "transcribe"},
            )
        if isinstance(out, list):
            out = out[0]
        return Transcript(text=str(out["text"]).strip(), words=())


def load_whisperx(device: str) -> WhisperX:
    """Load faster-whisper from its snapshot and the English alignment model from `TORCH_HOME`."""
    prepare(device, "whisperx", "faster_whisper")
    folder = str(model_dir(EXTRACT, WHISPERX_MODEL))
    cache_file(EXTRACT, ALIGN_MODEL)
    with library_errors(WHISPERX_MODEL):
        from faster_whisper import WhisperModel
        from whisperx.alignment import align, load_align_model
        from whisperx.log_utils import setup_logging

        setup_logging(level="error")
        model = WhisperModel(folder, device=device, compute_type="default", local_files_only=True)
        align_model, metadata = load_align_model(LANGUAGE, device)
    return WhisperX(model, align_model, metadata, align, device)


def load_crisper(device: str) -> CrisperWhisper:
    """Load CrisperWhisper from its snapshot; float16 needs the GPU."""
    prepare(device, "crisperwhisper", "transformers")
    folder = str(model_dir(EXTRACT, CRISPER_MODEL))
    with library_errors(CRISPER_MODEL):
        from crisperwhisper import CrisperWhisperModel

        compute_type = "float16" if device == "cuda" else "float32"
        model = CrisperWhisperModel(folder, device=device, compute_type=compute_type)
    return CrisperWhisper(model)


def load(device: str) -> tuple[WhisperX, CrisperWhisper]:
    """Return the clean stage's ASRs: the timing one (WhisperX), then the verbatim one."""
    return load_whisperx(device), load_crisper(device)


def load_turbo(device: str) -> WhisperTurbo:
    """Load Whisper large-v3-turbo from the score family's snapshot."""
    prepare(device, "transformers")
    folder = str(model_dir(SCORE, TURBO_MODEL))
    with library_errors(TURBO_MODEL):
        import torch
        from transformers import pipeline

        dtype = torch.float16 if device == "cuda" else torch.float32
        pipe = pipeline("automatic-speech-recognition", model=folder, device=device, dtype=dtype)
    return WhisperTurbo(pipe)
