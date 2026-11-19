"""Tests for the Nemotron streaming (GPU) and Parakeet (CPU) speech-to-text adapters.

Neither NeMo nor onnx-asr is installed where these tests run, so each adapter is handed a fake
library: a restore call or a model factory that records what it was given, and a model that
records every call and answers with scripted text. torch is real (the backend image and CI both
install it) and runs on the CPU, so the Nemotron stream's tensor slicing is the code under test,
not a stand-in. Audio is synthetic sine tones; transcripts carry a distinctive mark that must
never reach a log record or an error.
"""

import gc
import hashlib
import importlib
import logging
import os
import sys
import weakref
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

from backend.realtime.budget import PlacementConfig
from backend.realtime.engines import nemotron_stt, parakeet_stt
from backend.realtime.engines.nemotron_stt import (
    FilePin,
    NemoLibrary,
    NemotronStreamingTranscriber,
    stable_prefix_chars,
)
from backend.realtime.engines.parakeet_stt import OnnxAsrLibrary, ParakeetTranscriber
from backend.realtime.errors import RealtimeError, SpeechEngineError
from backend.realtime.lanes import LaneJobCancelled
from backend.realtime.speech import (
    CancelToken,
    StreamingTranscriber,
    TranscriptUpdate,
    WordTiming,
)
from scripts.model_bench.speech.probes import stt_engines as probe

RATE = 16000
FRAME = RATE * 20 // 1000
"""Samples in one 20 ms uplink frame, the size ingress pushes."""
CHUNK = RATE * 160 // 1000
"""Samples in one 160 ms streaming chunk."""
HOP = 160
"""Samples per feature frame of the fake preprocessor (10 ms)."""
MARK = "zebra quartz"
"""A distinctive transcript string no log record or error text may hold."""
WORDS = ("please", "zebra", "quartz", "turn", "the", "lights", "on", "now", "thanks", "again")
LOGGERS = (nemotron_stt.__name__, parakeet_stt.__name__)
NEMO_BYTES = b"synthetic nemo archive"


def tone(seconds: float, frequency: float = 220.0) -> np.ndarray:
    t = np.arange(int(seconds * RATE)) / RATE
    return (0.2 * np.sin(2 * np.pi * frequency * t)).astype(np.float32)


def push_frames(stream, audio: np.ndarray) -> None:
    for start in range(0, len(audio), FRAME):
        stream.push(audio[start : start + FRAME])


def _chain_text(exc: BaseException) -> str:
    """The text of an exception and of every exception a traceback of it would print."""
    parts = []
    seen: BaseException | None = exc
    while seen is not None:
        parts.append(f"{type(seen).__name__}: {seen}")
        seen = seen.__cause__ or (None if seen.__suppress_context__ else seen.__context__)
    return "\n".join(parts)


def _assert_private(records: list[logging.LogRecord], *marks: str) -> None:
    for record in records:
        message = record.getMessage()
        for mark in marks:
            assert mark not in message, message
        args = record.args if isinstance(record.args, tuple) else (record.args,)
        assert not [arg for arg in args if isinstance(arg, bytes | bytearray | np.ndarray)]
        if record.exc_info and record.exc_info[1] is not None:
            for mark in marks:
                assert mark not in _chain_text(record.exc_info[1])


# ---------------------------------------------------------------------------
# The fake NeMo library
# ---------------------------------------------------------------------------


class FakeHypothesis:
    """A NeMo hypothesis: how many encoder steps it has seen, and its text."""

    def __init__(self, steps: int, text: str) -> None:
        self.steps = steps
        self.text = text


@dataclass
class Step:
    """One `conformer_stream_step` call as the fake model saw it."""

    signal: torch.Tensor
    length: int
    keep_all_outputs: bool
    drop_extra_pre_encoded: int
    previous_steps: int
    cache: float


class FakePreprocessor:
    """Mean of each 10 ms hop, on two channels: features a slice can be checked against."""

    def __init__(self, lengths: list[int]) -> None:
        self.lengths = lengths

    def to(self, device):
        return self

    def __call__(self, *, input_signal: torch.Tensor, length: torch.Tensor):
        samples = int(length[0])
        self.lengths.append(samples)
        frames = samples // HOP
        hops = input_signal[0, : frames * HOP].reshape(frames, HOP).mean(dim=1)
        features = torch.stack([hops, -hops]).unsqueeze(0)
        return features, torch.tensor([frames])


class FakeEncoder:
    def __init__(self) -> None:
        self.att_context: list[int] | None = None
        self.streaming_setups = 0
        self.streaming_cfg = SimpleNamespace(
            chunk_size=[9, 16],
            shift_size=[9, 16],
            pre_encode_cache_size=[0, 9],
            drop_extra_pre_encoded=2,
        )

    def set_default_att_context_size(self, att_context_size) -> None:
        self.att_context = list(att_context_size)

    def setup_streaming_params(self) -> None:
        self.streaming_setups += 1

    def get_initial_cache_state(self, batch_size: int = 1):
        assert batch_size == 1
        return torch.zeros(1), torch.zeros(1), torch.zeros(1, dtype=torch.long)


class FakeNemoModel:
    """A cache-aware streaming model: one more word of `words` per encoder step.

    `replies`, when given, scripts the text after each step instead. The last step of a flush
    (`keep_all_outputs`) yields the whole text. `on_step` runs inside each step, before it
    returns, so a test can close the stream or set a token while a decode is in flight.
    """

    def __init__(self, words=WORDS, replies: list[str] | None = None) -> None:
        self.words = tuple(words)
        self.replies = replies
        self.encoder = FakeEncoder()
        # `window_stride` is the 10 ms hop `FakePreprocessor` frames at, in seconds.
        self._cfg = SimpleNamespace(
            preprocessor=SimpleNamespace(
                dither=1e-5, pad_to=16, sample_rate=RATE, window_stride=HOP / RATE
            )
        )
        self.preprocessor_cfgs: list[SimpleNamespace] = []
        self.feature_lengths: list[int] = []
        self.steps: list[Step] = []
        self.evals = 0
        self.fail_step: BaseException | None = None
        self.on_step: Callable[[], object] | None = None
        self._weight = torch.zeros(1)

    def parameters(self):
        return iter([self._weight])

    def eval(self):
        self.evals += 1
        return self

    def from_config_dict(self, cfg):
        self.preprocessor_cfgs.append(cfg)
        return FakePreprocessor(self.feature_lengths)

    def _text(self, steps: int, complete: bool) -> str:
        if self.replies is not None:
            return self.replies[min(steps, len(self.replies)) - 1]
        return " ".join(self.words if complete else self.words[:steps])

    def conformer_stream_step(
        self,
        *,
        processed_signal,
        processed_signal_length,
        cache_last_channel,
        cache_last_time,
        cache_last_channel_len,
        keep_all_outputs,
        previous_hypotheses,
        previous_pred_out,
        drop_extra_pre_encoded,
        return_transcription,
    ):
        assert return_transcription is True
        previous = previous_hypotheses[0].steps if previous_hypotheses else 0
        self.steps.append(
            Step(
                processed_signal.clone(),
                int(processed_signal_length[0]),
                keep_all_outputs,
                drop_extra_pre_encoded,
                previous,
                float(cache_last_channel[0]),
            )
        )
        if self.on_step is not None:
            self.on_step()
        if self.fail_step is not None:
            raise self.fail_step
        hypothesis = FakeHypothesis(previous + 1, self._text(previous + 1, keep_all_outputs))
        return (
            torch.ones(1),
            [hypothesis],
            cache_last_channel + 1,
            cache_last_time,
            cache_last_channel_len,
            [hypothesis],
        )


@dataclass
class FakeNemoLibrary:
    """Builds the `NemoLibrary` the adapter loads with, and records each restore."""

    model: FakeNemoModel | None = field(default_factory=FakeNemoModel)
    restores: list[tuple[str, str]] = field(default_factory=list)
    fail_restore: BaseException | None = None
    loads: int = 0
    models: list[FakeNemoModel] = field(default_factory=list)

    def restore_from(self, restore_path: str, map_location: str) -> FakeNemoModel:
        assert isinstance(restore_path, str)
        self.restores.append((restore_path, map_location))
        if self.fail_restore is not None:
            raise self.fail_restore
        model, self.model = self.model, None
        assert model is not None
        self.models.append(model)
        return model

    def __call__(self) -> NemoLibrary:
        self.loads += 1
        return NemoLibrary(restore_from=self.restore_from, torch=torch)


def nemo_path(models_dir: Path) -> Path:
    return (
        models_dir
        / "speech"
        / "nvidia"
        / "nemotron-speech-streaming-en-0.6b"
        / "nemotron-speech-streaming-en-0.6b.nemo"
    )


@dataclass
class NemotronEnv:
    models_dir: Path
    library: FakeNemoLibrary
    pin: FilePin

    @property
    def path(self) -> Path:
        return nemo_path(self.models_dir)

    @property
    def model(self) -> FakeNemoModel:
        return self.library.models[-1]

    def load(self, **kwargs) -> NemotronStreamingTranscriber:
        kwargs.setdefault("model_pin", self.pin)
        return NemotronStreamingTranscriber.load(
            PlacementConfig(models_dir=str(self.models_dir)),
            library_loader=self.library,
            **kwargs,
        )

    def fail_decode(self, error: BaseException) -> None:
        self.model.fail_step = error

    def during_decode(self, action: Callable[[], object]) -> None:
        self.model.on_step = action

    def model_calls(self) -> int:
        return len(self.model.steps)


def make_nemotron(tmp_path: Path, model: FakeNemoModel | None = None) -> NemotronEnv:
    models_dir = tmp_path / "models"
    path = nemo_path(models_dir)
    path.parent.mkdir(parents=True)
    path.write_bytes(NEMO_BYTES)
    pin = FilePin(size=len(NEMO_BYTES), sha256=hashlib.sha256(NEMO_BYTES).hexdigest())
    library = FakeNemoLibrary(model=model or FakeNemoModel())
    return NemotronEnv(models_dir, library, pin)


# ---------------------------------------------------------------------------
# The fake onnx-asr library
# ---------------------------------------------------------------------------


class FakeSessionOptions:
    """`onnxruntime.SessionOptions`: the thread settings the adapter sets."""

    def __init__(self) -> None:
        self.intra_op_num_threads = 0
        self.inter_op_num_threads = 0


class FakeOrtFail(Exception):  # noqa: N818
    """Stands in for onnxruntime's pybind errors, which derive from `Exception` alone."""


@dataclass
class FakeTimestampedResult:
    """onnx-asr's `TimestampedResult`: the text, each token's start in seconds, the tokens.

    Tokens are as the library returns them, its vocabulary's "▁" already read as a space.
    """

    text: str
    timestamps: list[float] | None = None
    tokens: list[str] | None = None
    logprobs: list[float] | None = None


def timed(text: str, step_s: float = 0.2) -> FakeTimestampedResult:
    """A result whose tokens are the text's words, each led by a space, `step_s` apart."""
    words = text.split()
    return FakeTimestampedResult(
        text,
        [round(index * step_s, 2) for index in range(len(words))],
        [f" {word}" for word in words],
    )


class FakeParakeetModel:
    """onnx-asr's timestamped adapter: one word of `words` per 200 ms of audio, or replies.

    A reply is a result, or a text whose words become its tokens 200 ms apart.
    """

    def __init__(
        self, words=WORDS, replies: list[str | FakeTimestampedResult] | None = None
    ) -> None:
        self.words = tuple(words)
        self.replies = replies
        self.calls: list[tuple[np.ndarray, dict]] = []
        self.fail: BaseException | None = None
        self.on_recognize: Callable[[], object] | None = None

    def recognize(self, waveform, **kwargs) -> FakeTimestampedResult:
        self.calls.append((np.array(waveform, copy=True), kwargs))
        if self.on_recognize is not None:
            self.on_recognize()
        if self.fail is not None:
            raise self.fail
        if self.replies is not None:
            reply = self.replies[min(len(self.calls), len(self.replies)) - 1]
            return timed(reply) if isinstance(reply, str) else reply
        return timed(" ".join(self.words[: len(waveform) * 5 // RATE]))


class FakeTextModel:
    """onnx-asr's text adapter, which `load_model` returns: only `with_timestamps` is called."""

    def __init__(self, library: "FakeOnnxAsrLibrary", timestamped: FakeParakeetModel) -> None:
        self._library = library
        self._timestamped: FakeParakeetModel | None = timestamped

    def with_timestamps(self) -> FakeParakeetModel:
        self._library.timestamp_calls += 1
        model, self._timestamped = self._timestamped, None
        assert model is not None
        return model

    def recognize(self, waveform, **kwargs):
        raise AssertionError("the text adapter was called; only timestamped results carry words")


@dataclass
class FakeOnnxAsrLibrary:
    model: FakeParakeetModel | None = field(default_factory=FakeParakeetModel)
    loads: int = 0
    calls: list[tuple[tuple, dict]] = field(default_factory=list)
    fail_load: BaseException | None = None
    models: list[FakeParakeetModel] = field(default_factory=list)
    timestamp_calls: int = 0

    def load_model(self, *args, **kwargs) -> FakeTextModel:
        self.calls.append((args, kwargs))
        if self.fail_load is not None:
            raise self.fail_load
        model, self.model = self.model, None
        assert model is not None
        self.models.append(model)
        return FakeTextModel(self, model)

    def __call__(self) -> OnnxAsrLibrary:
        self.loads += 1
        return OnnxAsrLibrary(
            load_model=self.load_model, session_options=FakeSessionOptions, errors=(FakeOrtFail,)
        )


def onnx_dir(models_dir: Path) -> Path:
    return models_dir / "speech" / "istupakov" / "parakeet-tdt-0.6b-v2-onnx"


@dataclass
class ParakeetEnv:
    models_dir: Path
    library: FakeOnnxAsrLibrary

    @property
    def path(self) -> Path:
        return onnx_dir(self.models_dir)

    @property
    def model(self) -> FakeParakeetModel:
        return self.library.models[-1]

    def load(self, **kwargs) -> ParakeetTranscriber:
        return ParakeetTranscriber.load(
            PlacementConfig(models_dir=str(self.models_dir)), library_loader=self.library, **kwargs
        )

    def fail_decode(self, error: BaseException) -> None:
        self.model.fail = error

    def during_decode(self, action: Callable[[], object]) -> None:
        self.model.on_recognize = action

    def model_calls(self) -> int:
        return len(self.model.calls)


def make_parakeet(tmp_path: Path, model: FakeParakeetModel | None = None) -> ParakeetEnv:
    models_dir = tmp_path / "models"
    weights = onnx_dir(models_dir)
    weights.mkdir(parents=True)
    for name in ("encoder-model.onnx", "decoder_joint-model.onnx", "vocab.txt"):
        (weights / name).write_bytes(b"synthetic")
    return ParakeetEnv(models_dir, FakeOnnxAsrLibrary(model=model or FakeParakeetModel()))


@pytest.fixture(params=["nemotron", "parakeet"])
def env(request, tmp_path):
    if request.param == "nemotron":
        return make_nemotron(tmp_path)
    return make_parakeet(tmp_path)


# ---------------------------------------------------------------------------
# Rules both adapters share
# ---------------------------------------------------------------------------


class TestStablePrefix:
    @pytest.mark.parametrize(
        ("previous", "current", "expected"),
        [
            ("", "turn on", 0),
            ("turn on", "turn on the", len("turn on")),
            ("turn on the", "turn off the lights", len("turn")),
            ("turn on the", "turn on the", len("turn on the")),
            ("lights", "turn on", 0),
            ("turn  on the", "turn  on  a", len("turn  on")),
        ],
    )
    def test_is_the_longest_common_word_prefix_of_the_last_two_partials(
        self, previous, current, expected
    ):
        assert stable_prefix_chars(previous, current) == expected


class TestShared:
    def test_meets_the_contract(self, env):
        transcriber: StreamingTranscriber = env.load()
        stream = transcriber.open_stream()
        push_frames(stream, tone(0.6))

        update = stream.decode_partial(CancelToken())

        assert isinstance(update, TranscriptUpdate)
        assert update.is_final is False
        assert isinstance(transcriber.model_id, str) and transcriber.model_id

    def test_partial_stable_chars_follow_the_last_two_partials(self, tmp_path, env):
        replies = ["turn on", "turn on the", "turn off the lights"]
        if isinstance(env, NemotronEnv):
            env = make_nemotron(tmp_path / "scripted", FakeNemoModel(replies=replies))
            seconds = 0.16
        else:
            env = make_parakeet(tmp_path / "scripted", FakeParakeetModel(replies=replies))
            seconds = 0.2
        stream = env.load().open_stream()

        stables = []
        for _ in replies:
            push_frames(stream, tone(seconds))
            stables.append(stream.decode_partial(CancelToken()))

        assert [update.text for update in stables][-1] == "turn off the lights"
        assert [update.stable_chars for update in stables] == [0, len("turn on"), len("turn")]

    def test_a_final_is_wholly_stable(self, env):
        stream = env.load().open_stream()
        push_frames(stream, tone(0.6))

        final = stream.decode_final(CancelToken())

        assert final.is_final is True
        assert final.text
        assert final.stable_chars == len(final.text)
        assert final.audio_ms == 600

    def test_partials_and_finals_carry_words_that_join_to_their_text(self, env):
        stream = env.load().open_stream()
        updates = []
        for _ in range(4):
            push_frames(stream, tone(0.5))
            updates.append(stream.decode_partial(CancelToken()))
        updates.append(stream.decode_final(CancelToken()))

        for update in updates:
            assert update.words is not None
            assert " ".join(word.text for word in update.words) == " ".join(update.text.split())
            assert all(word.start_ms <= word.end_ms <= update.audio_ms for word in update.words)
        assert updates[-1].words

    def test_whisper_scores_are_none_with_one_info_line_at_load(self, env, caplog):
        caplog.set_level(logging.DEBUG)
        stream = env.load().open_stream()
        push_frames(stream, tone(0.6))

        updates = [stream.decode_partial(CancelToken()), stream.decode_final(CancelToken())]

        assert all(u.no_speech_prob is None and u.avg_logprob is None for u in updates)
        lines = [r for r in caplog.records if "ingress" in r.getMessage().lower()]
        assert len(lines) == 1
        assert lines[0].levelno == logging.INFO
        assert "filter is off" in lines[0].getMessage()

    def test_weights_load_only_under_models_dir_and_no_hub_download_runs(self, env):
        hub = pytest.importorskip("huggingface_hub")
        refuse = mock.Mock(side_effect=AssertionError("hub download"))
        with (
            mock.patch.object(hub, "hf_hub_download", refuse),
            mock.patch.object(hub, "snapshot_download", refuse),
        ):
            env.load()

        refuse.assert_not_called()
        assert env.path.is_relative_to(env.models_dir)

    def test_a_set_token_skips_a_decode_not_yet_started(self, env):
        stream = env.load().open_stream()
        push_frames(stream, tone(0.6))
        token = CancelToken()
        token.set()
        before = env.model_calls()

        for decode in (stream.decode_partial, stream.decode_final):
            with pytest.raises(LaneJobCancelled):
                decode(token)

        assert env.model_calls() == before

    def test_library_errors_at_decode_become_speech_engine_errors_without_text(self, env):
        stream = env.load().open_stream()
        push_frames(stream, tone(0.6))
        env.fail_decode(RuntimeError(f"bad input near {MARK}"))

        for decode in (stream.decode_partial, stream.decode_final):
            with pytest.raises(SpeechEngineError) as caught:
                decode(CancelToken())
            assert MARK not in _chain_text(caught.value)

    def test_a_decode_after_close_is_cancelled_and_a_push_is_refused(self, env):
        stream = env.load().open_stream()
        push_frames(stream, tone(0.6))

        stream.close()
        stream.close()

        for decode in (stream.decode_partial, stream.decode_final):
            with pytest.raises(LaneJobCancelled):
                decode(CancelToken())
        with pytest.raises(RealtimeError):
            stream.push(tone(0.02))

    def test_a_close_during_a_decode_discards_its_result(self, env):
        stream = env.load().open_stream()
        push_frames(stream, tone(0.6))
        env.during_decode(stream.close)

        with pytest.raises(LaneJobCancelled):
            stream.decode_partial(CancelToken())

    def test_release_closes_open_streams_drops_the_model_and_refuses_later_calls(self, env):
        transcriber = env.load()
        stream = transcriber.open_stream()
        push_frames(stream, tone(0.6))
        stream.decode_partial(CancelToken())
        model_ref = weakref.ref(env.library.models.pop())
        gc.collect()
        assert model_ref() is not None

        transcriber.release()
        transcriber.release()
        gc.collect()

        assert model_ref() is None
        with pytest.raises(SpeechEngineError, match="released"):
            transcriber.open_stream()
        with pytest.raises(SpeechEngineError, match="released"):
            stream.push(tone(0.02))
        for decode in (stream.decode_partial, stream.decode_final):
            with pytest.raises(LaneJobCancelled):
                decode(CancelToken())

    def test_a_stream_closed_before_release_is_not_reopened(self, env):
        transcriber = env.load()
        closed = transcriber.open_stream()
        closed.close()

        transcriber.release()

        with pytest.raises(RealtimeError) as caught:
            closed.push(tone(0.02))
        assert not isinstance(caught.value, SpeechEngineError)

    def test_no_record_or_error_holds_text_or_audio(self, env, caplog):
        caplog.set_level(logging.DEBUG)
        transcriber = env.load()
        stream = transcriber.open_stream()
        push_frames(stream, tone(2.0))
        stream.decode_partial(CancelToken())
        final = stream.decode_final(CancelToken())
        assert MARK in final.text
        env.fail_decode(ValueError(f"cannot decode {MARK}"))
        with pytest.raises(SpeechEngineError) as caught:
            stream.decode_final(CancelToken())
        stream.close()
        transcriber.release()

        assert caplog.records
        assert {r.name for r in caplog.records} <= set(LOGGERS)
        _assert_private(caplog.records, MARK)
        assert MARK not in _chain_text(caught.value)


class TestLazyImport:
    BLOCKED = (
        "nemo",
        "nemo.collections",
        "nemo.collections.asr",
        "nemo.collections.asr.models",
        "torch",
        "onnx_asr",
        "onnxruntime",
        "onnxruntime.capi",
        "onnxruntime.capi.onnxruntime_pybind11_state",
    )

    def test_importing_the_modules_loads_no_engine_package(self, monkeypatch):
        for blocked in self.BLOCKED:
            monkeypatch.setitem(sys.modules, blocked, None)
        for name in LOGGERS:
            monkeypatch.delitem(sys.modules, name)

        nemotron = importlib.import_module(nemotron_stt.__name__)
        parakeet = importlib.import_module(parakeet_stt.__name__)

        assert nemotron.NemotronStreamingTranscriber.__name__ == "NemotronStreamingTranscriber"
        assert parakeet.ParakeetTranscriber.__name__ == "ParakeetTranscriber"
        with pytest.raises(SpeechEngineError, match="nemo"):
            nemotron.import_nemo()
        with pytest.raises(SpeechEngineError, match="onnx-asr"):
            parakeet.import_onnx_asr()

    def test_load_with_a_fake_library_imports_no_engine_package(self, tmp_path, monkeypatch):
        for blocked in ("nemo", "onnx_asr", "onnxruntime"):
            monkeypatch.setitem(sys.modules, blocked, None)

        assert make_nemotron(tmp_path / "n").load().engine_id == "nemotron-160"
        assert make_parakeet(tmp_path / "p").load().engine_id == "parakeet-cpu"


# ---------------------------------------------------------------------------
# Nemotron: load and the file pin
# ---------------------------------------------------------------------------


class TestNemotronLoad:
    def test_restores_the_nemo_file_on_the_device_with_the_probes_call(self, tmp_path):
        env = make_nemotron(tmp_path)

        transcriber = env.load()

        assert env.library.restores == [(str(env.path), "cuda")]
        assert env.model.evals == 1
        assert transcriber.engine_id == "nemotron-160"
        assert transcriber.model_id == (
            f"nvidia/nemotron-speech-streaming-en-0.6b@sha256:{nemotron_stt.MODEL_SHA256}"
        )

    def test_sets_160_ms_streaming_as_the_probe_does(self, tmp_path):
        env = make_nemotron(tmp_path)

        env.load()

        assert env.model.encoder.att_context == list(probe.NEMOTRON_ATT_CONTEXT) == [70, 1]
        assert env.model.encoder.streaming_setups == 1
        (cfg,) = env.model.preprocessor_cfgs
        assert (cfg.dither, cfg.pad_to) == (0.0, 0)
        assert (env.model._cfg.preprocessor.dither, env.model._cfg.preprocessor.pad_to) == (
            1e-5,
            16,
        )

    @pytest.mark.parametrize("device", ["cpu", "cuda:0", "mps"])
    def test_accepts_cuda_only(self, tmp_path, device):
        env = make_nemotron(tmp_path)

        with pytest.raises(SpeechEngineError, match=device):
            env.load(device=device)

        assert env.library.loads == 0

    def test_the_default_pin_is_the_recorded_file(self):
        assert NemotronStreamingTranscriber.load.__kwdefaults__["model_pin"] == FilePin(
            size=2473041920,
            sha256="283638054c44f6794e74fe9af9048d78a6d9d6c058c12131856c7859a62ac9cd",
        )

    def test_a_file_with_another_sha256_raises_naming_its_path(self, tmp_path):
        env = make_nemotron(tmp_path)
        env.path.write_bytes(b"synthetic nemo archivE")

        with pytest.raises(SpeechEngineError, match="sha256") as caught:
            env.load()

        assert str(env.path) in str(caught.value)
        assert env.library.loads == 0

    def test_a_file_of_another_size_raises_naming_its_path_without_hashing(self, tmp_path):
        env = make_nemotron(tmp_path)
        env.path.write_bytes(NEMO_BYTES + b"!")

        with (
            mock.patch.object(nemotron_stt, "_sha256", side_effect=AssertionError("hashed")),
            pytest.raises(SpeechEngineError, match="bytes") as caught,
        ):
            env.load()

        assert str(env.path) in str(caught.value)

    def test_a_missing_file_raises_naming_its_path(self, tmp_path):
        env = make_nemotron(tmp_path)
        env.path.unlink()

        with pytest.raises(SpeechEngineError) as caught:
            env.load()

        assert str(env.path) in str(caught.value)
        assert env.library.loads == 0

    def test_a_verified_file_is_hashed_once_per_process(self, tmp_path):
        env = make_nemotron(tmp_path)
        real = nemotron_stt._sha256
        with mock.patch.object(nemotron_stt, "_sha256", side_effect=real) as hashed:
            env.load().release()
            env.library.model = FakeNemoModel()
            env.load()

        assert hashed.call_count == 1

    def test_a_changed_file_is_hashed_again(self, tmp_path):
        env = make_nemotron(tmp_path)
        env.load().release()
        env.library.model = FakeNemoModel()
        before = env.path.stat().st_mtime_ns
        env.path.write_bytes(b"synthetic nemo archivE")
        later = before + 1_000_000_000
        os.utime(env.path, ns=(later, later))

        with pytest.raises(SpeechEngineError, match="sha256"):
            env.load()

    @pytest.mark.parametrize(
        "error", [RuntimeError("cuda"), OSError("archive"), KeyError("cfg"), AttributeError("x")]
    )
    def test_library_load_errors_become_speech_engine_errors(self, tmp_path, error):
        env = make_nemotron(tmp_path)
        env.library.fail_restore = error

        with pytest.raises(SpeechEngineError, match="nemotron-160") as caught:
            env.load()

        assert caught.value.__cause__ is error


# ---------------------------------------------------------------------------
# Nemotron: streaming
# ---------------------------------------------------------------------------


def run_probe(audio: np.ndarray) -> tuple[FakeNemoModel, str]:
    """The bench probe's streaming adapter, fed 160 ms chunks as `stt_probe.stream_timed` does."""
    model = FakeNemoModel()
    engine = probe.NemoStreamingTranscriber(model)
    for start in range(0, len(audio), CHUNK):
        engine.feed(audio[start : start + CHUNK])
    return model, engine.finish()


def _same_steps(left: list[Step], right: list[Step]) -> None:
    assert len(left) == len(right)
    for ours, theirs in zip(left, right):
        assert torch.equal(ours.signal, theirs.signal)
        assert (ours.length, ours.keep_all_outputs, ours.drop_extra_pre_encoded) == (
            theirs.length,
            theirs.keep_all_outputs,
            theirs.drop_extra_pre_encoded,
        )
        assert (ours.previous_steps, ours.cache) == (theirs.previous_steps, theirs.cache)


class TestNemotronStreaming:
    def test_push_only_buffers_and_partials_feed_whole_160_ms_chunks(self, tmp_path):
        env = make_nemotron(tmp_path)
        stream = env.load().open_stream()

        push_frames(stream, tone(0.5))
        assert env.model.feature_lengths == []
        assert env.model.steps == []

        update = stream.decode_partial(CancelToken())

        assert env.model.feature_lengths == [CHUNK, 2 * CHUNK, 3 * CHUNK]
        assert update.audio_ms == 480
        assert update.text == " ".join(WORDS[: len(env.model.steps)])

    def test_matches_the_probe_step_for_step_whatever_the_partial_cadence(self, tmp_path):
        audio = tone(1.37, 330.0) + tone(1.37, 95.0)
        probe_model, probe_text = run_probe(audio)
        env = make_nemotron(tmp_path)
        stream = env.load().open_stream()

        for index, start in enumerate(range(0, len(audio), FRAME)):
            stream.push(audio[start : start + FRAME])
            if index % 7 == 3:
                stream.decode_partial(CancelToken())
        final = stream.decode_final(CancelToken())

        _same_steps(env.model.steps, probe_model.steps)
        assert env.model.feature_lengths == probe_model.feature_lengths
        assert final.text == probe_text
        assert [step.keep_all_outputs for step in env.model.steps][-1] is True

    def test_a_partial_without_new_audio_returns_the_hypothesis_without_decoding(self, tmp_path):
        env = make_nemotron(tmp_path)
        stream = env.load().open_stream()
        push_frames(stream, tone(0.5))
        first = stream.decode_partial(CancelToken())
        steps, features = len(env.model.steps), len(env.model.feature_lengths)

        push_frames(stream, tone(0.1))
        again = stream.decode_partial(CancelToken())

        assert (len(env.model.steps), len(env.model.feature_lengths)) == (steps, features)
        assert again.text == first.text
        assert again.stable_chars == len(first.text)

    def test_each_partial_runs_only_the_steps_new_audio_completes(self, tmp_path):
        env = make_nemotron(tmp_path)
        stream = env.load().open_stream()
        push_frames(stream, tone(0.8))
        stream.decode_partial(CancelToken())
        before = len(env.model.steps)

        push_frames(stream, tone(0.8))
        stream.decode_partial(CancelToken())

        new = env.model.steps[before:]
        assert new
        assert [step.previous_steps for step in new] == list(range(before, len(env.model.steps)))

    def test_an_utterance_shorter_than_a_chunk_still_gets_a_final(self, tmp_path):
        env = make_nemotron(tmp_path)
        stream = env.load().open_stream()
        push_frames(stream, tone(0.1))

        partial = stream.decode_partial(CancelToken())
        final = stream.decode_final(CancelToken())

        assert (partial.text, partial.audio_ms) == ("", 0)
        assert final.text == " ".join(WORDS)
        assert env.model.feature_lengths == [1600, 1600]

    def test_a_final_of_no_audio_is_empty_and_runs_nothing(self, tmp_path):
        env = make_nemotron(tmp_path)
        stream = env.load().open_stream()

        final = stream.decode_final(CancelToken())

        assert (final.text, final.is_final, final.audio_ms) == ("", True, 0)
        assert env.model.steps == []

    def test_a_final_flushes_without_ending_the_stream(self, tmp_path):
        env = make_nemotron(tmp_path)
        stream = env.load().open_stream()
        push_frames(stream, tone(0.5))
        stream.decode_partial(CancelToken())
        live = len(env.model.steps)

        stream.decode_final(CancelToken())
        flushed = env.model.steps[live:]
        push_frames(stream, tone(0.5))
        resumed = stream.decode_partial(CancelToken())
        resumed_steps = env.model.steps[live + len(flushed) :]

        assert flushed[-1].keep_all_outputs is True
        assert resumed_steps[0].previous_steps == live
        assert not any(step.keep_all_outputs for step in resumed_steps)
        assert resumed.audio_ms == 960
        assert stream.decode_final(CancelToken()).audio_ms == 1000

    def test_a_token_set_between_chunks_stops_and_the_next_decode_resumes(self, tmp_path):
        env = make_nemotron(tmp_path)
        stream = env.load().open_stream()
        push_frames(stream, tone(1.0))
        token = CancelToken()
        env.during_decode(token.set)

        with pytest.raises(LaneJobCancelled):
            stream.decode_partial(token)
        stopped = len(env.model.steps)
        env.during_decode(None)
        stream.decode_partial(CancelToken())

        assert stopped >= 1
        previous = [step.previous_steps for step in env.model.steps]
        assert previous == list(range(len(previous)))

    def test_each_stream_has_its_own_caches(self, tmp_path):
        env = make_nemotron(tmp_path)
        transcriber = env.load()
        first, second = transcriber.open_stream(), transcriber.open_stream()
        push_frames(first, tone(0.5))
        push_frames(second, tone(0.5))

        first.decode_partial(CancelToken())
        count = len(env.model.steps)
        second.decode_partial(CancelToken())

        assert env.model.steps[count].previous_steps == 0
        assert env.model.steps[count].cache == 0.0


def W(text: str, start_ms: int, end_ms: int) -> WordTiming:  # noqa: N802
    return WordTiming(text, start_ms, end_ms)


class TestNemotronWordTimings:
    """The fake's steps cover feature frames [0, 9), then 16 at a time: 0-90 ms, 90-250 ms,
    250-410 ms, 410-570 ms, at the config's 10 ms `window_stride`.
    """

    def test_each_word_spans_the_step_that_emitted_it(self, tmp_path):
        env = make_nemotron(tmp_path)
        stream = env.load().open_stream()
        push_frames(stream, tone(0.5))

        update = stream.decode_partial(CancelToken())

        assert len(env.model.steps) == 3
        assert update.words == (W("please", 0, 90), W("zebra", 90, 250), W("quartz", 250, 410))

    def test_a_changed_word_keeps_its_first_step_and_ends_at_the_last_step_that_changed_it(
        self, tmp_path
    ):
        replies = ["Mi", "Mist,", "Mist, what", "Mist, what's"]
        env = make_nemotron(tmp_path, FakeNemoModel(replies=replies))
        stream = env.load().open_stream()
        push_frames(stream, tone(0.7))

        update = stream.decode_partial(CancelToken())

        assert len(env.model.steps) == 4
        assert update.text == "Mist, what's"
        assert update.words == (W("Mist,", 0, 250), W("what's", 250, 570))

    def test_a_word_no_later_step_changed_keeps_its_span_across_partials(self, tmp_path):
        replies = ["mist", "mist", "mist", "mist stop"]
        env = make_nemotron(tmp_path, FakeNemoModel(replies=replies))
        stream = env.load().open_stream()

        push_frames(stream, tone(0.5))
        first = stream.decode_partial(CancelToken())
        push_frames(stream, tone(0.2))
        second = stream.decode_partial(CancelToken())

        assert first.words == (W("mist", 0, 90),)
        assert second.words == (W("mist", 0, 90), W("stop", 410, 570))

    def test_a_word_a_step_drops_is_dropped(self, tmp_path):
        replies = ["miss it", "mist"]
        env = make_nemotron(tmp_path, FakeNemoModel(replies=replies))
        stream = env.load().open_stream()
        push_frames(stream, tone(0.34))

        update = stream.decode_partial(CancelToken())

        assert len(env.model.steps) == 2
        assert update.words == (W("mist", 0, 250),)

    def test_the_final_stamps_its_flush_on_a_copy_and_the_live_stream_keeps_its_own(self, tmp_path):
        env = make_nemotron(tmp_path)
        stream = env.load().open_stream()
        push_frames(stream, tone(0.5))
        stream.decode_partial(CancelToken())

        final = stream.decode_final(CancelToken())
        push_frames(stream, tone(0.5))
        resumed = stream.decode_partial(CancelToken())

        # The flush's last step covers frames 41 to 50, the audio's end, and yields every word.
        head = (W("please", 0, 90), W("zebra", 90, 250), W("quartz", 250, 410))
        assert final.words == head + tuple(W(word, 410, 500) for word in WORDS[3:])
        assert resumed.words == head + (
            W("turn", 410, 570),
            W("the", 570, 730),
            W("lights", 730, 890),
        )

    def test_a_final_shorter_than_a_chunk_spans_its_one_flush_step(self, tmp_path):
        env = make_nemotron(tmp_path, FakeNemoModel(replies=["mist"]))
        stream = env.load().open_stream()
        push_frames(stream, tone(0.1))

        final = stream.decode_final(CancelToken())

        assert final.words == (W("mist", 0, 90),)

    def test_no_audio_gives_no_words(self, tmp_path):
        env = make_nemotron(tmp_path)
        stream = env.load().open_stream()

        assert stream.decode_partial(CancelToken()).words == ()
        assert stream.decode_final(CancelToken()).words == ()

    def test_a_config_without_window_stride_gives_no_words_and_one_line_at_load(
        self, tmp_path, caplog
    ):
        caplog.set_level(logging.DEBUG)
        model = FakeNemoModel()
        del model._cfg.preprocessor.window_stride
        env = make_nemotron(tmp_path, model)
        stream = env.load().open_stream()
        push_frames(stream, tone(0.5))

        updates = [stream.decode_partial(CancelToken()), stream.decode_final(CancelToken())]

        assert [update.words for update in updates] == [None, None]
        assert updates[0].text == "please zebra quartz"
        lines = [r for r in caplog.records if "window_stride" in r.getMessage()]
        assert len(lines) == 1
        assert lines[0].levelno == logging.WARNING
        assert lines[0].name == nemotron_stt.__name__

    @pytest.mark.parametrize("stride", [0, -0.01, "10ms", None])
    def test_a_window_stride_that_is_not_a_positive_number_counts_as_absent(self, tmp_path, stride):
        model = FakeNemoModel()
        model._cfg.preprocessor.window_stride = stride
        env = make_nemotron(tmp_path, model)
        stream = env.load().open_stream()
        push_frames(stream, tone(0.5))

        assert stream.decode_partial(CancelToken()).words is None


# ---------------------------------------------------------------------------
# Parakeet
# ---------------------------------------------------------------------------


class TestParakeetLoad:
    def test_loads_with_the_probes_call_on_four_threads_per_session(self, tmp_path):
        env = make_parakeet(tmp_path)

        transcriber = env.load()

        ((args, kwargs),) = env.library.calls
        assert args == (probe.ONNX_ASR_PARAKEET_ID,) == ("nemo-parakeet-tdt-0.6b-v2",)
        assert kwargs["path"] == str(env.path)
        assert kwargs["providers"] == ["CPUExecutionProvider"]
        options = kwargs["sess_options"]
        assert (options.intra_op_num_threads, options.inter_op_num_threads) == (4, 1)
        assert set(kwargs) == {"path", "providers", "sess_options"}
        assert transcriber.engine_id == "parakeet-cpu"
        assert transcriber.model_id == "nemo-parakeet-tdt-0.6b-v2"

    def test_load_wraps_the_model_with_timestamps_and_decodes_on_that(self, tmp_path):
        env = make_parakeet(tmp_path)

        stream = env.load().open_stream()
        push_frames(stream, tone(0.4))
        stream.decode_partial(CancelToken())

        assert env.library.timestamp_calls == 1
        assert len(env.model.calls) == 1

    def test_the_weights_are_where_the_probe_reads_them(self, tmp_path):
        env = make_parakeet(tmp_path)

        assert env.path == env.models_dir / "speech" / probe.WEIGHT_PATHS["parakeet-cpu"]

    @pytest.mark.parametrize("device", ["cuda", "cuda:0"])
    def test_accepts_cpu_only(self, tmp_path, device):
        env = make_parakeet(tmp_path)

        with pytest.raises(SpeechEngineError, match=device):
            env.load(device=device)

        assert env.library.loads == 0

    def test_a_missing_weights_directory_raises_naming_its_path(self, tmp_path):
        env = make_parakeet(tmp_path)
        for path in env.path.iterdir():
            path.unlink()
        env.path.rmdir()

        with pytest.raises(SpeechEngineError) as caught:
            env.load()

        assert str(env.path) in str(caught.value)
        assert env.library.loads == 0

    @pytest.mark.parametrize("error", [FakeOrtFail("no file"), OSError("x"), ValueError("id")])
    def test_library_load_errors_become_speech_engine_errors(self, tmp_path, error):
        env = make_parakeet(tmp_path)
        env.library.fail_load = error

        with pytest.raises(SpeechEngineError, match="parakeet-cpu") as caught:
            env.load()

        assert caught.value.__cause__ is error


class TestParakeetDecoding:
    def test_each_partial_decodes_the_whole_utterance_so_far(self, tmp_path):
        env = make_parakeet(tmp_path)
        stream = env.load().open_stream()
        audio = tone(1.0)

        push_frames(stream, audio[:6400])
        first = stream.decode_partial(CancelToken())
        push_frames(stream, audio[6400:])
        second = stream.decode_partial(CancelToken())

        (one, kwargs), (two, _) = env.model.calls
        assert kwargs == {"sample_rate": RATE}
        assert one.dtype == np.float32 and one.ndim == 1
        assert np.array_equal(one, audio[:6400])
        assert np.array_equal(two, audio)
        assert (first.text, first.audio_ms) == ("please zebra", 400)
        assert (second.text, second.audio_ms) == (" ".join(WORDS[:5]), 1000)

    def test_the_final_decodes_the_whole_utterance_and_the_stream_continues(self, tmp_path):
        env = make_parakeet(tmp_path)
        stream = env.load().open_stream()
        audio = tone(1.0)
        push_frames(stream, audio)

        final = stream.decode_final(CancelToken())
        push_frames(stream, tone(0.4))
        later = stream.decode_partial(CancelToken())

        assert np.array_equal(env.model.calls[0][0], audio)
        assert final.text == " ".join(WORDS[:5])
        assert later.audio_ms == 1400

    def test_no_audio_makes_no_engine_call(self, tmp_path):
        env = make_parakeet(tmp_path)
        stream = env.load().open_stream()

        assert stream.decode_partial(CancelToken()).text == ""
        assert stream.decode_final(CancelToken()).text == ""
        assert env.model.calls == []

    def test_onnxruntime_errors_become_speech_engine_errors(self, tmp_path):
        env = make_parakeet(tmp_path)
        stream = env.load().open_stream()
        push_frames(stream, tone(0.4))
        env.fail_decode(FakeOrtFail(f"bad {MARK}"))

        with pytest.raises(SpeechEngineError) as caught:
            stream.decode_partial(CancelToken())

        assert MARK not in _chain_text(caught.value)

    def test_the_pushed_frame_is_copied(self, tmp_path):
        env = make_parakeet(tmp_path)
        stream = env.load().open_stream()
        frame = tone(0.4)

        stream.push(frame)
        frame[:] = 0.0
        stream.decode_partial(CancelToken())

        assert np.array_equal(env.model.calls[0][0], tone(0.4))


def decode_reply(
    tmp_path: Path, reply: FakeTimestampedResult, seconds: float = 1.5
) -> tuple[TranscriptUpdate, TranscriptUpdate]:
    """A partial and a final of `seconds` of audio, each answered with `reply`."""
    env = make_parakeet(tmp_path, FakeParakeetModel(replies=[reply]))
    stream = env.load().open_stream()
    push_frames(stream, tone(seconds))
    return stream.decode_partial(CancelToken()), stream.decode_final(CancelToken())


class TestParakeetWordTimings:
    """A word starts at its first character's token and ends where the next word starts."""

    def test_punctuation_tokens_join_their_word_and_the_last_word_ends_with_the_audio(
        self, tmp_path
    ):
        reply = FakeTimestampedResult(
            "Mist, what's the time?",
            [0.24, 0.48, 0.64, 0.8, 0.96, 1.12, 1.36],
            [" Mist", ",", " what", "'s", " the", " time", "?"],
        )

        partial, final = decode_reply(tmp_path, reply)

        words = (
            W("Mist,", 240, 640),
            W("what's", 640, 960),
            W("the", 960, 1120),
            W("time?", 1120, 1500),
        )
        assert (partial.text, partial.words) == ("Mist, what's the time?", words)
        assert (final.text, final.words) == ("Mist, what's the time?", words)

    def test_the_librarys_spacing_rule_drops_a_space_before_punctuation(self, tmp_path):
        # onnx-asr's `DECODE_SPACE_PATTERN` removes a leading space and a space before a
        # non-word character, and keeps one before a word.
        reply = FakeTimestampedResult(
            "Mist, hi.", [0.08, 0.24, 0.32, 0.56, 0.72], [" Mist", " ", ",", " hi", " ."]
        )

        partial, _ = decode_reply(tmp_path, reply, seconds=1.0)

        assert partial.words == (W("Mist,", 80, 560), W("hi.", 560, 1000))

    @pytest.mark.parametrize("space", [" ", "▁"], ids=["read", "raw"])
    def test_a_lone_space_token_separates_two_words(self, tmp_path, space):
        reply = FakeTimestampedResult(
            "mist stop", [0.16, 0.4, 0.48, 0.56], [f"{space}mi", "st", space, "stop"]
        )

        partial, final = decode_reply(tmp_path, reply, seconds=1.0)

        assert partial.words == final.words == (W("mist", 160, 560), W("stop", 560, 1000))

    def test_a_word_split_across_tokens_starts_at_its_first(self, tmp_path):
        reply = FakeTimestampedResult(
            "missed the bus", [0.08, 0.16, 0.4, 0.64], [" mi", "ssed", " the", " bus"]
        )

        partial, _ = decode_reply(tmp_path, reply, seconds=1.0)

        assert partial.words == (W("missed", 80, 400), W("the", 400, 640), W("bus", 640, 1000))

    def test_the_text_stays_the_librarys_text_stripped(self, tmp_path):
        reply = FakeTimestampedResult("  mist  ", [0.08], [" mist"])

        partial, final = decode_reply(tmp_path, reply, seconds=0.5)

        assert (partial.text, final.text) == ("mist", "mist")
        assert partial.words == (W("mist", 80, 500),)

    @pytest.mark.parametrize(
        "reply",
        [
            FakeTimestampedResult("mist stop", [0.08], [" missed"]),
            FakeTimestampedResult("mist stop", [0.08, 0.4], [" mist", "stop"]),
            FakeTimestampedResult("mist stop", None, [" mist", " stop"]),
            FakeTimestampedResult("mist stop", [0.08, 0.4], None),
            FakeTimestampedResult("mist stop", [0.08], [" mist", " stop"]),
            FakeTimestampedResult("mist stop", [0.4, 0.08], [" mist", " stop"]),
        ],
        ids=["other-text", "joined", "no-timestamps", "no-tokens", "short", "backwards"],
    )
    def test_tokens_that_do_not_give_the_text_give_no_words(self, tmp_path, reply):
        partial, final = decode_reply(tmp_path, reply, seconds=1.0)

        assert (partial.text, partial.words) == ("mist stop", None)
        assert (final.text, final.words) == ("mist stop", None)

    def test_no_audio_gives_no_words(self, tmp_path):
        env = make_parakeet(tmp_path)
        stream = env.load().open_stream()

        assert stream.decode_partial(CancelToken()).words == ()
        assert stream.decode_final(CancelToken()).words == ()

    def test_a_result_with_no_tokens_is_empty_with_no_words(self, tmp_path):
        partial, final = decode_reply(tmp_path, FakeTimestampedResult("", [], []), seconds=0.4)

        assert (partial.text, partial.words) == ("", ())
        assert (final.text, final.words) == ("", ())
