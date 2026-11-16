"""The harness's real embedder must make the production embedder's encode calls exactly.

One recording fake is patched in as `SentenceTransformer` in both the harness and
`backend/knowledge/embeddings/embedding_generator.py`, so the two classes drive the same fake
and any difference in their encode arguments, or in what they do without encoding, fails here.

The reranker side: real models load from `/opt/models` only, `--rerank-model` included, and
the trace counts a pair's tokens with the tokenizer of the cross-encoder the reranker loaded.

Placement: `--device` places the reranker and `--embedder-device`, defaulting to `--device`,
places the embedder; each record names the embedder's device.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import backend.knowledge.embeddings.embedding_generator as generator_module
import scripts.eval_harness.memory_recall_run as harness
from backend.knowledge.embeddings.embedding_generator import EmbeddingGenerator
from scripts.eval_harness.memory_recall_run import EMBEDDER_PATH, SentenceTransformerEmbedder


class RecordingSentenceTransformer:
    """Records its constructor and every encode call; vectors are hashes of the text."""

    instances: list[RecordingSentenceTransformer] = []

    def __init__(self, model_name_or_path: str, device: str | None = None) -> None:
        self.path = model_name_or_path
        self.device = device
        self.encodes: list[tuple[Any, dict[str, Any]]] = []
        RecordingSentenceTransformer.instances.append(self)

    def encode(self, sentences: str | list[str], **kwargs: Any) -> np.ndarray:
        self.encodes.append((sentences, dict(kwargs)))
        if isinstance(sentences, str):
            return _vector(sentences)
        return np.array([_vector(text) for text in sentences])


def _vector(text: str) -> np.ndarray:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return np.array([digest[i % len(digest)] / 255.0 for i in range(384)], dtype=np.float32)


@pytest.fixture(autouse=True)
def _gate_off(monkeypatch: pytest.MonkeyPatch) -> None:
    # A run without `--need-gate` takes the config's mode, `on` by default, and no gate model
    # file ships yet; these runs measure placement, so the gate is off through the environment.
    monkeypatch.setenv("MIST_MEMORY_NEED_GATE", "off")


@pytest.fixture(autouse=True)
def recording(monkeypatch: pytest.MonkeyPatch) -> type[RecordingSentenceTransformer]:
    RecordingSentenceTransformer.instances = []
    monkeypatch.setattr(harness, "SentenceTransformer", RecordingSentenceTransformer)
    monkeypatch.setattr(generator_module, "SentenceTransformer", RecordingSentenceTransformer)
    return RecordingSentenceTransformer


def _pair() -> tuple[SentenceTransformerEmbedder, EmbeddingGenerator]:
    return (
        SentenceTransformerEmbedder(str(EMBEDDER_PATH), device="cpu"),
        EmbeddingGenerator(str(EMBEDDER_PATH), device="cpu"),
    )


def _encodes(owner: object) -> list[tuple[Any, dict[str, Any]]]:
    calls: list[tuple[Any, dict[str, Any]]] = []
    for instance in RecordingSentenceTransformer.instances:
        if getattr(owner, "_model", None) is instance:
            calls.extend(instance.encodes)
    return calls


def test_the_constructor_gets_the_model_path_and_the_device() -> None:
    SentenceTransformerEmbedder(str(EMBEDDER_PATH), device="cuda")

    [model] = RecordingSentenceTransformer.instances
    assert (model.path, model.device) == ("/opt/models/all-MiniLM-L6-v2", "cuda")


@pytest.mark.parametrize(
    "call",
    [
        pytest.param(lambda e: e.generate_embedding("Wren works at Tidewater"), id="a-text"),
        pytest.param(lambda e: e.generate_embedding("   "), id="a-blank-text"),
        pytest.param(lambda e: e.generate_embeddings([]), id="an-empty-list"),
        pytest.param(
            lambda e: e.generate_embeddings(["kimchi", "  ", "focaccia"]), id="a-blank-entry"
        ),
        pytest.param(
            lambda e: e.generate_embeddings([f"note {n}" for n in range(101)]), id="101-texts"
        ),
    ],
)
def test_the_harness_encodes_as_the_production_embedder(call: Callable[[Any], Any]) -> None:
    ours, production = _pair()

    our_vectors = call(ours)
    production_vectors = call(production)

    assert our_vectors == production_vectors
    assert _encodes(ours) == _encodes(production)
    models = {(model.path, model.device) for model in RecordingSentenceTransformer.instances}
    assert models == {("/opt/models/all-MiniLM-L6-v2", "cpu")}


def test_a_blank_text_is_384_zeros_without_encoding() -> None:
    ours, _ = _pair()

    assert ours.generate_embedding("") == [0.0] * 384
    assert _encodes(ours) == []


def test_a_batch_over_100_texts_shows_progress() -> None:
    ours, _ = _pair()

    ours.generate_embeddings([f"t{n}" for n in range(101)])
    ours.generate_embeddings(["one", ""])

    [(_, many), (texts, few)] = _encodes(ours)
    assert many == {"convert_to_numpy": True, "show_progress_bar": True, "batch_size": 32}
    assert few == {"convert_to_numpy": True, "show_progress_bar": False, "batch_size": 32}
    assert texts == ["one", " "]


def test_real_models_load_from_opt_models_only() -> None:
    with pytest.raises(ValueError, match="/opt/models"):
        harness._model_path("/home/user/all-MiniLM-L6-v2")
    assert harness._model_path("/opt/models/ms-marco-MiniLM-L-6-v2") == (
        "/opt/models/ms-marco-MiniLM-L-6-v2"
    )


def test_rerank_model_overrides_the_configured_cross_encoder() -> None:
    args = harness._parser().parse_args(
        ["--output", "run.jsonl", "--rerank-model", "/opt/models/bge-reranker-base"]
    )

    assert harness.config_from_args(args).rerank_model == "/opt/models/bge-reranker-base"


class RecordingTokenizer:
    """Records each call; a pair's ids are its words, cut to `max_length` when truncating."""

    model_max_length = 512

    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []

    def __call__(self, *texts: str, **kwargs: Any) -> dict[str, list[int]]:
        self.calls.append((texts, dict(kwargs)))
        ids = [0] * sum(len(text.split()) for text in texts)
        limit = kwargs.get("max_length") or self.model_max_length
        return {"input_ids": ids[:limit] if kwargs.get("truncation") else ids}


class FakeCrossEncoder:
    """The `CrossEncoder` attributes the trace reads: its tokenizer and its `max_length`."""

    instances: list[FakeCrossEncoder] = []

    def __init__(self, model_name: str, *, device: str, local_files_only: bool) -> None:
        self.args = (model_name, device, local_files_only)
        self.tokenizer = RecordingTokenizer()
        self.max_length: int | None = 4
        FakeCrossEncoder.instances.append(self)


def test_pair_tokens_come_from_the_loaded_cross_encoders_tokenizer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(harness, "CrossEncoder", FakeCrossEncoder)
    capture = harness.CrossEncoderCapture()

    model = capture("/opt/models/ms-marco-MiniLM-L-6-v2", device="cpu", local_files_only=True)

    assert model.args == ("/opt/models/ms-marco-MiniLM-L-6-v2", "cpu", True)
    assert capture.pair_tokens("one two", "three") == 3
    assert capture.pair_tokens("one two", "three four five") == 4  # truncated to max_length
    # Truncated as CrossEncoder.predict truncates: the pair together, longest first.
    assert model.tokenizer.calls[-1] == (
        ("one two", "three four five"),
        {"truncation": "longest_first", "max_length": 4},
    )


def test_the_real_reranker_is_loaded_through_the_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    class StubCrossEncoderReranker:
        def __init__(self, model_path: str, *, device: str, cross_encoder_cls: Any) -> None:
            self.model = cross_encoder_cls(model_path, device=device, local_files_only=True)

    FakeCrossEncoder.instances = []
    monkeypatch.setattr(harness, "CrossEncoder", FakeCrossEncoder)
    monkeypatch.setattr(harness, "CrossEncoderReranker", StubCrossEncoderReranker)

    built = harness.build_reranker("real", device="cpu", config=harness.MemoryConfig())

    [model] = FakeCrossEncoder.instances
    assert built.path == "/opt/models/ms-marco-MiniLM-L-6-v2"
    assert built.pair_tokens("a b", "c") == 3 and model.tokenizer.calls


def test_the_fake_reranker_counts_whitespace_words() -> None:
    built = harness.build_reranker("fake", device="cpu", config=harness.MemoryConfig())

    assert built.path is None
    assert built.pair_tokens("which  drink", "Wren orders a flat white.") == 7


@pytest.mark.parametrize(
    ("p", "expected"),
    [(0.5, 0.0), (0.75, math.log(3.0)), (0.0, math.log(1e-7 / (1 - 1e-7)))],
    ids=["even", "three-to-one", "zero-clamped"],
)
def test_logit_inverts_the_sigmoid_within_the_clamp(p: float, expected: float) -> None:
    assert harness.logit(p) == pytest.approx(expected)
    assert harness.logit(1.0) == pytest.approx(-harness.logit(0.0))


@pytest.fixture
def placements(monkeypatch: pytest.MonkeyPatch) -> dict[str, list[str]]:
    """The device each model of the run is built on."""
    seen: dict[str, list[str]] = {"embedder": [], "reranker": []}
    build_embedder, build_reranker = harness.build_embedder, harness.build_reranker

    def embedder(choice: str, *, device: str) -> Any:
        seen["embedder"].append(device)
        return build_embedder(choice, device=device)  # type: ignore[arg-type]

    def reranker(choice: str, *, device: str, config: Any) -> Any:
        seen["reranker"].append(device)
        return build_reranker(choice, device=device, config=config)  # type: ignore[arg-type]

    monkeypatch.setattr(harness, "build_embedder", embedder)
    monkeypatch.setattr(harness, "build_reranker", reranker)
    return seen


@pytest.mark.parametrize(
    ("argv", "embedder", "reranker"),
    [
        pytest.param([], "cpu", "cpu", id="default"),
        pytest.param(["--device", "cuda"], "cuda", "cuda", id="one-placement"),
        pytest.param(["--device", "cuda", "--embedder-device", "cpu"], "cpu", "cuda", id="split"),
        pytest.param(["--embedder-device", "cuda"], "cuda", "cpu", id="embedder-on-cuda"),
    ],
)
def test_device_places_the_reranker_and_embedder_device_the_embedder(
    tmp_path: Path,
    placements: dict[str, list[str]],
    argv: list[str],
    embedder: str,
    reranker: str,
) -> None:
    output = tmp_path / "run.jsonl"

    assert harness.main(["--output", str(output), *argv]) == 0

    assert placements == {"embedder": [embedder], "reranker": [reranker]}
    lines = output.read_text(encoding="utf-8").splitlines()
    assert {json.loads(line)["embedder_device"] for line in lines} == {embedder}


def test_a_rerank_model_outside_opt_models_is_refused(tmp_path: Path) -> None:
    output = tmp_path / "run.jsonl"

    with pytest.raises(ValueError, match="/opt/models"):
        harness.main(["--output", str(output), "--rerank-model", "/home/user/ms-marco"])
    assert not output.exists()
