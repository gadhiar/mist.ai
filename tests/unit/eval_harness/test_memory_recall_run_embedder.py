"""The harness's real embedder must make the production embedder's encode calls exactly.

One recording fake is patched in as `SentenceTransformer` in both the harness and
`backend/knowledge/embeddings/embedding_generator.py`, so the two classes drive the same fake
and any difference in their encode arguments, or in what they do without encoding, fails here.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable
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
