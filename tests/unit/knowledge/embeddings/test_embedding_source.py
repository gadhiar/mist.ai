"""The embedder loads the copy the backend image bakes, and falls back to the name elsewhere.

The backend image bakes all-MiniLM-L6-v2 at `/opt/models/all-MiniLM-L6-v2`
(Dockerfile step 7, pinned by `docker/backend/models.lock`). Loading that
directory with `local_files_only=True` keeps the production embedder off the
network; outside the image the directory is absent and the generator loads by
name, as it always did. The model identity stays `model_name` either way, so
`compose_model_hash` and the stored vectors do not change.

No test here loads a model: `SentenceTransformer` is replaced with a recorder,
and `DEFAULT_MODELS_ROOT` points under `tmp_path`, so the result does not depend
on whether the suite runs inside the image.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from backend.knowledge.embeddings import embedding_generator as embedding_generator_module
from backend.knowledge.embeddings.embedding_generator import EmbeddingGenerator

_MODEL = "all-MiniLM-L6-v2"


class _RecordingSentenceTransformer:
    """Stands in for `SentenceTransformer`, recording each construction's arguments."""

    calls: list[tuple[tuple, dict]] = []

    def __init__(self, *args, **kwargs) -> None:
        type(self).calls.append((args, kwargs))


@pytest.fixture
def recorded_constructions(monkeypatch: pytest.MonkeyPatch) -> list[tuple[tuple, dict]]:
    _RecordingSentenceTransformer.calls = []
    monkeypatch.setattr(
        embedding_generator_module, "SentenceTransformer", _RecordingSentenceTransformer
    )
    return _RecordingSentenceTransformer.calls


@pytest.fixture
def models_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """A models root the module constant points at, empty until a test fills it."""
    root = tmp_path / "models"
    root.mkdir()
    monkeypatch.setattr(embedding_generator_module, "DEFAULT_MODELS_ROOT", str(root))
    return root


class TestLocalCopyPresent:
    @pytest.mark.parametrize(
        "device",
        [pytest.param(None, id="device-none"), pytest.param("cpu", id="device-cpu")],
    )
    def test_loads_the_local_directory_offline_with_the_device(
        self,
        recorded_constructions: list[tuple[tuple, dict]],
        models_root: Path,
        device: str | None,
    ) -> None:
        (models_root / _MODEL).mkdir()
        generator = EmbeddingGenerator(_MODEL, device=device)

        generator._get_model()

        assert recorded_constructions == [
            ((str(models_root / _MODEL),), {"device": device, "local_files_only": True})
        ]

    def test_model_name_stays_the_identity(
        self, recorded_constructions: list[tuple[tuple, dict]], models_root: Path
    ) -> None:
        (models_root / _MODEL).mkdir()
        generator = EmbeddingGenerator(_MODEL, device="cpu")

        generator._get_model()

        assert generator.model_name == _MODEL

    def test_logs_the_local_path_it_loaded(
        self,
        recorded_constructions: list[tuple[tuple, dict]],
        models_root: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        (models_root / _MODEL).mkdir()
        generator = EmbeddingGenerator(_MODEL, device="cpu")

        with caplog.at_level(logging.INFO, logger=embedding_generator_module.__name__):
            generator._get_model()
            generator._get_model()

        loaded = [r.getMessage() for r in caplog.records if "loaded from" in r.getMessage()]
        assert loaded == [f"Embedding model {_MODEL} loaded from {models_root / _MODEL}"]


class TestLocalCopyAbsent:
    @pytest.mark.parametrize(
        "device",
        [pytest.param(None, id="device-none"), pytest.param("cpu", id="device-cpu")],
    )
    def test_missing_directory_loads_by_name_with_the_device(
        self,
        recorded_constructions: list[tuple[tuple, dict]],
        models_root: Path,
        device: str | None,
    ) -> None:
        generator = EmbeddingGenerator(_MODEL, device=device)

        generator._get_model()

        assert recorded_constructions == [((_MODEL,), {"device": device})]

    def test_regular_file_at_the_path_loads_by_name(
        self, recorded_constructions: list[tuple[tuple, dict]], models_root: Path
    ) -> None:
        (models_root / _MODEL).write_text("not a model directory", encoding="utf-8")
        generator = EmbeddingGenerator(_MODEL, device="cpu")

        generator._get_model()

        assert recorded_constructions == [((_MODEL,), {"device": "cpu"})]

    def test_absolute_model_path_loads_as_given(
        self, recorded_constructions: list[tuple[tuple, dict]], models_root: Path
    ) -> None:
        """A caller that already names a path, as the recall eval harness does, is unchanged.

        Joining an absolute name onto the root yields the name itself, which would
        silently switch such a caller to `local_files_only=True`.
        """
        (models_root / _MODEL).mkdir()
        generator = EmbeddingGenerator(str(models_root / _MODEL), device="cpu")

        generator._get_model()

        assert recorded_constructions == [((str(models_root / _MODEL),), {"device": "cpu"})]

    def test_logs_the_name_it_loaded(
        self,
        recorded_constructions: list[tuple[tuple, dict]],
        models_root: Path,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        generator = EmbeddingGenerator(_MODEL, device="cpu")

        with caplog.at_level(logging.INFO, logger=embedding_generator_module.__name__):
            generator._get_model()

        loaded = [r.getMessage() for r in caplog.records if "loaded from" in r.getMessage()]
        assert loaded == [f"Embedding model {_MODEL} loaded from its name"]


class TestModelsRootResolution:
    def test_explicit_models_root_overrides_the_constant(
        self,
        recorded_constructions: list[tuple[tuple, dict]],
        models_root: Path,
        tmp_path: Path,
    ) -> None:
        explicit = tmp_path / "explicit"
        (explicit / _MODEL).mkdir(parents=True)
        generator = EmbeddingGenerator(_MODEL, device="cpu", models_root=str(explicit))

        generator._get_model()

        assert recorded_constructions == [
            ((str(explicit / _MODEL),), {"device": "cpu", "local_files_only": True})
        ]

    def test_constant_is_read_at_construction_not_import(
        self,
        recorded_constructions: list[tuple[tuple, dict]],
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        late = tmp_path / "late"
        (late / _MODEL).mkdir(parents=True)
        monkeypatch.setattr(embedding_generator_module, "DEFAULT_MODELS_ROOT", str(late))
        generator = EmbeddingGenerator(_MODEL, device="cpu")

        generator._get_model()

        assert recorded_constructions == [
            ((str(late / _MODEL),), {"device": "cpu", "local_files_only": True})
        ]

    def test_default_models_root_is_the_image_path(self) -> None:
        assert embedding_generator_module.DEFAULT_MODELS_ROOT == "/opt/models"

    def test_construction_does_not_load_the_model(
        self, recorded_constructions: list[tuple[tuple, dict]], models_root: Path
    ) -> None:
        (models_root / _MODEL).mkdir()

        EmbeddingGenerator(_MODEL, device="cpu")

        assert recorded_constructions == []
