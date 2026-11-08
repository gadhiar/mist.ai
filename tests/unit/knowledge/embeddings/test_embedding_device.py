"""The configured embedding device reaches `SentenceTransformer` at every construction site.

`EmbeddingConfig.device` used to be read from `EMBEDDING_DEVICE` and then
dropped: every `EmbeddingGenerator` called `SentenceTransformer(model_name)`
with no device, and sentence-transformers resolves no device to CUDA when it
is available, so each generator kept its own MiniLM copy on the GPU whatever
the configuration said. These tests pin the device from the environment
variable through the config and the generator to the model constructor, and
pin that no construction site in `backend/` or `scripts/` omits it.

No test here loads a model: `SentenceTransformer` is replaced with a recorder.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from backend.knowledge.config import (
    EmbeddingConfig,
    ExtractionConfig,
    KnowledgeConfig,
    LLMConfig,
    Neo4jConfig,
)
from backend.knowledge.embeddings import embedding_generator as embedding_generator_module
from backend.knowledge.embeddings.embedding_generator import EmbeddingGenerator

_REPO_ROOT = Path(__file__).resolve().parents[4]

# The offline calibration tool builds its own generator outside the
# configured runtime and is deliberately out of scope.
_EXCLUDED_DIRS = (_REPO_ROOT / "scripts" / "dedup_calibration",)

# The thirteen runtime construction sites at the time this test was written:
# five in `backend/factories.py`, one in `backend/extraction_backlog/cutover.py`
# and seven in `scripts/mist_admin.py`. The walk must see at least these, so a
# broken walk cannot pass by finding nothing.
_MIN_CONSTRUCTION_SITES = 13


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


class TestGeneratorPassesDevice:
    @pytest.mark.parametrize("device", [None, "cpu", "cuda"])
    def test_get_model_passes_device_to_sentence_transformer(
        self, recorded_constructions: list[tuple[tuple, dict]], device: str | None
    ) -> None:
        generator = EmbeddingGenerator("all-MiniLM-L6-v2", device=device)

        generator._get_model()

        assert recorded_constructions == [(("all-MiniLM-L6-v2",), {"device": device})]

    def test_device_defaults_to_none(self, recorded_constructions) -> None:
        generator = EmbeddingGenerator()

        generator._get_model()

        assert generator.device is None
        assert recorded_constructions == [(("all-MiniLM-L6-v2",), {"device": None})]

    def test_construction_does_not_load_the_model(self, recorded_constructions) -> None:
        EmbeddingGenerator(device="cpu")

        assert recorded_constructions == []


class TestEmbeddingDeviceConfig:
    @pytest.mark.parametrize("value", ["auto", "cpu", "cuda", "cuda:0", "cuda:1", "cuda:12"])
    def test_from_env_accepts_supported_values(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("EMBEDDING_DEVICE", value)

        assert EmbeddingConfig.from_env().device == value

    def test_from_env_defaults_to_cpu_when_unset(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("EMBEDDING_DEVICE", raising=False)

        assert EmbeddingConfig.from_env().device == "cpu"

    @pytest.mark.parametrize(
        "value", ["", "gpu", "CPU", "cuda:", "cuda:x", "cuda:-1", "mps", " cpu", "cuda:0 "]
    )
    def test_from_env_rejects_other_values_naming_the_variable(
        self, monkeypatch: pytest.MonkeyPatch, value: str
    ) -> None:
        monkeypatch.setenv("EMBEDDING_DEVICE", value)

        with pytest.raises(ValueError, match="EMBEDDING_DEVICE"):
            EmbeddingConfig.from_env()

    def test_resolved_device_is_none_for_auto(self) -> None:
        assert EmbeddingConfig(device="auto").resolved_device() is None

    @pytest.mark.parametrize("value", ["cpu", "cuda", "cuda:1"])
    def test_resolved_device_returns_the_value_otherwise(self, value: str) -> None:
        assert EmbeddingConfig(device=value).resolved_device() == value


def _is_embedding_generator_call(node: ast.AST) -> bool:
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if isinstance(func, ast.Name):
        return func.id == "EmbeddingGenerator"
    if isinstance(func, ast.Attribute):
        return func.attr == "EmbeddingGenerator"
    return False


def _construction_sites() -> list[tuple[Path, ast.Call]]:
    sites: list[tuple[Path, ast.Call]] = []
    for top in ("backend", "scripts"):
        for path in sorted((_REPO_ROOT / top).rglob("*.py")):
            if any(path.is_relative_to(excluded) for excluded in _EXCLUDED_DIRS):
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            sites.extend(
                (path, node) for node in ast.walk(tree) if _is_embedding_generator_call(node)
            )
    return sites


class TestEveryConstructionSitePassesDevice:
    def test_no_construction_site_omits_device(self) -> None:
        sites = _construction_sites()

        missing = [
            f"{path.relative_to(_REPO_ROOT).as_posix()}:{node.lineno}"
            for path, node in sites
            if not any(keyword.arg == "device" for keyword in node.keywords)
        ]

        assert len(sites) >= _MIN_CONSTRUCTION_SITES, sites
        assert missing == []


class _FakeConnection:
    """Satisfies the `GraphConnection` protocol with no I/O."""

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    def is_connected(self) -> bool:
        return True

    def execute_query(self, query: str, params: dict | None = None) -> list[dict]:
        return []

    def execute_write(self, query: str, params: dict | None = None) -> list[dict]:
        return []


class TestFactoryHonoursConfiguredDevice:
    def test_build_graph_store_yields_generator_on_configured_cpu(
        self, recorded_constructions: list[tuple[tuple, dict]]
    ) -> None:
        from backend.factories import build_graph_store

        config = KnowledgeConfig(
            neo4j=Neo4jConfig(),
            llm=LLMConfig(),
            embedding=EmbeddingConfig(device="cpu"),
            extraction=ExtractionConfig(),
        )

        store = build_graph_store(config, connection=_FakeConnection())
        generator = store.embedding_generator
        generator._get_model()

        assert isinstance(generator, EmbeddingGenerator)
        assert generator.device == "cpu"
        assert recorded_constructions == [((config.embedding.model_name,), {"device": "cpu"})]
