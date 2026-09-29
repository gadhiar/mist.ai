"""Embedding text, scoring and CLI exit codes, driven by fake embedders (no model, no network)."""

from __future__ import annotations

import json
import sys
import types
import zlib

import pytest

from backend.errors import EmbeddingError
from backend.knowledge.curation.deduplication import SIMILARITY_THRESHOLD
from backend.knowledge.embeddings.embedding_text import embedding_text_for
from scripts.dedup_calibration import cli
from scripts.dedup_calibration.dataset import (
    DEFAULT_DATASET_PATH,
    SLICE_EXTRACTED_VS_EXTRACTED,
    SLICE_EXTRACTED_VS_SEED,
    LabelledPair,
    load_dataset,
)
from scripts.dedup_calibration.embedder import EmbedderUnavailableError, build_real_embedder
from scripts.dedup_calibration.scoring import ScoredPair, probe_text, score_pairs, stored_text

_DIM = 256


class TrigramEmbedder:
    """Deterministic character-trigram embedder; similar strings get similar vectors."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def generate_embedding(self, text: str) -> list[float]:
        self.calls.append(text)
        padded = f"  {text.casefold()}  "
        vec = [0.0] * _DIM
        for i in range(len(padded) - 2):
            vec[zlib.crc32(padded[i : i + 3].encode()) % _DIM] += 1.0
        return vec

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self.generate_embedding(t) for t in texts]


class TableEmbedder:
    """Returns scripted vectors by text."""

    def __init__(self, table: dict[str, list[float]]) -> None:
        self._table = table
        self.calls: list[str] = []

    def generate_embedding(self, text: str) -> list[float]:
        self.calls.append(text)
        return self._table[text]

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self.generate_embedding(t) for t in texts]


def _pair(slice_name: str = SLICE_EXTRACTED_VS_EXTRACTED, **kw) -> LabelledPair:
    fields = {
        "slice": slice_name,
        "entity_type": "Technology",
        "label": "duplicate",
        "a": "Postgres",
        "b": "PostgreSQL",
    }
    fields.update(kw)
    return LabelledPair(**fields)


def _factory(embedder=None, name="fake-model"):
    provider = embedder if embedder is not None else TrigramEmbedder()
    return lambda: (provider, name)


class TestEmbeddingText:
    def test_probe_embeds_the_display_name(self):
        assert probe_text(_pair(a="Neo4j graph database")) == "Neo4j graph database"

    def test_extracted_stored_side_embeds_the_display_name(self):
        assert stored_text(_pair(b="PostgreSQL")) == "PostgreSQL"

    def test_seed_stored_side_without_description_is_the_display_name(self):
        pair = _pair(SLICE_EXTRACTED_VS_SEED, b="Neo4j")
        assert stored_text(pair) == "Neo4j" == embedding_text_for("Neo4j", None, "Neo4j")

    def test_seed_stored_side_with_description_uses_the_seed_text_builder(self):
        pair = _pair(SLICE_EXTRACTED_VS_SEED, b="Neo4j", b_description="Graph database")
        assert stored_text(pair) == "Neo4j " + chr(0x2014) + " Graph database"

    def test_score_pairs_sends_exactly_those_texts_to_the_provider(self):
        embedder = TrigramEmbedder()
        pairs = [
            _pair(a="Neo4j graph database", b="Neo4j"),
            _pair(SLICE_EXTRACTED_VS_SEED, a="neo4j DB", b="Neo4j", b_description="Graph DB"),
        ]
        score_pairs(pairs, embedder)
        assert embedder.calls == [
            "Neo4j graph database",
            "Neo4j",
            "neo4j DB",
            "Neo4j " + chr(0x2014) + " Graph DB",
        ]

    def test_each_distinct_text_is_embedded_once(self):
        embedder = TrigramEmbedder()
        score_pairs([_pair(a="Python", b="Rust"), _pair(a="Python", b="Go")], embedder)
        assert embedder.calls.count("Python") == 1


class TestScoring:
    def test_cosine_and_neo4j_score_from_scripted_vectors(self):
        embedder = TableEmbedder({"a": [1.0, 0.0], "b": [0.6, 0.8]})
        (scored,) = score_pairs([_pair(a="a", b="b")], embedder)
        assert isinstance(scored, ScoredPair)
        assert scored.cosine == pytest.approx(0.6)
        assert scored.neo4j_score == pytest.approx(0.8)

    def test_identical_names_score_one(self):
        (scored,) = score_pairs([_pair(a="Docker", b="docker")], TrigramEmbedder())
        assert scored.neo4j_score == pytest.approx(1.0)

    def test_a_zero_vector_is_an_error_not_a_score(self):
        embedder = TableEmbedder({"a": [0.0, 0.0], "b": [1.0, 0.0]})
        with pytest.raises(ValueError, match="zero vector"):
            score_pairs([_pair(a="a", b="b")], embedder)


class TestCliSuccess:
    def test_exit_zero_and_report_shape(self, tmp_path, capsys):
        out = tmp_path / "nested" / "report.json"
        code = cli.main(["--output", str(out)], embedder_factory=_factory())
        assert code == cli.EXIT_OK

        report = json.loads(out.read_text(encoding="utf-8"))
        pairs = load_dataset(DEFAULT_DATASET_PATH)
        assert report["model"] == "fake-model"
        assert len(report["pairs"]) == len(pairs)
        assert set(report["slices"]) == {
            "overall",
            SLICE_EXTRACTED_VS_EXTRACTED,
            SLICE_EXTRACTED_VS_SEED,
        }
        overall = report["slices"]["overall"]
        assert overall["n_duplicate"] + overall["n_distinct"] == len(pairs)
        for key in ("duplicate", "distinct"):
            for scale in ("cosine", "neo4j_score"):
                assert overall[key][scale]["n"] > 0
        for key in ("zero_false_merge", "best_f1", "at_current_threshold"):
            assert key in overall

        table = capsys.readouterr().out
        assert "=== overall:" in table
        assert f"=== {SLICE_EXTRACTED_VS_SEED}:" in table
        assert "zero-false-merge point" in table
        assert "best-F1 point" in table

    def test_reports_the_live_threshold_on_both_scales(self, tmp_path):
        out = tmp_path / "r.json"
        cli.main(["--output", str(out)], embedder_factory=_factory())
        current = json.loads(out.read_text(encoding="utf-8"))["current_threshold"]
        assert current["neo4j_score"] == SIMILARITY_THRESHOLD
        assert current["raw_cosine"] == pytest.approx(2 * SIMILARITY_THRESHOLD - 1)

    def test_threshold_override(self, tmp_path):
        out = tmp_path / "r.json"
        cli.main(["--threshold", "0.5", "--output", str(out)], embedder_factory=_factory())
        report = json.loads(out.read_text(encoding="utf-8"))
        assert report["current_threshold"]["neo4j_score"] == 0.5
        assert report["slices"]["overall"]["at_current_threshold"]["threshold"] == 0.5

    def test_output_is_optional(self, capsys):
        assert cli.main([], embedder_factory=_factory()) == cli.EXIT_OK
        assert "MIST dedup threshold calibration" in capsys.readouterr().out

    def test_scores_the_committed_pairs_in_dataset_order(self, tmp_path):
        out = tmp_path / "r.json"
        cli.main(["--output", str(out)], embedder_factory=_factory())
        report = json.loads(out.read_text(encoding="utf-8"))
        expected = [(p.a, p.b) for p in load_dataset(DEFAULT_DATASET_PATH)]
        assert [(r["a"], r["b"]) for r in report["pairs"]] == expected


class TestCliExitCodes:
    def test_exit_two_when_the_model_cannot_load(self, tmp_path, capsys):
        def broken():
            raise EmbedderUnavailableError("no cached model and no network")

        out = tmp_path / "r.json"
        code = cli.main(["--output", str(out)], embedder_factory=broken)
        assert code == cli.EXIT_MODEL_UNAVAILABLE == 2
        err = capsys.readouterr().err
        assert "refusing to run" in err
        assert "no cached model and no network" in err
        assert not out.exists()

    def test_exit_two_when_the_model_fails_while_embedding(self, tmp_path):
        class Failing:
            def generate_embedding(self, text):
                raise EmbeddingError("encode failed")

            def generate_embeddings(self, texts):
                raise EmbeddingError("encode failed")

        out = tmp_path / "r.json"
        code = cli.main(["--output", str(out)], embedder_factory=_factory(Failing()))
        assert code == 2
        assert not out.exists()

    def test_exit_two_on_a_zero_vector_from_the_model(self):
        class Zeros:
            def generate_embedding(self, text):
                return [0.0] * 8

            def generate_embeddings(self, texts):
                return [[0.0] * 8 for _ in texts]

        assert cli.main([], embedder_factory=_factory(Zeros())) == 2

    def test_default_factory_exits_two_when_the_real_model_load_fails(self, monkeypatch, capsys):
        """The real build path, with the model layer stubbed to fail as an offline load would."""

        class OfflineGenerator:
            def __init__(self, model_name):
                self.model_name = model_name

            def warmup(self):
                raise OSError("We couldn't connect to the hub and no cached files were found")

        stub = types.ModuleType("backend.knowledge.embeddings.embedding_generator")
        stub.EmbeddingGenerator = OfflineGenerator
        monkeypatch.setitem(sys.modules, "backend.knowledge.embeddings.embedding_generator", stub)
        assert cli.main([]) == 2
        assert "cannot load the embedding model" in capsys.readouterr().err

    def test_exit_one_for_a_missing_dataset(self, tmp_path, capsys):
        code = cli.main(["--dataset", str(tmp_path / "missing.json")], embedder_factory=_factory())
        assert code == cli.EXIT_BAD_INPUT == 1
        assert "cannot read dataset" in capsys.readouterr().err

    def test_exit_one_for_an_invalid_dataset_and_the_model_is_never_built(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"schema_version": 1, "pairs": []}), encoding="utf-8")

        def must_not_run():
            raise AssertionError("embedder factory called for an invalid dataset")

        assert cli.main(["--dataset", str(bad)], embedder_factory=must_not_run) == 1

    def test_exit_one_when_the_output_path_is_unwritable(self, tmp_path):
        blocker = tmp_path / "file"
        blocker.write_text("x", encoding="utf-8")
        code = cli.main(["--output", str(blocker / "r.json")], embedder_factory=_factory())
        assert code == 1

    @pytest.mark.parametrize("value", ["-0.1", "1.5"])
    def test_exit_one_for_a_threshold_outside_the_score_range(self, value):
        assert cli.main(["--threshold", value], embedder_factory=_factory()) == 1

    def test_usage_error_is_not_confused_with_model_unavailable(self):
        with pytest.raises(SystemExit) as info:
            cli.main(["--no-such-flag"], embedder_factory=_factory())
        assert info.value.code == cli.EXIT_USAGE == 64


class TestBuildRealEmbedder:
    def _install(self, monkeypatch, generator_cls):
        stub = types.ModuleType("backend.knowledge.embeddings.embedding_generator")
        stub.EmbeddingGenerator = generator_cls
        monkeypatch.setitem(sys.modules, "backend.knowledge.embeddings.embedding_generator", stub)

    def test_builds_the_generator_from_the_configured_model_name_and_warms_it(self, monkeypatch):
        built: list = []

        class Generator:
            def __init__(self, model_name):
                self.model_name = model_name
                self.warmed = False
                built.append(self)

            def warmup(self):
                self.warmed = True

        self._install(monkeypatch, Generator)
        monkeypatch.setenv("EMBEDDING_MODEL", "some-other-model")
        provider, name = build_real_embedder()
        assert name == "some-other-model"
        assert provider is built[0] and built[0].warmed
        assert built[0].model_name == "some-other-model"

    def test_defaults_to_minilm(self, monkeypatch):
        class Generator:
            def __init__(self, model_name):
                self.model_name = model_name

            def warmup(self):
                pass

        self._install(monkeypatch, Generator)
        monkeypatch.delenv("EMBEDDING_MODEL", raising=False)
        _, name = build_real_embedder()
        assert name == "all-MiniLM-L6-v2"

    @pytest.mark.parametrize(
        "error", [OSError("offline"), ImportError("no torch"), RuntimeError("x")]
    )
    def test_load_failures_become_embedder_unavailable(self, monkeypatch, error):
        class Generator:
            def __init__(self, model_name):
                pass

            def warmup(self):
                raise error

        self._install(monkeypatch, Generator)
        with pytest.raises(EmbedderUnavailableError, match="cannot load the embedding model"):
            build_real_embedder()
