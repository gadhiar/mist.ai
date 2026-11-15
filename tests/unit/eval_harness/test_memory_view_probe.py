"""Tests for scripts/eval_harness/memory_view_probe.py, H13's fixed selection rule.

The rule is checked on hand-built score matrices, where every figure can be worked out by hand,
and the command line end to end on the dev fixture with the harness's weight-free fakes
(`TokenEmbedder`, `TokenReranker`), so no model is loaded and nothing is downloaded.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

import scripts.eval_harness.memory_recall_run as harness
import scripts.eval_harness.memory_view_probe as view_probe
from backend.memory.sources import NOTE_VECTOR_POOL
from scripts.eval_harness.memory_recall import FIXTURE_DIR, ItemRef, Probe, load_probes
from scripts.eval_harness.memory_recall_gate import PR1_GATES
from scripts.eval_harness.memory_recall_run import TokenEmbedder, TokenReranker
from scripts.eval_harness.memory_view_probe import (
    BLEND,
    CE,
    EXIT_FLAG,
    EXIT_INPUT,
    EXIT_PROCEED,
    EXIT_STOP,
    GATED_CLASSES,
    GRID,
    GateClass,
    PassageEmbedder,
    ProbeScores,
    Setting,
    ViewPair,
    band,
    build_parser,
    choose_k,
    evaluate,
    fused_positions,
    main,
    model_dir,
    select,
)

_NOW = datetime(2026, 3, 16, 8, 30, tzinfo=UTC)
_FACT_ONLY = (GateClass("fact", 0.85, lambda p: p.kind == "fact"),)
_FACT_AND_NOTE = (
    GateClass("fact", 0.85, lambda p: p.kind == "fact"),
    GateClass("note", 0.85, lambda p: p.kind == "note"),
)


def _probe(probe_id: str, kind: str = "fact", *, targets: int = 1) -> Probe:
    ref_kind = "note" if kind == "note" else "fact"
    expected = (
        ()
        if kind == "noise"
        else tuple(ItemRef(kind=ref_kind, id=f"{probe_id}-{n}") for n in range(targets))
    )
    return Probe(
        probe_id=probe_id,
        kind=kind,
        lexical=False,
        query=f"query {probe_id}",
        now=_NOW,
        conversation_start=False,
        expected=expected,
        acceptable=(),
        forbidden=(),
        distractors=(),
        correction_id=None,
        rationale="hand-built",
    )


def _scores(
    probe: Probe,
    ce: list[float],
    cos: list[float] | None = None,
    *,
    targets: tuple[int | None, ...] | None = None,
    vector_ranks: tuple[int | None, ...] | None = None,
    pools: tuple[dict[str, int], ...] | None = None,
) -> ProbeScores:
    count = len(ce)
    return ProbeScores(
        probe=probe,
        item_ids=tuple(f"fact:{probe.probe_id}-item{n}" for n in range(count)),
        ce=np.asarray(ce, dtype=np.float64),
        cos=np.asarray(cos if cos is not None else [0.0] * count, dtype=np.float64),
        vector_ranks=vector_ranks if vector_ranks is not None else tuple(range(1, count + 1)),
        pool_ranks=pools if pools is not None else tuple({} for _ in range(count)),
        targets=targets if targets is not None else tuple(range(len(probe.expected))),
    )


def _pair(*probes: ProbeScores, fact_view: str = "F0", note_view: str = "N0") -> ViewPair:
    return ViewPair(fact_view=fact_view, note_view=note_view, probes=tuple(probes))


def _ce(fact_view: str = "F0", note_view: str = "N0") -> Setting:
    return Setting(fact_view=fact_view, note_view=note_view, scorer=CE)


class TestTStar:
    def test_t_star_is_the_highest_noise_maximum_so_every_noise_probe_is_empty(self):
        noise_a = _scores(_probe("noise-a", "noise"), [0.2, 0.7])
        noise_b = _scores(_probe("noise-b", "noise"), [0.4])
        fact = _scores(_probe("fact-a", targets=2), [0.9, 0.7])
        pair = _pair(noise_a, noise_b, fact)

        outcome = evaluate(pair, _ce(), _FACT_ONLY)

        assert outcome.t_star == 0.7
        for scores in (noise_a, noise_b):
            assert float(scores.ce.max()) <= outcome.t_star
        # A target passes only above t*: the one at 0.7 exactly does not.
        assert outcome.classes[0].figure == 0.5

    def test_no_noise_probe_is_refused(self):
        with pytest.raises(ValueError, match="noise"):
            evaluate(_pair(_scores(_probe("fact-a"), [0.9])), _ce(), _FACT_ONLY)

    def test_a_class_without_probes_is_refused(self):
        pair = _pair(_scores(_probe("noise-a", "noise"), [0.1]), _scores(_probe("f"), [0.9]))
        with pytest.raises(ValueError, match="note"):
            evaluate(pair, _ce(), _FACT_AND_NOTE)


class TestClassFigure:
    def test_figure_is_the_mean_share_of_expected_targets_above_t_star(self):
        noise = _scores(_probe("noise-a", "noise"), [0.5])
        half = _scores(_probe("fact-a", targets=2), [0.9, 0.1])
        whole = _scores(_probe("fact-b"), [0.9])
        # The target has no candidate (no version current at `now`): it never passes.
        absent = _scores(_probe("fact-c"), [0.9], targets=(None,))

        outcome = evaluate(_pair(noise, half, whole, absent), _ce(), _FACT_ONLY)

        [result] = outcome.classes
        assert result.probes == 3
        assert result.figure == pytest.approx(0.5)
        assert result.margin == pytest.approx(-0.35)

    def test_a_probe_without_expected_items_is_not_counted(self):
        empty = _probe("fact-empty", targets=0)
        noise = _scores(_probe("noise-a", "noise"), [0.5])
        scored = _scores(_probe("fact-a"), [0.9])

        outcome = evaluate(_pair(noise, scored, _scores(empty, [0.9])), _ce(), _FACT_ONLY)

        assert outcome.classes[0].probes == 1
        assert outcome.classes[0].figure == 1.0


class TestMarginAndWinner:
    def test_the_margin_is_the_worst_class_figure_minus_its_floor(self):
        noise = _scores(_probe("noise-a", "noise"), [0.5])
        fact = _scores(_probe("fact-a"), [0.9])
        note = _scores(_probe("note-a", "note", targets=2), [0.9, 0.2])

        outcome = evaluate(_pair(noise, fact, note), _ce(), _FACT_AND_NOTE)

        assert [r.margin for r in outcome.classes] == [pytest.approx(0.15), pytest.approx(-0.35)]
        assert outcome.margin == pytest.approx(-0.35)
        assert outcome.worst_class == "note"

    def test_the_largest_margin_wins_and_ties_go_to_the_first_setting(self):
        weak = _pair(
            _scores(_probe("noise-a", "noise"), [0.7]),
            _scores(_probe("fact-a"), [0.6]),
            fact_view="F0",
        )
        strong = _pair(
            _scores(_probe("noise-a", "noise"), [0.7]),
            _scores(_probe("fact-a"), [0.9]),
            fact_view="F1",
        )

        selection = select([weak, strong], _FACT_ONLY)

        # MAX over a zero cosine equals CE, so CE ties with every MAX setting and comes first.
        assert selection.winner.setting == _ce("F1")
        assert selection.winner.margin == pytest.approx(0.15)
        assert selection.tied == 1 + len(GRID)
        assert (selection.band, selection.exit_code) == ("proceed", EXIT_PROCEED)


class TestBands:
    @pytest.mark.parametrize(
        ("margin", "expected"),
        [
            (0.15, ("proceed", EXIT_PROCEED)),
            (0.05, ("proceed", EXIT_PROCEED)),
            (0.0499, ("flag", EXIT_FLAG)),
            (0.0, ("flag", EXIT_FLAG)),
            (-0.001, ("stop", EXIT_STOP)),
        ],
    )
    def test_band_edges(self, margin, expected):
        assert band(margin) == expected

    @pytest.mark.parametrize(
        ("targets", "passing", "name", "code"),
        [
            (8, 8, "proceed", EXIT_PROCEED),
            # 0.90 - 0.85 is 0.05000000000000004 in floats; the rounded margin is the edge.
            (10, 9, "proceed", EXIT_PROCEED),
            (8, 7, "flag", EXIT_FLAG),
            (8, 4, "stop", EXIT_STOP),
        ],
    )
    def test_each_band_sets_its_exit_code(self, targets, passing, name, code):
        ce = [0.9] * passing + [0.1] * (targets - passing)
        pair = _pair(
            _scores(_probe("noise-a", "noise"), [0.5]),
            _scores(_probe("fact-a", targets=targets), ce),
        )

        selection = select([pair], _FACT_ONLY)

        assert selection.winner.margin == pytest.approx(passing / targets - 0.85)
        assert (selection.band, selection.exit_code) == (name, code)
        # The winner is CE, so b is 0 and every pass is a cross-encoder-only one.
        assert selection.ce_only["passes"] == passing


class TestK:
    def test_k_is_the_smallest_step_holding_95_percent(self):
        assert choose_k([1] * 19 + [25]).value == 20
        choice = choose_k([10, 25, 25, 40])
        assert (choice.value, choice.flagged) == (50, False)
        assert choice.coverage[20] == 0.25
        assert choice.coverage[30] == 0.75

    def test_k_is_100_with_a_flag_when_no_step_reaches_95_percent(self):
        choice = choose_k([1, 150, 150, 150])
        assert (choice.value, choice.flagged) == (100, True)
        assert choice.coverage[100] == 0.25
        assert choose_k([]).flagged

    def test_k_reads_only_the_winners_fact_targets_that_pass(self):
        noise = _scores(_probe("noise-a", "noise"), [0.5])
        fact = _scores(_probe("fact-a", targets=2), [0.9, 0.1], vector_ranks=(5, 500))

        selection = select([_pair(noise, fact)], _FACT_ONLY)

        assert selection.k.targets == 1
        assert (selection.k.value, selection.k.flagged) == (20, False)


class TestBlend:
    def test_the_blend_is_reported_and_never_wins(self):
        # Each noise probe fools one model; the target is half-liked by both. Only a blend
        # separates them, and it must still not be chosen.
        noise_ce = _scores(_probe("noise-a", "noise"), [0.9], [0.0])
        noise_bi = _scores(_probe("noise-b", "noise"), [0.0], [0.9])
        fact = _scores(_probe("fact-a"), [0.5], [0.8])

        selection = select([_pair(noise_ce, noise_bi, fact)], _FACT_ONLY)

        assert selection.winner.setting.scorer != BLEND
        assert selection.winner.margin == pytest.approx(-0.85)
        assert selection.best_by_scorer[BLEND].margin == pytest.approx(0.15)
        assert selection.blend_flag
        assert (selection.band, selection.exit_code) == ("stop", EXIT_STOP)
        assert any(o.setting.scorer == BLEND for o in selection.outcomes)


class TestFusedPositions:
    def test_the_vector_pool_is_cut_at_k(self):
        scores = _scores(
            _probe("fact-a"),
            [0.0, 0.0, 0.0],
            vector_ranks=(3, 25, None),
            pools=({"fts": 1}, {}, {NOTE_VECTOR_POOL: 2}),
        )

        assert fused_positions(scores, 20) == [1, None, 2]
        assert fused_positions(scores, 30) == [1, 3, 2]


class TestDevClasses:
    def test_the_gated_classes_are_the_gates_recall_gates(self):
        recall = [gate for gate in PR1_GATES if gate.measure == "recall"]

        assert len(recall) == 12
        assert [(c.name, c.floor, c.member) for c in GATED_CLASSES] == [
            (gate.name, gate.floors["idle"], gate.predicate) for gate in recall
        ]
        assert not hasattr(view_probe, "PR1_CLASSES")

    def test_dev_class_sizes_match_the_plan(self):
        probes = load_probes()
        sizes = {gate.name: sum(1 for p in probes if gate.holds(p)) for gate in GATED_CLASSES}

        assert sizes == {
            "recall fact": 69,
            "recall fact, paraphrase": 49,
            "recall fact, plain shape": 43,
            "recall fact, plain shape, paraphrase": 27,
            "recall fact, list_item": 12,
            "recall fact, list_item, paraphrase": 8,
            "recall fact, type_scoped": 14,
            "recall note": 29,
            "recall note, paraphrase": 12,
            "recall correction with expected": 5,
            "recall phase 1 pooled": 103,
            "recall phase 1 pooled, paraphrase": 63,
        }
        assert sum(1 for p in probes if p.kind == "noise") == 17


class TestCommandLine:
    def test_the_parser_has_no_probe_file_option(self, tmp_path):
        parser = build_parser()
        assert not any("probe" in option for option in parser._option_string_actions)
        with pytest.raises(SystemExit) as raised:
            main(
                [
                    "--embedder",
                    "/opt/models/e",
                    "--reranker",
                    "/opt/models/r",
                    "--output",
                    str(tmp_path / "out.json"),
                    "--probes",
                    str(FIXTURE_DIR / "probes.jsonl"),
                ]
            )
        assert raised.value.code == EXIT_INPUT

    @pytest.mark.parametrize(
        "path", ["/tmp/model", "/opt/models", "/opt/models/../etc/model", "opt/models/model"]
    )
    def test_a_model_path_outside_opt_models_is_refused(self, path, tmp_path, monkeypatch):
        def no_load(*args, **kwargs):
            raise AssertionError("a refused path must not load")

        monkeypatch.setattr(view_probe, "load_models", no_load)
        output = tmp_path / "out.json"

        code = main(["--embedder", path, "--reranker", "/opt/models/r", "--output", str(output)])

        assert code == EXIT_INPUT
        assert not output.exists()
        with pytest.raises(ValueError):
            model_dir(path)

    def test_any_directory_under_opt_models_is_accepted(self):
        path = "/opt/models/candidates/bge-small-en-v1.5"
        assert model_dir(path) == Path(path)

    def test_a_missing_model_directory_is_an_input_error(self, tmp_path):
        code = main(
            [
                "--embedder",
                "/opt/models/no-such-embedder-here",
                "--reranker",
                "/opt/models/no-such-reranker-here",
                "--output",
                str(tmp_path / "out.json"),
            ]
        )
        assert code == EXIT_INPUT

    def test_an_unknown_view_is_an_input_error(self, tmp_path):
        with pytest.raises(SystemExit) as raised:
            main(
                [
                    "--embedder",
                    "/opt/models/e",
                    "--reranker",
                    "/opt/models/r",
                    "--output",
                    str(tmp_path / "out.json"),
                    "--fact-views",
                    "F0,F9",
                ]
            )
        assert raised.value.code == EXIT_INPUT


class _RecordingEmbedder:
    def __init__(self) -> None:
        self.inner = TokenEmbedder()
        self.texts: list[str] = []

    def generate_embedding(self, text: str) -> list[float]:
        self.texts.append(text)
        return self.inner.generate_embedding(text)

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        self.texts.extend(texts)
        return self.inner.generate_embeddings(texts)


class _RecordingReranker:
    def __init__(self) -> None:
        self.inner = TokenReranker()
        self.calls: list[tuple[str, list[str]]] = []

    def score(self, query, texts, budget_ms=None):
        self.calls.append((query, list(texts)))
        return self.inner.score(query, texts)

    def pairs_within(self, budget_ms):
        return self.inner.pairs_within(budget_ms)


_MODELS = [
    "--embedder",
    "/opt/models/candidates/bge-small-en-v1.5",
    "--reranker",
    "/opt/models/candidates/mxbai-rerank-xsmall-v1",
]


def _dev_run(directory: Path, argv: list[str]) -> dict[str, object]:
    """Run `main` over the dev fixture with recording fakes in place of the models."""
    embedder = _RecordingEmbedder()
    reranker = _RecordingReranker()
    loaded: list[tuple[Path, Path, str]] = []

    def fake_load(embedder_path, reranker_path, *, device):
        loaded.append((embedder_path, reranker_path, device))
        return embedder, reranker

    probe_files: list[Path] = []
    real_load_probes = harness.load_probes

    def recording_load_probes(path):
        probe_files.append(Path(path))
        return real_load_probes(path)

    output = directory / "view_probe.json"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(view_probe, "load_models", fake_load)
        patch.setattr(harness, "load_probes", recording_load_probes)
        code = main([*_MODELS, *argv, "--output", str(output)])
    return {
        "code": code,
        "result": json.loads(output.read_text(encoding="utf-8")),
        "embedder": embedder,
        "reranker": reranker,
        "loaded": loaded,
        "probe_files": probe_files,
    }


def _measured_queries() -> set[str]:
    return {p.query for p in load_probes() if view_probe.is_measured(p)}


class TestPrefixes:
    def test_both_prefixes_default_to_empty(self):
        args = build_parser().parse_args([*_MODELS, "--output", "out.json"])
        assert (args.query_prefix, args.passage_prefix) == ("", "")

    def test_the_passage_embedder_prepends_its_prefix_to_every_text(self):
        inner = _RecordingEmbedder()
        passages = PassageEmbedder(inner, "passage: ")

        passages.generate_embedding("one")
        passages.generate_embeddings(["two", "one", "three"])

        # Each text is embedded once, always with the prefix.
        assert inner.texts == ["passage: one", "passage: two", "passage: three"]

    def test_the_passage_embedder_passes_text_through_by_default(self):
        inner = _RecordingEmbedder()
        PassageEmbedder(inner).generate_embeddings(["one", "two"])
        assert inner.texts == ["one", "two"]


class TestDefaultPrefixRun:
    """A dev run with neither prefix: the bi-encoder reads the bare query and view texts."""

    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory):
        directory = tmp_path_factory.mktemp("view-probe-bare")
        return _dev_run(directory, ["--fact-views", "F1", "--note-views", "N0"])

    def test_queries_and_passages_are_embedded_bare(self, run):
        embedded = set(run["embedder"].texts)
        assert _measured_queries() <= embedded
        assert "dimension check" in embedded
        assert any(text.startswith("You ") for text in embedded)
        assert not any(text.startswith(("query: ", "passage: ")) for text in embedded)
        assert (run["result"]["query_prefix"], run["result"]["passage_prefix"]) == ("", "")


class TestDevRun:
    """One run over the dev fixture with the fakes, two fact views and one note view."""

    @pytest.fixture(scope="class")
    def run(self, tmp_path_factory):
        directory = tmp_path_factory.mktemp("view-probe")
        return _dev_run(
            directory,
            [
                "--fact-views",
                "F0,F2",
                "--note-views",
                "N1",
                "--query-prefix",
                "query: ",
                "--passage-prefix",
                "passage: ",
            ],
        )

    def test_reads_the_dev_set_only(self, run):
        assert run["probe_files"] == [FIXTURE_DIR / "probes.jsonl"]
        assert run["result"]["probe_set"] == "dev"
        assert run["result"]["probes"] == 103 + 17

    def test_loads_the_models_named(self, run):
        assert run["loaded"] == [
            (
                Path("/opt/models/candidates/bge-small-en-v1.5"),
                Path("/opt/models/candidates/mxbai-rerank-xsmall-v1"),
                "cpu",
            )
        ]

    def test_the_prefixes_reach_the_embedder_and_not_the_reranker(self, run):
        embedded = run["embedder"].texts
        assert embedded
        queries = _measured_queries()
        # A query carries the query prefix; every other text the bi-encoder embeds is a
        # passage (a fact view or a note chunk) and carries the passage prefix.
        bare_queries = {
            text.removeprefix("query: ") for text in embedded if text.startswith("query: ")
        }
        assert bare_queries == queries | {"dimension check"}
        passages = [text for text in embedded if not text.startswith("query: ")]
        assert passages
        assert all(text.startswith("passage: ") for text in passages)
        assert any(text.startswith("passage: You ") for text in passages)

        calls = run["reranker"].calls
        assert calls
        assert {query for query, _ in calls} == queries
        for _, texts in calls:
            assert not any(text.startswith(("query: ", "passage: ")) for text in texts)
        assert run["result"]["query_prefix"] == "query: "
        assert run["result"]["passage_prefix"] == "passage: "

    def test_the_output_holds_each_combination_the_winner_k_and_the_checks(self, run):
        result = run["result"]
        per_pair = 1 + 2 * len(GRID) + len(view_probe.BLEND_WEIGHTS) * len(GRID)
        assert len(result["combinations"]) == 2 * per_pair
        combination = result["combinations"][0]
        assert {"t_star", "margin", "worst_class", "classes"} <= set(combination)
        assert set(combination["classes"]) == {gate.name for gate in GATED_CLASSES}
        assert {"figure", "floor", "margin", "probes"} <= set(combination["classes"]["recall fact"])
        assert result["winner"]["scorer"] in ("CE", "BI", "MAX")
        assert result["winner"]["fact_view"] in ("F0", "F2")
        assert {"value", "flag", "targets", "coverage"} <= set(result["k"])
        assert result["k"]["value"] in view_probe.K_STEPS
        checks = result["checks"]
        assert {"ce_only_in_fused", "unit_norms", "sidecar_identity"} <= set(checks)
        # The fake embedder gives unit vectors, so the identity holds to float32 precision.
        assert checks["sidecar_identity"]["sampled"] > 0
        assert checks["sidecar_identity"]["max_abs_error"] < 1e-5
        assert checks["unit_norms"]["passages"]["max_deviation"] < 1e-9

    def test_the_exit_code_is_the_winners_band(self, run):
        result = run["result"]
        assert band(result["winner"]["margin"]) == (result["band"], run["code"])
        assert result["exit_code"] == run["code"]
