"""The selection grid (scripts/eval_harness/memory_selection_grid.py) and its fixed rule.

Nothing here loads a real model. The fact embedder and the reranker are the harness's fakes; a
gate embedder is either a `TokenEmbedder` (its probabilities vary from probe to probe, which the
bit-for-bit and derivation tests need) or a label oracle that maps each text to its fixture
label's vector, so its gate separates need from no-need and its variants are eligible. The
fixed rule's pure parts (`choose`, `load_check`, `followup_counts`, `derive_record`, the
variants and the preferred embedder) are tested on hand-built values.
"""

from __future__ import annotations

import asyncio
import json
import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from safetensors import SafetensorError

import backend.memory.need_gate as need_gate_module
import scripts.eval_harness.memory_selection_grid as grid
from backend.errors import MemoryRerankError, MistError
from backend.memory.need_gate import (
    FORM_CANDIDATES,
    CvResult,
    FormSpec,
    NeedBank,
    NeedFitError,
    context_text,
    fit_form,
    load_bank,
    load_models,
    theta,
)
from scripts.eval_harness.memory_recall import FIXTURE_DIR, ItemRef, Probe
from scripts.eval_harness.memory_recall_run import (
    FAKE_EMBEDDER_NAME,
    FAKE_EMBEDDER_REVISION,
    Fixture,
    RunSetting,
    TokenEmbedder,
    TokenReranker,
    run_settings,
)
from scripts.eval_harness.memory_selection_grid import (
    BASE_CONFIG,
    Followup,
    FollowupScore,
    GateEmbedder,
    GridPoint,
    Models,
    Scored,
    Setting,
    Variant,
    build_gate,
    build_parser,
    choose,
    common_set,
    derive_record,
    fit_gate,
    fixture_shadow_runner,
    followup_counts,
    gate_report,
    gate_variants,
    grid_config,
    judge_load,
    load_check,
    load_followups,
    main,
    need_model_record,
    preferred_embedder,
    run_grid,
    score_setting,
    variant_model,
)

# The fakes' fetches are timed on a shared runner, so the walks here give them time to finish;
# the derivation test compares blocks, not latency.
BASE = replace(BASE_CONFIG, deadline_ms=10_000)
POINT = GridPoint(fact_view="F3", select_k=10, rerank_top_n=20, facets=True)
_VECTORS = {"need": [1.0, 0.0], "no_need": [0.0, 1.0]}
BGE = "bge-small-en-v1.5"
GRANITE = "granite-embedding-small-english-r2"
_LATENCY = ("critical_path_ms", "timings_ms")
_REPORT_KEYS = {
    "fixed",
    "cv",
    "preferred",
    "gates",
    "settings",
    "load_checks",
    "followups",
    "reported",
    "need_context",
    "choice",
    "fresh_check",
    "need_model",
}


# --- shared inputs -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fixture() -> Fixture:
    return Fixture.load()


@pytest.fixture(scope="module")
def bank() -> NeedBank:
    return load_bank()


@pytest.fixture(scope="module")
def followups() -> tuple[Followup, ...]:
    return load_followups()


class _LabelOracle:
    """A gate embedder that sees only fixture labels: each known text maps to its label's vector.

    A text it does not know (the harness's warm-up query) is no-need. A prefix, when given, is
    taken off before the lookup, so the oracle can stand for a gate embedder that has one.
    """

    def __init__(self, labels: Mapping[str, str], prefix: str = "") -> None:
        self._labels = dict(labels)
        self._prefix = prefix

    def generate_embedding(self, text: str) -> list[float]:
        if self._prefix and text.startswith(self._prefix):
            text = text[len(self._prefix) :]
        return list(_VECTORS[self._labels.get(text, "no_need")])

    def generate_embeddings(self, texts: list[str]) -> list[list[float]]:
        return [self.generate_embedding(text) for text in texts]


def _labels(bank: NeedBank, fixture: Fixture, followups: Sequence[Followup]) -> dict[str, str]:
    labels = {exemplar.text: exemplar.label for exemplar in bank.exemplars}
    for probe in fixture.probes:
        labels.setdefault(probe.query, "no_need" if probe.kind == "noise" else "need")
    for row in followups:
        for text in (row.previous, row.query, context_text(row.previous, row.query)):
            labels.setdefault(text, row.label)
    return labels


def _oracle(
    name: str, bank: NeedBank, fixture: Fixture, followups: Sequence[Followup]
) -> GateEmbedder:
    prefix = need_gate_module.NEED_EMBEDDERS[name].query_prefix
    return GateEmbedder(
        name=name,
        key_name=name,
        revision="oracle-v1",
        embedder=_LabelOracle(_labels(bank, fixture, followups), prefix),
    )


def _shared(embedder: TokenEmbedder) -> GateEmbedder:
    return GateEmbedder(
        name="shared",
        key_name=FAKE_EMBEDDER_NAME,
        revision=FAKE_EMBEDDER_REVISION,
        embedder=embedder,
    )


# --- hand-built values ---------------------------------------------------------------------------

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _probe(probe_id: str, kind: str, expected: Sequence[tuple[str, str]] = ()) -> Probe:
    return Probe(
        probe_id=probe_id,
        kind=kind,
        lexical=True,
        query=f"query {probe_id}",
        now=_NOW,
        conversation_start=False,
        expected=tuple(ItemRef(k, i) for k, i in expected),
        acceptable=(),
        forbidden=(),
        distractors=(),
        correction_id=None,
        rationale="",
    )


def _need(p: float | None, **overrides: object) -> dict[str, object]:
    need: dict[str, object] = {
        "mode": "shadow",
        "band": "open",
        "p": p,
        "theta_low": 0.5,
        "theta_high": 0.5,
        "band_k": 1,
        "need_ids": ["a-need-001"],
        "no_need_ids": ["a-no-need-001"],
        "context": False,
        "context_p": None,
        "degraded": None,
    }
    need.update(overrides)
    return need


def _items(refs: Sequence[ItemRef], start: int = 1) -> list[dict[str, object]]:
    return [
        {"ref": f"m{n}", "kind": ref.kind, "id": ref.id, "score": 1.0, "rank": n}
        for n, ref in enumerate(refs, start=start)
    ]


def _record(
    probe: Probe,
    p: float | None,
    *,
    drop: bool = False,
    critical_path_ms: float = 5.0,
    mode: str = "shadow",
    **need: object,
) -> dict[str, object]:
    """A shadow run's record serving exactly the probe's expected items (none for noise)."""
    refs = () if drop or probe.kind == "noise" else probe.expected
    items = _items(refs)
    tokens = 10 * len(items)
    record: dict[str, object] = {
        "probe_id": probe.probe_id,
        "items": items,
        "tokens": tokens,
        "budget_tokens": 500,
        "critical_path_ms": critical_path_ms,
        "timings_ms": {"stage1": 1.0, "stage2": 1.0, "rerank": 1.0, "queue_wait": 0.0},
        "reranked": 10,
        "shortlisted": 10,
        "degraded_reason": None,
        "delta_tokens_est": tokens + 4,
        "cpu_load_procs": 0,
        "embedder_device": "cpu",
        "need": _need(p, mode=mode, **need),
        "stage_ids": {
            "fused": [f"{r.kind}:{r.id}" for r in refs],
            "shortlisted": [f"{r.kind}:{r.id}" for r in refs],
            "selected": [f"{r.kind}:{r.id}" for r in refs],
        },
    }
    if items and mode == "shadow":
        record["cuts"] = {
            "1": {"items": items[:1], "tokens": 10},
            "2": {"items": items[:2], "tokens": 10 * len(items[:2])},
        }
    return record


def _variant(
    low: float = 0.5, high: float = 0.7, *, r: float = 0.95, k: int = 1, eligible: bool = True
) -> Variant:
    return Variant(recall_target=r, band_k=k, theta_low=low, theta_high=high, eligible=eligible)


def _scored(
    *,
    view: str = "F3",
    select_k: int = 10,
    rerank_top_n: int = 30,
    facets: bool = True,
    embedder: str = "shared",
    r: float = 0.95,
    k: int = 2,
    passed: bool = True,
    margin: float | None = 0.1,
    cost: float | None = 100.0,
) -> Scored:
    return Scored(
        setting=Setting(
            point=GridPoint(view, select_k, rerank_top_n, facets),
            embedder=embedder,
            variant=_variant(r=r, k=k),
        ),
        passed=passed,
        margin=margin,
        cost=cost,
        failures=() if passed else ("recall fact",),
        report={},
    )


def _cv(name: str, closure: Sequence[Sequence[float]]) -> CvResult:
    spec = FormSpec("logistic", 1.0, None)
    folds = tuple(tuple(fold) for fold in closure)
    return CvResult(spec, folds, folds, (0.9, 0.9), (0.3, 0.3), (0.1, 0.1))


# --- the command line ----------------------------------------------------------------------------


def test_parser_has_no_probes_option() -> None:
    # The grid reads the dev set and the dev follow-ups only, so no flag can point it elsewhere.
    parser = build_parser()
    assert "--probes" not in parser._option_string_actions
    assert not any("probe" in flag for flag in parser._option_string_actions)
    with pytest.raises(SystemExit) as exc:
        main(_argv("--probes", "/tmp/heldout.jsonl"))
    assert exc.value.code == 2


def _argv(*extra: str, output: str = "/tmp/grid.json") -> list[str]:
    return [
        "--embedder",
        "/opt/models/all-MiniLM-L6-v2",
        "--reranker",
        "/opt/models/ms-marco-MiniLM-L-6-v2",
        "--device",
        "cpu",
        "--embedder-device",
        "cpu",
        "--output",
        output,
        *extra,
    ]


def test_need_embedder_flag_parses_name_path_and_revision() -> None:
    args = build_parser().parse_args(
        _argv(
            "--need-embedder",
            f"{BGE}=/opt/models/candidates/{BGE}@5c38ec7c",
            "--need-embedder",
            f"{GRANITE}=/opt/models/candidates/{GRANITE}@abc@def",
        )
    )
    assert args.need_embedder == [
        (BGE, f"/opt/models/candidates/{BGE}", "5c38ec7c"),
        (GRANITE, f"/opt/models/candidates/{GRANITE}@abc", "def"),
    ]


@pytest.mark.parametrize(
    "value",
    [
        f"/opt/models/candidates/{BGE}@5c38",  # no name
        f"{BGE}=/opt/models/candidates/{BGE}",  # no revision
        f"{BGE}=/opt/models/candidates/{BGE}@",  # empty revision
        "shared=/opt/models/all-MiniLM-L6-v2@x",  # shared is the fact embedder, never a path
        "minilm=/opt/models/x@y",  # not a NEED_EMBEDDERS name
        f"{BGE}=/tmp/{BGE}@5c38",  # outside /opt/models
    ],
)
def test_malformed_need_embedder_exits_2(value: str) -> None:
    with pytest.raises(SystemExit) as exc:
        main(_argv("--need-embedder", value))
    assert exc.value.code == 2


def test_a_need_embedder_named_twice_exits_2() -> None:
    flag = f"{BGE}=/opt/models/candidates/{BGE}@5c38"
    with pytest.raises(SystemExit) as exc:
        main(_argv("--need-embedder", flag, "--need-embedder", flag))
    assert exc.value.code == 2


def test_a_model_outside_opt_models_exits_2() -> None:
    argv = _argv()
    argv[1] = "/tmp/all-MiniLM-L6-v2"
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 2


def test_a_fact_embedder_missing_from_the_lock_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "models"
    monkeypatch.setattr(grid, "MODELS_ROOT", root)
    for name in ("fact", "reranker"):
        (root / name).mkdir(parents=True)
    argv = _argv(output=str(tmp_path / "grid.json"))
    argv[argv.index("--embedder") + 1] = str(root / "fact")
    argv[argv.index("--reranker") + 1] = str(root / "reranker")
    lock = tmp_path / "models.lock"
    lock.write_text(
        '[[model]]\nname = "other"\nrepo = "x/other"\nrevision = "1"\nlicense = "MIT"\n'
        'target = "other"\n',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        main(argv, models_lock=lock)
    assert exc.value.code == 2
    assert "has no entry named 'fact'" in capsys.readouterr().err


@pytest.mark.parametrize("flag", ["--embedder", "--reranker", "--need-embedder"])
def test_a_model_directory_that_does_not_exist_exits_2(
    flag: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    # A path that cannot load is an input error, not the exit 1 of a grid where nothing passes.
    root = tmp_path / "models"
    monkeypatch.setattr(grid, "MODELS_ROOT", root)
    for name in ("fact", "reranker"):
        (root / name).mkdir(parents=True)
    absent = str(root / "absent")
    argv = _argv(output=str(tmp_path / "grid.json"))
    argv[argv.index("--embedder") + 1] = str(root / "fact")
    argv[argv.index("--reranker") + 1] = str(root / "reranker")
    if flag == "--need-embedder":
        argv += ["--need-embedder", f"{BGE}={absent}@5c38"]
    else:
        argv[argv.index(flag) + 1] = absent
    lock = tmp_path / "models.lock"
    lock.write_text(
        '[[model]]\nname = "fact"\nrepo = "x/fact"\nrevision = "1"\nlicense = "MIT"\n'
        'target = "fact"\n',
        encoding="utf-8",
    )
    with pytest.raises(SystemExit) as exc:
        main(argv, models_lock=lock)
    assert exc.value.code == 2
    assert absent in capsys.readouterr().err


# --- exit 4: a model that exists but fails to load -----------------------------------------------

_FACT_LOCK = (
    '[[model]]\nname = "fact"\nrepo = "x/fact"\nrevision = "1"\nlicense = "MIT"\n'
    'target = "fact"\n'
)
# Each loader `load_real_models` calls, and the directory under /opt/models it loads.
_LOADERS = {
    "SentenceTransformerEmbedder": "fact",
    "build_reranker": "reranker",
    "EmbeddingGenerator": BGE,
    "warmup": BGE,
}


def _present_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], Path]:
    """Argv naming three model directories that exist, and a lock with the fact embedder."""
    root = tmp_path / "models"
    monkeypatch.setattr(grid, "MODELS_ROOT", root)
    for name in ("fact", "reranker", BGE):
        (root / name).mkdir(parents=True)
    argv = _argv("--need-embedder", f"{BGE}={root / BGE}@5c38", output=str(tmp_path / "grid.json"))
    argv[argv.index("--embedder") + 1] = str(root / "fact")
    argv[argv.index("--reranker") + 1] = str(root / "reranker")
    lock = tmp_path / "models.lock"
    lock.write_text(_FACT_LOCK, encoding="utf-8")
    return argv, lock


def _stub_loaders(
    monkeypatch: pytest.MonkeyPatch,
    *,
    failing: str | None = None,
    error: BaseException | None = None,
) -> list[str]:
    """Replace the four loaders: the one named `failing` raises `error`, every other one loads.

    Unstubbed, the first loader would already fail on an empty directory (and `build_reranker`
    refuses any path outside the real /opt/models), so a test would never reach a later loader.
    Returns the paths each gate embedder's `warmup` ran for.
    """
    warmed: list[str] = []

    def check(name: str) -> None:
        if name == failing:
            assert error is not None
            raise error

    class Fact:
        def __init__(self, path: str, *, device: str) -> None:
            check("SentenceTransformerEmbedder")

    def build_reranker(choice: str, *, device: str, config: object) -> SimpleNamespace:
        check("build_reranker")
        return SimpleNamespace(reranker=TokenReranker())

    class Generator:
        def __init__(self, path: str, *, device: str, **kwargs: object) -> None:
            check("EmbeddingGenerator")
            self.path = path

        def warmup(self) -> None:
            check("warmup")
            warmed.append(self.path)

    monkeypatch.setattr(grid, "SentenceTransformerEmbedder", Fact)
    monkeypatch.setattr(grid, "build_reranker", build_reranker)
    monkeypatch.setattr(grid, "EmbeddingGenerator", Generator)
    return warmed


def test_load_errors_are_reranks_model_errors_and_the_rerank_and_import_errors() -> None:
    assert (
        OSError,
        ValueError,
        RuntimeError,
        ImportError,
        SafetensorError,
        MemoryRerankError,
    ) == grid._LOAD_ERRORS
    assert issubclass(grid.GridLoadError, MistError)


def test_load_real_models_loads_every_model_and_warms_each_gate_embedder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, lock = _present_models(tmp_path, monkeypatch)
    warmed = _stub_loaders(monkeypatch)
    models = grid.load_real_models(build_parser().parse_args(argv), models_lock=lock)
    assert [g.name for g in models.gate_embedders] == ["shared", BGE]
    assert warmed == [str(tmp_path / "models" / BGE)]


@pytest.mark.parametrize(
    "error",
    [OSError, ValueError, RuntimeError, ImportError, SafetensorError, MemoryRerankError],
)
@pytest.mark.parametrize("loader", list(_LOADERS))
def test_a_model_that_fails_to_load_raises_grid_load_error_naming_it(
    loader: str, error: type[Exception], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv, lock = _present_models(tmp_path, monkeypatch)
    _stub_loaders(monkeypatch, failing=loader, error=error(f"{loader} stub failure"))
    with pytest.raises(grid.GridLoadError) as exc:
        grid.load_real_models(build_parser().parse_args(argv), models_lock=lock)
    directory = tmp_path / "models" / _LOADERS[loader]
    assert str(exc.value) == f"{directory}: {error.__name__}: {loader} stub failure"
    assert type(exc.value.__cause__) is error


def test_a_type_error_from_a_loader_is_not_caught(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A programming error keeps its traceback and Python's exit 1.
    argv, lock = _present_models(tmp_path, monkeypatch)
    _stub_loaders(monkeypatch, failing="SentenceTransformerEmbedder", error=TypeError("stub"))
    with pytest.raises(TypeError, match="stub"):
        grid.load_real_models(build_parser().parse_args(argv), models_lock=lock)


def test_main_exits_4_with_one_line_and_no_report_when_a_model_fails_to_load(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "grid.json"

    def models(_args: object) -> Models:
        raise grid.GridLoadError("/opt/models/fact: OSError: stub")

    code = main(_argv(output=str(output)), models=models)
    assert code == grid.EXIT_LOAD == 4
    assert not output.exists()
    err = capsys.readouterr().err
    assert err == "memory_selection_grid: exit 4, /opt/models/fact: OSError: stub\n"


def test_a_gate_embedder_whose_warmup_fails_exits_4_before_any_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv, lock = _present_models(tmp_path, monkeypatch)
    _stub_loaders(monkeypatch, failing="warmup", error=OSError("no weights"))

    def no_fit(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("a model failed to load, so nothing is fit")

    monkeypatch.setattr(grid, "fit_gate", no_fit)
    assert main(argv, models_lock=lock) == 4
    assert not (tmp_path / "grid.json").exists()
    directory = tmp_path / "models" / BGE
    assert capsys.readouterr().err == (
        f"memory_selection_grid: exit 4, {directory}: OSError: no weights\n"
    )


# --- step 1: the gate alone ----------------------------------------------------------------------


def _hand_probes(n_need: int, n_noise: int) -> list[Probe]:
    probes = [_probe(f"f{n}", "fact", [("fact", f"k{n}")]) for n in range(n_need)]
    probes += [_probe(f"z{n}", "noise") for n in range(n_noise)]
    probes.append(_probe("e0", "episode", [("episode", "ep0")]))  # not a PR 1 recall class
    return probes


def test_variants_use_theta_over_the_recall_probes() -> None:
    probes = _hand_probes(20, 3)
    p = {f"f{n}": 0.40 + n * 0.03 for n in range(20)}
    p |= {"z0": 0.01, "z1": 0.02, "z2": 0.03, "e0": 0.0}
    variants = gate_variants(p, probes)
    need = [p[f"f{n}"] for n in range(20)]
    assert [(v.recall_target, v.band_k) for v in variants] == [
        (0.95, 1),
        (0.95, 2),
        (0.98, 1),
        (0.98, 2),
    ]
    for v in variants:
        assert v.theta_low == theta(need, v.recall_target)
        assert v.theta_high == theta(need, round(v.recall_target - 0.10, 2))
        assert v.eligible
    # The episode probe's p of 0 never lowers a threshold: it is not a PR 1 recall probe.
    assert variants[0].theta_low > 0.3


def test_a_noise_probe_at_theta_low_makes_the_variant_ineligible() -> None:
    probes = _hand_probes(20, 2)
    p = {f"f{n}": 0.40 + n * 0.03 for n in range(20)} | {"e0": 0.0}
    need = [p[f"f{n}"] for n in range(20)]
    low_95 = theta(need, 0.95)
    # One noise probe exactly at R 0.95's theta_low, which is at or above R 0.98's.
    p |= {"z0": 0.01, "z1": low_95}
    variants = {v.recall_target: v for v in gate_variants(p, probes)}
    assert not variants[0.95].eligible  # a p equal to theta_low is not below it
    assert theta(need, 0.98) <= low_95
    assert not variants[0.98].eligible
    p["z1"] = math.nextafter(theta(need, 0.98), 0.0)
    variants = {v.recall_target: v for v in gate_variants(p, probes)}
    assert variants[0.98].eligible and variants[0.95].eligible


def test_theta_high_is_never_below_theta_low() -> None:
    probes = _hand_probes(20, 1)
    # Ties at the cut and a flat top: the k-th for R - 0.10 and for R can be the same value.
    p = {f"f{n}": 0.9 if n >= 2 else 0.2 for n in range(20)} | {"z0": 0.0, "e0": 0.0}
    for v in gate_variants(p, probes):
        assert v.theta_high >= v.theta_low
    flat = {f"f{n}": 0.7 for n in range(20)} | {"z0": 0.0, "e0": 0.0}
    for v in gate_variants(flat, probes):
        assert v.theta_high == v.theta_low == 0.35


@pytest.mark.parametrize(
    ("other", "preferred"),
    [
        ([[0.85, 0.85], [0.85, 0.85]], BGE),  # 0.05 above and above in every fold and target
        ([[0.849, 0.849], [0.849, 0.849]], "shared"),  # 0.049 above
        ([[0.95, 0.95], [0.80, 0.80]], "shared"),  # 0.075 above, equal in fold 1
        ([[0.95, 0.95], [0.85, 0.79]], "shared"),  # below in fold 1 at R 0.98
    ],
)
def test_preferred_embedder_needs_005_and_every_fold_and_target(
    other: list[list[float]], preferred: str
) -> None:
    scores = {"shared": _cv("shared", [[0.80, 0.80], [0.80, 0.80]]), BGE: _cv(BGE, other)}
    assert preferred_embedder(scores) == preferred


def test_preferred_embedder_is_the_highest_of_several() -> None:
    scores = {
        "shared": _cv("shared", [[0.80, 0.80], [0.80, 0.80]]),
        BGE: _cv(BGE, [[0.86, 0.86], [0.86, 0.86]]),
        GRANITE: _cv(GRANITE, [[0.90, 0.90], [0.90, 0.90]]),
    }
    assert preferred_embedder(scores) == GRANITE
    # Without a form for `shared` nothing is preferred, and rule 1 yields nothing.
    assert preferred_embedder({BGE: scores[BGE]}) is None


def test_fit_gate_scores_every_dev_probe_and_followup_from_the_gate_alone(
    fixture: Fixture, bank: NeedBank, followups: tuple[Followup, ...]
) -> None:
    fit = fit_gate(_oracle(GRANITE, bank, fixture, followups), bank, fixture.probes, followups)
    assert fit.form == FormSpec("logistic", 0.1, None)  # every candidate closes all: smoothest
    assert set(fit.p) == {probe.probe_id for probe in fixture.probes}
    assert [s.followup_id for s in fit.followups] == [f.followup_id for f in followups]
    assert all(v.eligible for v in fit.variants)
    noise = [fit.p[p.probe_id] for p in fixture.probes if p.kind == "noise"]
    assert len(noise) == 17
    assert max(noise) < min(v.theta_low for v in fit.variants)


def test_a_failed_fit_is_reported_and_skipped(
    monkeypatch: pytest.MonkeyPatch,
    fixture: Fixture,
    bank: NeedBank,
    followups: tuple[Followup, ...],
) -> None:
    # Logistic c 0.1 fails in cross-validation; c 1 cross-validates but fails the final fit.
    real = need_gate_module.fit_form

    def cv_stub(vectors: object, labels: object, spec: FormSpec) -> dict[str, object]:
        if spec == FormSpec("logistic", 0.1, None):
            raise NeedFitError("logistic c=0.1 reached 100 iterations (stub)")
        return real(vectors, labels, spec)

    def final_stub(vectors: object, labels: object, spec: FormSpec) -> dict[str, object]:
        if spec == FormSpec("logistic", 1.0, None):
            raise NeedFitError("logistic c=1 reached 100 iterations (stub)")
        return fit_form(vectors, labels, spec)

    monkeypatch.setattr(need_gate_module, "fit_form", cv_stub)
    oracle = _oracle(GRANITE, bank, fixture, followups)
    fit = fit_gate(oracle, bank, fixture.probes, followups, fit=final_stub)
    assert fit.form == FormSpec("logistic", 10.0, None)
    assert fit.failed == (FormSpec("logistic", 0.1, None), FormSpec("logistic", 1.0, None))
    report = grid.cv_report(fit)
    by_form = {row["form"]: row for row in report["candidates"]}
    assert by_form["logistic c=0.1"]["failed"] == "cross-validation"
    assert by_form["logistic c=1"]["failed"] == "final fit"
    assert by_form["logistic c=10"]["failed"] is None
    assert report["form"] == "logistic c=10"
    # Each row names the writer each fold scores, a failed candidate's too.
    writers = sorted({exemplar.writer for exemplar in bank.exemplars})
    assert [row["scored"] for row in report["candidates"]] == [
        list(result.scored) for result in fit.cv
    ]
    assert all(row["scored"] == writers for row in report["candidates"])


def test_three_writers_give_three_folds_and_the_preference_reads_every_fold(
    fixture: Fixture, followups: tuple[Followup, ...]
) -> None:
    # A bank of three writers: the cross-validation table holds a fold per writer, not two.
    three = load_bank(need_gate_module.DEFAULT_BANK_PATHS[:3])
    writers = sorted({exemplar.writer for exemplar in three.exemplars})
    assert len(writers) == 3
    fit = fit_gate(_oracle(BGE, three, fixture, followups), three, fixture.probes, followups)
    rows = grid.cv_report(fit)["candidates"]
    assert len(rows) == len(FORM_CANDIDATES)
    for row in rows:
        assert row["scored"] == writers
        if row["failed"] is None:
            assert len(row["closure"]) == len(row["hard_closure"]) == 3
            assert all(len(fold) == len(grid.RECALL_TARGETS) for fold in row["closure"])
            assert len(row["auroc"]) == len(row["log_loss"]) == len(row["brier"]) == 3
    # The preferred-embedder rule compares every fold: above in two folds and equal in the
    # third, at one target, is not above `shared` in each fold.
    base = _cv("shared", [[0.80, 0.80]] * 3)
    above = _cv(BGE, [[0.90, 0.90]] * 3)
    equal_in_third = _cv(BGE, [[0.90, 0.90], [0.90, 0.90], [0.90, 0.80]])
    below_in_third = _cv(BGE, [[0.95, 0.95], [0.95, 0.95], [0.79, 0.95]])
    assert preferred_embedder({"shared": base, BGE: above}) == BGE
    assert preferred_embedder({"shared": base, BGE: equal_in_third}) == "shared"
    assert preferred_embedder({"shared": base, BGE: below_in_third}) == "shared"


def test_every_candidate_failing_exits_1(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fixture: Fixture,
    bank: NeedBank,
    followups: tuple[Followup, ...],
) -> None:
    monkeypatch.setattr(need_gate_module, "NEWTON_MAX_ITERATIONS", 0)
    shared = TokenEmbedder()
    models = Models(
        fact_embedder=shared,
        reranker=TokenReranker(),
        gate_embedders=(_shared(shared), _oracle(GRANITE, bank, fixture, followups)),
    )
    output = tmp_path / "grid.json"

    def never(*_: object) -> list[list[dict[str, object]]]:
        raise AssertionError("no gate, so no shadow run")

    code = main(
        _argv(output=str(output)),
        models=lambda _args: models,
        points=(POINT,),
        base=BASE,
        shadow_runner=never,
        load_runner=never,
    )
    assert code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    for name in ("shared", GRANITE):
        assert report["cv"][name]["form"] is None
        assert {row["failed"] for row in report["cv"][name]["candidates"]} == {"cross-validation"}
    assert report["choice"] is None and report["need_model"] is None
    # The real fresh file is read, but with no choice there is no gate to check it with.
    assert report["fresh_check"] is None


# --- step 2: shadow runs -------------------------------------------------------------------------


def test_shadow_runs_one_run_settings_call_per_view_and_embedder(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fixture: Fixture,
    bank: NeedBank,
    followups: tuple[Followup, ...],
) -> None:
    calls: list[dict[str, object]] = []

    async def recorder(fx: Fixture, **kwargs: object) -> list[list[dict[str, object]]]:
        calls.append(kwargs)
        settings = kwargs["settings"]
        assert isinstance(settings, list)
        gate = settings[0].need_gate
        p = {pr.probe_id: gate.decide(gate.embed_query(pr.query)).p for pr in fx.probes}
        return [[_record(pr, p[pr.probe_id]) for pr in fx.probes] for _ in settings]

    monkeypatch.setattr(grid, "run_settings", recorder)
    fact = TokenEmbedder()
    embedders = (_oracle(BGE, bank, fixture, followups), _oracle(GRANITE, bank, fixture, followups))
    points = (
        GridPoint("F2", 5, 20, False),
        GridPoint("F3", 5, 20, False),
        GridPoint("F3", 8, 20, True),
    )
    outcome = run_grid(
        fixture,
        followups,
        bank,
        embedders,
        points=points,
        base=BASE,
        shadow_runner=fixture_shadow_runner(
            fixture, fact_embedder=fact, reranker=TokenReranker(), base=BASE, workdir=tmp_path
        ),
        load_runner=_passing_load_runner(fixture),
    )
    assert len(calls) == 4  # two fact views by two gate embedders
    seen = set()
    for call in calls:
        settings = call["settings"]
        views = {s.config.fact_view for s in settings}
        assert len(views) == 1
        assert {s.config.need_gate for s in settings} == {"shadow"}
        assert len({id(s.need_gate) for s in settings}) == 1
        assert call["warm_up"] is True
        assert call["probes"] is None and call["followups"] == ()
        assert call["embedder"] is fact
        names = {s.config.need_embedder for s in settings}
        seen.add((views.pop(), names.pop(), len(settings)))
    assert seen == {("F2", BGE, 1), ("F3", BGE, 2), ("F2", GRANITE, 1), ("F3", GRANITE, 2)}
    # Every retrieval setting by every eligible variant of each embedder is a setting.
    assert len(outcome.report["settings"]) == 3 * 4 * 2


# --- step 3: the derivation ----------------------------------------------------------------------


def test_derive_a_degraded_shadow_gate_closes_with_its_reason() -> None:
    probe = _probe("f1", "fact", [("fact", "k1"), ("fact", "k2")])
    record = _record(probe, None, band="closed", degraded="deadline")
    derived = derive_record(record, _variant())
    assert derived["items"] == [] and derived["tokens"] == 0
    assert derived["degraded_reason"] == "deadline"
    assert derived["reranked"] == 0 and derived["shortlisted"] == 0
    assert derived["stage_ids"] == {"fused": [], "shortlisted": [], "selected": []}
    assert derived["delta_tokens_est"] == 4
    assert "cuts" not in derived
    assert derived["need"]["mode"] == "on" and derived["need"]["band"] == "closed"
    assert derived["need"]["degraded"] == "deadline"
    assert derived["critical_path_ms"] == record["critical_path_ms"]
    # `need_unknown` is kept as well, even with a p from the query.
    record = _record(probe, 0.9, band="closed", degraded="need_unknown")
    assert derive_record(record, _variant())["degraded_reason"] == "need_unknown"


def test_derive_a_closed_variant_empties_the_block() -> None:
    probe = _probe("f1", "fact", [("fact", "k1"), ("fact", "k2")])
    derived = derive_record(_record(probe, 0.3), _variant(0.5, 0.7))
    assert derived["items"] == [] and derived["tokens"] == 0
    assert derived["degraded_reason"] is None
    assert derived["need"] == _need(
        0.3, mode="on", band="closed", theta_low=0.5, theta_high=0.7, band_k=1
    )


def test_derive_a_band_variant_takes_the_cut_of_band_k() -> None:
    probe = _probe("f1", "fact", [("fact", "k1"), ("fact", "k2"), ("fact", "k3")])
    record = _record(probe, 0.6)
    one = derive_record(record, _variant(0.5, 0.7, k=1))
    two = derive_record(record, _variant(0.5, 0.7, k=2))
    assert one["items"] == record["cuts"]["1"]["items"] and one["tokens"] == 10
    assert two["items"] == record["cuts"]["2"]["items"] and two["tokens"] == 20
    assert two["delta_tokens_est"] == 24
    assert two["reranked"] == 10 and two["stage_ids"] == record["stage_ids"]
    assert two["need"]["band"] == "band" and two["need"]["band_k"] == 2
    # A band probe whose fetch built no block derives as an empty block.
    bare = {k: v for k, v in record.items() if k != "cuts"}
    assert derive_record(bare, _variant(0.5, 0.7))["items"] == []
    # The input is not changed.
    assert record["need"]["mode"] == "shadow" and "cuts" in record


def test_derive_an_open_variant_keeps_the_block() -> None:
    probe = _probe("f1", "fact", [("fact", "k1")])
    record = _record(probe, 0.9)
    derived = derive_record(record, _variant(0.5, 0.7))
    expected = {k: v for k, v in record.items() if k not in ("cuts", "need")}
    assert {k: v for k, v in derived.items() if k != "need"} == expected
    assert derived["need"]["band"] == "open" and derived["need"]["mode"] == "on"


def test_derive_refuses_a_record_without_a_shadow_trace() -> None:
    probe = _probe("f1", "fact", [("fact", "k1")])
    with pytest.raises(ValueError, match="shadow"):
        derive_record(_record(probe, 0.9, mode="on"), _variant())


@pytest.fixture(scope="module")
def shared_shadow(
    tmp_path_factory: pytest.TempPathFactory,
    fixture: Fixture,
    bank: NeedBank,
    followups: tuple[Followup, ...],
) -> tuple[object, TokenEmbedder, list[dict[str, object]]]:
    """A `shared` gate on the fake fact embedder, fit, and its shadow run at `POINT`."""
    fact = TokenEmbedder()
    fit = fit_gate(_shared(fact), bank, fixture.probes, followups)
    runner = fixture_shadow_runner(
        fixture,
        fact_embedder=fact,
        reranker=TokenReranker(),
        base=BASE,
        workdir=tmp_path_factory.mktemp("shared-shadow"),
    )
    [records] = runner(fit, "F3", [POINT])
    return fit, fact, records


def _refs(items: Sequence[Mapping[str, object]]) -> list[object]:
    return list(dict.fromkeys(item["ref"] for item in items))


def test_derived_records_equal_gated_runs(
    tmp_path: Path,
    fixture: Fixture,
    shared_shadow: tuple[object, TokenEmbedder, list[dict[str, object]]],
) -> None:
    fit, fact, shadow = shared_shadow
    # A probe whose open block starts with a list: its one-entry cut keeps two or more facts
    # under one ref. And one whose open block has a single entry, so its cut is the block.
    list_start = next(
        r
        for r in shadow
        if "cuts" in r
        and len(r["cuts"]["1"]["items"]) > 1
        and len(_refs(r["cuts"]["1"]["items"])) == 1
        and len(_refs(r["items"])) > 1
    )
    one_entry = next(r for r in shadow if r["items"] and len(_refs(r["items"])) == 1)
    variants = []
    for record, k in ((list_start, 1), (one_entry, 2)):
        p = record["need"]["p"]
        variants.append(_variant(p, math.nextafter(p, 1.0), k=k))
    settings = [
        RunSetting(
            grid_config(BASE, POINT, need_gate="on", need_embedder="shared", need_context=False),
            build_gate(fit, variant_model(fit, v, source="test")),
        )
        for v in variants
    ]
    gated = asyncio.run(
        run_settings(
            fixture,
            settings=settings,
            embedder=fact,
            reranker=TokenReranker(),
            workdir=tmp_path,
            warm_up=True,
        )
    )
    bands = {"closed": 0, "band": 0, "open": 0}
    for variant, records in zip(variants, gated):
        assert len(records) == len(shadow)
        for source, real in zip(shadow, records):
            derived = derive_record(source, variant)
            assert {k: v for k, v in derived.items() if k not in _LATENCY} == {
                k: v for k, v in real.items() if k not in _LATENCY
            }, source["probe_id"]
            bands[real["need"]["band"]] += 1
    by_id = {r["probe_id"]: r for r in gated[0]}
    assert by_id[list_start["probe_id"]]["need"]["band"] == "band"
    assert by_id[list_start["probe_id"]]["items"] == list_start["cuts"]["1"]["items"]
    assert {r["probe_id"]: r for r in gated[1]}[one_entry["probe_id"]]["items"] == one_entry[
        "items"
    ]
    assert all(bands.values())  # each band occurs, so each branch of the derivation was compared


@pytest.mark.parametrize("which", ["shared", "own embedder and prefix"])
def test_grid_p_equals_the_fetchers_need_p_bit_for_bit(
    which: str,
    tmp_path: Path,
    fixture: Fixture,
    bank: NeedBank,
    followups: tuple[Followup, ...],
    shared_shadow: tuple[object, TokenEmbedder, list[dict[str, object]]],
) -> None:
    if which == "shared":
        fit, _fact, records = shared_shadow
    else:
        # A gate embedder of its own, apart from the fact embedder, with bge's query prefix.
        own = GateEmbedder(name=BGE, key_name=BGE, revision="fake", embedder=TokenEmbedder())
        fit = fit_gate(own, bank, fixture.probes, followups)
        assert fit.gate.query_prefix and not fit.gate.shares_query_vector
        runner = fixture_shadow_runner(
            fixture,
            fact_embedder=TokenEmbedder(),
            reranker=TokenReranker(),
            base=BASE,
            workdir=tmp_path,
        )
        [records] = runner(fit, "F3", [POINT])
    assert len(records) == len(fixture.probes) == len(fit.p)
    distinct = set()
    for record in records:
        recorded = record["need"]["p"]
        assert isinstance(recorded, float)
        assert recorded.hex() == fit.p[record["probe_id"]].hex(), record["probe_id"]
        distinct.add(recorded)
    assert len(distinct) > 10  # the probabilities vary, so the comparison is not vacuous


# --- scoring, the common set and the choice ------------------------------------------------------


def _pass_records(probes: Sequence[Probe], **kwargs: object) -> list[dict[str, object]]:
    return [_record(probe, 0.9, **kwargs) for probe in probes]


def test_a_setting_that_fails_only_a_latency_gate_does_not_pass(fixture: Fixture) -> None:
    setting = _scored().setting
    ok = score_setting(setting, _pass_records(fixture.probes), fixture.probes, frozenset())
    assert ok.passed and ok.failures == ()
    slow = score_setting(
        setting, _pass_records(fixture.probes, critical_path_ms=60.0), fixture.probes, frozenset()
    )
    assert not slow.passed and slow.failures == ("critical_path_p95_ms",)
    assert slow.margin == ok.margin  # the recall classes still meet their floors
    assert choose([slow], "shared", ("shared",)) is None


def test_a_setting_needs_every_noise_probe_empty_and_no_need_unknown(fixture: Fixture) -> None:
    setting = _scored().setting
    records = _pass_records(fixture.probes)
    noise = next(i for i, p in enumerate(fixture.probes) if p.kind == "noise")
    leaky = [dict(r) for r in records]
    leaky[noise] = {**leaky[noise], "items": _items([ItemRef("fact", "any")])}
    scored = score_setting(setting, leaky, fixture.probes, frozenset())
    assert scored.failures == ("empty_on_noise",)  # 16 of 17 clears 0.85 but not 1.0
    unknown = [dict(r) for r in records]
    unknown[0] = {**unknown[0], "degraded_reason": "need_unknown"}
    assert score_setting(setting, unknown, fixture.probes, frozenset()).failures == (
        "need_unknown",
    )


def test_cost_is_the_mean_block_tokens_over_the_common_set(fixture: Fixture) -> None:
    records = _pass_records(fixture.probes)
    common = frozenset(r["probe_id"] for r in records[:3])
    expected = sum(r["tokens"] for r in records[:3]) / 3
    scored = score_setting(_scored().setting, records, fixture.probes, common)
    assert scored.cost == expected


def test_the_common_set_is_served_by_every_eligible_variant_of_every_embedder() -> None:
    probes = [_probe(f"f{n}", "fact", [("fact", f"k{n}")]) for n in range(4)]
    probes.append(_probe("z0", "noise"))
    p = {
        "shared": {"f0": 0.9, "f1": 0.6, "f2": 0.2, "f3": 0.9, "z0": 0.9},
        BGE: {"f0": 0.9, "f1": 0.9, "f2": 0.9, "f3": 0.3, "z0": 0.9},
    }
    variants = {
        "shared": (_variant(0.5, 0.7), _variant(0.1, 0.7, eligible=False)),
        BGE: (_variant(0.4, 0.8),),
    }
    # f1 is band for shared and open for bge: served. f2 closed by shared, f3 by bge. The
    # ineligible variant would close nothing extra, and noise is never in the set.
    assert common_set(p, variants, probes) == frozenset({"f0", "f1"})


def test_choose_takes_the_lowest_cost_among_the_preferred_embedders_settings() -> None:
    cheap_other = _scored(embedder=BGE, cost=10.0)
    preferred = _scored(cost=50.0, select_k=15)
    pricier = _scored(cost=60.0, select_k=5)
    thin = _scored(cost=5.0, margin=0.04)
    assert choose([cheap_other, preferred, pricier, thin], "shared", ("shared", BGE)) == preferred
    # Preferring bge, its own setting wins.
    assert choose([cheap_other, preferred, pricier, thin], BGE, ("shared", BGE)) == cheap_other


def test_choose_takes_any_embedder_when_the_preferred_has_no_setting_with_margin() -> None:
    other = _scored(embedder=BGE, cost=80.0)
    other_cheap = _scored(embedder=GRANITE, cost=70.0)
    thin = _scored(cost=5.0, margin=0.01)
    order = ("shared", BGE, GRANITE)
    assert choose([other, other_cheap, thin], "shared", order) == other_cheap
    assert choose([other, other_cheap, thin], None, order) == other_cheap


def test_choose_takes_the_largest_margin_when_no_setting_has_005() -> None:
    small = _scored(margin=0.01, cost=1.0)
    larger = _scored(margin=0.03, cost=500.0, embedder=BGE)
    failing = _scored(margin=0.04, passed=False)
    assert choose([small, larger, failing], "shared", ("shared", BGE)) == larger


def test_choose_gives_none_when_nothing_passes() -> None:
    assert (
        choose([_scored(passed=False), _scored(passed=False, margin=0.2)], "shared", ("shared",))
        is None
    )
    assert choose([], "shared", ("shared",)) is None


@pytest.mark.parametrize(
    ("winner", "loser"),
    [
        ({"select_k": 5}, {"select_k": 8}),
        ({"rerank_top_n": 20}, {"rerank_top_n": 30}),
        ({"view": "F3"}, {"view": "F2"}),
        ({"embedder": "shared"}, {"embedder": BGE}),
        ({"facets": False}, {"facets": True}),
        ({"r": 0.98}, {"r": 0.95}),
        ({"k": 1}, {"k": 2}),
        # Each tie-break outranks every later one.
        ({"select_k": 5, "rerank_top_n": 50}, {"select_k": 8, "rerank_top_n": 20}),
        ({"rerank_top_n": 20, "view": "F2"}, {"rerank_top_n": 30, "view": "F3"}),
        ({"view": "F3", "embedder": BGE}, {"view": "F2", "embedder": "shared"}),
        ({"embedder": "shared", "facets": True}, {"embedder": BGE, "facets": False}),
        ({"facets": False, "r": 0.95}, {"facets": True, "r": 0.98}),
        ({"r": 0.98, "k": 2}, {"r": 0.95, "k": 1}),
    ],
)
def test_choose_breaks_ties_in_order(winner: dict, loser: dict) -> None:
    a = _scored(**winner)
    b = _scored(**loser)
    order = ("shared", BGE)
    # Rule 2 (bge's own settings are not preferred here), and rule 3 by margin.
    assert choose([b, a], None, order) == a
    assert choose([b, a], "shared", order) == a or a.setting.embedder != "shared"
    small_a = replace(a, margin=0.02)
    small_b = replace(b, margin=0.02)
    assert choose([small_b, small_a], None, order) == small_a


@pytest.mark.parametrize("order", [("shared", BGE, GRANITE), ("shared", GRANITE, BGE)])
def test_a_tie_between_two_other_embedders_goes_by_need_embedder_order(
    order: tuple[str, ...],
) -> None:
    # Plan v9 M57: two gate embedders other than `shared`, equal on every earlier key, go by
    # `--need-embedder` order. H18's command lists bge-small before granite-r2 (the first order).
    bge = _scored(embedder=BGE)
    granite = _scored(embedder=GRANITE)
    first = order[1]
    for results in ([bge, granite], [granite, bge]):
        assert choose(results, None, order).setting.embedder == first  # rule 2
        assert choose(results, "shared", order).setting.embedder == first  # rule 1 is empty
        thin = [replace(r, margin=0.02) for r in results]
        assert choose(thin, None, order).setting.embedder == first  # rule 3


# --- the load check and need_context -------------------------------------------------------------


def _followup_record(fid: str, turn: str, ms: float = 5.0) -> dict[str, object]:
    return {
        "followup_id": fid,
        "turn": turn,
        "need": _need(0.9, mode="on"),
        "critical_path_ms": ms,
        "timings_ms": {"stage1": 1.0, "stage2": 1.0, "rerank": 1.0, "queue_wait": 0.0},
        "reranked": 5,
        "shortlisted": 5,
        "degraded_reason": None,
    }


def _passing_load_runner(fixture: Fixture):
    def run(setting: RunSetting) -> list[dict[str, object]]:
        assert setting.config.need_gate == "on"
        records = _pass_records(fixture.probes, mode="on")
        return records + [_followup_record("fu-001", t) for t in ("previous", "query")]

    return run


def test_judge_load_takes_the_p95_over_recall_and_followup_fetches(fixture: Fixture) -> None:
    recall = _pass_records(fixture.probes, mode="on")
    fast = recall + [_followup_record(f"fu-{n:03d}", "query") for n in range(20)]
    assert judge_load(fast, fixture.probes).failures == ()
    # The follow-up fetches alone break the P95: 20 slow of 198 fetches is above 5%.
    slow = recall + [_followup_record(f"fu-{n:03d}", "query", ms=80.0) for n in range(20)]
    assert judge_load(slow, fixture.probes).failures == ("critical_path_p95_ms",)
    unknown = recall + [_followup_record("fu-001", "query")]
    unknown[-1]["need"] = _need(None, mode="on", band="closed", degraded="need_unknown")
    assert judge_load(unknown, fixture.probes).failures == ("need_unknown",)


def test_judge_load_takes_every_latency_gate_over_recall_and_followup_fetches(
    fixture: Fixture,
) -> None:
    recall = _pass_records(fixture.probes, mode="on")
    late = [
        {**_followup_record(f"fu-{n:03d}", "query"), "degraded_reason": "deadline"}
        for n in range(20)
    ]
    # 20 late follow-ups of 198 fetches: a deadline share of 0.101, above the 0.10 load floor.
    judged = judge_load(recall + late, fixture.probes)
    assert judged.failures == ("deadline_degraded_share",)
    assert judged.figures["deadline_degraded_share"]["n"] == len(recall) + len(late)
    shallow = [
        {**_followup_record(f"fu-{n:03d}", "query"), "reranked": 0, "shortlisted": 20}
        for n in range(20)
    ]
    assert judge_load(recall + shallow, fixture.probes).failures == ("rerank_depth",)


def test_load_check_strikes_a_setting_and_replaces_it(fixture: Fixture) -> None:
    first = _scored(cost=10.0)
    second = _scored(cost=20.0, embedder=BGE)
    third = _scored(cost=30.0)
    calls: list[tuple[str, float | None, bool]] = []
    recall = _pass_records(fixture.probes, mode="on")

    def run(setting: Setting, need_context: bool) -> list[dict[str, object]]:
        calls.append((setting.embedder, setting, need_context))
        ms = 80.0 if setting == first.setting else 5.0
        return recall + [_followup_record(f"fu-{n:03d}", "query", ms=ms) for n in range(20)]

    def context(setting: Setting) -> bool:
        return setting.embedder == "shared"

    outcome = load_check(
        [first, second, third],
        None,
        ("shared", BGE),
        probes=fixture.probes,
        need_context=context,
        run=run,
    )
    assert outcome.choice == second
    # need_context is set before each check and recomputed when the strike moves the embedder.
    assert [(name, ctx) for name, _s, ctx in calls] == [("shared", True), (BGE, False)]
    assert [check["struck"] for check in outcome.checks] == [True, False]
    assert outcome.checks[0]["failures"] == ["critical_path_p95_ms"]
    assert outcome.need_context is False


def test_three_strikes_stop(fixture: Fixture) -> None:
    results = [_scored(cost=float(n)) for n in range(1, 6)]
    recall = _pass_records(fixture.probes, mode="on")

    def run(setting: Setting, need_context: bool) -> list[dict[str, object]]:
        return [{**r, "critical_path_ms": 90.0} for r in recall]

    outcome = load_check(
        results, "shared", ("shared",), probes=fixture.probes, need_context=lambda s: False, run=run
    )
    assert outcome.choice is None
    assert len(outcome.checks) == 3 and all(c["struck"] for c in outcome.checks)


def test_followup_counts_a_and_b_out_of_closed() -> None:
    variant = _variant(0.5, 0.7)
    scores = [
        FollowupScore("fu-001", "need", 0.9, 0.3, 0.6),  # out of closed into band: A
        FollowupScore("fu-002", "need", 0.6, 0.3, 0.8),  # previous band served; to open: A
        FollowupScore("fu-003", "need", 0.9, 0.6, 0.9),  # query already band: not moved
        FollowupScore("fu-004", "need", 0.2, 0.3, 0.9),  # previous closed: not served
        FollowupScore("fu-005", "no_need", 0.9, 0.1, 0.55),  # out of closed: B
        FollowupScore("fu-006", "no_need", 0.9, 0.1, 0.4),  # still closed
    ]
    counts = followup_counts(scores, variant)
    assert (counts.a, counts.b) == (2, 1)
    assert counts.closed_previous == ("fu-004",)
    assert counts.need_context  # A > B
    tie = followup_counts(scores[:1] + scores[4:5], variant)
    assert (tie.a, tie.b) == (1, 1) and not tie.need_context


# --- reported figures and the model record -------------------------------------------------------


def test_reported_auroc_and_shares_come_from_the_recorded_p() -> None:
    probes = [_probe(f"f{n}", "fact", [("fact", f"k{n}")]) for n in range(3)]
    probes += [_probe("z0", "noise"), _probe("z1", "noise"), _probe("t0", "temporal")]
    p = {"f0": 0.9, "f1": 0.8, "f2": 0.3, "z0": 0.4, "z1": 0.1, "t0": 0.6}
    variants = (_variant(0.35, 0.85),)
    report = gate_report(p, variants, probes)
    # Need over noise: 0.9 and 0.8 beat both, 0.3 beats 0.1 only: 5 of 6 pairs.
    assert report["auroc"] == pytest.approx(5 / 6)
    shares = report["variants"][0]["shares"]
    assert shares["pr1_recall"] == {"closed": 1 / 3, "band": 1 / 3, "open": 1 / 3}
    assert shares["noise"] == {"closed": 0.5, "band": 0.5, "open": 0.0}
    assert shares["temporal"] == {"closed": 0.0, "band": 1.0, "open": 0.0}
    # Banded: f1, z0 and t0; one of them is a PR 1 recall probe.
    assert report["variants"][0]["banded_need_share"] == pytest.approx(1 / 3)


def test_the_need_model_record_is_one_load_models_accepts(
    tmp_path: Path, fixture: Fixture, bank: NeedBank, followups: tuple[Followup, ...]
) -> None:
    fit = fit_gate(_oracle(BGE, bank, fixture, followups), bank, fixture.probes, followups)
    variant = fit.variants[3]
    record = need_model_record(fit, variant, source="H18 test")
    path = tmp_path / "need_model.json"
    path.write_text(json.dumps({"records": [record]}), encoding="utf-8")
    [model] = load_models(path)
    assert model.embedder_name == BGE and model.embedder_revision == "oracle-v1"
    assert model.query_prefix == need_gate_module.NEED_EMBEDDERS[BGE].query_prefix
    assert model.bank_sha256 == bank.sha256
    assert (model.theta_low, model.theta_high) == (variant.theta_low, variant.theta_high)
    assert (model.band_k, model.recall_target) == (variant.band_k, variant.recall_target)
    assert model.form == fit.form.form


# --- the command's exits -------------------------------------------------------------------------


def _hand_runner(drop: frozenset[str] = frozenset()):
    def run(fit: object, view: str, points: Sequence[GridPoint]) -> list[list[dict[str, object]]]:
        probes = Fixture.load().probes
        return [
            [_record(pr, fit.p[pr.probe_id], drop=pr.probe_id in drop) for pr in probes]
            for _ in points
        ]

    return run


def _models(bank: NeedBank, fixture: Fixture, followups: Sequence[Followup]) -> Models:
    shared = TokenEmbedder()
    return Models(
        fact_embedder=shared,
        reranker=TokenReranker(),
        gate_embedders=(_shared(shared), _oracle(GRANITE, bank, fixture, followups)),
    )


def test_main_exits_0_and_writes_the_need_model(
    tmp_path: Path, fixture: Fixture, bank: NeedBank, followups: tuple[Followup, ...]
) -> None:
    output, model_out = tmp_path / "grid.json", tmp_path / "need_model.json"
    models = _models(bank, fixture, followups)
    code = main(
        _argv("--need-model-output", str(model_out), output=str(output)),
        models=lambda _args: models,
        points=(POINT,),
        base=BASE,
        shadow_runner=_hand_runner(),
        load_runner=_passing_load_runner(fixture),
    )
    assert code == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["choice"]["margin"] >= 0.05
    assert report["choice"]["setting"]["embedder"] == GRANITE
    # No `fresh_probes`: the real fresh file is read. The chosen oracle gate knows none of its
    # texts, so it maps each to the no-need vector, below every eligible `theta_low`.
    fresh = report["fresh_check"]
    assert fresh["passed"] is True and fresh["open"] == []
    assert len(fresh["p"]) == 23
    [model] = load_models(model_out)
    assert json.loads(model_out.read_text(encoding="utf-8")) == {"records": [report["need_model"]]}
    assert model.key == report["need_model"]["key"]


def test_main_exits_3_when_the_choice_has_margin_under_005(
    tmp_path: Path, fixture: Fixture, bank: NeedBank, followups: tuple[Followup, ...]
) -> None:
    # One type_scoped probe of the dev set served nothing: 13 of 14 is 0.929 against 0.90.
    typed = [p for p in fixture.probes if p.kind == "fact" and p.shape == "type_scoped"]
    assert len(typed) == 14
    output = tmp_path / "grid.json"
    code = main(
        _argv(output=str(output)),
        models=lambda _args: _models(bank, fixture, followups),
        points=(POINT,),
        base=BASE,
        shadow_runner=_hand_runner(frozenset({typed[0].probe_id})),
        load_runner=_passing_load_runner(fixture),
    )
    assert code == 3
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["choice"]["margin"] == pytest.approx(13 / 14 - 0.90)
    # The real fresh file, as in the exit 0 test: the chosen oracle gate closes every probe.
    assert report["choice"]["setting"]["embedder"] == GRANITE
    assert report["fresh_check"]["passed"] is True


def test_main_exits_1_when_no_setting_passes(
    tmp_path: Path, fixture: Fixture, bank: NeedBank, followups: tuple[Followup, ...]
) -> None:
    facts = frozenset(p.probe_id for p in fixture.probes if p.kind == "fact")
    output = tmp_path / "grid.json"
    code = main(
        _argv("--need-model-output", str(tmp_path / "m.json"), output=str(output)),
        models=lambda _args: _models(bank, fixture, followups),
        points=(POINT,),
        base=BASE,
        shadow_runner=_hand_runner(facts),
        load_runner=_passing_load_runner(fixture),
        # A fresh probe the oracle gate would open: with no choice nothing is checked, so it
        # changes neither the exit nor the report.
        fresh_probes=(_open_fresh(fixture, "fresh-open-01"),),
    )
    assert code == 1
    assert not (tmp_path / "m.json").exists()
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["choice"] is None and report["settings"]
    assert all(s["verdict"] == "fail" for s in report["settings"])
    assert report["fresh_check"] is None


# --- the fresh check -----------------------------------------------------------------------------


def _closed_fresh(probe_id: str) -> Probe:
    """A fresh noise probe whose query the label oracle does not know: the no-need vector."""
    return _probe(probe_id, "noise")


def _open_fresh(fixture: Fixture, probe_id: str) -> Probe:
    """A fresh noise probe worded as a dev need probe, so the label oracle gives it need's p."""
    need = next(probe for probe in fixture.probes if grid.is_pr1_recall(probe))
    return replace(_probe(probe_id, "noise"), query=need.query)


def _oracle_models(bank: NeedBank, fixture: Fixture, followups: Sequence[Followup]) -> Models:
    """The label oracle as the only gate embedder.

    The fresh check's tests skip `shared`'s fit, the slowest part of a run: the oracle is the
    choice either way.
    """
    return Models(
        fact_embedder=TokenEmbedder(),
        reranker=TokenReranker(),
        gate_embedders=(_oracle(GRANITE, bank, fixture, followups),),
    )


def _fresh_main(
    tmp_path: Path,
    bank: NeedBank,
    fixture: Fixture,
    followups: Sequence[Followup],
    fresh: Sequence[Probe],
) -> tuple[int, dict[str, object], Path]:
    output, model_out = tmp_path / "grid.json", tmp_path / "need_model.json"
    code = main(
        _argv("--need-model-output", str(model_out), output=str(output)),
        models=lambda _args: _oracle_models(bank, fixture, followups),
        points=(POINT,),
        base=BASE,
        shadow_runner=_hand_runner(),
        load_runner=_passing_load_runner(fixture),
        fresh_probes=fresh,
    )
    return code, json.loads(output.read_text(encoding="utf-8")), model_out


def test_no_fresh_probe_may_open_by_ruling() -> None:
    # Raj's ruling on plan v9 Decision 1 (16 Nov 2026): none of the fresh noise probes may open
    # the chosen gate.
    assert grid.FRESH_OPEN_ALLOWED == 0
    assert grid.EXIT_FRESH == 5


def test_a_fresh_probe_at_theta_low_is_open() -> None:
    low = 0.4
    p = {"fresh-a": low, "fresh-b": math.nextafter(low, 0.0)}
    assert grid.fresh_check(p, low) == {
        "passed": False,
        "allowed": 0,
        "open": ["fresh-a"],
        "p": p,
    }


def test_the_real_fresh_file_loads_as_23_noise_probes() -> None:
    probes = grid.load_fresh_probes()
    assert len(probes) == 23 and {probe.kind for probe in probes} == {"noise"}


def test_every_fresh_probe_closed_passes_and_writes_the_need_model(
    tmp_path: Path, fixture: Fixture, bank: NeedBank, followups: tuple[Followup, ...]
) -> None:
    fresh = (_closed_fresh("fresh-a"), _closed_fresh("fresh-b"))
    code, report, model_out = _fresh_main(tmp_path, bank, fixture, followups, fresh)
    assert code == 0  # the choice's margin is at least 0.05
    check = report["fresh_check"]
    assert check["passed"] is True and check["allowed"] == 0 and check["open"] == []
    assert set(check["p"]) == {"fresh-a", "fresh-b"}
    low = report["choice"]["setting"]["theta_low"]
    assert all(value < low for value in check["p"].values())
    assert report["need_model"] is not None
    assert json.loads(model_out.read_text(encoding="utf-8")) == {"records": [report["need_model"]]}


def test_one_open_fresh_probe_fails_with_exit_5_and_no_need_model(
    tmp_path: Path,
    fixture: Fixture,
    bank: NeedBank,
    followups: tuple[Followup, ...],
    capsys: pytest.CaptureFixture[str],
) -> None:
    fresh = (_closed_fresh("fresh-a"), _open_fresh(fixture, "fresh-open-01"))
    code, report, model_out = _fresh_main(tmp_path, bank, fixture, followups, fresh)
    assert code == grid.EXIT_FRESH == 5
    check = report["fresh_check"]
    assert check["passed"] is False and check["open"] == ["fresh-open-01"]
    assert check["p"]["fresh-open-01"] >= report["choice"]["setting"]["theta_low"]
    assert report["need_model"] is None
    assert not model_out.exists()
    # The choice stands in the report; there is no second choice after a failed check.
    assert report["choice"]["setting"]["embedder"] == GRANITE
    assert len(report["load_checks"]) == 1
    assert "exit 5" in capsys.readouterr().err


def test_with_one_open_probe_allowed_one_passes_and_two_fail(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    fixture: Fixture,
    bank: NeedBank,
    followups: tuple[Followup, ...],
) -> None:
    monkeypatch.setattr(grid, "FRESH_OPEN_ALLOWED", 1)
    one = (_closed_fresh("fresh-a"), _open_fresh(fixture, "fresh-open-01"))
    code, report, model_out = _fresh_main(tmp_path / "one", bank, fixture, followups, one)
    assert code == 0
    check = report["fresh_check"]
    assert check["passed"] is True and check["allowed"] == 1
    assert check["open"] == ["fresh-open-01"]
    assert model_out.exists()
    two = (*one, _open_fresh(fixture, "fresh-open-02"))
    code, report, model_out = _fresh_main(tmp_path / "two", bank, fixture, followups, two)
    assert code == 5
    assert report["fresh_check"]["open"] == ["fresh-open-01", "fresh-open-02"]
    assert report["need_model"] is None and not model_out.exists()


def test_the_fresh_probes_change_no_score_and_no_choice(
    fixture: Fixture, bank: NeedBank, followups: tuple[Followup, ...]
) -> None:
    embedders = _oracle_models(bank, fixture, followups).gate_embedders

    def run(fresh: Sequence[Probe]) -> dict[str, object]:
        outcome = run_grid(
            fixture,
            followups,
            bank,
            embedders,
            points=(POINT,),
            base=BASE,
            shadow_runner=_hand_runner(),
            load_runner=_passing_load_runner(fixture),
            source="H18 test",
            fresh=fresh,
        )
        return json.loads(json.dumps(outcome.report))

    without = run(())
    assert without["fresh_check"] is None
    passing = run((_closed_fresh("fresh-a"),))
    assert passing["fresh_check"]["passed"] is True
    drop = {"fresh_check"}
    assert {k: v for k, v in passing.items() if k not in drop} == {
        k: v for k, v in without.items() if k not in drop
    }
    # A failed check also withholds the need model, and changes nothing else.
    failing = run((_open_fresh(fixture, "fresh-open-01"),))
    assert failing["fresh_check"]["passed"] is False and failing["need_model"] is None
    drop = {"fresh_check", "need_model"}
    assert {k: v for k, v in failing.items() if k not in drop} == {
        k: v for k, v in without.items() if k not in drop
    }


_DEV_FILES = (
    "probes.jsonl",
    "facts.jsonl",
    "vault_notes.jsonl",
    "corrections.jsonl",
    grid.FOLLOWUP_FILE,
)


@pytest.mark.parametrize("case", ["not noise", "empty", "missing"])
def test_a_fresh_file_that_is_missing_empty_or_not_all_noise_exits_2(
    case: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The command never runs without the fresh check: the file is a dev input like the others.
    directory = tmp_path / "fixture"
    directory.mkdir()
    for name in _DEV_FILES:
        shutil.copyfile(FIXTURE_DIR / name, directory / name)
    rows = (FIXTURE_DIR / grid.FRESH_NOISE_FILE).read_text(encoding="utf-8").splitlines()
    fresh = directory / grid.FRESH_NOISE_FILE
    if case == "not noise":
        row = json.loads(rows[-1])
        row["kind"] = "fact"
        fresh.write_text("\n".join([*rows[:-1], json.dumps(row)]) + "\n", encoding="utf-8")
    elif case == "empty":
        fresh.write_text("", encoding="utf-8")

    def never(_args: object) -> Models:
        raise AssertionError("the fresh file is refused before any model loads")

    output = tmp_path / "grid.json"
    with pytest.raises(SystemExit) as exc:
        main(_argv(output=str(output)), models=never, fixture_dir=directory)
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert grid.FRESH_NOISE_FILE in err
    if case == "not noise":
        assert json.loads(rows[-1])["probe_id"] in err
    assert not output.exists()


def test_two_setting_grid_end_to_end(
    tmp_path: Path, fixture: Fixture, bank: NeedBank, followups: tuple[Followup, ...]
) -> None:
    fact = TokenEmbedder()
    embedders = (_shared(fact), _oracle(GRANITE, bank, fixture, followups))
    points = (POINT, GridPoint("F3", 5, 30, False))
    outcome = run_grid(
        fixture,
        followups,
        bank,
        embedders,
        points=points,
        base=BASE,
        shadow_runner=fixture_shadow_runner(
            fixture, fact_embedder=fact, reranker=TokenReranker(), base=BASE, workdir=tmp_path
        ),
        load_runner=_passing_load_runner(fixture),
    )
    report = json.loads(json.dumps(outcome.report))
    assert set(report) == _REPORT_KEYS
    assert set(report["cv"]) == {"shared", GRANITE}
    assert len(report["cv"][GRANITE]["candidates"]) == len(FORM_CANDIDATES)
    assert set(report["gates"][GRANITE]) == {"form", "variants"}
    assert len(report["gates"][GRANITE]["variants"]) == 4
    assert {"recall_target", "band_k", "theta_low", "theta_high", "eligible", "shares"} <= set(
        report["gates"][GRANITE]["variants"][0]
    )
    assert report["settings"]
    setting = report["settings"][0]
    assert {
        "setting",
        "figures",
        "margin",
        "cost",
        "p95_tokens",
        "mean_items",
        "facet_share",
        "stage_shares",
        "latency_p95_ms",
        "deadline_share",
        "verdict",
        "failures",
    } <= set(setting)
    assert set(report["followups"][GRANITE][0]) >= {"a", "b", "closed_previous"}
    assert set(report["reported"]) >= {"auroc", "common_set", "dated_window_probes"}
    assert report["preferred"] in ("shared", GRANITE)
    assert outcome.exit_code in (0, 1, 3)
