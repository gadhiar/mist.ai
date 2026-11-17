"""Hand-built probe files and runs pin the phase-1 recall gate's verdicts and exit codes.

H4 judges each recall run with `scripts/eval_harness/memory_recall_gate.py`, so a gate that
passed a failing run, or failed on a reported-only figure, would decide the first PR on a wrong
verdict. Each test writes a small probe set and a run into `tmp_path`, points the script at that
directory instead of the fixture, and reads the gate lines it prints. The one test that reads
the real fixture counts the dev set's classes; nothing here opens the held-out file.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.eval_harness.memory_recall import FIXTURE_DIR, load_probes  # noqa: E402
from scripts.eval_harness.memory_recall_gate import (  # noqa: E402
    PR1_GATES,
    main,
    nearest_rank_p95,
)

# -- the hand-built probe set ----------------------------------------------------------------
#
# One probe per leaf of the class table, so every recall class has a probe and a single miss
# can be aimed at one class. The tuple is (probe_id, kind, shape, lexical, expected, forbidden),
# each item a (kind, id) pair.
_PROBES = (
    ("f-plain-lex", "fact", "plain", True, [("fact", "fa")], []),
    ("f-plain-para", "fact", "plain", False, [("fact", "fb")], []),
    ("f-list-lex", "fact", "list_item", True, [("fact", "fc")], []),
    ("f-list-para", "fact", "list_item", False, [("fact", "fd")], []),
    ("f-type", "fact", "type_scoped", False, [("fact", "fe")], []),
    ("n-lex", "note", "plain", True, [("note", "notes/a.md")], []),
    ("n-para", "note", "plain", False, [("note", "notes/b.md")], []),
    ("c-correct", "correction", "plain", False, [("fact", "fg")], [("fact", "fh")]),
    ("c-forget", "correction", "plain", True, [], [("fact", "fi")]),
    ("noise-1", "noise", "plain", False, [], []),
    ("ep-1", "episode", "plain", True, [("episode", "e1")], []),
    ("co-1", "continuity", "plain", True, [("exchange", "x1")], []),
    ("t-1", "temporal", "plain", False, [("episode", "e2")], []),
)
_EXPECTED = {probe_id: expected for probe_id, _, _, _, expected, _ in _PROBES}
_FORBIDDEN = {probe_id: forbidden for probe_id, _, _, _, _, forbidden in _PROBES}

PROBE_FILES = {"dev": "probes.jsonl", "heldout": "heldout_probes.jsonl"}

RECALL_GATES = [gate.name for gate in PR1_GATES if gate.predicate is not None]


def _probe_row(probe_id, kind, shape, lexical, expected, forbidden) -> dict[str, object]:
    return {
        "probe_id": probe_id,
        "kind": kind,
        "lexical": lexical,
        "query": f"query for {probe_id}",
        "now": "2026-03-16T09:00:00+00:00",
        "conversation_start": False,
        "expected": [{"kind": k, "id": i} for k, i in expected],
        "acceptable": [],
        "forbidden": [{"kind": k, "id": i} for k, i in forbidden],
        "distractors": [],
        "correction_id": "corr-1" if kind == "correction" else None,
        "rationale": "hand-built",
        "shape": shape,
    }


def _write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _fixture(tmp_path: Path, *, sets: Iterable[str] = ("dev", "heldout")) -> Path:
    """A fixture directory holding the hand-built probe set under each named set's file."""
    directory = tmp_path / "fixture"
    directory.mkdir()
    for name in sets:
        _write_jsonl(directory / PROBE_FILES[name], (_probe_row(*probe) for probe in _PROBES))
    return directory


def _record(probe_id: str, items: Iterable[tuple[str, str]], **overrides) -> dict[str, object]:
    row: dict[str, object] = {
        "probe_id": probe_id,
        "items": [
            {"ref": f"m{n}", "kind": kind, "id": item_id, "score": 0.9, "rank": n}
            for n, (kind, item_id) in enumerate(items, start=1)
        ],
        "tokens": 120,
        "budget_tokens": 500,
        "critical_path_ms": 12.0,
        "reranked": 10,
        "shortlisted": 14,
        "degraded_reason": None,
    }
    row.update(overrides)
    return row


def _run(
    tmp_path: Path,
    *,
    miss: Iterable[str] = (),
    serve: dict[str, list[tuple[str, str]]] | None = None,
    overrides: dict[str, dict[str, object]] | None = None,
    drop: Iterable[str] = (),
) -> Path:
    """A run that serves every expected item, less `miss`, plus `serve`, with `overrides`."""
    missed, dropped = set(miss), set(drop)
    serve, overrides = serve or {}, overrides or {}
    rows = []
    for probe_id, *_ in _PROBES:
        if probe_id in dropped:
            continue
        items = [] if probe_id in missed else list(_EXPECTED[probe_id])
        items += serve.get(probe_id, [])
        rows.append(_record(probe_id, items, **overrides.get(probe_id, {})))
    return _write_jsonl(tmp_path / "run.jsonl", rows)


def _gate(
    capsys: pytest.CaptureFixture[str],
    fixture: Path,
    run: Path,
    *,
    probe_set: str = "heldout",
    load: str = "idle",
) -> tuple[int, dict[str, list[str]]]:
    """Run the gate; return its exit code and its lines keyed by gate name."""
    code = main(["--run", str(run), "--set", probe_set, "--load", load], fixture_dir=fixture)
    lines: dict[str, list[str]] = {}
    for line in capsys.readouterr().out.splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        if len(cells) == 5:
            lines[cells[0]] = cells
    return code, lines


def _failing(lines: dict[str, list[str]]) -> set[str]:
    return {name for name, cells in lines.items() if cells[4] == "fail"}


# -- the table -------------------------------------------------------------------------------


class TestTable:
    def test_dev_class_counts_are_pinned(self) -> None:
        """The table's Probes column, counted over the dev probes by each class's predicate.

        A fixture edit that moved a probe into or out of a class would move that class's gate
        without any other test noticing, so the counts are fixed here.
        """
        probes = load_probes(FIXTURE_DIR / "probes.jsonl")
        counts = {
            gate.name: sum(1 for probe in probes if gate.predicate(probe))
            for gate in PR1_GATES
            if gate.predicate is not None
        }

        assert counts == {
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

    def test_floors_and_loads_follow_the_table(self) -> None:
        gates = {gate.name: gate for gate in PR1_GATES}

        assert {name: gates[name].floors for name in RECALL_GATES} == {
            "recall fact": {"idle": 0.85},
            "recall fact, paraphrase": {"idle": 0.75},
            "recall fact, plain shape": {"idle": 0.85},
            "recall fact, plain shape, paraphrase": {"idle": 0.75},
            "recall fact, list_item": {"idle": 0.85},
            "recall fact, list_item, paraphrase": {"idle": 0.75},
            "recall fact, type_scoped": {"idle": 0.90},
            "recall note": {"idle": 0.85},
            "recall note, paraphrase": {"idle": 0.75},
            "recall correction with expected": {"idle": 0.85},
            "recall phase 1 pooled": {"idle": 0.85},
            "recall phase 1 pooled, paraphrase": {"idle": 0.75},
        }
        assert all(gates[name].sets == frozenset({"heldout"}) for name in RECALL_GATES)
        both = {"idle": None, "load": None}
        assert {name: gates[name].floors for name in gates if name not in RECALL_GATES} == {
            "empty_on_noise": dict.fromkeys(both, 0.85),
            "forgotten_leak": dict.fromkeys(both, 0.0),
            "budget_ok": dict.fromkeys(both, 1.0),
            "critical_path_p95_ms": dict.fromkeys(both, 50.0),
            "deadline_degraded_share": {"idle": 0.05, "load": 0.10},
            "rerank_depth": dict.fromkeys(both, 1.0),
        }
        assert all(
            gates[name].sets == frozenset({"dev", "heldout"})
            for name in gates
            if name not in RECALL_GATES
        )

    def test_p95_is_nearest_rank(self) -> None:
        """Nearest rank: the ceil(0.95 * n)-th smallest value, no interpolation."""
        assert nearest_rank_p95([float(v) for v in range(1, 21)]) == 19.0
        assert nearest_rank_p95([float(v) for v in range(1, 14)]) == 13.0
        assert nearest_rank_p95([7.0]) == 7.0


# -- verdicts --------------------------------------------------------------------------------


class TestPassing:
    @pytest.mark.parametrize(
        ("probe_set", "load"),
        [("heldout", "idle"), ("heldout", "load"), ("dev", "idle"), ("dev", "load")],
    )
    def test_every_gate_passing_exits_zero(self, tmp_path, capsys, probe_set, load) -> None:
        code, lines = _gate(
            capsys, _fixture(tmp_path), _run(tmp_path), probe_set=probe_set, load=load
        )

        assert code == 0
        assert _failing(lines) == set()
        assert {name for name, cells in lines.items() if cells[4] == "pass"} >= {
            "empty_on_noise",
            "forgotten_leak",
            "budget_ok",
            "critical_path_p95_ms",
            "deadline_degraded_share",
            "rerank_depth",
        }

    def test_each_line_carries_figure_floor_and_verdict(self, tmp_path, capsys) -> None:
        _, lines = _gate(capsys, _fixture(tmp_path), _run(tmp_path))

        assert lines["recall fact"][1:] == ["1.000", ">= 0.85", "n=5", "pass"]
        assert lines["forgotten_leak"][1:] == ["0.000", "<= 0.00", "n=2", "pass"]
        assert lines["critical_path_p95_ms"][1:] == ["12.000", "<= 50.00", "n=13", "pass"]
        assert lines["deadline_degraded_share"][1:] == ["0.000", "<= 0.05", "n=13", "pass"]


class TestRecallFloors:
    @pytest.mark.parametrize("name", RECALL_GATES)
    def test_a_class_missed_fails_under_heldout_and_is_named(self, tmp_path, capsys, name):
        """The run misses every probe of the class, so that class's recall is 0."""
        gate = next(gate for gate in PR1_GATES if gate.name == name)
        fixture = _fixture(tmp_path)
        probes = load_probes(fixture / "heldout_probes.jsonl")
        missed = [probe.probe_id for probe in probes if gate.predicate(probe)]

        code, lines = _gate(capsys, fixture, _run(tmp_path, miss=missed))

        assert code == 1
        assert lines[name][1] == "0.000"
        assert lines[name][4] == "fail"
        assert _failing(lines) <= set(RECALL_GATES)

    def test_a_fact_miss_leaves_note_classes_passing(self, tmp_path, capsys) -> None:
        code, lines = _gate(capsys, _fixture(tmp_path), _run(tmp_path, miss=["f-type"]))

        assert code == 1
        assert "recall fact, type_scoped" in _failing(lines)
        assert lines["recall note"][4] == "pass"
        assert lines["recall correction with expected"][4] == "pass"

    def test_a_floor_missed_under_dev_is_reported_and_exits_zero(self, tmp_path, capsys):
        code, lines = _gate(
            capsys, _fixture(tmp_path), _run(tmp_path, miss=["f-type"]), probe_set="dev"
        )

        assert code == 0
        assert lines["recall fact, type_scoped"][1] == "0.000"
        assert lines["recall fact, type_scoped"][4] == "reported, below floor"
        assert lines["recall note"][4] == "reported"

    def test_the_same_miss_under_heldout_exits_one(self, tmp_path, capsys) -> None:
        code, lines = _gate(capsys, _fixture(tmp_path), _run(tmp_path, miss=["f-type"]))

        assert code == 1
        assert lines["recall fact, type_scoped"][4] == "fail"

    def test_recall_floors_are_reported_on_a_load_run(self, tmp_path, capsys) -> None:
        """Under load only the all-probe and latency gates apply, on either set."""
        code, lines = _gate(
            capsys, _fixture(tmp_path), _run(tmp_path, miss=["f-type"]), load="load"
        )

        assert code == 0
        assert lines["recall fact, type_scoped"][4] == "reported, below floor"


class TestAllProbeGates:
    @pytest.mark.parametrize("probe_set", ["dev", "heldout"])
    def test_a_noise_probe_served_fails_alone(self, tmp_path, capsys, probe_set) -> None:
        run = _run(tmp_path, serve={"noise-1": [("fact", "fa")]})

        code, lines = _gate(capsys, _fixture(tmp_path), run, probe_set=probe_set)

        assert code == 1
        assert _failing(lines) == {"empty_on_noise"}

    @pytest.mark.parametrize("probe_set", ["dev", "heldout"])
    def test_a_forgotten_fact_served_fails_alone(self, tmp_path, capsys, probe_set) -> None:
        run = _run(tmp_path, serve={"c-forget": [("fact", "fi")]})

        code, lines = _gate(capsys, _fixture(tmp_path), run, probe_set=probe_set)

        assert code == 1
        assert _failing(lines) == {"forgotten_leak"}

    @pytest.mark.parametrize("probe_set", ["dev", "heldout"])
    @pytest.mark.parametrize("load", ["idle", "load"])
    def test_a_block_over_budget_fails_alone(self, tmp_path, capsys, probe_set, load) -> None:
        run = _run(tmp_path, overrides={"ep-1": {"tokens": 501}})

        code, lines = _gate(capsys, _fixture(tmp_path), run, probe_set=probe_set, load=load)

        assert code == 1
        assert _failing(lines) == {"budget_ok"}


class TestLatencyGates:
    @pytest.mark.parametrize("load", ["idle", "load"])
    def test_a_slow_p95_fails_alone(self, tmp_path, capsys, load) -> None:
        """Thirteen fetches put the nearest-rank P95 on the slowest one."""
        run = _run(tmp_path, overrides={"t-1": {"critical_path_ms": 50.5}})

        code, lines = _gate(capsys, _fixture(tmp_path), run, load=load)

        assert code == 1
        assert _failing(lines) == {"critical_path_p95_ms"}
        assert lines["critical_path_p95_ms"][1] == "50.500"

    def test_one_deadline_in_thirteen_fails_idle_and_passes_load(self, tmp_path, capsys):
        """1 / 13 is 7.7%: over the 5% idle floor, under the 10% load floor."""
        deadline = {"degraded_reason": "deadline", "reranked": 0}
        fixture = _fixture(tmp_path)
        run = _run(tmp_path, overrides={"ep-1": deadline})

        idle_code, idle = _gate(capsys, fixture, run, load="idle")
        load_code, under_load = _gate(capsys, fixture, run, load="load")

        assert idle_code == 1
        assert _failing(idle) == {"deadline_degraded_share"}
        assert load_code == 0
        assert under_load["deadline_degraded_share"][2:] == ["<= 0.10", "n=13", "pass"]

    def test_two_deadlines_in_thirteen_fail_under_load(self, tmp_path, capsys) -> None:
        deadline = {"degraded_reason": "deadline", "reranked": 0}
        run = _run(tmp_path, overrides={"ep-1": deadline, "co-1": deadline})

        code, lines = _gate(capsys, _fixture(tmp_path), run, load="load")

        assert code == 1
        assert _failing(lines) == {"deadline_degraded_share"}

    def test_another_degraded_reason_is_not_a_deadline(self, tmp_path, capsys) -> None:
        """A fetch degraded for another reason is outside both latency share and rerank depth."""
        run = _run(
            tmp_path, overrides={"ep-1": {"degraded_reason": "reranker_unavailable", "reranked": 0}}
        )

        code, lines = _gate(capsys, _fixture(tmp_path), run)

        assert code == 0
        assert lines["rerank_depth"][3] == "n=12"

    @pytest.mark.parametrize("load", ["idle", "load"])
    def test_a_fetch_reranking_too_few_pairs_fails_alone(self, tmp_path, capsys, load) -> None:
        run = _run(tmp_path, overrides={"f-type": {"reranked": 9, "shortlisted": 14}})

        code, lines = _gate(capsys, _fixture(tmp_path), run, load=load)

        assert code == 1
        assert _failing(lines) == {"rerank_depth"}

    def test_a_short_shortlist_needs_only_its_own_size(self, tmp_path, capsys) -> None:
        """`min(10, shortlisted)`: three pairs reranked of a shortlist of three is enough."""
        run = _run(tmp_path, overrides={"f-type": {"reranked": 3, "shortlisted": 3}})

        code, lines = _gate(capsys, _fixture(tmp_path), run)

        assert code == 0
        assert lines["rerank_depth"][1] == "1.000"


class TestReportedOnly:
    def test_episode_continuity_and_temporal_figures_print_and_never_fail(
        self, tmp_path, capsys
    ) -> None:
        run = _run(tmp_path, miss=["ep-1", "co-1", "t-1"])

        code, lines = _gate(capsys, _fixture(tmp_path), run)

        assert code == 0
        for name in ("recall episode", "recall continuity", "recall temporal"):
            assert lines[name][1] == "0.000"
            assert lines[name][2] == "-"
            assert lines[name][4] == "reported"
        assert lines["mrr"][4] == "reported"
        assert lines["precision_at_block"][4] == "reported"


class TestInputErrors:
    def test_a_missing_heldout_file_exits_two(self, tmp_path, capsys) -> None:
        code = main(
            ["--run", str(_run(tmp_path)), "--set", "heldout", "--load", "idle"],
            fixture_dir=_fixture(tmp_path, sets=["dev"]),
        )

        assert code == 2
        assert "heldout_probes.jsonl" in capsys.readouterr().err

    @pytest.mark.parametrize("probe_set", ["dev", "heldout"])
    def test_a_run_missing_a_probe_exits_two(self, tmp_path, capsys, probe_set) -> None:
        code, lines = _gate(
            capsys, _fixture(tmp_path), _run(tmp_path, drop=["noise-1"]), probe_set=probe_set
        )

        assert code == 2
        assert lines == {}

    def test_a_record_missing_a_latency_key_exits_two(self, tmp_path, capsys) -> None:
        run = _run(tmp_path)
        rows = [json.loads(line) for line in run.read_text(encoding="utf-8").splitlines()]
        del rows[3]["shortlisted"]
        _write_jsonl(run, rows)

        code = main(
            ["--run", str(run), "--set", "dev", "--load", "idle"], fixture_dir=_fixture(tmp_path)
        )

        assert code == 2
        assert "shortlisted" in capsys.readouterr().err

    def test_a_missing_run_file_exits_two(self, tmp_path, capsys) -> None:
        code = main(
            ["--run", str(tmp_path / "absent.jsonl"), "--set", "dev", "--load", "idle"],
            fixture_dir=_fixture(tmp_path),
        )

        assert code == 2
