"""The phase-1 recall gate: judge one recall run against the read path's first-PR gates.

H4 runs this over every gated run of `memory_recall_run.py`, idle and under load, of each probe
set, and the first PR merges only when every run exits 0. The gates are plan mist-read-path's
table, held here as `PR1_GATES`:

- Recall@block per class, on idle runs of the held-out set. Each class is a predicate on the
  probe's `kind`, `shape`, lexical tag (paraphrase is `lexical: false`) and whether `expected`
  is empty; the run's records are filtered to the class's probes and scored with
  `memory_recall.score_run`. On the dev set, and on load runs, the same figures print marked
  reported and never fail: every threshold is tuned on the dev set, so gating on it would check
  the probes the thresholds were chosen on.
- Over every probe of either set, idle and under load: `empty_on_noise` at least 0.85,
  `forgotten_leak` 0 and `budget_ok` 1.0.
- Latency, over every run: the critical path's nearest-rank P95 at most 50 ms; the share of
  fetches degraded by the deadline at most 5% idle and 10% under load; and every fetch that is
  not degraded reranks at least `min(10, shortlisted)` pairs, so a fast P95 cannot come from
  skipping the reranker.

MRR, precision and the episode, continuity and temporal recall print as reported only.

The probe file comes from `--set`, never from a path argument, so a run is always judged on the
whole set it claims: `probes.jsonl` for `dev`, `heldout_probes.jsonl` for `heldout`. A missing
probe file is an input error, as is a run without a record for every probe, or a record without
the latency keys the harness writes (`critical_path_ms`, `degraded_reason`, `reranked`,
`shortlisted`).

Output is one line per gate, `name | figure | floor | n=probes | verdict`, then a summary line.
Exit 0 when every gate applied passes, 1 when one fails, 2 on an input error.

Usage, from the repository root:
    python -m scripts.eval_harness.memory_recall_gate --run run.jsonl
        --set dev|heldout --load idle|load
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from scripts.eval_harness.memory_recall import (
    FIXTURE_DIR,
    Probe,
    RecallReport,
    RunRecord,
    load_probes,
    parse_run_record,
    score_run,
)

PROBE_FILES: Mapping[str, str] = {"dev": "probes.jsonl", "heldout": "heldout_probes.jsonl"}

BOTH_SETS = frozenset({"dev", "heldout"})
HELDOUT_ONLY = frozenset({"heldout"})

# A mean of per-probe shares can land a rounding step under a floor it meets exactly.
_TOLERANCE = 1e-9


# ---------------------------------------------------------------------------
# The gate table
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Gate:
    """One gate: what it measures, its floor under each load it applies to, and its sets.

    `floors` maps each load the gate applies to onto its floor; under a load it does not name,
    and on a set outside `sets`, the gate prints as reported against its idle floor. A recall
    class carries the `predicate` that picks its probes; the other gates measure every probe.
    """

    name: str
    measure: str
    op: Literal[">=", "<="]
    floors: Mapping[str, float]
    sets: frozenset[str]
    predicate: Callable[[Probe], bool] | None = None


def _fact(probe: Probe) -> bool:
    return probe.kind == "fact" and bool(probe.expected)


def _note(probe: Probe) -> bool:
    return probe.kind == "note" and bool(probe.expected)


def _correction(probe: Probe) -> bool:
    return probe.kind == "correction" and bool(probe.expected)


def _pooled(probe: Probe) -> bool:
    return _fact(probe) or _note(probe) or _correction(probe)


def _paraphrase(predicate: Callable[[Probe], bool]) -> Callable[[Probe], bool]:
    return lambda probe: predicate(probe) and not probe.lexical


def _shape(shape: str) -> Callable[[Probe], bool]:
    return lambda probe: _fact(probe) and probe.shape == shape


def _recall(name: str, floor: float, predicate: Callable[[Probe], bool]) -> Gate:
    return Gate(
        name=f"recall {name}",
        measure="recall",
        op=">=",
        floors={"idle": floor},
        sets=HELDOUT_ONLY,
        predicate=predicate,
    )


def _all_probe(name: str, op: Literal[">=", "<="], floor: float) -> Gate:
    return Gate(
        name=name, measure=name, op=op, floors={"idle": floor, "load": floor}, sets=BOTH_SETS
    )


PR1_GATES: tuple[Gate, ...] = (
    _recall("fact", 0.85, _fact),
    _recall("fact, paraphrase", 0.75, _paraphrase(_fact)),
    _recall("fact, plain shape", 0.85, _shape("plain")),
    _recall("fact, plain shape, paraphrase", 0.75, _paraphrase(_shape("plain"))),
    _recall("fact, list_item", 0.85, _shape("list_item")),
    _recall("fact, list_item, paraphrase", 0.75, _paraphrase(_shape("list_item"))),
    # Every dev type_scoped probe is a paraphrase; the class is the shape, so a lexical one
    # authored later still counts.
    _recall("fact, type_scoped", 0.90, _shape("type_scoped")),
    _recall("note", 0.85, _note),
    _recall("note, paraphrase", 0.75, _paraphrase(_note)),
    # Five probes at 0.85 is five of five when each expects one item.
    _recall("correction with expected", 0.85, _correction),
    _recall("phase 1 pooled", 0.85, _pooled),
    _recall("phase 1 pooled, paraphrase", 0.75, _paraphrase(_pooled)),
    _all_probe("empty_on_noise", ">=", 0.85),
    _all_probe("forgotten_leak", "<=", 0.0),
    _all_probe("budget_ok", ">=", 1.0),
    _all_probe("critical_path_p95_ms", "<=", 50.0),
    Gate(
        name="deadline_degraded_share",
        measure="deadline_degraded_share",
        op="<=",
        floors={"idle": 0.05, "load": 0.10},
        sets=BOTH_SETS,
    ),
    _all_probe("rerank_depth", ">=", 1.0),
)

# Printed for the reader, never gated: the kinds the held-out set lacks until phase 2, and the
# ranking metrics no floor is set on.
REPORTED_KINDS = ("episode", "continuity", "temporal")

# The pairs a fetch that is not degraded must rerank, unless its shortlist is shorter.
RERANK_MIN_PAIRS = 10


# ---------------------------------------------------------------------------
# Reading the run
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FetchLatency:
    """The keys of one run record the latency gates read (memory_recall_run.py `run_record`)."""

    critical_path_ms: float
    degraded_reason: str | None
    reranked: int
    shortlisted: int


def _latency(row: Mapping[str, object], where: str) -> FetchLatency:
    missing = sorted({"critical_path_ms", "degraded_reason", "reranked", "shortlisted"} - set(row))
    if missing:
        raise ValueError(f"{where}: missing latency key(s) {missing}")
    critical = row["critical_path_ms"]
    if isinstance(critical, bool) or not isinstance(critical, int | float) or critical < 0:
        raise ValueError(f"{where}: critical_path_ms must be a non-negative number")
    reason = row["degraded_reason"]
    if reason is not None and (not isinstance(reason, str) or not reason):
        raise ValueError(f"{where}: degraded_reason must be null or a non-empty string")
    counts = {}
    for key in ("reranked", "shortlisted"):
        value = row[key]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{where}: {key} must be a non-negative integer, got {value!r}")
        counts[key] = value
    return FetchLatency(
        critical_path_ms=float(critical),
        degraded_reason=reason,
        reranked=counts["reranked"],
        shortlisted=counts["shortlisted"],
    )


def load_gate_run(path: Path) -> tuple[tuple[RunRecord, ...], tuple[FetchLatency, ...]]:
    """Each record's scorer fields and latency fields, naming the line of any bad row."""
    records: list[RunRecord] = []
    latencies: list[FetchLatency] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        where = f"{path.name}:{number}"
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{where}: not valid JSON ({exc.msg})") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{where}: a row must be a JSON object")
        try:
            records.append(parse_run_record(row))
        except ValueError as exc:
            raise ValueError(f"{where}: {exc}") from exc
        latencies.append(_latency(row, where))
    return tuple(records), tuple(latencies)


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------


def nearest_rank_p95(values: Sequence[float]) -> float:
    """The ceil(0.95 * n)-th smallest value: a P95 that is always one of the measured values."""
    if not values:
        raise ValueError("a P95 needs at least one value")
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


@dataclass(frozen=True, slots=True)
class Figure:
    """A measured value and the number of probes or fetches it is taken over."""

    value: float | None
    n: int


def _recall_figure(
    predicate: Callable[[Probe], bool], probes: Sequence[Probe], records: Sequence[RunRecord]
) -> Figure:
    chosen = [probe for probe in probes if predicate(probe)]
    ids = {probe.probe_id for probe in chosen}
    if not chosen:
        return Figure(None, 0)
    overall = score_run(chosen, [record for record in records if record.probe_id in ids]).overall
    return Figure(overall.recall_at_block, overall.recall_probes)


def _measure(
    gate: Gate,
    probes: Sequence[Probe],
    records: Sequence[RunRecord],
    latencies: Sequence[FetchLatency],
    report: RecallReport,
) -> Figure:
    overall = report.overall
    if gate.predicate is not None:
        return _recall_figure(gate.predicate, probes, records)
    if gate.measure == "empty_on_noise":
        return Figure(overall.empty_on_noise, overall.noise_probes)
    if gate.measure == "forgotten_leak":
        return Figure(overall.forgotten_leak, overall.forgotten_probes)
    if gate.measure == "budget_ok":
        return Figure(overall.budget_ok, overall.probes)
    if gate.measure == "critical_path_p95_ms":
        return Figure(nearest_rank_p95([f.critical_path_ms for f in latencies]), len(latencies))
    if gate.measure == "deadline_degraded_share":
        deadline = sum(1 for f in latencies if f.degraded_reason == "deadline")
        return Figure(deadline / len(latencies), len(latencies))
    if gate.measure == "rerank_depth":
        whole = [f for f in latencies if f.degraded_reason is None]
        if not whole:
            return Figure(None, 0)
        deep = sum(1 for f in whole if f.reranked >= min(RERANK_MIN_PAIRS, f.shortlisted))
        return Figure(deep / len(whole), len(whole))
    raise ValueError(f"gate {gate.name} has no measure {gate.measure!r}")


# ---------------------------------------------------------------------------
# Judging and printing
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GateLine:
    """One printed line: the figure, the floor it is held to and the verdict."""

    name: str
    figure: Figure
    floor: str
    verdict: Literal["pass", "fail", "reported", "reported, below floor"]

    def render(self, name_width: int) -> str:
        """The line as printed, its cells separated by `|` so a reader can split them."""
        value = "-" if self.figure.value is None else f"{self.figure.value:.3f}"
        return (
            f"{self.name:<{name_width}} | {value:>8} | {self.floor:<8} | "
            f"n={self.figure.n:<4} | {self.verdict}"
        )


def _meets(gate: Gate, value: float | None, floor: float) -> bool:
    """Whether a figure meets a floor; a figure no probe defines cannot meet one."""
    if value is None:
        return False
    if gate.op == ">=":
        return value >= floor - _TOLERANCE
    return value <= floor + _TOLERANCE


def judge(
    probes: Sequence[Probe],
    records: Sequence[RunRecord],
    latencies: Sequence[FetchLatency],
    *,
    probe_set: str,
    load: str,
) -> list[GateLine]:
    """Every gate's line for one run, then the reported-only figures."""
    report = score_run(probes, records)
    lines: list[GateLine] = []
    for gate in PR1_GATES:
        figure = _measure(gate, probes, records, latencies, report)
        gated = probe_set in gate.sets and load in gate.floors
        floor = gate.floors[load] if load in gate.floors else gate.floors["idle"]
        meets = _meets(gate, figure.value, floor)
        if gated:
            verdict: Literal["pass", "fail", "reported", "reported, below floor"] = (
                "pass" if meets else "fail"
            )
        else:
            verdict = "reported" if meets or figure.value is None else "reported, below floor"
        lines.append(GateLine(gate.name, figure, f"{gate.op} {floor:.2f}", verdict))
    for kind in REPORTED_KINDS:
        metrics = report.by_kind[kind]
        figure = Figure(metrics.recall_at_block, metrics.recall_probes)
        lines.append(GateLine(f"recall {kind}", figure, "-", "reported"))
    overall = report.overall
    lines.append(GateLine("mrr", Figure(overall.mrr, overall.recall_probes), "-", "reported"))
    lines.append(
        GateLine(
            "precision_at_block",
            Figure(overall.precision_at_block, overall.precision_probes),
            "-",
            "reported",
        )
    )
    return lines


def main(argv: Sequence[str] | None = None, *, fixture_dir: Path = FIXTURE_DIR) -> int:
    """Judge one run; `fixture_dir` is where the set's probe file is looked up (tests move it)."""
    parser = argparse.ArgumentParser(description="Judge a memory recall run against PR 1's gates.")
    parser.add_argument("--run", type=Path, required=True, help="run file (JSONL)")
    parser.add_argument("--set", dest="probe_set", choices=sorted(PROBE_FILES), required=True)
    parser.add_argument("--load", choices=("idle", "load"), required=True)
    args = parser.parse_args(argv)

    probe_path = fixture_dir / PROBE_FILES[args.probe_set]
    try:
        if not probe_path.is_file():
            raise ValueError(f"probe file {probe_path} does not exist")
        probes = load_probes(probe_path)
        records, latencies = load_gate_run(args.run)
        lines = judge(probes, records, latencies, probe_set=args.probe_set, load=args.load)
    except (OSError, ValueError) as exc:
        sys.stderr.write(f"memory_recall_gate: input error: {exc}\n")
        return 2

    width = max(len(line.name) for line in lines)
    for line in lines:
        sys.stdout.write(line.render(width) + "\n")
    failed = [line.name for line in lines if line.verdict == "fail"]
    gated = sum(1 for line in lines if line.verdict in ("pass", "fail"))
    summary = f"fail: {'; '.join(failed)}" if failed else "pass"
    sys.stdout.write(
        f"memory_recall_gate: set={args.probe_set} load={args.load}: {gated} gated, {summary}\n"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
