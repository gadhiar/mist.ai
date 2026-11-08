"""Score raw extraction runs with the scorer and classifier of a pinned checkout.

The ceiling runs are compared with r6 and r7-1 under one scorer, the one at 43f1b51, whose
Metric id rebuild needs that checkout's `canonical_id.py`; the miss classifier exists only
there. So this script is run by path from inside a `git archive` of that checkout, and
`--code-root` names the archive. It puts the code root first on `sys.path`, drops its own
directory from it, and imports `scripts.eval_harness.score_extraction_run` (the scorer) and
`scripts.eval_harness.diagnose_list_misses` (the classifier) from there. Neither is copied:
a copy would drift from what the reproductions check. After the imports, and again after
scoring, it exits 2 if any loaded `scripts.*` or `backend.*` module's file lies outside the
code root, so a score never mixes this worktree's code with the pinned one.

Scoring is the call sequence of the bench probe that wrote the raw files
(`scripts/model_bench/probes/extraction_service.py` at 43f1b51): `build_produced_index`
(`:374-390`) passes each successful case's payload through `parse_produced` into a `Produced`,
keyed by the whitespace-normalised utterance, and `score_run(probes, index)` (`:519`) scores
it. A raw row holds that case's `utterance` and its `response` dump, so the same two calls on
each row's `response.payload` give the bench probe's index without a running service. A probe
with no raw row is absent from the index and scores as all false negatives, as it does there.

Per run it prints relation recall over every gold relation, the same over the probes that
have a raw row, relation precision (matched over produced), entity recall and precision, and
the classifier's class counts and recall over the relations the prompt asks for: the
classifier's `_beyond_prompt` sets aside the relations the current gold adds to the pre-H8
gold, and `diagnose` classifies the rest. Every figure is the scorer's or the classifier's own.

`--expect NAME=R` compares run NAME's relation recall at 3 decimals. When any expectation
fails, only the expected runs' lines print, nothing is written, and the exit is 1, so no
other run's figure appears before the reproductions hold.

Exit codes: 0 scored, 1 a usage or input error or a failed expectation, 2 a module loaded
from outside `--code-root`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

SCORER_MODULE = "scripts.eval_harness.score_extraction_run"
CLASSIFIER_MODULE = "scripts.eval_harness.diagnose_list_misses"
_GUARDED_PACKAGES = frozenset({"scripts", "backend"})

RawRow = dict[str, object]


class ScoreRunsError(Exception):
    """A run that cannot be scored as asked; `exit_code` is the process exit status."""

    exit_code = 1


class UsageError(ScoreRunsError):
    """The arguments are malformed or contradict each other."""


class InputError(ScoreRunsError):
    """A gold or raw file is missing or unreadable, or a module cannot be imported."""


class CodeRootError(ScoreRunsError):
    """A `scripts.*` or `backend.*` module was loaded from outside `--code-root`."""

    exit_code = 2


class _Parser(argparse.ArgumentParser):
    """An argument parser whose errors exit 1, so that exit 2 means the code-root guard only."""

    def error(self, message: str) -> None:  # type: ignore[override]
        raise UsageError(message)


@dataclass(frozen=True, slots=True)
class Ratio:
    """A figure with the counts behind it."""

    value: float
    numerator: int
    denominator: int

    @classmethod
    def of(cls, numerator: int, denominator: int) -> Ratio:
        """A ratio computed here, `nan` over an empty denominator rather than a guess."""
        value = numerator / denominator if denominator else math.nan
        return cls(value, numerator, denominator)

    def __str__(self) -> str:
        shown = "n/a" if math.isnan(self.value) else f"{self.value:.3f}"
        return f"{shown} ({self.numerator}/{self.denominator})"


@dataclass(frozen=True, slots=True)
class RunScore:
    """One run's printed figures."""

    name: str
    relation_recall: Ratio
    covered_recall: Ratio
    covered_probes: int
    total_probes: int
    relation_precision: Ratio
    entity_recall: Ratio
    entity_precision: Ratio
    classifier_recall: Ratio
    class_counts: tuple[tuple[str, int], ...]

    def lines(self) -> list[str]:
        """The run's lines, each prefixed `[name]` so a reader can tell which runs printed."""
        prefix = f"[{self.name}]"
        out = [
            f"{prefix} relation recall: {self.relation_recall}",
            f"{prefix} relation recall over probes with a raw row: "
            f"{self.covered_recall.value:.3f} ({self.covered_recall.numerator}/"
            f"{self.covered_recall.denominator}, probes {self.covered_probes} of "
            f"{self.total_probes})",
            f"{prefix} relation precision: {self.relation_precision}",
            f"{prefix} entity recall: {self.entity_recall}",
            f"{prefix} entity precision: {self.entity_precision}",
            f"{prefix} classifier recall without the relations beyond the prompt: "
            f"{self.classifier_recall}",
        ]
        out.extend(f"{prefix} class {cls}: {count}" for cls, count in self.class_counts)
        return out


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = _Parser(description=__doc__.splitlines()[0])
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pre-h8-gold", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--expect", action="append", default=[], metavar="NAME=R")
    return parser.parse_args(argv)


def _named_runs(specs: list[str]) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for spec in specs:
        name, sep, path = spec.partition("=")
        if not sep or not name or not path:
            raise UsageError(f"--run takes NAME=PATH, got {spec!r}")
        if name in runs:
            raise UsageError(f"--run names {name} twice")
        runs[name] = Path(path)
    return runs


def _expectations(specs: list[str], runs: dict[str, Path]) -> dict[str, float]:
    expects: dict[str, float] = {}
    for spec in specs:
        name, sep, value = spec.partition("=")
        if not sep:
            raise UsageError(f"--expect takes NAME=R, got {spec!r}")
        if name not in runs:
            raise UsageError(f"--expect names {name!r}, but there is no --run named {name}")
        try:
            expects[name] = float(value)
        except ValueError as exc:
            raise UsageError(f"--expect {name}: {value!r} is not a number") from exc
    return expects


def _bind_code_root(code_root: Path) -> None:
    """Make `scripts.*` and `backend.*` resolve to the code root and nowhere else first."""
    own_dir = Path(__file__).resolve().parent
    sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != own_dir]
    sys.path.insert(0, str(code_root))


def _import(name: str) -> ModuleType:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise InputError(f"cannot import {name} from the code root: {exc}") from exc


def _check_code_root(code_root: Path) -> None:
    outside = []
    for name, module in list(sys.modules.items()):
        if name.partition(".")[0] not in _GUARDED_PACKAGES:
            continue
        file = getattr(module, "__file__", None)
        if file is None:
            continue  # A namespace package has no file of its own; its modules are checked.
        path = Path(file).resolve()
        if not path.is_relative_to(code_root):
            outside.append(f"{name} ({path})")
    if outside:
        raise CodeRootError(
            f"modules loaded from outside --code-root {code_root}: " + ", ".join(sorted(outside))
        )


def _load_gold(scorer: ModuleType, path: Path) -> list[object]:
    try:
        return list(scorer.iter_gold_probes(path))
    except (OSError, ValueError, KeyError) as exc:
        raise InputError(f"cannot read gold {path}: {exc}") from exc


def _read_raw(name: str, path: Path) -> dict[str, RawRow]:
    """One run's raw rows by probe id, in file order; the classifier looks rows up by id."""
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputError(f"run {name}: cannot read {path}: {exc}") from exc
    rows: dict[str, RawRow] = {}
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise InputError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
        if not (
            isinstance(row, dict)
            and isinstance(row.get("id"), str)
            and isinstance(row.get("utterance"), str)
        ):
            raise InputError(f"{path}:{line_no}: a raw row needs a string id and utterance")
        if row["id"] in rows:
            raise InputError(f"{path}:{line_no}: probe {row['id']} has a second row")
        rows[row["id"]] = row
    return rows


def _produced_index(scorer: ModuleType, path: Path, rows: dict[str, RawRow]) -> dict[str, object]:
    """The bench probe's `build_produced_index`, fed from raw rows instead of live cases."""
    index: dict[str, object] = {}
    for probe_id, row in rows.items():
        response = row.get("response")
        if response is None:
            continue  # A failed case: absent from the index, as in the bench probe.
        if not isinstance(response, dict) or not isinstance(response.get("payload"), dict):
            raise InputError(f"{path}: probe {probe_id} has a response with no payload object")
        ok, entities, type_by_id, rels = scorer.parse_produced(json.dumps(response["payload"]))
        utterance = scorer._norm_ws(row["utterance"])
        index[utterance] = scorer.Produced(
            utterance=utterance,
            parse_ok=ok,
            entities=entities,
            entity_type_by_id=type_by_id,
            relationships=rels,
        )
    return index


def _score_run(
    scorer: ModuleType,
    classifier: ModuleType,
    name: str,
    path: Path,
    probes: list[object],
    beyond_prompt: set[object],
) -> RunScore:
    rows = _read_raw(name, path)
    index = _produced_index(scorer, path, rows)
    report = scorer.score_run(probes, index)
    covered = [p for p in probes if p.tag in rows]  # type: ignore[attr-defined]
    covered_report = scorer.score_run(covered, index)
    diagnosis = classifier.diagnose(probes, rows, beyond_prompt)
    counts = {cls: 0 for cls in classifier.CLASSES}
    for miss in diagnosis.misses:
        counts[miss.cls] = counts.get(miss.cls, 0) + 1
    return RunScore(
        name=name,
        relation_recall=Ratio(report.rel_recall, report.rel_tp, report.rel_recall_denominator),
        covered_recall=Ratio(
            covered_report.rel_recall,
            covered_report.rel_tp,
            covered_report.rel_recall_denominator,
        ),
        covered_probes=len(covered),
        total_probes=len(probes),
        relation_precision=Ratio(
            report.rel_precision, report.rel_tp, report.rel_precision_denominator
        ),
        entity_recall=Ratio(
            report.entity_recall, report.entity_tp, report.entity_recall_denominator
        ),
        entity_precision=Ratio(
            report.entity_precision, report.entity_tp, report.entity_precision_denominator
        ),
        classifier_recall=Ratio.of(diagnosis.found, diagnosis.total),
        class_counts=tuple(counts.items()),
    )


def _run(argv: Sequence[str] | None) -> int:
    args = _parse_args(argv)
    runs = _named_runs(args.run)
    expects = _expectations(args.expect, runs)
    code_root = args.code_root.resolve()
    if not code_root.is_dir():
        raise UsageError(f"--code-root {code_root} is not a directory")
    for path in (args.gold, args.pre_h8_gold, *runs.values()):
        if not path.is_file():
            raise InputError(f"no such file: {path}")

    _bind_code_root(code_root)
    scorer = _import(SCORER_MODULE)
    classifier = _import(CLASSIFIER_MODULE)
    _check_code_root(code_root)

    probes = _load_gold(scorer, args.gold)
    beyond_prompt = classifier._beyond_prompt(probes, _load_gold(scorer, args.pre_h8_gold))
    scores = {
        name: _score_run(scorer, classifier, name, path, probes, beyond_prompt)
        for name, path in runs.items()
    }
    _check_code_root(code_root)

    print(f"scorer: {Path(scorer.__file__).resolve()}")
    print(f"classifier: {Path(classifier.__file__).resolve()}")
    failed = [
        f"expect {name}: wanted {want:.3f}, got {scores[name].relation_recall.value:.3f}"
        for name, want in expects.items()
        if f"{scores[name].relation_recall.value:.3f}" != f"{want:.3f}"
    ]
    shown = [name for name in runs if name in expects] if failed else list(runs)
    for name in shown:
        print()
        print("\n".join(scores[name].lines()))
    print()
    if failed:
        print("\n".join(failed))
        return 1
    for name, want in expects.items():
        print(f"expect {name}: {want:.3f} holds")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _run(argv)
    except ScoreRunsError as exc:
        print(f"score_runs: {exc}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
