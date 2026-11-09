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

With two or more runs, one line per unordered pair of `--run` names, in `--run` order (n1|n2,
n1|n3, ..., then n2|n3 and so on), compares the runs on the same probes:

    [paired X|Y] relation recall over probes both runs completed: X 0.000 (tp/g), Y 0.000
    (tp/g), probes n of t

on one line. The probes run X completed are the gold probes whose raw row in X has a non-null
`response`, narrower than the per-run rule above, which counts any raw row. The pair's subset
is the probes both runs completed, in gold order; n is its size and t the gold probe count.
Each run's tp and g are the scorer's `rel_tp` and `rel_recall_denominator` from `score_run`
over that subset with the run's own index, so two runs are compared on the same probes even
when one completes few of them. g is the gold relations over the subset, so it must be
the same for both runs; a difference is an input error. An empty subset prints `n/a (0/0)`,
not the scorer's 1.0 for an empty denominator. The lines follow the expectation lines and
precede the worksheet line, and none prints when an expectation fails.

`--expect NAME=R` compares run NAME's relation recall at 3 decimals. When any expectation
fails, only the expected runs' lines print, nothing is written, and the exit is 1, so no
other run's figure appears before the reproductions hold.

`--worksheet-run NAME --worksheet OUT` writes NAME's adjudication worksheet in the same
invocation, after every expectation holds, so the worksheet comes from the scorer the
reproductions just checked. An existing OUT is refused before anything is scored, so a
committed worksheet is never rewritten; a later check writes to a scratch path and compares
bytes. One JSON line per gold probe, in gold order, holds the tag and utterance, gold and
produced entities and relations (scorer-canonical ids), the counts `gold`, `found`, `produced`
and `matched` with `beyond_prompt_gold` and `beyond_prompt_found`, and the items: one per
missed gold relation (`<tag>:miss:<n>`) and one per produced relation gold lacks
(`<tag>:fp:<n>`), numbered in the scorer's sorted `rel_fns` and `rel_fps` order, so the ids
are stable across runs. A miss carries the classifier's class: from the same diagnosis the
printed class counts come from, or, for a relation beyond the prompt, which that diagnosis
sets aside, from a diagnosis that sets nothing aside. The worksheet holds no recall or
precision figure: the adjudicator classifies items without a score in view.

Exit codes: 0 scored, 1 a usage or input error or a failed expectation, 2 a module loaded
from outside `--code-root`.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol

SCORER_MODULE = "scripts.eval_harness.score_extraction_run"
CLASSIFIER_MODULE = "scripts.eval_harness.diagnose_list_misses"
_GUARDED_PACKAGES = frozenset({"scripts", "backend"})

RawRow = dict[str, object]
Key = tuple[str, str, str]


class _Entity(Protocol):
    """The pinned scorer's `GoldEntity`, as far as this script reads it."""

    id: str
    type: str


class _Relation(Protocol):
    """The pinned scorer's `GoldRel`, as far as this script reads it."""

    source: str
    predicate: str
    target: str


class _Probe(Protocol):
    """The pinned scorer's `GoldProbe`, as far as this script reads it."""

    tag: str
    utterance: str
    entities: Sequence[_Entity]
    relationships: Sequence[_Relation]


class _Produced(Protocol):
    """The pinned scorer's `Produced`, as far as this script reads it."""

    entities: Sequence[_Entity]
    relationships: Sequence[Mapping[str, str]]


class ScoreRunsError(Exception):
    """A run that cannot be scored as asked; `exit_code` is the process exit status."""

    exit_code = 1


class UsageError(ScoreRunsError):
    """The arguments are malformed or contradict each other."""


class InputError(ScoreRunsError):
    """A gold or raw file is missing or unreadable, or a module cannot be imported."""


class WorksheetError(ScoreRunsError):
    """The worksheet cannot be written: its path exists, or scorer and classifier disagree."""


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


@dataclass(frozen=True, slots=True)
class PairScore:
    """Two runs' relation recall over the probes both completed."""

    first: str
    second: str
    first_recall: Ratio
    second_recall: Ratio
    shared_probes: int
    total_probes: int

    def line(self) -> str:
        """The pair's line, prefixed `[paired ` so no per-run `[name] ` prefix matches it."""
        return (
            f"[paired {self.first}|{self.second}] relation recall over probes both runs "
            f"completed: {self.first} {self.first_recall}, {self.second} {self.second_recall}, "
            f"probes {self.shared_probes} of {self.total_probes}"
        )


@dataclass(frozen=True, slots=True)
class ScoredRun:
    """What the worksheet is built from.

    The run's rows and index, and the scorer's per-probe missed and false-positive relation
    keys in the scorer's sorted order, which fixes the item numbering.
    """

    rows: dict[str, RawRow]
    index: dict[str, _Produced]
    misses: dict[str, list[Key]]
    fps: dict[str, list[Key]]


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = _Parser(description=__doc__.splitlines()[0])
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--pre-h8-gold", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--expect", action="append", default=[], metavar="NAME=R")
    parser.add_argument("--worksheet-run", metavar="NAME")
    parser.add_argument("--worksheet", type=Path, metavar="OUT")
    return parser.parse_args(argv)


def _worksheet_target(args: argparse.Namespace, runs: dict[str, Path]) -> Path | None:
    if (args.worksheet_run is None) != (args.worksheet is None):
        raise UsageError("--worksheet-run and --worksheet go together")
    if args.worksheet is None:
        return None
    if args.worksheet_run not in runs:
        raise UsageError(
            f"--worksheet-run names {args.worksheet_run!r}, "
            f"but there is no --run named {args.worksheet_run}"
        )
    if args.worksheet.exists():
        raise WorksheetError(f"worksheet {args.worksheet} already exists; nothing was scored")
    return args.worksheet


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


def _load_gold(scorer: ModuleType, path: Path) -> list[_Probe]:
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


def _produced_index(
    scorer: ModuleType, path: Path, rows: dict[str, RawRow]
) -> dict[str, _Produced]:
    """The bench probe's `build_produced_index`, fed from raw rows instead of live cases."""
    index: dict[str, _Produced] = {}
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
    probes: list[_Probe],
    beyond_prompt: set[tuple[str, Key]],
) -> tuple[RunScore, ScoredRun]:
    rows = _read_raw(name, path)
    index = _produced_index(scorer, path, rows)
    report = scorer.score_run(probes, index)
    covered = [p for p in probes if p.tag in rows]
    covered_report = scorer.score_run(covered, index)
    diagnosis = classifier.diagnose(probes, rows, beyond_prompt)
    counts = {cls: 0 for cls in classifier.CLASSES}
    for miss in diagnosis.misses:
        counts[miss.cls] = counts.get(miss.cls, 0) + 1
    score = RunScore(
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
    return score, ScoredRun(
        rows=rows,
        index=index,
        misses={r["tag"]: [_key(k) for k in r["rel_fns"]] for r in report.per_probe},
        fps={r["tag"]: [_key(k) for k in r["rel_fps"]] for r in report.per_probe},
    )


def _completed(probes: list[_Probe], run: ScoredRun) -> set[str]:
    """The probes the run completed: a raw row whose `response` is not null."""
    return {
        p.tag for p in probes if p.tag in run.rows and run.rows[p.tag].get("response") is not None
    }


def _paired_scores(
    scorer: ModuleType, probes: list[_Probe], runs: dict[str, ScoredRun]
) -> list[PairScore]:
    """Each unordered pair in `--run` order, scored by the scorer over the shared probes."""
    names = list(runs)
    completed = {name: _completed(probes, run) for name, run in runs.items()}
    pairs: list[PairScore] = []
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            shared = [p for p in probes if p.tag in completed[first] & completed[second]]
            one = scorer.score_run(shared, runs[first].index)
            two = scorer.score_run(shared, runs[second].index)
            if one.rel_recall_denominator != two.rel_recall_denominator:
                raise InputError(
                    f"paired {first}|{second}: the scorer counts "
                    f"{one.rel_recall_denominator} gold relations over the shared probes for "
                    f"{first} and {two.rel_recall_denominator} for {second}"
                )
            pairs.append(
                PairScore(
                    first=first,
                    second=second,
                    first_recall=Ratio.of(one.rel_tp, one.rel_recall_denominator),
                    second_recall=Ratio.of(two.rel_tp, two.rel_recall_denominator),
                    shared_probes=len(shared),
                    total_probes=len(probes),
                )
            )
    return pairs


def _key(parts: Sequence[str]) -> Key:
    source, predicate, target = parts
    return source, predicate, target


def _unique(keys: list[Key]) -> list[Key]:
    return list(dict.fromkeys(keys))


def _worksheet_lines(
    scorer: ModuleType,
    classifier: ModuleType,
    probes: list[_Probe],
    run: ScoredRun,
    beyond_prompt: set[tuple[str, Key]],
) -> list[dict[str, object]]:
    """One line per gold probe, its items numbered in the scorer's sorted miss and fp order."""
    asked = {
        (m.tag, m.key): m.cls for m in classifier.diagnose(probes, run.rows, beyond_prompt).misses
    }
    every = {(m.tag, m.key): m.cls for m in classifier.diagnose(probes, run.rows, set()).misses}
    lines: list[dict[str, object]] = []
    for probe in probes:
        tag = probe.tag
        produced = run.index.get(scorer._norm_ws(probe.utterance))
        gold_keys = _unique([(r.source, r.predicate, r.target) for r in probe.relationships])
        produced_keys: list[Key] = []
        produced_entities: list[list[str]] = []
        if produced is not None:
            produced_keys = _unique(
                [(r["source"], r["predicate"], r["target"]) for r in produced.relationships]
            )
            produced_entities = [[e.id, e.type] for e in produced.entities]
        misses = run.misses[tag]
        fps = run.fps[tag]
        items: list[dict[str, object]] = []
        for n, key in enumerate(misses, start=1):
            beyond = (tag, key) in beyond_prompt
            cls = (every if beyond else asked).get((tag, key))
            if cls is None:
                raise WorksheetError(
                    f"{tag}: the scorer misses {' '.join(key)}, the classifier does not"
                )
            items.append(
                {
                    "id": f"{tag}:miss:{n}",
                    "kind": "miss",
                    "relation": list(key),
                    "class": cls,
                    "beyond_prompt": beyond,
                }
            )
        for n, key in enumerate(fps, start=1):
            items.append(
                {
                    "id": f"{tag}:fp:{n}",
                    "kind": "fp",
                    "relation": list(key),
                    "class": None,
                    "beyond_prompt": False,
                }
            )
        beyond_gold = sum(1 for key in gold_keys if (tag, key) in beyond_prompt)
        beyond_missed = sum(1 for key in misses if (tag, key) in beyond_prompt)
        lines.append(
            {
                "tag": tag,
                "utterance": probe.utterance,
                "gold_entities": [[e.id, e.type] for e in probe.entities],
                "gold_relations": [list(key) for key in gold_keys],
                "produced_entities": produced_entities,
                "produced_relations": [list(key) for key in produced_keys],
                "counts": {
                    "gold": len(gold_keys),
                    "found": len(gold_keys) - len(misses),
                    "produced": len(produced_keys),
                    "matched": len(produced_keys) - len(fps),
                    "beyond_prompt_gold": beyond_gold,
                    "beyond_prompt_found": beyond_gold - beyond_missed,
                },
                "items": items,
            }
        )
    return lines


def _write_worksheet(path: Path, lines: list[dict[str, object]]) -> None:
    text = "".join(json.dumps(line, ensure_ascii=False) + "\n" for line in lines)
    try:
        # Exclusive create: a worksheet that appeared since the start is still not rewritten.
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise WorksheetError(f"worksheet {path} already exists") from exc
    except OSError as exc:
        raise WorksheetError(f"cannot write worksheet {path}: {exc}") from exc


def _run(argv: Sequence[str] | None) -> int:
    args = _parse_args(argv)
    runs = _named_runs(args.run)
    expects = _expectations(args.expect, runs)
    worksheet = _worksheet_target(args, runs)
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
    scored = {
        name: _score_run(scorer, classifier, name, path, probes, beyond_prompt)
        for name, path in runs.items()
    }
    scores = {name: score for name, (score, _) in scored.items()}
    failed = [
        f"expect {name}: wanted {want:.3f}, got {scores[name].relation_recall.value:.3f}"
        for name, want in expects.items()
        if f"{scores[name].relation_recall.value:.3f}" != f"{want:.3f}"
    ]
    pairs: list[PairScore] = []
    if not failed:
        pairs = _paired_scores(scorer, probes, {name: run for name, (_, run) in scored.items()})
    lines: list[dict[str, object]] = []
    if worksheet is not None and not failed:
        lines = _worksheet_lines(
            scorer, classifier, probes, scored[args.worksheet_run][1], beyond_prompt
        )
    _check_code_root(code_root)

    print(f"scorer: {Path(scorer.__file__).resolve()}")
    print(f"classifier: {Path(classifier.__file__).resolve()}")
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
    for pair in pairs:
        print(pair.line())
    if worksheet is not None:
        _write_worksheet(worksheet, lines)
        run = scored[args.worksheet_run][1]
        misses = sum(len(keys) for keys in run.misses.values())
        fps = sum(len(keys) for keys in run.fps.values())
        print(f"worksheet {args.worksheet_run}: probes {len(lines)}, misses {misses}, fps {fps}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    try:
        return _run(argv)
    except ScoreRunsError as exc:
        print(f"score_runs: {exc}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
