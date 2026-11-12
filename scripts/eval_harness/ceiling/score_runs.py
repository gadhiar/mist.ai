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
`--pre-h8-gold` is optional: without it the gold's own `beyond_prompt` flags set relations
aside (`asked-for basis: the gold's beyond_prompt flags, N relations set aside`), and a gold
with neither (the 60, held-out) prints `asked-for basis: n/a (no beyond-prompt flags)` and
the classifier line reads `classifier recall over every relation`.

When the code root's scorer has aligned scoring (plan v8, D8: `score_run` takes
`accepted_pairs`), each run also prints

    [aligned X] recall R (tp/n) ci95 [lo, hi]; precision P (tp/m) ci95 [lo, hi]; f1 F ci95
    [lo, hi]; id agreement A (rail/aligned); typing T (ok/total); list-only recall L (tp/n);
    kinds exact=N ...; pending P accepted Q

on one line, and each pair of runs, after the `[paired ...]` lines,

    [paired-aligned X|Y] recall diff D ci95 [lo, hi]; precision diff D ci95 [lo, hi]; probes
    n of t

with D second minus first over the probes both completed. Intervals are the gauntlet's
cluster bootstrap by probe tag (`scripts/model_bench/probes/extraction.py`, imported from the
code root), 10,000 replicates, seed 0, 95%. Typing is the scorer's relation-typing accuracy;
list-only recall pools the aligned counts of probes `extraction_report.is_list_probe` names.
`--accepted-pairs PATH` (repeatable) reads `{"tag", "gold_id", "produced_id", "accept"}` rows,
and a typo or paraphrase pair counts only when a row accepts it. `--pairs-out DIR` writes each
run's pending pairs to `DIR/<name>.jsonl` with neither the run's name nor the accepted flag in
a row, so a blind adjudicator can read them pooled; it is refused, before anything is scored
or written, for a gold whose file name starts with `heldout-` or `selection-`, and for a
pairs file that exists. A scorer without aligned scoring prints neither line and refuses both options.

When the scorer also has column scoring (plan v15, T50: `produced_from_response` takes
`relation_rails`), each run then prints the headline, the model-written column, that
column's aligned figures and what the relation rails did:

    [exact k1-4 X] recall R (tp/n) ci95 [lo, hi]; precision P (tp/m) ci95 [lo, hi]; f1 F
    ci95 [lo, hi]
    [model-written X] (the same figures, the rows scored with the relation rails off)
    [aligned model-written X] (the aligned line's figures for that column)
    [relation rails X] relation_rails_added=A relation_rails_removed=B
    found_only_by_relation_rails=C

The headline counts only relations found through pairs of alignment kinds 1 to 4, the ids
the graph stores as one node. Each pair of runs then prints, after the `[paired-aligned`
line, `[paired-exact k1-4 X|Y]` and `[paired-model-written X|Y]`: recall, precision and F1
diffs, second minus first, with paired cluster-bootstrap intervals. The headline pair
prints only when every completed row of both runs carries a `batch_record`, and, when
either run's rows carry a `turn_plan`, both runs' do; otherwise it reads
`[paired-exact k1-4 X|Y] withheld: <the run and the missing field>; compared on
model-written`, since a rail that reads either field fires only on a row that has it.

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

`--prior-code-root DIR` re-scores the same runs under the scorer at DIR (this file run again,
in a second interpreter, with DIR as its code root) and prints, after the paired lines, one
line per run in `--run` order:

    [scorer effect X] relations found under the prior scorer p/g, under this scorer c/g,
    diff D; prior scorer_sha=S

on one line, so a moved score names the share the scorer change made (design h.5).

Every line that holds a figure (each `[...` line and each `expect` line) ends with
` | scorer_sha=... gold_version=... pack_prompt_hash=... pack_rails_hash=...`, and an
`identity:` line after the provenance lines prints the same. They are taken before anything
is scored, by `scripts/eval_harness/governance.py` of this script's own tree (loaded by
path, so the rule is the one every summary is written under): `scorer_sha` over the code
root's harness and engine rails, `gold_version` from the gold file, its LABELLING guide and
the code root's pack version, and the code root's pack hashes; `none` for a code root with
no pack registry.

Exit codes: 0 scored, 1 a usage or input error or a failed expectation, 2 a module loaded
from outside `--code-root`.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import inspect
import json
import math
import re
import subprocess  # nosec B404 - re-runs this file under the prior code root, a fixed argv
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Protocol

SCORER_MODULE = "scripts.eval_harness.score_extraction_run"
CLASSIFIER_MODULE = "scripts.eval_harness.diagnose_list_misses"
# Loaded only for a scorer with aligned scoring: the cluster bootstrap and `is_list_probe`.
BOOTSTRAP_MODULE = "scripts.model_bench.probes.extraction"
REPORT_MODULE = "scripts.eval_harness.extraction_report"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 0
BOOTSTRAP_CONFIDENCE = 0.95
REGISTRY_MODULE = "backend.knowledge.ontologies.registry"
REGISTRY_FILE = Path("backend") / "knowledge" / "ontologies" / "registry.py"
_GOVERNANCE_PATH = Path(__file__).resolve().parents[1] / "governance.py"
_GOVERNANCE_NAME = "_score_runs_governance"
# The lines that hold a figure, each of which ends with the reading's identity.
_FIGURE_LINE_STARTS = ("[", "expect ")
# A run's relation recall line, as this script prints it under any code root.
_RECALL_LINE_RE = re.compile(r"^\[(?P<name>.+?)\] relation recall: \S+ \((?P<tp>\d+)/(?P<n>\d+)\)")
PAIR_FIELDS = ("tag", "utterance", "gold_id", "gold_type", "produced_id", "produced_type", "kind")
# The gold files whose pending pairs are never written: (file name prefix, what it is).
SEALED_GOLD_PREFIXES: tuple[tuple[str, str], ...] = (
    ("heldout-", "held-out"),
    ("selection-", "selection"),
)
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


class _AlignedReport(Protocol):
    """A scorer `Report` with aligned scoring, as far as this script reads it."""

    per_probe: list[dict[str, object]]
    aligned_rel_tp: int
    aligned_rel_fp: int
    aligned_rel_fn: int
    aligned_rel_recall: float
    aligned_rel_precision: float
    aligned_rel_f1: float
    id_rail_pairs: int
    id_aligned_pairs: int
    id_agreement: float
    typing_ok: int
    typing_total: int
    typing_accuracy: float
    alignment_kinds: dict[str, int]
    pending_pairs: list[dict[str, object]]


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
    # False when nothing marks the relations beyond the prompt, so the classifier counts all.
    asked_for: bool = True

    def lines(self) -> list[str]:
        """The run's lines, each prefixed `[name]` so a reader can tell which runs printed."""
        prefix = f"[{self.name}]"
        classifier_basis = (
            "without the relations beyond the prompt" if self.asked_for else "over every relation"
        )
        out = [
            f"{prefix} relation recall: {self.relation_recall}",
            f"{prefix} relation recall over probes with a raw row: "
            f"{self.covered_recall.value:.3f} ({self.covered_recall.numerator}/"
            f"{self.covered_recall.denominator}, probes {self.covered_probes} of "
            f"{self.total_probes})",
            f"{prefix} relation precision: {self.relation_precision}",
            f"{prefix} entity recall: {self.entity_recall}",
            f"{prefix} entity precision: {self.entity_precision}",
            f"{prefix} classifier recall {classifier_basis}: {self.classifier_recall}",
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
    aligned: AlignedScore | None = None
    pending_pairs: tuple[dict[str, object], ...] = ()
    columns: Columns | None = None


def _ci(bounds: tuple[float, float] | None, *, signed: bool = False) -> str:
    if bounds is None:
        return "ci95 n/a"
    spec = "+.3f" if signed else ".3f"
    return f"ci95 [{format(bounds[0], spec)}, {format(bounds[1], spec)}]"


@dataclass(frozen=True, slots=True)
class AlignedScore:
    """One run's relations over aligned entity ids, with their intervals."""

    name: str
    recall: Ratio
    recall_ci: tuple[float, float] | None
    precision: Ratio
    precision_ci: tuple[float, float] | None
    f1: float
    f1_ci: tuple[float, float] | None
    id_agreement: Ratio
    typing: Ratio
    list_only_recall: Ratio
    kinds: tuple[tuple[str, int], ...]
    pending: int
    accepted: int

    def line(self) -> str:
        """The run's aligned line, prefixed `[aligned ` so no `[name] ` prefix matches it."""
        kinds = " ".join(f"{kind}={count}" for kind, count in self.kinds)
        return (
            f"[aligned {self.name}] recall {self.recall} {_ci(self.recall_ci)}; "
            f"precision {self.precision} {_ci(self.precision_ci)}; "
            f"f1 {self.f1:.3f} {_ci(self.f1_ci)}; id agreement {self.id_agreement}; "
            f"typing {self.typing}; list-only recall {self.list_only_recall}; "
            f"kinds {kinds}; pending {self.pending} accepted {self.accepted}"
        )


@dataclass(frozen=True, slots=True)
class AlignedPairScore:
    """Second minus first of two runs' aligned recall and precision over the shared probes."""

    first: str
    second: str
    recall_diff: float
    recall_ci: tuple[float, float] | None
    precision_diff: float
    precision_ci: tuple[float, float] | None
    shared_probes: int
    total_probes: int

    def line(self) -> str:
        """The pair's line, prefixed `[paired-aligned ` so no `[paired ` filter matches it."""

        def diff(value: float) -> str:
            return "n/a" if math.isnan(value) else f"{value:+.3f}"

        return (
            f"[paired-aligned {self.first}|{self.second}] recall diff {diff(self.recall_diff)} "
            f"{_ci(self.recall_ci, signed=True)}; precision diff {diff(self.precision_diff)} "
            f"{_ci(self.precision_ci, signed=True)}; probes {self.shared_probes} of "
            f"{self.total_probes}"
        )


HEADLINE = "exact k1-4"
MODEL_WRITTEN = "model-written"


@dataclass(frozen=True, slots=True)
class ColumnScore:
    """One run's relations in one column: recall, precision and F1 with their intervals."""

    column: str
    name: str
    recall: Ratio
    recall_ci: tuple[float, float] | None
    precision: Ratio
    precision_ci: tuple[float, float] | None
    f1: float
    f1_ci: tuple[float, float] | None

    def line(self) -> str:
        """The run's line in the column, prefixed `[<column> <name>] `."""
        return (
            f"[{self.column} {self.name}] recall {self.recall} {_ci(self.recall_ci)}; "
            f"precision {self.precision} {_ci(self.precision_ci)}; "
            f"f1 {self.f1:.3f} {_ci(self.f1_ci)}"
        )


@dataclass(frozen=True, slots=True)
class ColumnPairScore:
    """Second minus first of two runs' recall, precision and F1 in one column."""

    column: str
    first: str
    second: str
    recall_diff: float
    recall_ci: tuple[float, float] | None
    precision_diff: float
    precision_ci: tuple[float, float] | None
    f1_diff: float
    f1_ci: tuple[float, float] | None
    shared_probes: int
    total_probes: int

    def line(self) -> str:
        """The pair's line, prefixed `[paired-<column> ` so no `[paired ` filter matches it."""

        def diff(value: float) -> str:
            return "n/a" if math.isnan(value) else f"{value:+.3f}"

        return (
            f"[paired-{self.column} {self.first}|{self.second}] recall diff "
            f"{diff(self.recall_diff)} {_ci(self.recall_ci, signed=True)}; precision diff "
            f"{diff(self.precision_diff)} {_ci(self.precision_ci, signed=True)}; f1 diff "
            f"{diff(self.f1_diff)} {_ci(self.f1_ci, signed=True)}; probes "
            f"{self.shared_probes} of {self.total_probes}"
        )


@dataclass(frozen=True, slots=True)
class Columns:
    """One run's headline and model-written columns, and what the relation rails did."""

    model_index: dict[str, _Produced]
    headline: ColumnScore
    model_written: ColumnScore
    model_written_aligned: AlignedScore
    rails_counts: dict[str, int]

    def lines(self) -> list[str]:
        """The headline, then the model-written column with its aligned figures, then the counts."""
        counts = " ".join(f"{field}={count}" for field, count in self.rails_counts.items())
        return [
            self.headline.line(),
            self.model_written.line(),
            self.model_written_aligned.line(),
            f"[relation rails {self.headline.name}] {counts}",
        ]


@dataclass(frozen=True, slots=True)
class Aligned:
    """The modules and inputs the aligned lines need, loaded only for a scorer that has them.

    `columns` is True for a scorer that also has the headline and the model-written column.
    """

    bootstrap: ModuleType
    report: ModuleType
    accepted: frozenset[tuple[str, str, str]]
    columns: bool = False


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = _Parser(description=__doc__.splitlines()[0])
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument(
        "--pre-h8-gold",
        type=Path,
        help="the gold before H8; its diff sets relations aside in place of the gold's "
        "beyond_prompt flags",
    )
    parser.add_argument("--run", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--expect", action="append", default=[], metavar="NAME=R")
    parser.add_argument("--worksheet-run", metavar="NAME")
    parser.add_argument("--worksheet", type=Path, metavar="OUT")
    parser.add_argument(
        "--pairs-out",
        type=Path,
        metavar="DIR",
        help="write each run's pending typo and paraphrase pairs to DIR/<name>.jsonl",
    )
    parser.add_argument(
        "--prior-code-root",
        type=Path,
        metavar="DIR",
        help="the checkout of the scorer the runs were last scored under; prints a "
        "`scorer effect` line per run",
    )
    parser.add_argument(
        "--accepted-pairs",
        type=Path,
        action="append",
        default=[],
        metavar="PATH",
        help="an adjudicated pairs file, repeatable: {tag, gold_id, produced_id, accept} rows",
    )
    return parser.parse_args(argv)


def _pairs_targets(args: argparse.Namespace, runs: dict[str, Path]) -> dict[str, Path]:
    """Each run's pairs file; refused for a sealed gold and for a file that exists.

    A held-out or selection gold is never read probe by probe, and a pairs row holds a
    probe's utterance and ids, so neither gold's pairs are written.
    """
    if args.pairs_out is None:
        return {}
    for prefix, kind in SEALED_GOLD_PREFIXES:
        if args.gold.name.startswith(prefix):
            raise UsageError(
                f"--pairs-out with a {kind} gold ({args.gold.name}, `{prefix}`): {kind} pairs "
                "are never written; nothing was scored"
            )
    targets = {name: args.pairs_out / f"{name}.jsonl" for name in runs}
    existing = [str(path) for path in targets.values() if path.exists()]
    if existing:
        raise UsageError(f"pairs file already exists: {', '.join(existing)}; nothing was scored")
    return targets


def _read_accepted(paths: list[Path]) -> frozenset[tuple[str, str, str]]:
    """The accepted (tag, gold id, produced id) rows; a row two files disagree on is refused."""
    verdicts: dict[tuple[str, str, str], bool] = {}
    for path in paths:
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise InputError(f"cannot read accepted pairs {path}: {exc}") from exc
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise InputError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            if not (
                isinstance(row, dict)
                and all(isinstance(row.get(k), str) for k in ("tag", "gold_id", "produced_id"))
                and isinstance(row.get("accept"), bool)
            ):
                raise InputError(
                    f"{path}:{line_no}: a pairs row needs string tag, gold_id and produced_id "
                    "and a true or false accept"
                )
            key = (row["tag"], row["gold_id"], row["produced_id"])
            if verdicts.setdefault(key, row["accept"]) != row["accept"]:
                raise InputError(f"{path}:{line_no}: {' '.join(key)} is both accepted and not")
    return frozenset(key for key, accept in verdicts.items() if accept)


def _has_aligned_scoring(scorer: ModuleType) -> bool:
    return "accepted_pairs" in inspect.signature(scorer.score_run).parameters


def _has_column_scoring(scorer: ModuleType) -> bool:
    """True for a scorer with the headline and the model-written column (plan v15, T50)."""
    produce = getattr(scorer, "produced_from_response", None)
    return (
        produce is not None
        and "relation_rails" in inspect.signature(produce).parameters
        and hasattr(scorer, "relation_rails_counts")
    )


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


def _governance() -> ModuleType:
    """This checkout's `governance.py`, loaded by path under a name the guard does not hold.

    The identity rule is the one every summary is written under, so it comes from this
    script's own tree, not from the pinned code root, which may predate it.
    """
    spec = importlib.util.spec_from_file_location(_GOVERNANCE_NAME, _GOVERNANCE_PATH)
    if spec is None or spec.loader is None:
        raise InputError(f"cannot load {_GOVERNANCE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _reading_identity(code_root: Path, gold: Path) -> str:
    """The code root's scorer sha, the gold version and the code root's active pack hashes.

    A code root with no pack registry (a checkout from before the packs) names none.
    """
    governance = _governance()
    pack = None
    if (code_root / REGISTRY_FILE).is_file():
        pack = _import(REGISTRY_MODULE).active_pack()
    identity = governance.reading_identity(
        root=code_root,
        gold=gold,
        pack_version=getattr(pack, "version", None),
        pack_prompt_hash=getattr(pack, "pack_prompt_hash", None),
        pack_rails_hash=getattr(pack, "pack_rails_hash", None),
    )
    text: str = governance.identity_text(identity)
    return text


@dataclass(frozen=True, slots=True)
class PriorScores:
    """Each run's relations found under the prior scorer, and that scorer's sha."""

    scorer_sha: str
    found: dict[str, tuple[int, int]]


def _prior_scores(args: argparse.Namespace, runs: dict[str, Path]) -> PriorScores:
    """Score the runs again under `--prior-code-root`, in a second interpreter.

    One process holds `scripts.*` and `backend.*` to one code root, so the prior scorer
    runs this file again with the prior root as its code root, and its recall lines are
    read back. Nothing but the gold, the pre-H8 gold and the runs is passed on.
    """
    prior = args.prior_code_root.resolve()
    argv = [sys.executable, str(Path(__file__).resolve()), "--code-root", str(prior)]
    argv += ["--gold", str(args.gold.resolve())]
    if args.pre_h8_gold is not None:
        argv += ["--pre-h8-gold", str(args.pre_h8_gold.resolve())]
    for name, path in runs.items():
        argv += ["--run", f"{name}={path.resolve()}"]
    done = subprocess.run(  # nosec B603 - a fixed argv: this interpreter on this file
        argv, cwd=prior, capture_output=True, text=True, check=False
    )
    if done.returncode != 0:
        tail = (done.stderr.strip().splitlines() or ["no message"])[-1]
        raise InputError(f"the prior scorer at {prior} exited {done.returncode}: {tail}")
    found: dict[str, tuple[int, int]] = {}
    for line in done.stdout.splitlines():
        match = _RECALL_LINE_RE.match(line)
        if match is not None and match["name"] in runs:
            found[match["name"]] = (int(match["tp"]), int(match["n"]))
    missing = [name for name in runs if name not in found]
    if missing:
        raise InputError(f"the prior scorer at {prior} printed no recall for {', '.join(missing)}")
    return PriorScores(scorer_sha=_governance().scorer_sha(prior), found=found)


def _scorer_effect_line(name: str, score: RunScore, prior: PriorScores) -> str:
    """One run's found count under the prior scorer and this one, and the difference."""
    before, total_before = prior.found[name]
    now, total = score.relation_recall.numerator, score.relation_recall.denominator
    return (
        f"[scorer effect {name}] relations found under the prior scorer {before}/"
        f"{total_before}, under this scorer {now}/{total}, diff {now - before:+d}; "
        f"prior scorer_sha={prior.scorer_sha}"
    )


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
    scorer: ModuleType, path: Path, rows: dict[str, RawRow], *, relation_rails: bool = True
) -> dict[str, _Produced]:
    """The bench probe's `build_produced_index`, fed from raw rows instead of live cases.

    `relation_rails=False` gives the model-written column's index; only a scorer with
    column scoring is asked for it.
    """
    switch = {} if relation_rails else {"relation_rails": False}
    index: dict[str, _Produced] = {}
    for probe_id, row in rows.items():
        response = row.get("response")
        if response is None:
            continue  # A failed case: absent from the index, as in the bench probe.
        if not isinstance(response, dict) or not isinstance(response.get("payload"), dict):
            raise InputError(f"{path}: probe {probe_id} has a response with no payload object")
        if hasattr(scorer, "produced_from_response"):
            # The rails read the row's scope label, batch record and plan, as on the service.
            produced = scorer.produced_from_response(row["utterance"], response, **switch)
            index[produced.utterance] = produced
            continue
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
    beyond_prompt: set[tuple[str, Key]] | None,
    aligned: Aligned | None,
) -> tuple[RunScore, ScoredRun]:
    rows = _read_raw(name, path)
    index = _produced_index(scorer, path, rows)
    if aligned is None:
        report = scorer.score_run(probes, index)
    else:
        # The accepted pairs move only the aligned figures; the exact ones do not read them.
        report = scorer.score_run(probes, index, accepted_pairs=aligned.accepted)
    covered = [p for p in probes if p.tag in rows]
    covered_report = scorer.score_run(covered, index)
    diagnosis = classifier.diagnose(probes, rows, beyond_prompt or set())
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
        asked_for=beyond_prompt is not None,
    )
    return score, ScoredRun(
        rows=rows,
        index=index,
        misses={r["tag"]: [_key(k) for k in r["rel_fns"]] for r in report.per_probe},
        fps={r["tag"]: [_key(k) for k in r["rel_fps"]] for r in report.per_probe},
        aligned=None if aligned is None else _aligned_score(aligned, name, report),
        pending_pairs=() if aligned is None else tuple(report.pending_pairs),
        columns=(
            _columns(scorer, aligned, name, path, rows, probes, index, report)
            if aligned is not None and aligned.columns
            else None
        ),
    )


def _headline_counts(per_probe: Sequence[Mapping]) -> dict[str, tuple[int, int, int]]:
    """(tp, fp, fn) per probe tag on the headline, as the bootstrap reads aligned counts."""
    return {
        str(probe["tag"]): (
            probe["gold_relationships"] - len(probe["headline_rel_fns"]),
            len(probe["headline_rel_fps"]),
            len(probe["headline_rel_fns"]),
        )
        for probe in per_probe
    }


class _HeadlineReport(Protocol):
    """A scorer `Report` with the headline counts, as far as this script reads it."""

    per_probe: list[dict[str, object]]
    headline_rel_recall: float
    headline_rel_precision: float
    headline_rel_f1: float


def _column_score(aligned: Aligned, column: str, name: str, report: _HeadlineReport) -> ColumnScore:
    """A report's headline counts, as one column's line, intervals by the cluster bootstrap."""
    counts = _headline_counts(report.per_probe)
    tp = sum(c[0] for c in counts.values())
    fp = sum(c[1] for c in counts.values())
    fn = sum(c[2] for c in counts.values())
    return ColumnScore(
        column=column,
        name=name,
        recall=Ratio(report.headline_rel_recall, tp, tp + fn),
        recall_ci=_bootstrap_ci(aligned, counts, "recall"),
        precision=Ratio(report.headline_rel_precision, tp, tp + fp),
        precision_ci=_bootstrap_ci(aligned, counts, "precision"),
        f1=report.headline_rel_f1,
        f1_ci=_bootstrap_ci(aligned, counts, "f1"),
    )


def _columns(
    scorer: ModuleType,
    aligned: Aligned,
    name: str,
    path: Path,
    rows: dict[str, RawRow],
    probes: list[_Probe],
    index: dict[str, _Produced],
    report: _HeadlineReport,
) -> Columns:
    """The run's headline, and the same rows scored with the relation rails switched off."""
    model_index = _produced_index(scorer, path, rows, relation_rails=False)
    model_report = scorer.score_run(probes, model_index, accepted_pairs=aligned.accepted)
    return Columns(
        model_index=model_index,
        headline=_column_score(aligned, HEADLINE, name, report),
        model_written=_column_score(aligned, MODEL_WRITTEN, name, model_report),
        model_written_aligned=_aligned_score(aligned, f"{MODEL_WRITTEN} {name}", model_report),
        rails_counts=dict(scorer.relation_rails_counts(probes, index, model_index)),
    )


def _bootstrap_ci(
    aligned: Aligned, counts: dict[str, tuple[int, int, int]], kind: str
) -> tuple[float, float] | None:
    result: tuple[float, float] | None = aligned.bootstrap.cluster_bootstrap_ratio_ci(
        counts,
        n_replicates=BOOTSTRAP_REPLICATES,
        seed=BOOTSTRAP_SEED,
        confidence=BOOTSTRAP_CONFIDENCE,
        kind=kind,
    )
    return result


def _aligned_score(aligned: Aligned, name: str, report: _AlignedReport) -> AlignedScore:
    """The scorer's aligned figures, intervals by the gauntlet's cluster bootstrap."""
    counts = aligned.bootstrap.per_probe_aligned_rel_counts(report.per_probe)
    listed = [c for tag, c in counts.items() if aligned.report.is_list_probe(tag)]
    tp, fp, fn = report.aligned_rel_tp, report.aligned_rel_fp, report.aligned_rel_fn
    return AlignedScore(
        name=name,
        recall=Ratio(report.aligned_rel_recall, tp, tp + fn),
        recall_ci=_bootstrap_ci(aligned, counts, "recall"),
        precision=Ratio(report.aligned_rel_precision, tp, tp + fp),
        precision_ci=_bootstrap_ci(aligned, counts, "precision"),
        f1=report.aligned_rel_f1,
        f1_ci=_bootstrap_ci(aligned, counts, "f1"),
        id_agreement=Ratio(report.id_agreement, report.id_rail_pairs, report.id_aligned_pairs),
        typing=Ratio(report.typing_accuracy, report.typing_ok, report.typing_total),
        list_only_recall=Ratio.of(sum(c[0] for c in listed), sum(c[0] + c[2] for c in listed)),
        kinds=tuple(report.alignment_kinds.items()),
        pending=len(report.pending_pairs),
        accepted=sum(1 for pair in report.pending_pairs if pair["accepted"]),
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


def _paired_aligned_scores(
    scorer: ModuleType, aligned: Aligned, probes: list[_Probe], runs: dict[str, ScoredRun]
) -> list[AlignedPairScore]:
    """Each unordered pair in `--run` order: second minus first over the shared probes."""
    names = list(runs)
    completed = {name: _completed(probes, run) for name, run in runs.items()}
    pairs: list[AlignedPairScore] = []
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            shared = [p for p in probes if p.tag in completed[first] & completed[second]]
            one = scorer.score_run(shared, runs[first].index, accepted_pairs=aligned.accepted)
            two = scorer.score_run(shared, runs[second].index, accepted_pairs=aligned.accepted)
            one_counts = aligned.bootstrap.per_probe_aligned_rel_counts(one.per_probe)
            two_counts = aligned.bootstrap.per_probe_aligned_rel_counts(two.per_probe)

            def interval(kind: str, a=one_counts, b=two_counts) -> tuple[float, float] | None:
                result: tuple[float, float] | None = (
                    aligned.bootstrap.cluster_bootstrap_paired_difference_ci(
                        a,
                        b,
                        n_replicates=BOOTSTRAP_REPLICATES,
                        seed=BOOTSTRAP_SEED,
                        confidence=BOOTSTRAP_CONFIDENCE,
                        kind=kind,
                    )
                )
                return result

            empty = not shared
            pairs.append(
                AlignedPairScore(
                    first=first,
                    second=second,
                    recall_diff=(
                        math.nan if empty else two.aligned_rel_recall - one.aligned_rel_recall
                    ),
                    recall_ci=interval("recall"),
                    precision_diff=(
                        math.nan if empty else two.aligned_rel_precision - one.aligned_rel_precision
                    ),
                    precision_ci=interval("precision"),
                    shared_probes=len(shared),
                    total_probes=len(probes),
                )
            )
    return pairs


def _paired_column_score(
    scorer: ModuleType,
    aligned: Aligned,
    column: str,
    pair: tuple[str, str],
    shared: list[_Probe],
    indexes: tuple[dict[str, _Produced], dict[str, _Produced]],
    total: int,
) -> ColumnPairScore:
    """Second minus first in one column over the shared probes, with paired intervals."""
    one = scorer.score_run(shared, indexes[0], accepted_pairs=aligned.accepted)
    two = scorer.score_run(shared, indexes[1], accepted_pairs=aligned.accepted)
    one_counts = _headline_counts(one.per_probe)
    two_counts = _headline_counts(two.per_probe)

    def interval(kind: str) -> tuple[float, float] | None:
        result: tuple[float, float] | None = (
            aligned.bootstrap.cluster_bootstrap_paired_difference_ci(
                one_counts,
                two_counts,
                n_replicates=BOOTSTRAP_REPLICATES,
                seed=BOOTSTRAP_SEED,
                confidence=BOOTSTRAP_CONFIDENCE,
                kind=kind,
            )
        )
        return result

    def diff(field: str) -> float:
        return math.nan if not shared else getattr(two, field) - getattr(one, field)

    return ColumnPairScore(
        column=column,
        first=pair[0],
        second=pair[1],
        recall_diff=diff("headline_rel_recall"),
        recall_ci=interval("recall"),
        precision_diff=diff("headline_rel_precision"),
        precision_ci=interval("precision"),
        f1_diff=diff("headline_rel_f1"),
        f1_ci=interval("f1"),
        shared_probes=len(shared),
        total_probes=total,
    )


@dataclass(frozen=True, slots=True)
class WithheldPair:
    """A pair the headline does not compare, and the field one of its runs lacks."""

    first: str
    second: str
    reason: str

    def line(self) -> str:
        """The headline pair's line, naming the missing field and the column compared on."""
        return (
            f"[paired-{HEADLINE} {self.first}|{self.second}] withheld: {self.reason}; "
            f"compared on {MODEL_WRITTEN}"
        )


def _headline_withheld(
    names: tuple[str, str], probes: list[_Probe], runs: dict[str, ScoredRun]
) -> str | None:
    """Why two runs are not compared on the headline, or None when they are.

    A rail that reads the batch record or the turn plan fires only on a row that carries
    it, so the headline compares two runs only when every completed row of both carries a
    batch record, and, when either run had the plan on (a row carries a turn plan), both
    did (plan v15, T50 Acceptance 5).
    """
    responses = {
        name: [runs[name].rows[tag]["response"] for tag in sorted(_completed(probes, runs[name]))]
        for name in names
    }
    no_record = [
        name
        for name in names
        if not responses[name]
        or any(response.get("batch_record") is None for response in responses[name])
    ]
    if no_record:
        return " and ".join(f"{name}'s" for name in no_record) + " rows carry no batch_record"
    plan_on = [
        name
        for name in names
        if any(response.get("turn_plan") is not None for response in responses[name])
    ]
    if len(plan_on) == 1:
        (off,) = (name for name in names if name not in plan_on)
        return f"{off}'s rows carry no turn_plan, and {plan_on[0]} had the plan on"
    return None


def _paired_columns(
    scorer: ModuleType, aligned: Aligned, probes: list[_Probe], runs: dict[str, ScoredRun]
) -> list[ColumnPairScore | WithheldPair]:
    """Each unordered pair in `--run` order: the headline or why not, then model-written."""
    names = list(runs)
    completed = {name: _completed(probes, run) for name, run in runs.items()}
    pairs: list[ColumnPairScore | WithheldPair] = []
    for i, first in enumerate(names):
        for second in names[i + 1 :]:
            shared = [p for p in probes if p.tag in completed[first] & completed[second]]
            one, two = runs[first].columns, runs[second].columns
            if one is None or two is None:
                continue
            withheld = _headline_withheld((first, second), probes, runs)
            for column, indexes in (
                (HEADLINE, (runs[first].index, runs[second].index)),
                (MODEL_WRITTEN, (one.model_index, two.model_index)),
            ):
                if column == HEADLINE and withheld is not None:
                    pairs.append(WithheldPair(first, second, withheld))
                    continue
                pairs.append(
                    _paired_column_score(
                        scorer, aligned, column, (first, second), shared, indexes, len(probes)
                    )
                )
    return pairs


def _write_pairs(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """A run's pending pairs without the run's name or the accepted flag, for a blind reader."""
    text = "".join(
        json.dumps({field: row[field] for field in PAIR_FIELDS}, ensure_ascii=False) + "\n"
        for row in rows
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
    except FileExistsError as exc:
        raise InputError(f"pairs file {path} already exists") from exc
    except OSError as exc:
        raise InputError(f"cannot write pairs file {path}: {exc}") from exc


def _asked_for_basis(
    scorer: ModuleType, classifier: ModuleType, probes: list[_Probe], pre_h8_gold: Path | None
) -> tuple[set[tuple[str, Key]] | None, str | None]:
    """The relations beyond the prompt, None when nothing marks them, and the line saying so.

    The pre-H8 gold's diff wins when it is given and prints no line, as before; otherwise the
    gold's own `beyond_prompt` flags are read.
    """
    if pre_h8_gold is not None:
        return classifier._beyond_prompt(probes, _load_gold(scorer, pre_h8_gold)), None
    flagged = {
        (p.tag, (r.source, r.predicate, r.target))
        for p in probes
        for r in p.relationships
        if getattr(r, "beyond_prompt", False)
    }
    if not flagged:
        return None, "asked-for basis: n/a (no beyond-prompt flags)"
    n = len(flagged)
    return flagged, (
        f"asked-for basis: the gold's beyond_prompt flags, {n} relation"
        f"{'' if n == 1 else 's'} set aside"
    )


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
    pairs_targets = _pairs_targets(args, runs)
    code_root = args.code_root.resolve()
    if not code_root.is_dir():
        raise UsageError(f"--code-root {code_root} is not a directory")
    if args.prior_code_root is not None and not args.prior_code_root.is_dir():
        raise UsageError(f"--prior-code-root {args.prior_code_root} is not a directory")
    for path in (args.gold, args.pre_h8_gold, *runs.values(), *args.accepted_pairs):
        if path is not None and not path.is_file():
            raise InputError(f"no such file: {path}")
    accepted = _read_accepted(args.accepted_pairs)

    _bind_code_root(code_root)
    scorer = _import(SCORER_MODULE)
    classifier = _import(CLASSIFIER_MODULE)
    aligned: Aligned | None = None
    if _has_aligned_scoring(scorer):
        aligned = Aligned(
            _import(BOOTSTRAP_MODULE),
            _import(REPORT_MODULE),
            accepted,
            columns=_has_column_scoring(scorer),
        )
    elif pairs_targets or args.accepted_pairs:
        raise InputError(
            "the code root's scorer has no aligned scoring, which --pairs-out and "
            "--accepted-pairs need"
        )
    # The reading's identity is taken from the code root before anything is scored.
    identity = _reading_identity(code_root, args.gold)
    _check_code_root(code_root)

    def say(line: str) -> None:
        """Print a line; a figure line ends with the identity it was scored under."""
        print(f"{line} | {identity}" if line.startswith(_FIGURE_LINE_STARTS) else line)

    probes = _load_gold(scorer, args.gold)
    beyond_prompt, basis_line = _asked_for_basis(scorer, classifier, probes, args.pre_h8_gold)
    scored = {
        name: _score_run(scorer, classifier, name, path, probes, beyond_prompt, aligned)
        for name, path in runs.items()
    }
    scores = {name: score for name, (score, _) in scored.items()}
    failed = [
        f"expect {name}: wanted {want:.3f}, got {scores[name].relation_recall.value:.3f}"
        for name, want in expects.items()
        if f"{scores[name].relation_recall.value:.3f}" != f"{want:.3f}"
    ]
    pairs: list[PairScore] = []
    aligned_pairs: list[AlignedPairScore] = []
    column_pairs: list[ColumnPairScore | WithheldPair] = []
    by_name = {name: run for name, (_, run) in scored.items()}
    prior: PriorScores | None = None
    if not failed:
        pairs = _paired_scores(scorer, probes, by_name)
        if aligned is not None:
            aligned_pairs = _paired_aligned_scores(scorer, aligned, probes, by_name)
            column_pairs = _paired_columns(scorer, aligned, probes, by_name)
        if args.prior_code_root is not None:
            prior = _prior_scores(args, runs)
    lines: list[dict[str, object]] = []
    if worksheet is not None and not failed:
        lines = _worksheet_lines(
            scorer, classifier, probes, scored[args.worksheet_run][1], beyond_prompt or set()
        )
    _check_code_root(code_root)

    print(f"scorer: {Path(scorer.__file__).resolve()}")
    print(f"classifier: {Path(classifier.__file__).resolve()}")
    print(f"identity: {identity}")
    if basis_line is not None:
        print(basis_line)
    shown = [name for name in runs if name in expects] if failed else list(runs)
    for name in shown:
        print()
        for line in scores[name].lines():
            say(line)
        aligned_score = by_name[name].aligned
        if aligned_score is not None:
            say(aligned_score.line())
        columns = by_name[name].columns
        for line in [] if columns is None else columns.lines():
            say(line)
    print()
    if failed:
        for line in failed:
            say(line)
        return 1
    for name, want in expects.items():
        say(f"expect {name}: {want:.3f} holds")
    for pair in pairs:
        say(pair.line())
    for aligned_pair in aligned_pairs:
        say(aligned_pair.line())
    for column_pair in column_pairs:
        say(column_pair.line())
    if prior is not None:
        for name in runs:
            say(_scorer_effect_line(name, scores[name], prior))
    for name, target in pairs_targets.items():
        _write_pairs(target, by_name[name].pending_pairs)
        print(f"pairs {name}: {len(by_name[name].pending_pairs)} pending, written to {target}")
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
