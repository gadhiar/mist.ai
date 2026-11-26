"""The seed gate (VC-040 as amended 12 Nov), evaluated from a run's `report.json`.

`evaluate` is a pure function of the report and the effective gate thresholds, so each boundary
is tested without running a stage. The report comes from one of two sources. `gate --run E`
passes extract run E's `report.json` as it is. `gate --run E --clean C` passes E's report with
clean run C's labelled values merged in by `commands/gate.py`: `clean_minutes` (what survived
the labels and cleaning), `minutes_per_state` (per projected state, all five with labels), and
`one_speaker.kept` from C's report; `flagged_unconfirmed` counted over C's kept segments; and
the four `LABELLED_COUNTS` from C's removals. It reads these report keys:

- `clean_minutes`: number; `minutes_per_state` and `minutes_per_channel`: name to minutes;
- `sources`: one object per source with `source_id`, `codec`, `profile`, `bitrate_bps`,
  `sample_rate`, `layout`, `effective_bandwidth_hz` and `lossless`, copied from the inventory;
- `one_speaker`: counts `kept`, `kept_failing_retrieval`, `kept_failing_window` and
  `flagged_unconfirmed` (flagged segments in the seed that no label covers), and, from a clean
  run only, `LABELLED_COUNTS`, which are printed and not gated: a clip Raj removed never reaches
  the seed;
- `purity_p10`: the 10th percentile of kept segments' margin over the nearest other cluster, or
  null when no segment was kept;
- `rehearsal`: `{"precision": number, "thresholds": {...}}` from the rehearsal's score, or null;
- `thresholds`: the gate values the run used.

Extract writes `"rehearsal": null`, because the rehearsal runs in another root (A24). Its
precision reaches the gate as a record instead: `rehearsal score` writes
`rehearsal/scores/<R>/precision.json` (`rehearsal_record`), H8 copies it to the root's
`config/rehearsal.json` (`REHEARSAL_RECORD_REL`), and `gate --run` evaluates with the record's
`precision` and `thresholds` when the report's own `rehearsal` is null. The record's thresholds
are compared with the run's as a report's would be, so a record from a rehearsal at other values
fails as "thresholds differ"; `gate --check-record` makes the same comparison against the root's
effective gate values before any job runs (`record_differences`).

A criterion with `passed` `None` is reported, not gated: the labelled removals (from a clean run
only), purity, minutes per channel, and minutes per state, where a state under `seed.state_seconds_min` is flagged for state-prompted teachers
or conversion. Every line carries names and numbers only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .errors import InputError

SOURCE_FACTS = ("codec", "bitrate_bps", "layout", "effective_bandwidth_hz")
ONE_SPEAKER_COUNTS = ("kept_failing_retrieval", "kept_failing_window", "flagged_unconfirmed")
LABELLED_COUNTS = (
    "labelled_not_her",
    "labelled_flag_group",
    "labelled_overlap",
    "labelled_nearby_removed",
)
REHEARSAL_RECORD_REL = "config/rehearsal.json"
RECORD_SCHEMA = 1
RECORD_KIND = "rehearsal-precision"


@dataclass(frozen=True, slots=True)
class Criterion:
    """One line of the gate: a pass, a fail, or (`passed` None) a reported number."""

    name: str
    passed: bool | None
    detail: str


@dataclass(frozen=True, slots=True)
class SeedGateResult:
    """The gate's verdict and every criterion behind it."""

    passed: bool
    criteria: tuple[Criterion, ...]


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def _number(data: Mapping[str, object], key: str, where: str = "report") -> float:
    value = data.get(key)
    if not _is_number(value):
        raise InputError(f"{where}: {key} is missing or not a number")
    return float(value)  # type: ignore[arg-type]


def _object(data: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = data.get(key)
    if not isinstance(value, Mapping):
        raise InputError(f"report: {key} is missing or not an object")
    return value


def _table(table: Mapping[str, object], where: str) -> dict[str, float]:
    return {str(name): _number(table, str(name), where) for name in table}


def _sources(report: Mapping[str, object]) -> Sequence[Mapping[str, object]]:
    sources = report.get("sources")
    if not isinstance(sources, list) or not all(isinstance(s, Mapping) for s in sources):
        raise InputError("report: sources is missing or not a list of objects")
    for source in sources:
        if not isinstance(source.get("source_id"), str):
            raise InputError("report: a source has no source_id")
        _number(source, "sample_rate", "report source")
    return sources


def _rehearsal(
    report: Mapping[str, object], used: Mapping[str, float], minimum: float
) -> Criterion:
    name = "rehearsal precision"
    rehearsal = report.get("rehearsal")
    if rehearsal is None:
        return Criterion(name, False, f"missing (min {minimum:g})")
    if not isinstance(rehearsal, Mapping):
        raise InputError("report: rehearsal is not an object or null")
    precision = _number(rehearsal, "precision", "report rehearsal")
    thresholds = rehearsal.get("thresholds")
    if not isinstance(thresholds, Mapping):
        raise InputError("report rehearsal: thresholds is missing or not an object")
    theirs = _table(thresholds, "report rehearsal thresholds")
    differ = sorted(k for k in set(theirs) | set(used) if theirs.get(k) != used.get(k))
    detail = f"{precision:.3f} (min {minimum:g})"
    if differ:
        return Criterion(name, False, f"{detail}; thresholds differ: {', '.join(differ)}")
    return Criterion(name, precision >= minimum, f"{detail}, at the run's thresholds")


def evaluate(report: Mapping[str, object], thresholds: Mapping[str, float]) -> SeedGateResult:
    """Evaluate the seed gate.

    Args:
        report: A run's parsed `report.json`.
        thresholds: The effective gate values (`gates.Gates.effective()`).

    Raises:
        InputError: The report lacks a key the gate reads, or holds a wrong type.
    """
    if not isinstance(report, Mapping):
        raise InputError("report: not a JSON object")
    clean = _number(report, "clean_minutes")
    per_state = _table(_object(report, "minutes_per_state"), "report minutes_per_state")
    per_channel = _table(_object(report, "minutes_per_channel"), "report minutes_per_channel")
    sources = _sources(report)
    speaker = _object(report, "one_speaker")
    counts = {key: _number(speaker, key, "report one_speaker") for key in ONE_SPEAKER_COUNTS}
    kept = _number(speaker, "kept", "report one_speaker")
    purity = report.get("purity_p10")
    if purity is not None and not _is_number(purity):
        raise InputError("report: purity_p10 is not a number or null")
    used = _table(_object(report, "thresholds"), "report thresholds")

    minutes_min = thresholds["seed.clean_minutes_min"]
    rate_min = thresholds["seed.sample_rate_min_hz"]
    precision_min = thresholds["seed.rehearsal_precision_min"]
    state_min_s = thresholds["seed.state_seconds_min"]
    percentile = thresholds["seed.purity_percentile"]

    criteria = [
        Criterion("clean minutes", clean >= minutes_min, f"{clean:.2f} (min {minutes_min:g})")
    ]
    if sources:
        lowest = min(_number(s, "sample_rate", "report source") for s in sources)
        criteria.append(
            Criterion("sample rate", lowest >= rate_min, f"lowest {lowest:g} Hz (min {rate_min:g})")
        )
    else:
        criteria.append(Criterion("sample rate", False, "no sources"))
    missing = sorted(
        str(s["source_id"])
        for s in sources
        if any(s.get(k) is None for k in SOURCE_FACTS) or "profile" not in s
    )
    lossless = sum(1 for s in sources if s.get("lossless") is True)
    facts = f"{len(sources) - len(missing)} of {len(sources)} sources recorded"
    facts += f"; lossless {lossless} of {len(sources)} (best source available)"
    if missing:
        facts += f"; unrecorded: {', '.join(missing)}"
    criteria.append(Criterion("source facts", not missing, facts))
    failing = {k: v for k, v in counts.items() if v != 0}
    criteria.append(
        Criterion(
            "one speaker",
            not failing,
            f"kept {kept:g}; "
            + ", ".join(f"{k.replace('_', ' ')} {v:g}" for k, v in counts.items()),
        )
    )
    labelled = [key for key in LABELLED_COUNTS if key in speaker]
    if labelled:
        removed = [
            f"{key.removeprefix('labelled_').replace('_', ' ')} "
            f"{_number(speaker, key, 'report one_speaker'):g}"
            for key in labelled
        ]
        criteria.append(Criterion("labelled removals", None, ", ".join(removed)))
    criteria.append(_rehearsal(report, used, precision_min))
    shown = "not reported" if purity is None else f"{float(purity):.3f}"  # type: ignore[arg-type]
    criteria.append(Criterion("purity", None, f"p{percentile:g} margin {shown}"))
    for channel, minutes in sorted(per_channel.items()):
        criteria.append(Criterion(f"channel {channel}", None, f"{minutes:.2f} min"))
    for state, minutes in sorted(per_state.items()):
        detail = f"{minutes:.2f} min"
        if minutes * 60 < state_min_s:
            detail += (
                f"; under {state_min_s:g} s, flagged for state-prompted teachers or conversion"
            )
        criteria.append(Criterion(f"state {state}", None, detail))
    passed = all(c.passed is not False for c in criteria)
    return SeedGateResult(passed=passed, criteria=tuple(criteria))


def _record_thresholds(thresholds: object, where: str) -> dict[str, object]:
    if not isinstance(thresholds, Mapping):
        raise InputError(f"{where}: thresholds is missing or not an object")
    bad = sorted(str(k) for k, v in thresholds.items() if not _is_number(v))
    if bad:
        raise InputError(f"{where}: thresholds that are not numbers: {', '.join(bad)}")
    return {str(k): v for k, v in thresholds.items()}


def rehearsal_record(
    run: str, precision: float, thresholds: Mapping[str, object]
) -> dict[str, object]:
    """Return the precision record `rehearsal score` writes for extract run `run`.

    Args:
        run: The rehearsal's extract run.
        precision: The run's `extraction.precision`.
        thresholds: The run's `report.json` `thresholds`, every key, so the gate can tell whether
            the rehearsal ran at the values the gated run used.

    Raises:
        InputError: `thresholds` is not an object of numbers.
    """
    return {
        "schema": RECORD_SCHEMA,
        "kind": RECORD_KIND,
        "run": run,
        "precision": precision,
        "thresholds": _record_thresholds(thresholds, f"refused: run {run}'s report"),
    }


def read_rehearsal_record(raw: object) -> dict[str, object]:
    """Check a parsed precision record and return it.

    Raises:
        InputError: The record is not an object, has another `kind`, has no string `run`, no
            number `precision`, or no object of numbers as `thresholds`; the message names
            `REHEARSAL_RECORD_REL`, the file `gate` reads it from.
    """
    where = f"refused: {REHEARSAL_RECORD_REL}"
    if not isinstance(raw, Mapping):
        raise InputError(f"{where} is not a JSON object")
    if raw.get("kind") != RECORD_KIND:
        raise InputError(f"{where}: kind is not {RECORD_KIND}")
    if not isinstance(raw.get("run"), str):
        raise InputError(f"{where}: run is missing or not a string")
    if not _is_number(raw.get("precision")):
        raise InputError(f"{where}: precision is missing or not a number")
    record = dict(raw)
    record["thresholds"] = _record_thresholds(raw.get("thresholds"), where)
    return record


def record_differences(record: Mapping[str, object], thresholds: Mapping[str, object]) -> list[str]:
    """Return the sorted keys whose value differs, or that only one side has.

    A key present on one side only differs even when the other side's value would be `None`, so
    a gate added since the rehearsal ran is named rather than passed over.

    Raises:
        InputError: The record has no object `thresholds` (`read_rehearsal_record` checks it).
    """
    theirs = record.get("thresholds")
    if not isinstance(theirs, Mapping):
        raise InputError(f"refused: {REHEARSAL_RECORD_REL}: thresholds is missing or not an object")
    keys = set(theirs) | set(thresholds)
    return sorted(
        str(k) for k in keys if k not in theirs or k not in thresholds or theirs[k] != thresholds[k]
    )


def source_lines(report: Mapping[str, object]) -> list[str]:
    """Return one line per source: its id and codec facts, for Raj's "best source" review."""
    lines = []
    for source in _sources(report):
        lines.append(
            f"source {source.get('source_id')}: {source.get('codec')}"
            f" ({source.get('profile') or 'no profile'}), {source.get('layout')}, "
            f"{source.get('sample_rate')} Hz, {source.get('bitrate_bps')} bps, bandwidth "
            f"{source.get('effective_bandwidth_hz')} Hz, lossless {source.get('lossless')}"
        )
    return lines


def format_lines(result: SeedGateResult) -> list[str]:
    """Return the gate's printed lines: one per criterion, then the verdict."""
    marks = {True: "pass", False: "FAIL", None: "info"}
    lines = [f"  {marks[c.passed]:4}  {c.name}: {c.detail}" for c in result.criteria]
    return [*lines, f"seed gate: {'pass' if result.passed else 'fail'}"]
