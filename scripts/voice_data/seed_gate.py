"""The seed gate (VC-040 as amended 12 Nov), evaluated from a run's `report.json`.

`evaluate` is a pure function of the report and the effective gate thresholds, so each boundary
is tested without running a stage. It reads these report keys, which the extraction and cleaning
stages write:

- `clean_minutes`: number; `minutes_per_state` and `minutes_per_channel`: name to minutes;
- `sources`: one object per source with `source_id`, `codec`, `profile`, `bitrate_bps`,
  `sample_rate`, `layout`, `effective_bandwidth_hz` and `lossless`, copied from the inventory;
- `one_speaker`: counts `kept`, `kept_failing_retrieval`, `kept_failing_window` and
  `flagged_unconfirmed` (flagged segments in the seed that Raj has not confirmed);
- `purity_p10`: the 10th percentile of kept segments' margin over the nearest other cluster, or
  null when no segment was kept;
- `rehearsal`: `{"precision": number, "thresholds": {...}}` from the rehearsal's score, or null;
- `thresholds`: the gate values the run used.

A criterion with `passed` `None` is reported, not gated: purity, minutes per channel, and minutes
per state, where a state under `seed.state_seconds_min` is flagged for state-prompted teachers
or conversion. Every line carries names and numbers only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .errors import InputError

SOURCE_FACTS = ("codec", "bitrate_bps", "layout", "effective_bandwidth_hz")
ONE_SPEAKER_COUNTS = ("kept_failing_retrieval", "kept_failing_window", "flagged_unconfirmed")


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
