"""The room device's wake calibration: its streaming evaluation as `calibration.endpoint`.

A room device and a desktop can both hear the name at once, and the server then ranks the two
wakes. Their scores come from different models, so the ranking reads each spotter's tables in
the server wake manifest instead (`backend/realtime/speech.py`, `arbitration_key`). This writes
the device model's tables there, from the evaluation the lead runs on the `tune` split (H7):

    python -m scripts.wake.device_calibration <eval.json> --manifest <server manifest.json>

The evaluation is a JSON object with exactly three keys: `cutoffs`, the probability cutoffs the
streaming ROC was measured at; `false_accepts_per_hour`, the device's false accepts per hour of
`tune` media at each cutoff, in the same order; and `positive_scores`, the device model's score
for each `tune` positive. Probabilities run from 0 to 1, as the device's cutoff does.

The tables are built by `eval_wake`'s own builders, so a share or a rate means on the device's
scale exactly what it means on the server spotter's. A higher score must read as the stronger
wake: `turn.py`'s `_rank_key` ranks the lower false-accept rate first and the higher positive
share next, so rates that rise with the cutoff are refused rather than smoothed, since
they mean the evaluation is not the one the table claims to describe.

Only `calibration.endpoint` changes. The manifest is spliced as text, not re-serialised, so every
other byte, `calibration.server` included, stays as it was. The result is written beside the
manifest, loaded by `read_manifest` and only then moved over it, so a manifest the server would
refuse is never written. No audio is read.
"""

import argparse
import json
import logging
import math
import os
import sys
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from backend.realtime.engines.kws import read_manifest
from backend.realtime.errors import SpeechEngineError

from .eval_wake import false_accept_table, positive_quantile_table

logger = logging.getLogger(__name__)

SPOTTER = "endpoint"
EVAL_KEYS = ("cutoffs", "false_accepts_per_hour", "positive_scores")
_JSON_WHITESPACE = " \t\n\r"


class RefusedError(Exception):
    """The conversion cannot go ahead; the message says why, and nothing is written."""


@dataclass(frozen=True, slots=True)
class Calibration:
    """The device spotter's two tables, as `[score, value]` points with ascending scores."""

    false_accepts: list[tuple[float, float]]
    positive_quantile: list[tuple[float, float]]


# -- The evaluation ---------------------------------------------------------------------------


def _is_number(value: object) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    )


def _numbers(evaluation: dict, key: str) -> list[float]:
    values = evaluation[key]
    if not isinstance(values, list) or not values:
        raise RefusedError(f"{key} must be a non-empty list of numbers")
    if not all(_is_number(value) for value in values):
        raise RefusedError(f"{key} must hold finite numbers only, got {values!r}")
    return [float(value) for value in values]


def _probabilities(values: list[float], key: str) -> list[float]:
    outside = [value for value in values if not 0.0 <= value <= 1.0]
    if outside:
        raise RefusedError(
            f"{key} must be the device model's probabilities, from 0 to 1; got {outside}"
        )
    return values


def calibration_from_evaluation(evaluation: object) -> Calibration:
    """Check one evaluation and build the device's tables from it.

    Raises:
        RefusedError: A key is missing or unknown, a value is not a finite number, a probability
            lies outside 0 to 1, a cutoff appears twice, a rate is negative, or a rate rises with
            the cutoff.
    """
    if not isinstance(evaluation, dict):
        raise RefusedError("the evaluation must be a JSON object")
    unknown = sorted(set(evaluation) - set(EVAL_KEYS))
    missing = sorted(set(EVAL_KEYS) - set(evaluation))
    if unknown:
        raise RefusedError(f"the evaluation has unknown keys {unknown}")
    if missing:
        raise RefusedError(f"the evaluation is missing keys {missing}")

    cutoffs = _probabilities(_numbers(evaluation, "cutoffs"), "cutoffs")
    rates = _numbers(evaluation, "false_accepts_per_hour")
    scores = _probabilities(_numbers(evaluation, "positive_scores"), "positive_scores")
    if len(rates) != len(cutoffs):
        raise RefusedError(
            f"false_accepts_per_hour must hold one rate per cutoff: {len(rates)} rates for "
            f"{len(cutoffs)} cutoffs"
        )
    if len(set(cutoffs)) != len(cutoffs):
        raise RefusedError(f"cutoffs lists a cutoff twice: {cutoffs}")
    if any(rate < 0 for rate in rates):
        raise RefusedError(f"false_accepts_per_hour must be at least 0, got {rates}")

    points = sorted(zip(cutoffs, rates, strict=True))
    for (low, low_rate), (high, high_rate) in zip(points, points[1:], strict=False):
        if high_rate > low_rate:
            raise RefusedError(
                f"false accepts per hour rise with the cutoff, from {low_rate} at {low} to "
                f"{high_rate} at {high}; a higher cutoff cannot accept more"
            )
    # With rising rates refused, the running minimum changes nothing; it is called so the
    # table is the server spotter's by construction, not by a second definition.
    return Calibration(
        false_accepts=false_accept_table(points),
        positive_quantile=positive_quantile_table(scores, cutoffs),
    )


# -- The manifest splice ----------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Member:
    """One member of a JSON object in the manifest's text, by character offsets.

    `lead` is where the whitespace before the key starts, just after the `{` or `,`.
    """

    key: str
    lead: int
    key_start: int
    value_start: int
    value_end: int


def _skip(text: str, index: int) -> int:
    while index < len(text) and text[index] in _JSON_WHITESPACE:
        index += 1
    return index


def _members(text: str, start: int) -> tuple[list[_Member], int]:
    """The members of the object whose `{` is at `start`, and the offset of its `}`.

    The text has already parsed as JSON, so its structure is known to be well formed.
    """
    decoder = json.JSONDecoder()
    members: list[_Member] = []
    lead = start + 1
    index = _skip(text, lead)
    if text[index] == "}":
        return members, index
    while True:
        key, after_key = json.decoder.scanstring(text, index + 1)
        colon = _skip(text, after_key)
        value_start = _skip(text, colon + 1)
        _, value_end = decoder.raw_decode(text, value_start)
        members.append(_Member(key, lead, index, value_start, value_end))
        index = _skip(text, value_end)
        if text[index] == "}":
            return members, index
        lead = index + 1
        index = _skip(text, lead)


def _keyed(members: list[_Member], where: str) -> dict[str, _Member]:
    """The members by key; a key given twice is refused, since `json` keeps only the last."""
    keyed: dict[str, _Member] = {}
    for member in members:
        if member.key in keyed:
            raise RefusedError(f"{where} has the key {member.key!r} twice")
        keyed[member.key] = member
    return keyed


def _indent(text: str, position: int) -> str | None:
    """The whitespace before `position` on its line, or None when other text precedes it."""
    prefix = text[text.rfind("\n", 0, position) + 1 : position]
    return prefix if not prefix.strip(" \t\r") else None


def _render(calibration: Calibration, indent: str | None, unit: str, newline: str) -> str:
    """The entry as JSON, its keys one level below `indent`, or on one line when it is None."""
    tables = (
        ("false_accepts", calibration.false_accepts),
        ("positive_quantile", calibration.positive_quantile),
    )
    members = [
        f"{json.dumps(key)}: {json.dumps([list(point) for point in points])}"
        for key, points in tables
    ]
    if indent is None:
        return "{" + ", ".join(members) + "}"
    inner = newline + indent + unit
    return "{" + inner + ("," + inner).join(members) + newline + indent + "}"


def splice_endpoint(text: str, calibration: Calibration) -> str:
    """The manifest text with `calibration.endpoint` set and every other character kept.

    An existing entry's value is replaced in place. A new entry goes after the last one, laid
    out like it, or alone inside an empty `calibration` object, one level below the key.
    """
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RefusedError(f"the manifest is not valid JSON: {exc}") from exc
    if not isinstance(document, dict):
        raise RefusedError("the manifest must be a JSON object")
    root, _ = _members(text, _skip(text, 0))
    entry = _keyed(root, "the manifest").get("calibration")
    if entry is None:
        raise RefusedError("the manifest has no calibration object")
    if text[entry.value_start] != "{":
        raise RefusedError("the manifest's calibration must be an object")
    spotters, close = _members(text, entry.value_start)
    current = _keyed(spotters, "calibration").get(SPOTTER)

    newline = "\r\n" if "\r\n" in text else "\n"
    multiline = "\n" in text
    top = _indent(text, entry.key_start)
    unit = top if multiline and top else "  "

    if current is not None:
        indent = _indent(text, current.key_start) if multiline else None
        rendered = _render(calibration, indent, unit, newline)
        return text[: current.value_start] + rendered + text[current.value_end :]

    if spotters:
        last = spotters[-1]
        separator = text[last.lead : last.key_start]
        indent = _indent(text, last.key_start) if multiline else None
        member = f"{json.dumps(SPOTTER)}: {_render(calibration, indent, unit, newline)}"
        return text[: last.value_end] + "," + separator + member + text[last.value_end :]

    at = entry.value_start + 1
    if not multiline or top is None:
        member = f"{json.dumps(SPOTTER)}: {_render(calibration, None, unit, newline)}"
        return text[:at] + member + text[at:]
    indent = top + unit
    member = f"{json.dumps(SPOTTER)}: {_render(calibration, indent, unit, newline)}"
    closing = "" if "\n" in text[at:close] else newline + top
    return text[:at] + newline + indent + member + closing + text[at:]


# -- The command ------------------------------------------------------------------------------


def _read_evaluation(path: Path) -> object:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RefusedError(f"the evaluation {path} could not be read: {exc}") from exc
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise RefusedError(f"the evaluation {path} is not valid JSON: {exc}") from exc


def _write_checked(manifest: Path, text: str) -> None:
    """Write `text` beside the manifest, load it as the server would, then move it over."""
    handle, name = tempfile.mkstemp(
        dir=manifest.parent, prefix=".device-calibration-", suffix=".json"
    )
    staged = Path(name)
    try:
        with os.fdopen(handle, "wb") as out:
            out.write(text.encode("utf-8"))
        try:
            read_manifest(staged, manifest.parent.name)
        except SpeechEngineError as exc:
            reason = str(exc).removeprefix(f"{staged}: ")
            raise RefusedError(
                f"{manifest} would not load with the new table, so it is unchanged: {reason}"
            ) from exc
        os.replace(staged, manifest)
    finally:
        staged.unlink(missing_ok=True)


def _run(evaluation_path: Path, manifest: Path) -> None:
    calibration = calibration_from_evaluation(_read_evaluation(evaluation_path))
    try:
        text = manifest.read_bytes().decode("utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        raise RefusedError(f"the manifest {manifest} could not be read: {exc}") from exc
    _write_checked(manifest, splice_endpoint(text, calibration))
    logger.info(
        "Wrote calibration.%s to %s: %d false-accept points, %d quantile points",
        SPOTTER,
        manifest,
        len(calibration.false_accepts),
        len(calibration.positive_quantile),
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.wake.device_calibration", description=__doc__
    )
    parser.add_argument("evaluation", type=Path, help="the device model's evaluation JSON")
    parser.add_argument(
        "--manifest", required=True, type=Path, help="the server wake model's manifest.json"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Write the device's tables; 0 when the manifest is written, 2 when the run is refused."""
    args = _parser().parse_args(argv)
    try:
        _run(args.evaluation, args.manifest)
    except RefusedError as exc:
        print(f"device_calibration: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    sys.exit(main())
