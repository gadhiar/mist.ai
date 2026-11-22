"""The pre-flight gate: will the gated cascade still wake on the tune positives, and stay quiet?

    python -m scripts.wake.verifier.preflight --runs <dir> --verifier <dir> \
        --spotter-eval <spotter_eval.json> --out <preflight.json>

Run before the one final measurement, on training data only (Design 10):

- Each fold's table `v*_fold(c)`: the fold's model (`<runs>/fold-<k>/`, torch on the CPU) scores
  the calibration share's hit windows `calibrate.py` recorded in `<verifier>/calibration.json`,
  and `calibrate.threshold_table` makes the table from them, as it made the final one.
- `r(c)`: of the real train positives with a hit at c, the fraction whose best out-of-fold hit
  there scores at least its fold's `v*_fold(c)` (each fold's `oof.jsonl`).
- `W_s(c)`: v16's tune positives whose spotter-only score reaches c. Only `positive_scores` is
  read from the spotter evaluation. The predicted gated tune wakes, `W_s(c) * r(c)`, must reach
  `REQUIRED_WAKES` (one wake of margin over 14) at some cutoff; c1 is the highest that does.
- The final model's gated rate at `v*(c1)` on the check share must be at most `CHECK_BAR` an
  hour; ONNX Runtime's parity with torch (`export.json`) at most `export.PARITY_BOUND`; and the
  verifier directory must pass `read_verifier_manifest`, `provenance.json` included.
- The Raj-channel negatives' out-of-fold counts are reported, not gated.

It prints counts and rates only, never a file name, and writes them to `--out`. Exit 0 pass,
1 fail, 2 refused (a missing fold, a malformed input).
"""

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from backend.realtime.engines.verifier import GRID, read_verifier_manifest
from backend.realtime.errors import SpeechEngineError

from .calibrate import (
    CALIBRATION_FILE,
    MANIFEST_FILE,
    CalibrateError,
    ScoredHit,
    gated_counts,
    threshold_table,
)
from .dataset import PROVENANCE_FILE, DatasetError, read_wav_segment
from .export import PARITY_BOUND, REPORT_FILE
from .stage import WAV_SUFFIX
from .train import FOLDS, RUN_OOF, fold_run

REQUIRED_WAKES = 15
CHECK_BAR = 0.5
RAJ_CHANNEL = ("homophone", "media", "pre_name")


class PreflightError(Exception):
    """A refusal; the command exits 2 with the message."""


@dataclass(frozen=True, slots=True)
class OutOfFold:
    """One out-of-fold window's score, as `train.py` wrote it."""

    fold: int
    kind: str
    file_sha256: str
    cutoffs: tuple[float, ...]
    score: float


FoldScorer = Callable[[Path, Sequence[ScoredHit], str], list[float]]
"""Scores the calibration hit windows with a fold's model: (run dir, hits, AMI root)."""


def torch_fold_scorer(run: Path, hits: Sequence[ScoredHit], ami_root: str) -> list[float]:
    """Each hit window's score from the fold's torch model on the CPU."""
    import torch  # noqa: PLC0415 - only the real scorer needs it

    from .export import load_run  # noqa: PLC0415 - loads torch and torchaudio

    model = load_run(run)
    scores = []
    with torch.no_grad():
        for hit in hits:
            path = Path(ami_root, f"{hit.meeting}{WAV_SUFFIX}")
            samples = read_wav_segment(path, hit.start, hit.end - hit.start)
            audio = samples.astype(np.float32) / np.float32(32768.0)
            logit = model(torch.from_numpy(audio).reshape(1, -1))
            scores.append(float(torch.sigmoid(logit)[0]))
    return scores


def _json(path: Path, what: str) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise PreflightError(f"{what} could not be read: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"{what} must be a JSON object")
    return value


def positive_scores(path: Path) -> list[float]:
    """The spotter evaluation's `positive_scores`, the one key read from it.

    Raises:
        PreflightError: The key is missing or not a list of scores from 0 to 1.
    """
    scores = _json(path, "--spotter-eval").get("positive_scores")
    if not isinstance(scores, list) or not scores:
        raise PreflightError("--spotter-eval has no non-empty positive_scores list")
    if not all(
        isinstance(s, int | float) and not isinstance(s, bool) and 0 <= s <= 1 for s in scores
    ):
        raise PreflightError("--spotter-eval positive_scores must be numbers from 0 to 1")
    return [float(score) for score in scores]


def read_oof(runs: Path) -> list[OutOfFold]:
    """Every fold's out-of-fold scores.

    Raises:
        PreflightError: A fold's `oof.jsonl` is missing or malformed.
    """
    rows = []
    for fold in range(FOLDS):
        path = runs / fold_run(fold) / RUN_OOF
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                value = json.loads(line)
                rows.append(
                    OutOfFold(
                        fold,
                        str(value["kind"]),
                        str(value["file_sha256"]),
                        tuple(float(c) for c in value["cutoffs"]),
                        float(value["score"]),
                    )
                )
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise PreflightError(f"{fold_run(fold)}/{RUN_OOF} could not be read: {exc}") from exc
    return rows


def _hits(record: Mapping, share: str) -> list[ScoredHit]:
    try:
        return [
            ScoredHit(
                str(hit["meeting"]),
                int(hit["start"]),
                int(hit["end"]),
                tuple(float(c) for c in hit["cutoffs"]),
                float(hit["score"]),
            )
            for hit in record["hits"][share]
        ]
    except (KeyError, TypeError, ValueError) as exc:
        raise PreflightError(f"{CALIBRATION_FILE} has no readable {share} hits: {exc}") from exc


def pass_rates(
    oof: Sequence[OutOfFold], tables: Mapping[int, Sequence[float]]
) -> list[tuple[int, int]]:
    """At each cutoff, (real positives with a hit there, those whose best hit passes its fold).

    A positive is one recording, its windows' `file_sha256`; its best hit at c is its highest
    scoring out-of-fold window that is a hit at c.
    """
    best: dict[tuple[str, float], tuple[float, int]] = {}
    for row in oof:
        if row.kind != "real_positive":
            continue
        for cutoff in row.cutoffs:
            key = (row.file_sha256, cutoff)
            if key not in best or row.score > best[key][0]:
                best[key] = (row.score, row.fold)
    counts = []
    for row_index, cutoff in enumerate(GRID):
        found = [value for (_, c), value in best.items() if c == cutoff]
        passed = sum(1 for score, fold in found if score >= tables[fold][row_index])
        counts.append((len(found), passed))
    return counts


def raj_channel(
    oof: Sequence[OutOfFold], tables: Mapping[int, Sequence[float]]
) -> dict[str, list[int]]:
    """Per kind, at each cutoff, the out-of-fold negatives passing their fold's threshold there.

    A homophone or media window counts at the cutoffs it is a hit at; a pre-name window, which
    holds no hit, at every cutoff.
    """
    counts: dict[str, list[int]] = {kind: [0] * len(GRID) for kind in RAJ_CHANNEL}
    for row in oof:
        if row.kind not in counts:
            continue
        for index, cutoff in enumerate(GRID):
            at = row.kind == "pre_name" or cutoff in row.cutoffs
            if at and row.score >= tables[row.fold][index]:
                counts[row.kind][index] += 1
    return counts


def preflight(args: argparse.Namespace, scorer: FoldScorer) -> int:
    """Compute the gate's figures, write them, and return 0 on a pass and 1 on a failure."""
    verifier = args.verifier
    try:
        manifest = read_verifier_manifest(verifier / MANIFEST_FILE, verifier.name)
    except SpeechEngineError as exc:
        raise PreflightError(f"the verifier directory is refused: {exc}") from exc
    provenance = _json(verifier / PROVENANCE_FILE, PROVENANCE_FILE)
    if provenance.get("training") != manifest.training:
        raise PreflightError("provenance.json's training is not the manifest's")
    record = _json(verifier / CALIBRATION_FILE, CALIBRATION_FILE)
    report = _json(verifier / REPORT_FILE, REPORT_FILE)
    try:
        hours = {share: float(record["hours"][share]) for share in ("calibration", "check")}
        ami_root = str(record["ami_root"])
        drift = float(report["parity"]["max_abs_diff"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PreflightError(f"the calibration or export record is malformed: {exc}") from exc
    calibration = _hits(record, "calibration")
    check = _hits(record, "check")
    wakes = positive_scores(args.spotter_eval)
    oof = read_oof(args.runs)
    tables = {}
    for fold in range(FOLDS):
        scores = scorer(args.runs / fold_run(fold), calibration, ami_root)
        rescored = [
            ScoredHit(hit.meeting, hit.start, hit.end, hit.cutoffs, score)
            for hit, score in zip(calibration, scores, strict=True)
        ]
        tables[fold] = threshold_table(rescored, hours["calibration"])
    rates = pass_rates(oof, tables)
    rows = []
    for cutoff, (with_hit, passed) in zip(GRID, rates, strict=True):
        spotter_wakes = sum(1 for score in wakes if score >= cutoff)
        r = passed / with_hit if with_hit else 0.0
        rows.append(
            {
                "cutoff": cutoff,
                "spotter_wakes": spotter_wakes,
                "positives_with_hit": with_hit,
                "positives_passed": passed,
                "r": r,
                "predicted_wakes": spotter_wakes * r,
            }
        )
    c1 = max(
        (row["cutoff"] for row in rows if row["predicted_wakes"] >= REQUIRED_WAKES - 1e-9),
        default=None,
    )
    failures = []
    check_rate = None
    if c1 is None:
        failures.append(f"no cutoff predicts {REQUIRED_WAKES} gated tune wakes")
    else:
        row_index = GRID.index(c1)
        gated = gated_counts(check, manifest.scores)[row_index]
        check_rate = gated / hours["check"]
        if check_rate > CHECK_BAR:
            failures.append(f"the check share's gated rate at {c1} is {check_rate:.3f} an hour")
    if not math.isfinite(drift) or drift > PARITY_BOUND:
        failures.append(f"ONNX Runtime's parity with torch is {drift:.2e}")
    raj = raj_channel(oof, tables)
    sources = provenance.get("sources") or {}
    payload = {
        "version": 1,
        "pass": not failures,
        "failures": failures,
        "c1": c1,
        "required_wakes": REQUIRED_WAKES,
        "check_rate": check_rate,
        "check_bar": CHECK_BAR,
        "parity": drift,
        "rows": rows,
        "fold_tables": {str(fold): table for fold, table in tables.items()},
        "raj_channel": raj,
        "provenance": {
            name: {"files": entry.get("files"), "hours": entry.get("hours")}
            for name, entry in sorted(sources.items())
            if isinstance(entry, dict)
        },
    }
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for row in rows:
        if row["positives_with_hit"] or row["spotter_wakes"]:
            print(
                f"cutoff {row['cutoff']:.2f}: W_s {row['spotter_wakes']}, r "
                f"{row['positives_passed']}/{row['positives_with_hit']}, predicted "
                f"{row['predicted_wakes']:.2f}"
            )
    for name, entry in payload["provenance"].items():
        print(f"provenance {name}: {entry['files']} files, {entry['hours']} h")
    if c1 is not None:
        index = GRID.index(c1)
        counts = ", ".join(f"{kind} {raj[kind][index]}" for kind in RAJ_CHANNEL)
        print(f"c1 {c1:.2f}: check share {check_rate:.3f} an hour; Raj channel {counts}")
    print(f"parity {drift:.2e}")
    for failure in failures:
        print(f"fail: {failure}")
    print("pre-flight passed" if not failures else "pre-flight failed")
    return 0 if not failures else 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m scripts.wake.verifier.preflight", description=__doc__
    )
    parser.add_argument("--runs", required=True, type=Path)
    parser.add_argument("--verifier", required=True, type=Path)
    parser.add_argument("--spotter-eval", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None, *, scorer: FoldScorer | None = None) -> int:
    """Run the gate; 0 pass, 1 fail, 2 refused."""
    args = _parser().parse_args(argv)
    try:
        return preflight(args, scorer or torch_fold_scorer)
    except (PreflightError, CalibrateError, DatasetError) as exc:
        print(f"preflight: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
