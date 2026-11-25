"""Pre-fill the clip labels of an extract run, run as the `score` family's job.

Runs as `/opt/venv-score/bin/python -I -u -m scripts.voice_data.stages.prefill --data /data
--run R --device D`, with the extract run's folder `seed/runs/<R>/` mounted read-only and
`seed/prefill/<R>/` writable. The host's `prefill-plan.json` there names the run, `segment.min_s`
and the sha256 of `segments.jsonl`, `turns.jsonl` and each kept FLAC; each file is checked
against it before it is read, as `stages/student_score.py` checks its plan's files, because the
executor checks a job's outputs and not its inputs.

Each kept segment, in segment order, gets one `labels.PrefillRecord`: the emotion model's
arousal, dominance and valence on the clip, `labels.scale_from` of arousal as its energy and of
valence as its mood, and `labels.prefill_voices` from the run's turns for who is audible and the
keep range. The stage writes `prefill.jsonl`, `summary.json` and `outputs.json`.

A refusal names the kind of file ("a kept clip", "turns.jsonl"), never its path or a segment id,
and nothing printed carries more than counts and seconds. The model comes from `load_model`,
which the arousal binding serves; tests pass a fake through it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from .. import labels
from ..bindings.protocols import EmotionDimensions
from ..commands import extract as extract_cmd
from ..commands import labels as contract
from ..errors import InputError
from ..records import SegmentRecord, read_jsonl
from . import fetch

DIGITS = 4


class StageError(RuntimeError):
    """The pre-fill cannot proceed; the message names a file kind, never a path or an id."""


def load_model(device: str) -> EmotionDimensions:
    """Load the emotion model from the arousal binding, on `device`.

    The binding is imported here and not at the top, so that a module walk imports no model
    package.
    """
    from ..bindings import arousal

    return arousal.load(device)


# Files.


def _sha256(path: Path) -> str:
    return fetch.sha256_file(path)


def _write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    os.replace(part, path)


def _write_json(path: Path, payload: object) -> None:
    _write_bytes(path, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))


def _read_audio(path: Path) -> tuple[np.ndarray, int]:
    import soundfile

    audio, rate = soundfile.read(str(path), dtype="float32", always_2d=True)
    return audio.mean(axis=1), int(rate)


def _checked(path: Path, sha256: object, what: str) -> Path:
    if not path.is_file() or _sha256(path) != sha256:
        raise StageError(f"{what} is missing or does not match the plan")
    return path


# The plan.


def read_plan(data: Path, run_id: str) -> dict[str, object]:
    """Return the host's pre-fill plan for run `run_id`.

    Raises:
        StageError: It is missing, not JSON, not this run's schema 1 plan, or its `min_s` or
            `files` is malformed.
    """
    try:
        path = data / labels.prefill_rel(run_id) / labels.PREFILL_PLAN_NAME
        plan = json.loads(path.read_text(encoding="utf-8"))
    except InputError as exc:
        raise StageError("the run id is not one folder name") from exc
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise StageError("the pre-fill plan is missing or not JSON") from exc
    if not isinstance(plan, dict):
        raise StageError("the pre-fill plan is not a JSON object")
    if (plan.get("schema"), plan.get("run")) != (contract.PLAN_SCHEMA, run_id):
        raise StageError("the pre-fill plan is not this run's schema 1 plan")
    min_s = plan.get("min_s")
    if not isinstance(min_s, int | float) or isinstance(min_s, bool) or not min_s > 0:
        raise StageError("the pre-fill plan's min_s is not a positive number of seconds")
    files = plan.get("files")
    if not isinstance(files, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in files.items()
    ):
        raise StageError("the pre-fill plan's files are not paths and sha256s")
    return plan


# The pre-fill.


def _value(value: float) -> float:
    # A non-finite output would pass the clip to 0..1 as a number it is not.
    if not math.isfinite(value):
        raise StageError("the emotion model gave a value that is not a number")
    return round(min(1.0, max(0.0, float(value))), DIGITS)


def prefill(
    segment: SegmentRecord,
    turns: Sequence[extract_cmd.TurnRecord],
    min_s: float,
    values: tuple[float, float, float],
) -> labels.PrefillRecord:
    """Return one kept segment's pre-fill from the model's values and the run's turns.

    Energy and mood are `scale_from` of the stored, rounded arousal and valence, so a record's
    scale values always follow from its own model values.
    """
    arousal, dominance, valence = (_value(v) for v in values)
    voices, keep_from, keep_to, other_s = labels.prefill_voices(segment, turns, min_s)
    return labels.PrefillRecord(
        segment_id=segment.segment_id,
        source_id=segment.source_id,
        start_s=segment.start_s,
        end_s=segment.end_s,
        duration_s=round(segment.end_s - segment.start_s, 2),
        arousal=arousal,
        dominance=dominance,
        valence=valence,
        energy=labels.scale_from(arousal),
        mood=labels.scale_from(valence),
        voices=voices,
        keep_from_s=keep_from,
        keep_to_s=keep_to,
        other_s=other_s,
        overlap_ms=segment.overlap_ms,
    )


def _read_inputs(
    data: Path, run_id: str, files: Mapping[str, str]
) -> tuple[list[SegmentRecord], list[extract_cmd.TurnRecord]]:
    run_dir = extract_cmd.run_rel(run_id)
    segments_rel = f"{run_dir}/{contract.SEGMENTS_NAME}"
    turns_rel = f"{run_dir}/{contract.TURNS_NAME}"
    try:
        segments = read_jsonl(
            _checked(data / segments_rel, files.get(segments_rel), contract.SEGMENTS_NAME),
            SegmentRecord,
        )
        turns = read_jsonl(
            _checked(data / turns_rel, files.get(turns_rel), contract.TURNS_NAME),
            extract_cmd.TurnRecord,
        )
    except InputError as exc:
        raise StageError("segments.jsonl or turns.jsonl holds a malformed record") from exc
    kept = [s for s in segments if s.kept]
    clips = set(files) - {segments_rel, turns_rel}
    if not kept or sorted(clips) != sorted(s.audio for s in kept):
        raise StageError("the plan's clips are not the run's kept segments")
    return kept, turns


def run(data: Path, run_id: str, model: EmotionDimensions, device: str) -> list[str]:
    """Pre-fill every kept segment of run `run_id`; return the root-relative files written.

    Raises:
        StageError: The plan is not this run's, a planned file is missing or differs from the
            plan, or a value is not a number.
    """
    plan = read_plan(data, run_id)
    files: Mapping[str, str] = plan["files"]  # type: ignore[assignment]
    min_s = float(plan["min_s"])  # type: ignore[arg-type]
    kept, turns = _read_inputs(data, run_id, files)
    records: list[labels.PrefillRecord] = []
    for segment in kept:
        clip = _checked(data / segment.audio, files[segment.audio], "a kept clip")
        audio, rate = _read_audio(clip)
        values = model.dimensions(audio, rate)
        try:
            records.append(prefill(segment, turns, min_s, values))
        except InputError as exc:
            raise StageError(f"kept clip {len(records) + 1}: the pre-fill is not valid") from exc
    out_dir = labels.prefill_rel(run_id)
    seconds = round(sum(r.duration_s for r in records), 2)
    voices = {v: sum(r.voices == v for r in records) for v in contract.PREFILL_VOICES}
    summary = {
        "schema": contract.SUMMARY_SCHEMA,
        "run": run_id,
        "clips": len(records),
        "seconds": seconds,
        "device": device,
        "voices": voices,
    }
    prefill_rel = f"{out_dir}/{labels.PREFILL_NAME}"
    summary_rel = f"{out_dir}/{contract.SUMMARY_NAME}"
    text = "".join(json.dumps(r.to_json()) + "\n" for r in records)
    _write_bytes(data / prefill_rel, text.encode("utf-8"))
    _write_json(data / summary_rel, summary)
    print(f"pre-filled {len(records)} clips, {seconds:.2f} s")
    return [prefill_rel, summary_rel]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the stage; write `outputs.json` for the executor to verify."""
    parser = argparse.ArgumentParser(prog="scripts.voice_data.stages.prefill")
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--run", required=True)
    parser.add_argument("--device", default="cpu", choices=contract.DEVICES)
    args = parser.parse_args(argv)
    try:
        written = run(args.data, args.run, load_model(args.device), args.device)
    except StageError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    files = [{"rel": rel, "sha256": _sha256(args.data / rel)} for rel in written]
    outputs = args.data / labels.prefill_rel(args.run) / "outputs.json"
    _write_json(outputs, {"schema": 1, "files": files})
    return 0


if __name__ == "__main__":
    sys.exit(main())
