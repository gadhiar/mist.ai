"""`labels prefill --run R [--device cpu|cuda]`: pre-fill the clip labels of extract run R.

The labels pack asks Raj, per kept clip, who is audible and her energy and mood. This command
fills those in first, so he corrects values rather than writing each one: a job in the `score`
family's image, which already holds the emotion model, runs `stages/prefill.py` over R's kept
clips and writes `seed/prefill/<R>/prefill.jsonl` (`labels.PrefillRecord`), `summary.json` and
`outputs.json`. The output has its own folder because a job's output folder receives its
`outputs.json`, and R's folder must stay as extract left it.

`jobs.py` checks a job's outputs and not its inputs, so before the job this command writes
`seed/prefill/<R>/prefill-plan.json` with `segment.min_s` and the sha256 of `segments.jsonl`,
`turns.jsonl` and each kept FLAC; the stage refuses any file that is missing or differs. The
job's one input, R's folder mounted read-only, carries the digest of those shas in path order,
so `job.json` pins them too. The job runs on the CPU by default; on cuda it keeps the family's
`vram_gb` for the preflight. Seed stages stay on this host, so there is no executor option.

This command prints counts and seconds only.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from .. import gates as gates_mod
from .. import jobs
from .. import labels as labels_mod
from .. import root as root_mod
from ..errors import InputError, JobError
from ..records import JobInput, SegmentRecord, read_json, read_jsonl, sha256_file, write_json
from . import extract as extract_cmd

FAMILY = "score"
MODULE = "scripts.voice_data.stages.prefill"
DEVICES = extract_cmd.DEVICES
PLAN_SCHEMA = 1
SUMMARY_SCHEMA = 1
SUMMARY_NAME = "summary.json"
SEGMENTS_NAME = extract_cmd.SEGMENTS_NAME
TURNS_NAME = extract_cmd.TURNS_NAME
MIN_S_GATE = "segment.min_s"
PREFILL_VOICES = labels_mod.VOICES[:3]


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the `labels` command and its `prefill` action."""
    parser = subparsers.add_parser(
        "labels",
        help="clip labels: pre-fill a run's values for the labels pack (local jobs only)",
        description=(
            "Pre-fill each kept clip's energy, mood and who is audible for extract run R, from "
            "the emotion model and the run's diarizer turns, so the labels pack opens with "
            "values to correct. Runs on this host only."
        ),
    )
    actions = parser.add_subparsers(dest="labels_action", metavar="ACTION", required=True)
    prefill = actions.add_parser("prefill", help="pre-fill a run's labels in a score-image job")
    prefill.add_argument("--run", required=True, help="the extract run under seed/runs")
    prefill.add_argument("--device", choices=DEVICES, default="cpu", help="where the model runs")
    prefill.set_defaults(func=run_prefill)


def make_docker() -> jobs.DockerOps:
    """Return the docker CLI; tests replace this function."""
    return jobs.HostDocker()


def image_tag(family: str) -> str:
    """Return the local image of `family` at HEAD, as `images build` tags it."""
    return extract_cmd.image_tag(family)


def files_digest(files: Mapping[str, str]) -> str:
    """Return the digest of the plan's file sha256s in path order: the job input's sha256."""
    return hashlib.sha256("\n".join(files[rel] for rel in sorted(files)).encode()).hexdigest()


def _kept(root: root_mod.Root, run_id: str) -> list[SegmentRecord]:
    run_dir = extract_cmd.run_rel(run_id)
    segments = root.path / run_dir / SEGMENTS_NAME
    if not segments.is_file():
        raise InputError(f"refused: run {run_id} has no {SEGMENTS_NAME}; run extract first")
    if not (root.path / run_dir / TURNS_NAME).is_file():
        raise InputError(f"refused: run {run_id} has no {TURNS_NAME}; run extract again")
    kept = [r for r in read_jsonl(segments, SegmentRecord) if r.kept]
    if not kept:
        raise InputError(f"refused: run {run_id} has no kept segment to pre-fill")
    for record in kept:
        if not record.audio.startswith(f"{run_dir}/"):
            raise InputError(f"refused: {record.segment_id}'s audio is outside run {run_id}")
        if not (root.path / record.audio).is_file():
            raise InputError(f"refused: {record.segment_id}'s audio is missing")
    return kept


def write_plan(root: root_mod.Root, run_id: str) -> dict[str, object]:
    """Check run `run_id` and write its `prefill-plan.json`; return the plan.

    Raises:
        InputError: The run id is not one folder name, the run has no `segments.jsonl` or
            `turns.jsonl`, no kept segment, or a kept segment's audio is missing, or the pre-fill
            folder already holds `prefill.jsonl` or `summary.json`.
    """
    out_dir = labels_mod.prefill_rel(run_id)
    kept = _kept(root, run_id)
    finished = (labels_mod.PREFILL_NAME, SUMMARY_NAME)
    if any((root.path / out_dir / name).exists() for name in finished):
        raise InputError(f"refused: {out_dir} already holds a pre-fill of run {run_id}")
    run_dir = extract_cmd.run_rel(run_id)
    rels = [f"{run_dir}/{SEGMENTS_NAME}", f"{run_dir}/{TURNS_NAME}"]
    rels += [record.audio for record in kept]
    plan: dict[str, object] = {
        "schema": PLAN_SCHEMA,
        "run": run_id,
        "min_s": gates_mod.load(root.path).effective()[MIN_S_GATE],
        "files": {rel: sha256_file(root.path / rel) for rel in rels},
    }
    write_json(root, f"{out_dir}/{labels_mod.PREFILL_PLAN_NAME}", plan)
    return plan


def _job_id(root: root_mod.Root) -> str:
    now = datetime.now(UTC)
    number = 1
    while (root.path / "runs" / root_mod.job_id(now, number)).exists():
        number += 1
    return root_mod.job_id(now, number)


def run_prefill(args: argparse.Namespace) -> int:
    """Plan the pre-fill, run its job and print its counts."""
    root = root_mod.open_root(args.root)
    out_dir = labels_mod.prefill_rel(args.run)
    plan = write_plan(root, args.run)
    files: Mapping[str, str] = plan["files"]  # type: ignore[assignment]
    cuda = args.device == "cuda"
    job = jobs.JobSpec(
        job_id=_job_id(root),
        family=FAMILY,
        image=image_tag(FAMILY),
        module=MODULE,
        args=("--run", args.run, "--device", args.device),
        inputs=(JobInput(rel=extract_cmd.run_rel(args.run), sha256=files_digest(files)),),
        outputs=out_dir,
        gpu=cuda,
        network=False,
    )
    jobs.executor_for("local", root=root, docker=make_docker()).run(job)
    summary = read_json(root.path / out_dir / SUMMARY_NAME)
    if not isinstance(summary, dict):
        raise JobError(f"run {args.run}: the pre-fill's {SUMMARY_NAME} is not a JSON object")
    for line in summary_lines(args.run, summary):
        print(line)
    return 0


def _count(value: object) -> str:
    # Numbers only: anything else prints as `?`, so no text reaches stdout through a summary.
    return str(value) if isinstance(value, int) and not isinstance(value, bool) else "?"


def summary_lines(run_id: str, summary: Mapping[str, object]) -> Sequence[str]:
    """Return the summary as printed: the clip count, seconds and the `voices` counts."""
    seconds = summary.get("seconds")
    number = isinstance(seconds, int | float) and not isinstance(seconds, bool)
    shown = f"{seconds:.2f}" if number else "?"
    voices = summary.get("voices")
    counts = voices if isinstance(voices, Mapping) else {}
    return [
        f"prefill run {run_id}: {_count(summary.get('clips'))} clips, {shown} s",
        "  voices: " + ", ".join(f"{v} {_count(counts.get(v))}" for v in PREFILL_VOICES),
    ]
