"""`clean --run R --from E --target T [--device cpu|cuda]`: SIM-gated cleaning and references.

The host side of Design 5. `E` is a finished extract run and `R` a new run that holds its cleaned
copy: `segments.jsonl` with transcripts, `audio/` at 24 kHz, `gates.json`, `refs.json` and
`report.json`. The extract run is read-only to the job and stays as it was, so the clean-ab pack
can play each segment before (`E`) and after (`R`) cleaning. The chosen references are written
by the job into `R/refs/<state>/` and copied here to `targets/<T>/refs/<state>/`, where
`target build` reads them. Seed stages stay on this host, so there is no executor option.

Raj's labels (`seed/labels.csv`, from `pack import`) confirm segments when present: a listed
segment takes the state he gave it, and a kept segment he did not list is dropped as `not-her`.
Without labels, every kept segment is cleaned, and only primary-cluster segments count as
confirmed for the leave-one-out distribution and the references.

The stage imports the names below, so host and job share one definition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from .. import gates as gates_mod
from .. import jobs
from .. import root as root_mod
from ..errors import InputError, JobError
from ..packs import LABELS_REL
from ..records import JobInput, SegmentRecord, read_json, read_jsonl, sha256_file, write_json
from ..states import SPEAKING_STATES
from . import extract as extract_cmd

FAMILY = "extract"
MODULE = "scripts.voice_data.stages.clean"
DEVICES = extract_cmd.DEVICES
PLAN_NAME = "clean-plan.json"
SEGMENTS_NAME = "segments.jsonl"
GATES_NAME = "gates.json"
REFS_NAME = "refs.json"
REPORT_NAME = "report.json"
AUDIO_DIR = "audio"
REFS_DIR = "refs"
LABELS_HEADER = ["segment_id", "state"]
PRIMARY = "primary"
NOT_HER = "not-her"
RUN_ID = extract_cmd.RUN_ID
REF_REL = re.compile(r"refs/(?P<state>[a-z]+)/(?P<n>[1-9][0-9]*)\.wav")


def run_rel(run_id: str) -> str:
    """Return a run's folder, relative to the root."""
    return extract_cmd.run_rel(run_id)


def target_rel(target_id: str) -> str:
    """Return a target's folder, relative to the root."""
    return f"targets/{target_id}"


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the `clean` command."""
    parser = subparsers.add_parser(
        "clean",
        help="SIM-gated cleaning, transcripts and references (local jobs only)",
        description=(
            "Clean the kept segments of extract run E into the new run R: each enhancement step "
            "kept only by the cleaning gate, one loudness gain per source, 24 kHz mono, two ASR "
            "transcripts, and 2-4 references per state written to targets/<T>/refs/. Runs on "
            "this host only."
        ),
    )
    parser.add_argument("--run", required=True, help="the new run id, a folder under seed/runs")
    parser.add_argument("--from", dest="from_run", required=True, help="the extract run to clean")
    parser.add_argument("--target", required=True, help="the target id the references are for")
    parser.add_argument("--device", choices=DEVICES, default="cpu", help="where models run")
    parser.set_defaults(func=run)


def make_docker() -> jobs.DockerOps:
    """Return the docker CLI; tests replace this function."""
    return jobs.HostDocker()


def image_tag(family: str) -> str:
    """Return the local image of `family` at HEAD, as `images build` tags it."""
    return extract_cmd.image_tag(family)


def read_labels(root: root_mod.Root) -> dict[str, str] | None:
    """Return Raj's labels as segment id to state, or `None` when there are none yet.

    Errors name the row number only.

    Raises:
        InputError: The header is not `segment_id,state`, or a row is malformed or names a
            state outside the speaking states.
    """
    path = root.path / LABELS_REL
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"refused: {LABELS_REL} is unreadable") from exc
    rows = [(n, row) for n, row in enumerate(csv.reader(io.StringIO(text)), 1) if any(row)]
    if not rows or [cell.strip() for cell in rows[0][1]] != LABELS_HEADER:
        raise InputError(f"refused: {LABELS_REL} must start with the header segment_id,state")
    labels: dict[str, str] = {}
    for number, row in rows[1:]:
        if len(row) != 2 or row[1].strip() not in SPEAKING_STATES:
            raise InputError(f"refused: {LABELS_REL} row {number} is not a segment and a state")
        labels[row[0].strip()] = row[1].strip()
    return labels


def confirmed_ids(records: Sequence[SegmentRecord], labels: Mapping[str, str] | None) -> list[str]:
    """Return the kept segments that count as the target's: labelled ones, else primary ones."""
    kept = [r for r in records if r.kept]
    if labels is not None:
        return [r.segment_id for r in kept if r.segment_id in labels]
    return [r.segment_id for r in kept if r.cluster == PRIMARY]


def _run_input(root: root_mod.Root, from_dir: str, records: Sequence[SegmentRecord]) -> JobInput:
    # One mount for the extract run; its sha256 is a digest of segments.jsonl's and each kept
    # segment's audio sha256, so job.json pins the inputs.
    parts = [sha256_file(root.path / from_dir / SEGMENTS_NAME)]
    parts += [sha256_file(root.path / r.audio) for r in records if r.kept]
    digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
    return JobInput(rel=from_dir, sha256=digest)


def _job_id(root: root_mod.Root) -> str:
    now = datetime.now(UTC)
    number = 1
    while (root.path / "runs" / root_mod.job_id(now, number)).exists():
        number += 1
    return root_mod.job_id(now, number)


def _check_ids(args: argparse.Namespace) -> None:
    for name, value in (("run id", args.run), ("run id", args.from_run)):
        if not RUN_ID.fullmatch(value) or ".." in value:
            raise InputError(f"refused: a {name} is one folder name under seed/runs")
    if not RUN_ID.fullmatch(args.target) or ".." in args.target:
        raise InputError("refused: a target id is one folder name under targets")
    if args.run == args.from_run:
        raise InputError("refused: the clean run and the extract run must differ")


def _read_from_run(root: root_mod.Root, from_run: str) -> list[SegmentRecord]:
    from_dir = run_rel(from_run)
    path = root.path / from_dir / SEGMENTS_NAME
    if not path.is_file():
        raise InputError(f"refused: {from_dir}/{SEGMENTS_NAME} is missing; run extract first")
    records = read_jsonl(path, SegmentRecord)
    for record in records:
        if record.kept and not (root.path / record.audio).is_file():
            raise InputError(f"refused: {record.segment_id}'s audio {record.audio} is missing")
        if record.kept and not record.audio.startswith(f"{from_dir}/"):
            raise InputError(f"refused: {record.segment_id}'s audio is outside {from_dir}")
    return records


def run(args: argparse.Namespace) -> int:
    """Plan the run, run the clean job, place the references and print the report."""
    root = root_mod.open_root(args.root)
    _check_ids(args)
    thresholds = gates_mod.load(root.path).effective()
    records = _read_from_run(root, args.from_run)
    run_dir = run_rel(args.run)
    if any((root.path / run_dir / name).exists() for name in (SEGMENTS_NAME, REPORT_NAME)):
        raise InputError(f"refused: {run_dir} already holds a run; choose a new run id")
    refs_dir = f"{target_rel(args.target)}/{REFS_DIR}"
    if (root.path / refs_dir).exists():
        raise InputError(f"refused: {refs_dir} already holds references; choose a new target")
    labels = read_labels(root)
    kept = {r.segment_id for r in records if r.kept}
    if labels is not None and not kept & set(labels):
        raise InputError(f"refused: {LABELS_REL} labels no kept segment of run {args.from_run}")
    if len(confirmed_ids(records, labels)) < 2:
        raise InputError(
            f"refused: run {args.from_run} has fewer than two confirmed segments to clean"
        )

    plan = {
        "schema": 1,
        "run": args.run,
        "from": args.from_run,
        "target": args.target,
        "thresholds": thresholds,
        "labels": labels,
    }
    write_json(root, f"{run_dir}/{PLAN_NAME}", plan)
    job = jobs.JobSpec(
        job_id=_job_id(root),
        family=FAMILY,
        image=image_tag(FAMILY),
        module=MODULE,
        args=("--run", args.run, "--device", args.device),
        inputs=(_run_input(root, run_rel(args.from_run), records),),
        outputs=run_dir,
        gpu=args.device == "cuda",
        network=False,
    )
    jobs.executor_for("local", root=root, docker=make_docker()).run(job)
    return _finish(root, args.run, args.target)


def _finish(root: root_mod.Root, run_id: str, target_id: str) -> int:
    run_dir = run_rel(run_id)
    report = read_json(root.path / run_dir / REPORT_NAME)
    refs = read_json(root.path / run_dir / REFS_NAME)
    if not isinstance(report, dict) or not isinstance(refs, dict):
        raise JobError(f"run {run_id}: {REPORT_NAME} or {REFS_NAME} is not a JSON object")
    placed = place_references(root, run_dir, target_rel(target_id), refs)
    for line in report_lines(report):
        print(line)
    if placed:
        print(f"  references written to {target_rel(target_id)}/{REFS_DIR}")
    return 0


def place_references(
    root: root_mod.Root, run_dir: str, target_dir: str, refs: Mapping[str, object]
) -> int:
    """Copy each reference the job chose into the target folder; return how many.

    Raises:
        JobError: `refs.json` names a path outside the reference layout.
    """
    states = refs.get("states")
    count = 0
    for state, entry in (states if isinstance(states, Mapping) else {}).items():
        for ref in entry.get("refs", []) if isinstance(entry, Mapping) else []:
            rel = ref.get("audio") if isinstance(ref, Mapping) else None
            match = REF_REL.fullmatch(rel) if isinstance(rel, str) else None
            if match is None or match["state"] != state or state not in SPEAKING_STATES:
                raise JobError(f"{run_dir}/{REFS_NAME} names a reference outside refs/<state>/")
            data = (root.path / run_dir / rel).read_bytes()
            root.write_bytes(f"{target_dir}/{rel}", data)
            count += 1
    return count


def _num(value: object) -> str:
    # Numbers only: anything else prints as `?`, so no text can reach stdout through a report.
    if value is None:
        return "none"
    if isinstance(value, bool):
        return str(int(value))
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.3f}"
    return "?"


def _table(table: object) -> str:
    if not isinstance(table, Mapping):
        return "none"
    return ", ".join(f"{k} {_num(v)}" for k, v in table.items())


def _step_line(step: Mapping[str, object]) -> str:
    sim = step.get("sim") if isinstance(step.get("sim"), Mapping) else {}
    threshold = step.get("threshold") if isinstance(step.get("threshold"), Mapping) else {}
    sims = ", ".join(
        f"{name} {_num(value)}/{_num(threshold.get(name))}"  # type: ignore[union-attr]
        for name, value in sim.items()  # type: ignore[union-attr]
    )
    reasons = step.get("reasons")
    verdict = "kept" if step.get("kept") is True else "dropped"
    if verdict == "dropped" and isinstance(reasons, list):
        verdict += f" ({', '.join(str(r) for r in reasons)})"
    return (
        f"  step {step.get('step')}: {verdict}; sim/threshold {sims}; spectrum "
        f"{_num(step.get('spectrum_db'))} dB, max {_num(step.get('spectrum_max_db'))}"
    )


def report_lines(report: Mapping[str, object]) -> list[str]:
    """Return the report as printed: step decisions and counts, never text or file names."""
    lines = [f"clean run {report.get('run')} from {report.get('from')}:"]
    steps = report.get("steps")
    for step in steps if isinstance(steps, list) else []:
        if isinstance(step, Mapping):
            lines.append(_step_line(step))
    counts = report.get("segments")
    if isinstance(counts, Mapping):
        lines.append(
            f"  segments: in {_num(counts.get('in'))}, kept {_num(counts.get('kept'))}, "
            f"not her {_num(counts.get('not_her'))}, "
            f"asr disagree {_num(counts.get('asr_disagree'))}"
        )
    lines.append(f"  minutes per state: {_table(report.get('minutes_per_state'))}")
    lines.append(f"  references: {_table(report.get('references'))}")
    short = report.get("short_states")
    lines.append(
        "  short states: "
        + (", ".join(str(s) for s in short) if isinstance(short, list) and short else "none")
    )
    return lines
