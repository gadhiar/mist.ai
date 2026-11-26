"""`clean --run R --from E --target T [--device cpu|cuda]`: SIM-gated cleaning and references.

The host side of Design 5. `E` is a finished extract run and `R` a new run that holds its cleaned
copy: `segments.jsonl` with transcripts, `audio/` at 24 kHz, `gates.json`, `refs.json` and
`report.json`. The extract run is read-only to the job and stays as it was, so the clean-ab pack
can play each segment before (`E`) and after (`R`) cleaning. The chosen references, each a WAV
with its verbatim transcript beside it as `<n>.private.txt`, are written by the job into
`R/refs/<state>/` and copied here to `targets/<T>/refs/<state>/`, where `target build` reads
them. The transcript is private text: it is copied as bytes and never printed. Seed stages stay
on this host, so there is no executor option.

Raj's labels for extract run E (`seed/labels/<E>/labels.jsonl`, from `pack import`) replace
the old `seed/labels.csv` and its state column; they are read only through
`labels.read_labels`, which refuses that old file. When labels exist, the plan (schema 2)
carries each labelled clip's whole label but the note. The stage drops a clip whose label has
`removed` with that reason and a kept clip with no label as `unlabelled`, trims a `her_nearby`
clip to its keep range, and gives every labelled clip its projected state, which may be null.
The labelled clips that survive are the confirmed ones. Without labels, every kept segment is
cleaned, and only primary-cluster segments count as confirmed for the leave-one-out distribution
and the references.

The stage imports the names below, so host and job share one definition.
"""

from __future__ import annotations

import argparse
import hashlib
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from .. import gates as gates_mod
from .. import jobs
from .. import labels as labels_mod
from .. import root as root_mod
from ..errors import InputError, JobError
from ..records import JobInput, SegmentRecord, read_json, read_jsonl, sha256_file, write_json
from ..states import SPEAKING_STATES
from . import extract as extract_cmd

FAMILY = "extract"
MODULE = "scripts.voice_data.stages.clean"
DEVICES = extract_cmd.DEVICES
PLAN_SCHEMA = 2
PLAN_NAME = "clean-plan.json"
SEGMENTS_NAME = "segments.jsonl"
GATES_NAME = "gates.json"
REFS_NAME = "refs.json"
REPORT_NAME = "report.json"
AUDIO_DIR = "audio"
REFS_DIR = "refs"
PRIMARY = "primary"
NOT_HER = "not-her"
UNLABELLED = "unlabelled"
LABEL_KEYS = (
    "voices",
    "energy",
    "mood",
    "tags",
    "keep_from_s",
    "keep_to_s",
    "state",
    "removed",
    "carried_from",
)
RUN_ID = extract_cmd.RUN_ID
REF_REL = re.compile(r"refs/(?P<state>[a-z]+)/(?P<n>[1-9][0-9]*)\.wav")


def transcript_rel(state: str, number: int | str) -> str:
    """Return the path of a reference's transcript, beside `refs/<state>/<number>.wav`."""
    return f"{REFS_DIR}/{state}/{number}.private.txt"


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


def plan_label(record: labels_mod.LabelRecord) -> dict[str, object]:
    """Return a label as the plan carries it: `LABEL_KEYS`, never the note or `has_note`."""
    whole = record.to_json()
    return {key: whole[key] for key in LABEL_KEYS}


def removed_by(label: object) -> str | None:
    """Return a plan label's `removed`; a schema-1 plan's label (a state name) removes nothing."""
    if isinstance(label, Mapping):
        removed = label.get("removed")
        return removed if isinstance(removed, str) else None
    return None


def confirmed_ids(
    records: Sequence[SegmentRecord], labels: Mapping[str, object] | None
) -> list[str]:
    """Return the kept segments that count as the target's.

    With labels, the labelled ones no label removes; without, the primary-cluster ones. `labels`
    is a plan's: a schema-2 label maps to its fields, a schema-1 one to a state name.
    """
    kept = [r for r in records if r.kept]
    if labels is not None:
        return [
            r.segment_id
            for r in kept
            if r.segment_id in labels and removed_by(labels[r.segment_id]) is None
        ]
    return [r.segment_id for r in kept if r.cluster == PRIMARY]


def read_run_labels(
    root: root_mod.Root, from_run: str, records: Sequence[SegmentRecord]
) -> dict[str, dict[str, object]] | None:
    """Return extract run `from_run`'s labels as the plan carries them, or None without labels.

    Raises:
        InputError: `seed/labels.csv` exists; the labels are malformed, cover no kept segment
            of the run, or hold a carried label that no longer holds.
    """
    labels = labels_mod.read_labels(root, from_run)
    if labels is None:
        return None
    kept = {r.segment_id for r in records if r.kept}
    if not kept & {label.segment_id for label in labels}:
        raise InputError(
            f"refused: {labels_mod.labels_rel(from_run)} labels no kept segment of run {from_run}"
        )
    stale = labels_mod.carried_stale(root, labels)
    if stale:
        noun = "label no longer holds" if len(stale) == 1 else "labels no longer hold"
        raise InputError(
            f"refused: {len(stale)} carried {noun} in run {from_run}; ask them again with "
            f"pack labels --run {from_run} --replace"
        )
    return {label.segment_id: plan_label(label) for label in labels}


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
    labels = read_run_labels(root, args.from_run, records)
    if len(confirmed_ids(records, labels)) < 2:
        raise InputError(
            f"refused: run {args.from_run} has fewer than two confirmed segments to clean"
        )

    plan = {
        "schema": PLAN_SCHEMA,
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
    """Copy each reference the job chose, and its transcript, into the target folder.

    Every path is checked before anything is copied, so a bad `refs.json` leaves the target
    folder as it was. The transcript is copied as bytes and never read as text here.

    Returns:
        How many references were placed.

    Raises:
        JobError: `refs.json` names a path outside the reference layout, or a reference has no
            transcript beside it.
    """
    states = refs.get("states")
    rels: list[str] = []
    for state, entry in (states if isinstance(states, Mapping) else {}).items():
        for ref in entry.get("refs", []) if isinstance(entry, Mapping) else []:
            rel = ref.get("audio") if isinstance(ref, Mapping) else None
            match = REF_REL.fullmatch(rel) if isinstance(rel, str) else None
            if match is None or match["state"] != state or state not in SPEAKING_STATES:
                raise JobError(f"{run_dir}/{REFS_NAME} names a reference outside refs/<state>/")
            text_rel = transcript_rel(state, match["n"])
            if ref.get("transcript") != text_rel:
                raise JobError(
                    f"{run_dir}/{REFS_NAME}: reference {state}/{match['n']} has no transcript "
                    "beside it"
                )
            rels += [rel, text_rel]
    for rel in rels:
        root.write_bytes(f"{target_dir}/{rel}", (root.path / run_dir / rel).read_bytes())
    return len(rels) // 2


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
        lines.append(
            f"  clean minutes {_num(report.get('clean_minutes'))}; removed by label: "
            f"{_table(counts.get('removed_by_label'))}; trimmed {_num(counts.get('trimmed'))}"
        )
    lines.append(f"  minutes per state: {_table(report.get('minutes_per_state'))}")
    lines.append(f"  references: {_table(report.get('references'))}")
    short = report.get("short_states")
    lines.append(
        "  short states: "
        + (", ".join(str(s) for s in short) if isinstance(short, list) and short else "none")
    )
    tables = report.get("labels")
    if isinstance(tables, Mapping):
        lines += labels_mod.table_lines(tables)
    return lines
