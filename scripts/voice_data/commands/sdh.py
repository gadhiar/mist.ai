"""`sdh scan` and `sdh score --run R`: the target's SDH subtitle cues, and a recall check on them.

SDH subtitle tracks label a speaker's lines with a bracketed name. Raj writes the target's
labels in `config/sdh.json` (`{"labels": [...]}`); that file is never in a repository and
never printed. `sdh scan` runs one local job per source that finds the source's SDH stream,
converts it to SRT in memory, and writes only the times and two flags of her cues to
`seed/sdh/<source_id>.jsonl`. `extract` turns those cues into extra anchors, holding every fifth
span out, and `sdh score` measures the share of the held-out spans' seconds that a run's kept
segments cover.

Both commands build local jobs in the base image (family `score`), without the network or a GPU,
and print ids, counts, seconds and shares. This module is host-side: it never imports
`stages/sdh.py`; the stage imports the names below, so host and job share one definition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

from .. import gates as gates_mod
from .. import jobs
from .. import root as root_mod
from ..errors import InputError, JobError
from ..records import JobInput, SourceRecord, read_json, read_jsonl, sha256_file
from . import extract as extract_cmd
from . import ingest as ingest_cmd

FAMILY = "score"
MODULE = "scripts.voice_data.stages.sdh"
CONFIG_REL = "config/sdh.json"
SDH_REL = "seed/sdh"
RAW_REL = "seed/raw"
INVENTORY_REL = "seed/inventory.jsonl"
SCORE_NAME = "sdh-score.json"
STEPS = ("scan", "score")


def cues_name(source_id: str) -> str:
    """Return the name of a source's cue file in `seed/sdh/`."""
    return f"{source_id}.jsonl"


def scan_name(source_id: str) -> str:
    """Return the name of a source's scan counts in `seed/sdh/`."""
    return f"{source_id}.scan.json"


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the `sdh` command and its `scan` and `score` steps."""
    parser = subparsers.add_parser(
        "sdh",
        help="SDH subtitle cues of the target, and a recall check on them (local jobs only)",
        description=(
            "scan: find each source's SDH subtitle stream and write the times of the target's "
            "labelled cues into seed/sdh. score: the share of the held-out SDH spans' seconds "
            "that a run's kept segments cover. Runs on this host only."
        ),
    )
    steps = parser.add_subparsers(dest="sdh_step", metavar="STEP", required=True)
    scan = steps.add_parser("scan", help="write the target's SDH cue times per source")
    scan.set_defaults(func=run_scan)
    score = steps.add_parser("score", help="recall of a run's kept segments on held-out spans")
    score.add_argument("--run", required=True, help="an extract run under seed/runs")
    score.set_defaults(func=run_score)


def make_docker() -> jobs.DockerOps:
    """Return the docker CLI; tests replace this function."""
    return jobs.HostDocker()


def image_tag(family: str) -> str:
    """Return the local image of `family` at HEAD, as `images build` tags it."""
    return ingest_cmd.image_tag(family)


def check_config(root: root_mod.Root) -> Path:
    """Return `config/sdh.json` once it holds at least one label.

    Messages name the file and never its content, since a label names a character.

    Raises:
        InputError: The file is missing, not JSON, or not `{"labels": [<label>, ...]}`.
    """
    path = root.path / CONFIG_REL
    if not path.is_file():
        raise InputError(f"refused: {CONFIG_REL} is missing; write the target's SDH labels there")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        data = None
    labels = data.get("labels") if isinstance(data, dict) else None
    if (
        not isinstance(labels, list)
        or not labels
        or not all(isinstance(x, str) and x.strip() for x in labels)
    ):
        raise InputError(f'refused: {CONFIG_REL} must be {{"labels": [<label>, ...]}}')
    return path


def sdh_input(root: root_mod.Root) -> JobInput | None:
    """Return `seed/sdh/` as a job input, or `None` when no scan has run.

    The mount's sha256 is a digest of the cue files' names (source ids) and sha256s.
    """
    folder = root.path / SDH_REL
    if not folder.is_dir():
        return None
    pairs = sorted([p.name, sha256_file(p)] for p in folder.glob("src-*.jsonl") if p.is_file())
    digest = hashlib.sha256(json.dumps(pairs).encode("utf-8")).hexdigest()
    return JobInput(rel=SDH_REL, sha256=digest)


def _job_id(root: root_mod.Root) -> str:
    now = datetime.now(UTC)
    number = 1
    while (root.path / "runs" / root_mod.job_id(now, number)).exists():
        number += 1
    return root_mod.job_id(now, number)


def _job(
    root: root_mod.Root, args: tuple[str, ...], inputs: Sequence[JobInput], outputs: str
) -> jobs.JobSpec:
    return jobs.JobSpec(
        job_id=_job_id(root),
        family=FAMILY,
        image=image_tag(FAMILY),
        module=MODULE,
        args=args,
        inputs=tuple(inputs),
        outputs=outputs,
        gpu=False,
        network=False,
    )


def _records(root: root_mod.Root) -> list[SourceRecord]:
    inventory = root.path / INVENTORY_REL
    if not inventory.is_file():
        raise InputError(f"refused: {INVENTORY_REL} is missing; run ingest first")
    records = sorted(read_jsonl(inventory, SourceRecord), key=lambda r: r.source_id)
    if not records:
        raise InputError(f"refused: {INVENTORY_REL} lists no source")
    return records


def _raw_input(records: Sequence[SourceRecord]) -> JobInput:
    # The read-only `seed/raw` mount is pinned by the path digests and sha256s ingest recorded,
    # so job.json names no source file.
    pairs = sorted([hashlib.sha256(r.path.encode("utf-8")).hexdigest(), r.sha256] for r in records)
    digest = hashlib.sha256(json.dumps(pairs).encode("utf-8")).hexdigest()
    return JobInput(rel=RAW_REL, sha256=digest)


def _count(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise JobError(f"{SDH_REL}: a scan result lacks {key}")
    return value


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def run_scan(args: argparse.Namespace) -> int:
    """Scan every ingested source for its SDH stream and write her cues' times."""
    root = root_mod.open_root(args.root)
    config = check_config(root)
    records = _records(root)
    inputs = (
        _raw_input(records),
        JobInput(rel=INVENTORY_REL, sha256=sha256_file(root.path / INVENTORY_REL)),
        JobInput(rel=CONFIG_REL, sha256=sha256_file(config)),
    )
    executor = jobs.executor_for("local", root=root, docker=make_docker())
    with_stream = image_skipped = hers = 0
    for record in records:
        sid = record.source_id
        job = _job(root, ("--step", "scan", "--source", sid), inputs, SDH_REL)
        executor.run(job)
        row = read_json(root.path / SDH_REL / scan_name(sid))
        if not isinstance(row, dict) or row.get("source_id") != sid:
            raise JobError(f"{sid}: the scan wrote no result")
        skipped = _count(row, "image_skipped")
        image_skipped += skipped
        stream = row.get("stream")
        if stream is None:
            extra = f", {_plural(skipped, 'image stream')} skipped" if skipped else ""
            print(f"{sid}: no SDH stream{extra}")
            continue
        with_stream += 1
        hers += _count(row, "hers")
        print(
            f"{sid}: SDH stream {stream}, {_plural(skipped, 'image stream')} skipped, "
            f"{_count(row, 'cues')} cues, {_count(row, 'hers')} hers "
            f"(multi {_count(row, 'multi')}, offscreen {_count(row, 'offscreen')})"
        )
    print(
        f"sdh scan: {len(records)} sources, {with_stream} with an SDH stream, "
        f"{len(records) - with_stream} without, {_plural(image_skipped, 'image stream')} "
        f"skipped, {hers} her cues in {SDH_REL}"
    )
    return 0


def _share(value: object) -> str:
    if isinstance(value, float | int) and not isinstance(value, bool):
        return f"{value:.3f}"
    return "none"


def score_line(name: str, row: Mapping[str, object]) -> str:
    """Return one printed line of `sdh-score.json`: spans, seconds and the two shares."""
    seconds = row.get("held_out_s")
    seconds_text = f"{seconds:.1f}" if isinstance(seconds, float | int) else "?"
    return (
        f"{name}: {row.get('held_out_spans')} held-out spans, {seconds_text} s; "
        f"primary {_share(row.get('primary_share'))}, "
        f"flagged {_share(row.get('flagged_share'))}"
    )


def run_score(args: argparse.Namespace) -> int:
    """Measure a run's recall on the held-out SDH spans."""
    root = root_mod.open_root(args.root)
    if not extract_cmd.RUN_ID.fullmatch(args.run) or ".." in args.run:
        raise InputError("refused: a run id is one folder name under seed/runs")
    sdh = sdh_input(root)
    if sdh is None or not any((root.path / SDH_REL).glob("src-*.jsonl")):
        raise InputError(f"refused: {SDH_REL} holds no cues; run sdh scan first")
    run_dir = extract_cmd.run_rel(args.run)
    segments_rel = f"{run_dir}/{extract_cmd.SEGMENTS_NAME}"
    if not (root.path / segments_rel).is_file():
        raise InputError(f"refused: {segments_rel} is missing; run extract first")
    gates = gates_mod.load(root.path)
    min_s, max_s = gates.value("anchors.min_s"), gates.value("anchors.max_s")
    segments = JobInput(rel=segments_rel, sha256=sha256_file(root.path / segments_rel))
    job = _job(
        root,
        ("--step", "score", "--run", args.run, "--min-s", f"{min_s:g}", "--max-s", f"{max_s:g}"),
        (sdh, segments),
        run_dir,
    )
    jobs.executor_for("local", root=root, docker=make_docker()).run(job)
    score = read_json(root.path / run_dir / SCORE_NAME)
    overall = score.get("overall") if isinstance(score, dict) else None
    if not isinstance(overall, dict):
        raise JobError(f"run {args.run}: {SCORE_NAME} is not a score")
    print(score_line(f"sdh score {args.run}", overall))
    sources = score.get("sources")  # type: ignore[union-attr]
    for row in sources if isinstance(sources, list) else []:
        if isinstance(row, dict):
            print(score_line(f"  {row.get('source_id')}", row))
    return 0
