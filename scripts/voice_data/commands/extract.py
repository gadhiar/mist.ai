"""`extract --run R [--device cpu|cuda] [--vram-gb N]`: dialogue extraction and retrieval.

The host side of Design 4. It reads the inventory and the anchors, writes the run's plan, runs
one local job in the `extract` image, and prints the run's `report.json` as numbers. Seed stages
stay on this host, so there is no executor option (Design 3).

On cuda the job gets a VRAM budget, `--vram-gb` or the family's `vram_gb`: the preflight checks
free VRAM against it, and the job caps its allocator 1 GiB under it (`stages/vram.py`). A job that
outgrows the cap ends with an `out-of-memory` result naming the source, channel and step (exit 4),
and one that starts with too little free VRAM with a `vram-short` result (exit 2).

Anchors are solo clips in `seed/reference/`, rows of `seed/anchors.csv` (`source,start_s,end_s`,
the source relative to `seed/raw/`), or both. A row names a source by its file, so this command
turns each row into a source id before the plan is written: no file name reaches the job, its
argv or its output. No anchor at all exits 2. Anchors that disagree are flagged and never
refused (Raj, 16 Nov): the run goes on, and this command prints each anchor's SIMs and a flag
line before the report, then exits 0.

When `sdh scan` has run, `seed/sdh/` is mounted too and the job adds the target's SDH spans as
extra anchors, each checked against these (`stages/sdh.py`, `stages/extract.py`). They never
count toward exit 2 or the agreement flag: SDH spans alone are no anchor.

The stage imports the file names and patterns below, so host and job share one definition.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from .. import gates as gates_mod
from .. import jobs
from .. import root as root_mod
from ..errors import InputError, JobError
from ..records import (
    JobInput,
    SourceRecord,
    _Record,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)
from . import ingest as ingest_cmd
from . import sdh as sdh_cmd

FAMILY = "extract"
MODULE = "scripts.voice_data.stages.extract"
DEVICES = ("cpu", "cuda")
INVENTORY_REL = "seed/inventory.jsonl"
REFERENCE_REL = "seed/reference"
ANCHORS_REL = "seed/anchors.csv"
ANCHORS_HEADER = ["source", "start_s", "end_s"]
CLIP_SUFFIXES = frozenset({".wav", ".flac"})
PLAN_NAME = "extract-plan.json"
RESULT_NAME = "extract-result.json"
REPORT_NAME = "report.json"
SEGMENTS_NAME = "segments.jsonl"
TURNS_NAME = "turns.jsonl"
PROPOSALS_NAME = "proposals.jsonl"
AUDIO_DIR = "audio"
# Result statuses of a run stopped by the card's memory: an overrun of the allocator's cap, and a
# start with less free VRAM than the budget needs.
OUT_OF_MEMORY = "out-of-memory"
VRAM_SHORT = "vram-short"
# A SIM equal to a gate value up to rounding meets it, as in `stages/retrieve.py`.
EPS = 1e-9
# A run id names a folder under `seed/runs/`, so it is one plain path segment.
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")
# Source facts the report copies from the inventory: codec facts, never the path.
SOURCE_FACTS = (
    "codec",
    "profile",
    "bitrate_bps",
    "sample_rate",
    "channels",
    "layout",
    "effective_bandwidth_hz",
    "lossless",
    "primary_channel",
)


def run_rel(run_id: str) -> str:
    """Return a run's folder, relative to the root."""
    return f"seed/runs/{run_id}"


@dataclass(frozen=True, slots=True)
class TurnRecord(_Record):
    """One line of a run's `turns.jsonl`: a diarized turn, its role and what retrieval measured.

    Segment records cover admitted clusters only, so without this record a turn of a cluster
    that is not admitted leaves no trace, and a scorer cannot tell which cluster held a given
    speaker. Every turn of every source has one, whatever its role. It holds times and scores
    only, like a segment record: no file name, text or audio.

    `turn` is the turn's index in its source's turns, sorted by start. `sim_anchor` is its SIM
    to the anchor centroid retrieval used and `sim_target` its SIM to the target centroid;
    `gaps` is `retrieve.Verdict.gaps`, and `passed` and `reasons` are `retrieve.judge` on the
    turn's own embedding. All are `None` (and `reasons` empty) where they were not measured: a
    turn too short to embed, a run with no target model, or (for `gaps`) a source with no rival.
    """

    source_id: str
    turn: int
    start_s: float
    end_s: float
    diar_cluster: str
    role: str
    embedded: bool
    touches_overlap: bool
    sim_anchor: dict[str, float] | None
    sim_target: dict[str, float] | None
    gaps: dict[str, float] | None
    passed: bool | None
    reasons: tuple[str, ...]


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the `extract` command."""
    parser = subparsers.add_parser(
        "extract",
        help="dialogue extraction and target-speaker retrieval (local jobs only)",
        description=(
            "Extract the target's dialogue from every ingested source into seed/runs/<run>: "
            "segments.jsonl, turns.jsonl, audio/, proposals.jsonl and report.json. Runs on this "
            "host only."
        ),
    )
    parser.add_argument("--run", required=True, help="the run id, a new folder under seed/runs")
    parser.add_argument("--device", choices=DEVICES, default="cpu", help="where models run")
    parser.add_argument(
        "--vram-gb",
        type=float,
        default=None,
        help=(
            "the job's VRAM budget in GiB on cuda; the allocator is capped 1 GiB under it "
            "(default: the extract family's vram_gb)"
        ),
    )
    parser.set_defaults(func=run)


def make_docker() -> jobs.DockerOps:
    """Return the docker CLI; tests replace this function."""
    return jobs.HostDocker()


def image_tag(family: str) -> str:
    """Return the local image of `family` at HEAD, as `images build` tags it."""
    return ingest_cmd.image_tag(family)


def list_clips(root: root_mod.Root) -> list[Path]:
    """Return the solo clips in `seed/reference/`, sorted; hidden files are skipped."""
    folder = root.path / REFERENCE_REL
    if not folder.is_dir():
        return []
    return sorted(
        p
        for p in folder.iterdir()
        if p.is_file() and not p.name.startswith(".") and p.suffix.lower() in CLIP_SUFFIXES
    )


def read_anchors(
    root: root_mod.Root, records: Sequence[SourceRecord], min_s: float, max_s: float
) -> list[dict[str, object]]:
    """Return `seed/anchors.csv` as plan rows of source id, start and end.

    Errors name the row number, never the source file.

    Raises:
        InputError: The header is not `source,start_s,end_s`, a row names a source missing from
            the inventory, its times are not numbers, or the anchor is not `min_s`-`max_s` long.
    """
    path = root.path / ANCHORS_REL
    if not path.is_file():
        return []
    try:
        text = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise InputError(f"refused: {ANCHORS_REL} is unreadable") from exc
    rows = [(n, row) for n, row in enumerate(csv.reader(io.StringIO(text)), 1) if any(row)]
    if not rows:
        return []
    if [cell.strip() for cell in rows[0][1]] != ANCHORS_HEADER:
        raise InputError(f"refused: {ANCHORS_REL} must start with the header source,start_s,end_s")
    by_path = {r.path: r.source_id for r in records}
    anchors: list[dict[str, object]] = []
    for number, row in rows[1:]:
        where = f"{ANCHORS_REL} row {number}"
        if len(row) != 3:
            raise InputError(f"refused: {where} does not have three fields")
        source = PurePosixPath("seed/raw", row[0].strip().replace("\\", "/")).as_posix()
        sid = by_path.get(source)
        if sid is None:
            raise InputError(f"refused: {where} names a source that is not in the inventory")
        try:
            start, end = float(row[1]), float(row[2])
        except ValueError as exc:
            raise InputError(f"refused: {where} has times that are not numbers") from exc
        if not min_s <= end - start <= max_s:
            raise InputError(
                f"refused: {where} is {end - start:.2f} s; anchors are {min_s:g}-{max_s:g} s"
            )
        anchors.append({"source_id": sid, "start_s": start, "end_s": end})
    return anchors


def _reference_input(clips: Sequence[Path]) -> JobInput:
    # The mount's sha256 is a digest of name digests and file sha256s, so job.json pins the
    # clips without naming them.
    pairs = sorted(
        [hashlib.sha256(p.name.encode("utf-8")).hexdigest(), sha256_file(p)] for p in clips
    )
    digest = hashlib.sha256(repr(pairs).encode("utf-8")).hexdigest()
    return JobInput(rel=REFERENCE_REL, sha256=digest)


def _job_id(root: root_mod.Root) -> str:
    now = datetime.now(UTC)
    number = 1
    while (root.path / "runs" / root_mod.job_id(now, number)).exists():
        number += 1
    return root_mod.job_id(now, number)


def _plan_source(record: SourceRecord) -> dict[str, object]:
    facts = record.to_json()
    return {
        "source_id": record.source_id,
        "track": record.track,
        "codec": record.codec,
        "primary_channel": record.primary_channel,
        "sample_rate": record.sample_rate,
        "facts": {key: facts[key] for key in SOURCE_FACTS},
    }


def vram_budget(value: float | None) -> float:
    """Return the job's VRAM budget in GiB: `value`, or the extract family's `vram_gb`.

    Raises:
        InputError: The budget is not a positive number.
    """
    if value is None:
        value = float(jobs.load_families()[FAMILY]["vram_gb"])  # type: ignore[arg-type]
    budget = value
    if not budget > 0:
        raise InputError("refused: --vram-gb must be a positive number of GiB")
    return budget


def run(args: argparse.Namespace) -> int:
    """Plan the run, run the extract job, and print its report."""
    root = root_mod.open_root(args.root)
    if not RUN_ID.fullmatch(args.run) or ".." in args.run:
        raise InputError("refused: a run id is one folder name under seed/runs")
    budget = vram_budget(args.vram_gb)
    thresholds = gates_mod.load(root.path).effective()
    inventory = root.path / INVENTORY_REL
    if not inventory.is_file():
        raise InputError(f"refused: {INVENTORY_REL} is missing; run ingest first")
    records = sorted(read_jsonl(inventory, SourceRecord), key=lambda r: r.source_id)
    if not records:
        raise InputError(f"refused: {INVENTORY_REL} lists no source")
    clips = list_clips(root)
    anchors = read_anchors(root, records, thresholds["anchors.min_s"], thresholds["anchors.max_s"])
    if not clips and not anchors:
        raise InputError(
            f"refused: no anchor; place a solo clip in {REFERENCE_REL}/ or rows in {ANCHORS_REL}"
        )
    run_dir = run_rel(args.run)
    finished = (SEGMENTS_NAME, TURNS_NAME, REPORT_NAME)
    if any((root.path / run_dir / name).exists() for name in finished):
        raise InputError(f"refused: {run_dir} already holds a run; choose a new run id")
    for record in records:
        if not (root.path / record.track).is_file():
            raise InputError(f"refused: {record.source_id}'s track {record.track} is missing")

    plan = {
        "schema": 1,
        "run": args.run,
        "thresholds": thresholds,
        "sources": [_plan_source(r) for r in records],
        "anchors": anchors,
    }
    write_json(root, f"{run_dir}/{PLAN_NAME}", plan)
    inputs = [JobInput(rel=r.track, sha256=sha256_file(root.path / r.track)) for r in records]
    if clips:
        inputs.append(_reference_input(clips))
    sdh = sdh_cmd.sdh_input(root)
    if sdh is not None:
        inputs.append(sdh)
    cuda = args.device == "cuda"
    job_args = ("--run", args.run, "--device", args.device)
    if cuda:
        job_args += ("--vram-gb", f"{budget:g}")
    job = jobs.JobSpec(
        job_id=_job_id(root),
        family=FAMILY,
        image=image_tag(FAMILY),
        module=MODULE,
        args=job_args,
        inputs=tuple(inputs),
        outputs=run_dir,
        gpu=cuda,
        network=False,
        vram_gb=budget if cuda else None,
    )
    jobs.executor_for("local", root=root, docker=make_docker()).run(job)
    return _finish(root, args.run, thresholds)


def _finish(root: root_mod.Root, run_id: str, thresholds: Mapping[str, float]) -> int:
    result = read_json(root.path / run_rel(run_id) / RESULT_NAME)
    status = result.get("status") if isinstance(result, dict) else None
    if status == "no-anchor":
        raise InputError("refused: the job found no anchor")
    if status == "anchor-length":
        bad = ", ".join(str(a) for a in result.get("anchors", []))  # type: ignore[union-attr]
        raise InputError(f"refused: anchors must be 3-30 s; out of range: {bad}")
    if status == OUT_OF_MEMORY:
        raise JobError(oom_message(run_id, result))  # type: ignore[arg-type]
    if status == VRAM_SHORT:
        raise jobs.PreflightError(short_message(run_id, result))  # type: ignore[arg-type]
    if status != "ok":
        raise JobError(f"run {run_id}: the job wrote no result")
    report = read_json(root.path / run_rel(run_id) / REPORT_NAME)
    if not isinstance(report, dict):
        raise JobError(f"run {run_id}: {REPORT_NAME} is not a JSON object")
    if result.get("anchors_agree") is False:  # type: ignore[union-attr]
        for line in anchor_lines(result.get("anchors", []), thresholds):  # type: ignore[union-attr]
            print(line)
        print(
            f"anchors disagree in run {run_id}: flagged, not refused; each SIM above is to the "
            "others"
        )
    for line in report_lines(report, result.get("vram")):  # type: ignore[union-attr]
        print(line)
    return 0


def _gib(value: object) -> str:
    return (
        f"{value:g}" if isinstance(value, int | float) and not isinstance(value, bool) else "none"
    )


def _name(value: object) -> str:
    return str(value) if value is not None else "none"


def oom_message(run_id: str, result: Mapping[str, object]) -> str:
    """The line an `out-of-memory` result prints: where it happened, the budget and the peak."""
    return (
        f"run {run_id}: out of GPU memory at {_name(result.get('source'))} "
        f"{_name(result.get('channel'))} {_name(result.get('step'))}: budget "
        f"{_gib(result.get('budget_gib'))} GiB, allocator cap {_gib(result.get('cap_gib'))} GiB, "
        f"peak reserved {_name(result.get('peak_reserved_mib'))} MiB"
    )


def short_message(run_id: str, result: Mapping[str, object]) -> str:
    """The line a `vram-short` result prints."""
    return (
        f"run {run_id}: {_gib(result.get('free_gib'))} GiB VRAM free after the job's CUDA "
        f"context, {_gib(result.get('needed_gib'))} GiB needed for budget "
        f"{_gib(result.get('budget_gib'))} GiB"
    )


def vram_line(report: Mapping[str, object], card: object) -> str:
    """The report's VRAM line: the budget and cap, and the largest per-source reserved peak."""
    if not isinstance(card, Mapping):
        return "  vram: none"
    line = (
        f"  vram: budget {_gib(card.get('budget_gib'))} GiB, allocator cap "
        f"{_gib(card.get('cap_gib'))} GiB"
    )
    sources = report.get("sources")
    peaks = [
        (source["vram_peak_mib"]["reserved"], str(source.get("source_id")))
        for source in (sources if isinstance(sources, list) else [])
        if isinstance(source, Mapping)
        and isinstance(source.get("vram_peak_mib"), Mapping)
        and isinstance(source["vram_peak_mib"].get("reserved"), int)
    ]
    if peaks:
        reserved, source_id = max(peaks, key=lambda pair: pair[0])
        line += f", largest peak reserved {reserved} MiB at {source_id}"
    return line


def _num(value: object) -> str:
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


def anchor_lines(rows: object, thresholds: Mapping[str, float]) -> list[str]:
    """Return one line per anchor: its opaque name and its SIM to the others per embedder.

    Each embedder has its own Anchors value (`anchors.sim_min.<embedder>`), so a SIM under its
    embedder's value is marked `below`: one number cannot be read against another embedder's
    scale, and the mark names which value an anchor failed.
    """
    if not isinstance(rows, list):
        return []
    lines = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        sims = row.get("sim")
        parts = []
        for name, value in sims.items() if isinstance(sims, Mapping) else ():
            sim_min = thresholds.get(f"anchors.sim_min.{name}")
            below = isinstance(value, int | float) and sim_min is not None and value < sim_min - EPS
            parts.append(f"{name} {_num(value)}{' below' if below else ''}")
        lines.append(f"  anchor {row.get('anchor')}: {', '.join(parts) or 'none'}")
    return lines


def report_lines(report: Mapping[str, object], card: object = None) -> list[str]:
    """Return the report as printed: ids, names of states, channels and gates, and numbers.

    `card` is the `vram` object of the run's result, `None` on the CPU.
    """
    lines = [f"extract run {report.get('run')}:"]
    lines.append(f"  clean minutes {_num(report.get('clean_minutes'))}")
    lines.append(f"  minutes per state: {_table(report.get('minutes_per_state'))}")
    lines.append(f"  minutes per channel: {_table(report.get('minutes_per_channel'))}")
    lines.append(f"  minutes per set: {_table(report.get('minutes_per_set'))}")
    flagged = report.get("flagged")
    if isinstance(flagged, Mapping):
        lines.append(
            f"  flagged for the labels pack: {_num(flagged.get('segments'))} segments in "
            f"{_num(flagged.get('groups'))} groups, {_num(flagged.get('minutes'))} min"
        )
    lines.append(f"  kept from the downmix: {_table(report.get('channel_reasons'))}")
    lines.append(f"  one speaker: {_table(report.get('one_speaker'))}")
    lines.append(f"  purity p10 margin {_num(report.get('purity_p10'))}")
    lines.append(f"  sdh spans: {_table(report.get('sdh'))}")
    for key in ("distribution", "windows"):
        dist = report.get(key)
        if isinstance(dist, Mapping):
            for name, row in dist.items():
                lines.append(f"  {key} {name}: {_table(row)}")
        else:
            lines.append(f"  {key}: none")
    lines.append(f"  drops: {_table(report.get('drops'))}")
    lines.append(vram_line(report, card))
    sources = report.get("sources")
    for source in sources if isinstance(sources, list) else []:
        if not isinstance(source, Mapping):
            continue
        lines.append(
            f"  {source.get('source_id')}: separator {source.get('separator')} "
            f"({_table(source.get('separator_scores'))}); primary {_num(source.get('primary_s'))} "
            f"s; kept {_num(source.get('kept'))}, flagged {_num(source.get('flagged'))}, "
            f"dropped {_num(source.get('dropped'))}"
        )
        # A report written before the turn-based choice has no basis; its scores are the windows'.
        basis = source.get("separator_basis")
        if isinstance(basis, str):
            again = "diarized again" if source.get("separator_rediarized") else "diarized once"
            lines.append(
                f"    separator basis {basis}: provisional {source.get('separator_provisional')}, "
                f"chosen {source.get('separator')}, {_num(source.get('separator_turns'))} turns "
                f"scored, {again}"
            )
            lines.append(f"    window scores: {_table(source.get('separator_window_scores'))}")
        clusters = source.get("clusters")
        for cluster in clusters if isinstance(clusters, list) else []:
            lines.append(
                f"    {cluster.get('diar_cluster')} {cluster.get('role')}: "
                f"{_num(cluster.get('turns'))} turns, {_num(cluster.get('seconds'))} s, "
                f"median {_table(cluster.get('median_sim'))}"
            )
    thresholds = report.get("thresholds")
    if isinstance(thresholds, Mapping):
        lines.append(f"  thresholds: {_table(thresholds)}")
    return lines
