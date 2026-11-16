"""`rehearsal build|answer|score`: public stand-in episodes, their labels answers and their score.

All three run on a root of kind `rehearsal` only, so a private root's episodes, keys and labels
are never mixed with the stand-in's.

- `build [--episodes N] [--minutes M] [--seed S]` runs the `build` step of `rehearsal/mix.py` as
  a local job in the `score` image, with `public/` mounted read-only, and writes the episodes to
  `rehearsal/episodes/`, the ground truth to `rehearsal/truth.jsonl` and `rehearsal/build.json`,
  and the solo anchor clip to `seed/reference/rehearsal-anchor.wav`. Every package writer refuses
  `seed/raw/`, so the lead moves the episodes to `seed/raw/rehearsal/` before `ingest` (H6).
- `answer --pack P` answers a labels pack from the ground truth: each segment's true state, and
  "not her" for every segment that is not the target's. It writes
  `rehearsal/answers/<P>/ratings.csv` for `pack import`.
- `score --run R [--clean C]` scores extract run R (and the clean run C made from it, found by
  itself when there is one) against the ground truth; the effect measures run as the `effects`
  step of the same job module. It writes `rehearsal/scores/<R>/score.json`.

Output names ids, counts, scores and root-relative paths; no public or private file name.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime

from .. import gates as gates_mod
from .. import jobs, packs
from .. import rehearsal as pkg
from .. import root as root_mod
from ..errors import InputError, JobError, RootError
from ..records import (
    JobInput,
    SegmentRecord,
    SourceRecord,
    read_json,
    read_jsonl,
    sha256_file,
    write_json,
)
from ..rehearsal import score as score_mod
from . import clean as clean_cmd
from . import extract as extract_cmd
from . import ingest as ingest_cmd

FAMILY = "score"
MODULE = "scripts.voice_data.rehearsal.mix"
SOURCES_PATH = pkg.SOURCES_PATH
INVENTORY_REL = "seed/inventory.jsonl"
_PACK_ID = re.compile(r"[0-9a-f]{16}")


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the `rehearsal` command and its three actions."""
    parser = subparsers.add_parser(
        "rehearsal",
        help="public stand-in episodes with ground truth (rehearsal roots only)",
        description=(
            "Build stand-in episodes from public speech, answer a labels pack from their ground "
            "truth, or score an extraction run against it. Runs on a root of kind rehearsal."
        ),
    )
    actions = parser.add_subparsers(dest="rehearsal_action", metavar="ACTION", required=True)
    build = actions.add_parser("build", help="mix and encode the episodes and the ground truth")
    build.add_argument("--episodes", type=int, default=pkg.DEFAULT_EPISODES)
    build.add_argument("--minutes", type=float, default=pkg.DEFAULT_MINUTES)
    build.add_argument("--seed", type=int, default=pkg.DEFAULT_SEED)
    build.set_defaults(func=run_build)
    answer = actions.add_parser("answer", help="write a labels pack's ratings from the truth")
    answer.add_argument("--pack", required=True, help="the labels pack id")
    answer.set_defaults(func=run_answer)
    score = actions.add_parser("score", help="score an extraction run against the ground truth")
    score.add_argument("--run", required=True, help="the extract run under seed/runs")
    score.add_argument("--clean", help="the clean run made from it (default: found)")
    score.set_defaults(func=run_score)


def make_docker() -> jobs.DockerOps:
    """Return the docker CLI; tests replace this function."""
    return jobs.HostDocker()


def image_tag(family: str) -> str:
    """Return the local image of `family` at HEAD, as `images build` tags it."""
    return ingest_cmd.image_tag(family)


def open_rehearsal(arg: str | None) -> root_mod.Root:
    """Open the root and refuse it unless its kind is `rehearsal`.

    Raises:
        RootError: The root is unset, in a git work tree, unmarked, or of another kind.
    """
    root = root_mod.open_root(arg)
    if root.kind != pkg.KIND:
        raise RootError("refused: rehearsal commands run only on a root of kind rehearsal")
    return root


def _job_id(root: root_mod.Root) -> str:
    now = datetime.now(UTC)
    number = 1
    while (root.path / "runs" / root_mod.job_id(now, number)).exists():
        number += 1
    return root_mod.job_id(now, number)


def _public_input(files: Mapping[str, str]) -> JobInput:
    # The mount's sha256 is a digest of the files' paths and sha256s, as sources.json lists them.
    manifest = json.dumps(sorted(files.items())).encode("utf-8")
    return JobInput(rel="public", sha256=hashlib.sha256(manifest).hexdigest())


def _run_job(
    root: root_mod.Root, args: Sequence[str], inputs: Sequence[JobInput], out: str
) -> None:
    job = jobs.JobSpec(
        job_id=_job_id(root),
        family=FAMILY,
        image=image_tag(FAMILY),
        module=MODULE,
        args=tuple(args),
        inputs=tuple(inputs),
        outputs=out,
        gpu=False,
        network=False,
    )
    jobs.executor_for("local", root=root, docker=make_docker()).run(job)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def run_build(args: argparse.Namespace) -> int:
    """Plan and run the build job, place the anchor clip and print the summary."""
    root = open_rehearsal(args.root)
    if args.episodes < 1:
        raise InputError("refused: --episodes must be at least 1")
    if args.minutes < pkg.MIN_MINUTES:
        raise InputError(f"refused: --minutes must be at least {pkg.MIN_MINUTES:g}")
    sources = pkg.load_sources(SOURCES_PATH)
    if (root.path / pkg.BUILD_REL).exists():
        raise InputError(f"refused: {pkg.BUILD_REL} exists; this root already holds a build")
    if not (root.path / "public").is_dir():
        raise InputError("refused: public/ is missing; place H5's downloads first")
    plan = {
        "schema": pkg.SCHEMA,
        "episodes": args.episodes,
        "minutes": args.minutes,
        "seed": args.seed,
        "sources": sources.raw,
    }
    write_json(root, pkg.PLAN_REL, plan)
    _run_job(root, ("--step", "build"), (_public_input(sources.all_files()),), pkg.REHEARSAL_REL)
    build = read_json(root.path / pkg.BUILD_REL)
    if not isinstance(build, dict) or not isinstance(build.get("episodes"), list):
        raise JobError(f"{pkg.BUILD_REL} is not a build record")
    root.write_bytes(pkg.SEED_ANCHOR_REL, (root.path / pkg.ANCHOR_REL).read_bytes())
    for line in build_lines(build):
        print(line)
    return 0


def build_lines(build: Mapping[str, object]) -> list[str]:
    """Return the build summary as printed: ids, counts, codecs and root-relative paths."""
    episodes: list[Mapping[str, object]] = build["episodes"]  # type: ignore[assignment]
    imp: Mapping[str, object] = build["impostor"]  # type: ignore[assignment]
    anchor: Mapping[str, object] = build["anchor"]  # type: ignore[assignment]
    lines = [
        f"rehearsal build: {len(episodes)} episodes, seed {build['seed']}; "
        f"{build['files_checked']} public files checked against sources.json"
    ]
    for ep in episodes:
        lines.append(
            f"  {ep['file']}: {ep['format']} {ep['layout']} with {ep['video_codec']} video, "
            f"{float(ep['duration_s']):.1f} s, {ep['lines']} lines ({ep['target_lines']} target, "  # type: ignore[arg-type]
            f"{ep['impostor_lines']} impostor), {ep['fallback_lines']} fallback, "
            f"{ep['off_centre_lines']} off-centre, {ep['overlapping_lines']} overlapping, "
            f"sha256 {str(ep['sha256'])[:12]}"
        )
    lines.append(
        f"  impostor: speaker {imp['speaker']}, ECAPA SIM {float(imp['ecapa_sim']):.3f}, "  # type: ignore[arg-type]
        f"nearest of {len(imp['candidates'])} candidates of the target's sex"  # type: ignore[arg-type]
    )
    lines.append(
        f"  anchor: {pkg.SEED_ANCHOR_REL}, {float(anchor['duration_s']):.1f} s from "  # type: ignore[arg-type]
        f"{len(anchor['clips'])} clips no episode uses"  # type: ignore[arg-type]
    )
    lines.append(
        f"  ground truth: {pkg.TRUTH_REL} and {pkg.BUILD_REL}; move {pkg.EPISODES_REL}/ to "
        "seed/raw/rehearsal/ before ingest"
    )
    return lines


# ---------------------------------------------------------------------------
# Shared readers
# ---------------------------------------------------------------------------


def _episodes(root: root_mod.Root) -> tuple[score_mod.Episodes, dict[str, str]]:
    truth = pkg.read_truth(root.path / pkg.TRUTH_REL)
    inventory_path = root.path / INVENTORY_REL
    if not inventory_path.is_file():
        raise InputError(f"refused: {INVENTORY_REL} is missing; run ingest first")
    inventory = read_jsonl(inventory_path, SourceRecord)
    eps = score_mod.Episodes(truth, inventory)
    if not eps.lines:
        raise InputError("refused: no ingested source is a rehearsal episode (by sha256)")
    build = read_json(root.path / pkg.BUILD_REL)
    layout_by_sha = {
        str(ep.get("sha256")): str(ep.get("layout"))
        for ep in (build.get("episodes") if isinstance(build, dict) else None) or []
        if isinstance(ep, dict)
    }
    layouts = {r.source_id: layout_by_sha.get(r.sha256, "") for r in inventory}
    return eps, layouts


def _segments(root: root_mod.Root, run: str) -> list[SegmentRecord]:
    if not extract_cmd.RUN_ID.fullmatch(run) or ".." in run:
        raise InputError("refused: a run id is one folder name under seed/runs")
    path = root.path / extract_cmd.run_rel(run) / extract_cmd.SEGMENTS_NAME
    if not path.is_file():
        raise InputError(f"refused: run {run} has no {extract_cmd.SEGMENTS_NAME}")
    return read_jsonl(path, SegmentRecord)


# ---------------------------------------------------------------------------
# answer
# ---------------------------------------------------------------------------


def run_answer(args: argparse.Namespace) -> int:
    """Answer a labels pack from the ground truth and write its ratings file."""
    root = open_rehearsal(args.root)
    if not _PACK_ID.fullmatch(args.pack):
        raise InputError("refused: a pack id is 16 lower-case hex digits")
    page = root.path / packs.PACKS_DIR / args.pack / packs.PAGE_FILE
    if not page.is_file():
        raise InputError(f"refused: there is no pack {args.pack}")
    data = packs.page_data(page.read_text(encoding="utf-8"))
    key = packs.read_key(root, args.pack)
    if key["kind"] != "labels" or data.get("kind") != "labels":
        raise InputError("refused: rehearsal answer answers labels packs only")
    run = str(key.get("inputs", {}).get("run"))
    segments = {s.segment_id: s for s in _segments(root, run)}
    eps, _ = _episodes(root)
    rows = answer_rows(key["items"], segments, eps)
    columns = list(data["columns"])
    out = io.StringIO()
    writer = csv.writer(out, lineterminator="\n")
    writer.writerow(columns)
    for label in sorted(rows):
        writer.writerow([label if c == "label" else rows[label][c] for c in columns])
    rel = ratings_rel(args.pack)
    root.write_bytes(rel, out.getvalue().encode("utf-8"))
    marked = sum(r["not_her"] == "1" for r in rows.values())
    print(
        f"rehearsal answer: pack {args.pack}, {len(rows)} items, {len(rows) - marked} hers, "
        f"{marked} not her; ratings written to {rel}; import them with pack import --pack "
        f"{args.pack} --ratings <root>/{rel}"
    )
    return 0


def ratings_rel(pack: str) -> str:
    """Return the root-relative ratings file `answer` writes for a pack and tells the lead."""
    return f"{pkg.ANSWERS_REL}/{pack}/{packs.RATINGS_FILE}"


def answer_rows(
    items: Mapping[str, Mapping[str, object]],
    segments: Mapping[str, SegmentRecord],
    eps: score_mod.Episodes,
) -> dict[str, dict[str, str]]:
    """Return each item's answer: the true state of a target segment, else "not her".

    A segment is the target's when target lines cover half of it; its state is that of the line
    covering most of it. Any other segment is marked not her and keeps its proposed state, which
    `pack import` ignores for a dropped segment.

    Raises:
        InputError: The key names a segment the run does not hold, or one of a source that is no
            rehearsal episode.
    """
    rows = {}
    for label, info in items.items():
        seg = segments.get(str(info.get("segment_id")))
        if seg is None:
            raise InputError("refused: the pack key names a segment the run does not hold")
        if seg.source_id not in eps.lines:
            raise InputError(f"refused: {seg.source_id} is not a rehearsal episode (by sha256)")
        line = eps.target_line(seg) if eps.who(seg) == "target" else None
        if line is not None and line.state is not None:
            rows[label] = {"state": line.state, "not_her": "0"}
        else:
            rows[label] = {"state": seg.state_proposed, "not_her": "1"}
    return rows


# ---------------------------------------------------------------------------
# score
# ---------------------------------------------------------------------------


def find_clean_run(root: root_mod.Root, run: str) -> str | None:
    """Return the one clean run made from extract run `run`, or `None` when there is none.

    Raises:
        InputError: Several clean runs were made from it; `--clean` must choose.
    """
    found = []
    runs = root.path / "seed" / "runs"
    for folder in sorted(runs.iterdir()) if runs.is_dir() else []:
        plan = folder / clean_cmd.PLAN_NAME
        if not plan.is_file() or not (folder / clean_cmd.SEGMENTS_NAME).is_file():
            continue
        data = read_json(plan)
        if isinstance(data, dict) and data.get("from") == run:
            found.append(folder.name)
    if len(found) > 1:
        raise InputError(f"refused: {len(found)} clean runs come from run {run}; pass --clean")
    return found[0] if found else None


def _labels(root: root_mod.Root) -> dict[str, str] | None:
    return clean_cmd.read_labels(root)


def effects_plan(
    eps: score_mod.Episodes,
    clean: Sequence[SegmentRecord],
    files: Mapping[str, str],
) -> dict[str, object]:
    """Return the effects step's plan: the cleaned target segments, their clips, identity clips.

    Every clip named is a target clip the ground truth placed; the job checks each against its
    sha256 in `files` before reading it.
    """
    segments, clips = [], []
    for seg in clean:
        line = eps.target_line(seg) if seg.kept and seg.audio else None
        if line is None or eps.who(seg) != "target":
            continue
        segments.append(seg.audio)
        if line.clip not in clips:
            clips.append(line.clip)
    identity: dict[str, list[str]] = {}
    for lines in eps.lines.values():
        for line in lines:
            if line.speaker == "target" and line.state is not None:
                bucket = identity.setdefault(line.state, [])
                if line.clip not in bucket and len(bucket) < 5:
                    bucket.append(line.clip)
    named = set(clips) | {c for v in identity.values() for c in v}
    return {
        "schema": pkg.SCHEMA,
        "segments": segments,
        "clips": clips,
        "identity": identity,
        "sources": {rel: files[rel] for rel in sorted(named) if rel in files},
    }


def run_score(args: argparse.Namespace) -> int:
    """Score an extraction run, run the effects step, write and print the score."""
    root = open_rehearsal(args.root)
    segments = _segments(root, args.run)
    report = read_json(root.path / extract_cmd.run_rel(args.run) / extract_cmd.REPORT_NAME)
    if not isinstance(report, dict):
        raise InputError(f"refused: run {args.run}'s report is not a JSON object")
    clean_run = args.clean or find_clean_run(root, args.run)
    clean = _segments(root, clean_run) if clean_run else []
    eps, layouts = _episodes(root)
    labels = _labels(root)
    plan_doc = read_json(root.path / pkg.PLAN_REL)
    sources = pkg.parse_sources(plan_doc.get("sources") if isinstance(plan_doc, dict) else None)
    files = sources.all_files()
    plan = effects_plan(eps, clean, files)
    folder = f"{pkg.SCORES_REL}/{args.run}"
    write_json(root, f"{folder}/{pkg.EFFECTS_PLAN_NAME}", plan)
    inputs = [_public_input(files)]
    if clean_run:
        clean_dir = extract_cmd.run_rel(clean_run)
        parts = [sha256_file(root.path / clean_dir / clean_cmd.SEGMENTS_NAME)]
        parts += [sha256_file(root.path / rel) for rel in plan["segments"]]  # type: ignore[union-attr]
        digest = hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()
        inputs.append(JobInput(rel=clean_dir, sha256=digest))
    _run_job(root, ("--step", "effects", "--run", args.run), inputs, folder)
    effects = read_json(root.path / folder / pkg.EFFECTS_NAME)
    if not isinstance(effects, dict):
        raise JobError(f"{folder}/{pkg.EFFECTS_NAME} is not a JSON object")
    gates = gates_mod.load(root.path)
    h6 = {key: gates.value(key) for key, d in gates.defaults.items() if d.h6}
    clusters = score_mod.impostor_clusters(eps, segments, report)
    result = {
        "schema": pkg.SCHEMA,
        "run": args.run,
        "clean_run": clean_run,
        "episodes": len(eps.lines),
        "sources_not_in_truth": eps.unmatched,
        "extraction": score_mod.extraction(eps, segments, gates.value("segment.min_s")),
        "state_accuracy": score_mod.state_accuracy(eps, segments),
        "impostor": score_mod.impostor_turns(eps, segments, labels),
        "downmix": score_mod.downmix_recovery(eps, segments, layouts),
        "impostor_clusters": clusters,
        "retrieval": score_mod.retrieval_summary(clusters),
        "effects": {"comb": effects.get("comb"), "eq": effects.get("eq")},
        "identity_windows": effects.get("identity_windows"),
        "gates": score_mod.gate_values(
            eps,
            segments,
            effects,
            h6,
            score_mod.fallback_fc(read_json(root.path / pkg.BUILD_REL), eps),
        ),
    }
    write_json(root, f"{folder}/{pkg.SCORE_NAME}", result)
    for line in score_mod.score_lines(result):
        print(line)
    print(f"  written to {folder}/{pkg.SCORE_NAME}")
    return 0
