"""`corpus build --takes R [--takes R ...] --version C [--seed N]`: the student's corpus, aligned.

The corpus is the selected takes of one or more teacher runs `teachers/omnivoice/<R>/`, read in
the order given, plus the seed's kept real segments, written to `corpus/<C>/` in the layout the
pinned Pocket TTS training code reads:
`train_aligned.jsonl` and `valid_aligned.jsonl`, one `{path, duration, transcript, words}` row
per utterance, the names its fine-tune config gives `data.train_jsonl` and `data.valid_jsonl`.
A row carries only `Entry`'s fields, because the loader builds an `Entry` from each row. `path`
is the clip's path inside a job (`/data/<root-relative path>`), where every job mounts its
inputs, so the training job must mount the clips the same way.

Runs combine only when they were made the same way, which the build checks rather than assumes:
their score plans agree on every key but `run` and `lines_file` (anchor, embedders, `cer_max`,
the floor and its target, values and centroid hashes, the centroids, profile and spread); with
more than one run, each floor comes from the seed target, since a floor from a run's own takes
records only a percentile, and two such runs could pass with different real floors; their
generation plans agree on `take_n` (`rounds_max` and `thresholds` may differ); and their lines
files are different files sharing no line id and no `text_corpus.normalise`d text. Take ids
derive from the line id, so unique line ids give unique take ids across runs. A refusal names
the runs and the key or a count, never a value or a line's text.

The seed is the clean run behind the seed target the takes were scored against (`teacher score
--target`), which the runs share: its confirmed segments whose two ASR transcripts agree. Takes
scored under the seed fallback have no seed target, so their corpus is teacher-only.

One GPU job in the `train-pocket` family (`stages/corpus_build.py`) measures every clip and runs
upstream's CTC aligner (`training/scripts/align_data.py`) over all of them once: the loader
refuses a row without `words` (`UnalignedEntry`), so a row the aligner leaves out is dropped and
counted. After alignment the job holds out `corpus.val_fraction` of the teacher lines, drawn
once over every run's lines (whole lines, so a line's takes never sit in both splits), and
`corpus.seed_val_share` of the trusted seed seconds (whole sources, so no training segment
shares a recording with a validation one), caps the training seed by `corpus.seed_share_max` of
the training split's hours, every run's teacher seconds plus the seed counted after it is
oversampled 3x, and writes the splits. The held-out seed rows come first in
`valid_aligned.jsonl`, once each, so a validation pass that drops a short tail drops teacher
rows, not her real voice.
Every such choice follows the order of the sha256 of `--seed` (default 0) and the id, so the
seed the manifest records as `random_seed` reproduces the splits byte for byte.
The aligner's batch is 8, not upstream's 32, and the job asks for 3 GB of VRAM rather than the
family's 16, which the 12 GB card cannot give.

This side prints each run's selected takes, its takes not yet scored and its aligned hours, then
gates the aligned teacher hours of all runs together on `corpus.teacher_hours_min`: below it, it
writes `report.json` and refuses (exit 3) without a `manifest.json`, so nothing downstream takes
the corpus. Above it, the schema-2 `manifest.json` records the runs (`takes_runs`), each run's
aligned hours (`teacher_hours_by_run`), the hours, the shares, the counts and each split's
sha256. `teacher_runs` reads a manifest's runs, schema 1 or 2, for the student commands. A take
of an ear-test line (`scripts/model_bench/speech/lines.json`) is refused before the job; a seed
segment speaking one is left out by the job, and the manifest's count of ear-test rows is
asserted to be 0.

Nothing printed carries a transcript, a line's text, a clip's path or a source id: only take,
line, segment and run ids, counts and hours.
The stage imports the names and helpers below, so host and job share one definition.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from .. import gates as gates_mod
from .. import jobs, text_corpus
from .. import root as root_mod
from .. import target as target_mod
from ..errors import GateError, InputError, JobError
from ..records import JobInput, TakeRecord, read_json, read_jsonl, sha256_file, write_json
from . import clean as clean_cmd
from . import extract as extract_cmd
from . import teacher as teacher_cmd

SCHEMA = 2
FAMILY = "train-pocket"
MODULE = "scripts.voice_data.stages.corpus_build"
ALIGNER_MODEL = "wav2vec2-base-960h"
DEVICE = "cuda"
BATCH_SIZE = 8
VRAM_GB = 3.0
OVERSAMPLE = 3
PLAN_NAME = "corpus-plan.json"
SUMMARY_NAME = "summary.json"
REPORT_NAME = "report.json"
MANIFEST_NAME = "manifest.json"
OUTPUTS_NAME = "outputs.json"
TRAIN_NAME = "train_aligned.jsonl"
VALID_NAME = "valid_aligned.jsonl"
ALIGN_DIR = "align"
ALIGN_IN = "input.jsonl"
ALIGN_OUT = "output.jsonl"
# `Entry`'s fields that a row of the corpus carries; the loader builds an `Entry` from each.
ROW_KEYS = ("path", "duration", "transcript", "words")
TEACHER, SEED = "teacher", "seed"
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*")


def corpus_rel(version: str) -> str:
    """Return a corpus version's folder, relative to the root."""
    return f"corpus/{version}"


def _order(key: str, seed: int) -> str:
    # A stable order no input file's order can change, the sha256 of the seed and the id, so a
    # recorded seed reproduces every split and another seed draws a different one.
    return hashlib.sha256(f"{seed}\n{key}".encode()).hexdigest()


def holdout_lines(line_ids: Iterable[str], fraction: float, *, seed: int) -> set[str]:
    """Return the lines held out for validation: `fraction` of them, in the seed's order.

    With two or more lines and a positive fraction, at least one line is held out and at least
    one stays in training, so neither split is empty.
    """
    lines = sorted(set(line_ids), key=lambda line: _order(line, seed))
    if fraction <= 0 or len(lines) < 2:
        return set()
    count = min(max(1, round(fraction * len(lines))), len(lines) - 1)
    return set(lines[:count])


NO_SEED = "no trusted seed segment"
ONE_SOURCE = "one seed source, which training keeps"
NO_SHARE = "corpus.seed_val_share is 0"
# The only reasons a report prints for holding out no seed source; none names a source.
SEED_VAL_REASONS = (NO_SEED, ONE_SOURCE, NO_SHARE)


@dataclass(frozen=True, slots=True)
class SeedHoldout:
    """The seed sources held out for validation, or the reason none is."""

    sources: frozenset[str]
    reason: str | None


def holdout_sources(sources: Mapping[str, float], share: float, *, seed: int) -> SeedHoldout:
    """Return the seed sources held out for validation: whole sources, in the seed's order.

    `sources` maps each source with trusted segments to their seconds. A whole source is held
    out so no segment in training shares a recording with one the student is validated on.
    Sources are taken in the seed's order until the held-out seconds reach `share` of the total.
    With two or more sources, at least one is held out and at least one stays in training; with
    fewer, none is held out and `reason` says why without naming a source.
    """
    if not sources:
        return SeedHoldout(frozenset(), NO_SEED)
    if len(sources) < 2:
        return SeedHoldout(frozenset(), ONE_SOURCE)
    if share <= 0:
        return SeedHoldout(frozenset(), NO_SHARE)
    ordered = sorted(sources, key=lambda source: _order(source, seed))
    goal = share * sum(sources.values())
    held: list[str] = []
    seconds = 0.0
    for source in ordered[:-1]:
        if held and seconds >= goal - 1e-9:
            break
        held.append(source)
        seconds += sources[source]
    return SeedHoldout(frozenset(held), None)


def cap_seed(
    segments: Sequence[tuple[str, float]],
    teacher_train_s: float,
    share_max: float,
    oversample: int,
    *,
    seed: int,
) -> list[str]:
    """Return the seed segments that fit the share cap, filled greedily in the seed's order.

    The share is the oversampled seed's seconds over the training split's seconds (the teacher's
    training takes plus the oversampled seed). A segment that would push it past `share_max` is
    left out and the next one tried, so a short segment can still fit after a long one did not.
    """
    ordered = sorted(segments, key=lambda s: _order(s[0], seed))
    if share_max >= 1:
        return [segment_id for segment_id, _ in ordered]
    budget = share_max / (1.0 - share_max) * teacher_train_s if share_max > 0 else 0.0
    kept: list[str] = []
    total = 0.0
    for segment_id, seconds in ordered:
        if oversample * (total + seconds) <= budget + 1e-9:
            kept.append(segment_id)
            total += seconds
    return kept


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the `corpus` command with its `build` step."""
    parser = subparsers.add_parser(
        "corpus",
        help="the student's training corpus",
        description=(
            "Build the student's training corpus from the selected takes of one or more scored "
            "teacher runs made the same way and the seed's kept real segments, word-aligned in "
            "the Pocket TTS training layout."
        ),
    )
    steps = parser.add_subparsers(dest="step", metavar="STEP", required=True)
    build = steps.add_parser("build", help="align and split the corpus (a GPU job)")
    build.add_argument(
        "--takes",
        required=True,
        action="append",
        help="a scored teacher run under teachers/; repeat it once per run, read in that order",
    )
    build.add_argument("--version", required=True, help="the corpus version, a folder in corpus/")
    build.add_argument(
        "--seed",
        type=int,
        default=0,
        help="orders every split choice; recorded in the manifest as random_seed (default 0)",
    )
    build.set_defaults(func=run_build)


def make_docker() -> jobs.DockerOps:
    """Return the docker CLI; tests replace this function."""
    return jobs.HostDocker()


def image_tag(family: str) -> str:
    """Return the local image of `family` at HEAD, as `images build` tags it."""
    return extract_cmd.image_tag(family)


def utc_now() -> datetime:
    """Return the current time in UTC; tests replace this function."""
    return datetime.now(UTC)


def _check_id(value: str, what: str) -> None:
    if not _ID.fullmatch(value) or ".." in value:
        raise InputError(f"refused: a {what} is one folder name")


def _job_id(root: root_mod.Root) -> str:
    now = utc_now()
    number = 1
    while (root.path / "runs" / root_mod.job_id(now, number)).exists():
        number += 1
    return root_mod.job_id(now, number)


def _digest(parts: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _read_texts(root: root_mod.Root, rel: str) -> dict[str, str]:
    try:
        rows = (root.path / rel).read_text(encoding="utf-8").splitlines()
        parsed = [json.loads(row) for row in rows if row.strip()]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InputError(f"refused: {rel} is missing or not JSONL") from exc
    return {
        str(row["line_id"]): str(row["text"])
        for row in parsed
        if isinstance(row, dict) and "line_id" in row and "text" in row
    }


@dataclass(frozen=True, slots=True)
class RunTakes:
    """One teacher run's part of a corpus: its score plan, its lines and its selected takes."""

    run: str
    score_plan: Mapping[str, object]
    lines_file: str
    texts: Mapping[str, str]
    selected: tuple[TakeRecord, ...]
    unscored: int


def _read_run(root: root_mod.Root, run_id: str, hold_out: text_corpus.HoldOut) -> RunTakes:
    """Return a run's selected takes, each audio file inside the run, and each line its own.

    Each selected take's line must be in the run's own lines file and must not be an ear-test
    line. `unscored` counts the run's takes `teacher score` has not measured yet.
    """
    run_dir = teacher_cmd.run_rel(run_id)
    plan_path = root.path / run_dir / teacher_cmd.SCORE_PLAN_NAME
    takes_path = root.path / run_dir / teacher_cmd.TAKES_NAME
    if not plan_path.is_file() or not takes_path.is_file():
        raise InputError(f"refused: run {run_id} is not scored; run teacher score first")
    plan = read_json(plan_path)
    if not isinstance(plan, dict) or not isinstance(plan.get("lines_file"), str):
        raise InputError(f"refused: run {run_id}'s score plan names no lines file")
    records = read_jsonl(takes_path, TakeRecord)
    selected = tuple(t for t in records if t.selected and t.kept)
    if not selected:
        raise InputError(f"refused: run {run_id} has no selected take; run teacher score first")
    for take in selected:
        if not take.audio.startswith(f"{run_dir}/") or not (root.path / take.audio).is_file():
            raise InputError(f"refused: take {take.take_id}'s audio is missing from run {run_id}")
    lines_file = str(plan["lines_file"])
    texts = _read_texts(root, lines_file)
    missing = {t.line_id for t in selected} - set(texts)
    if missing:
        raise InputError(
            f"refused: {len(missing)} selected lines of run {run_id} are not in {lines_file}"
        )
    held = sum(hold_out.holds(texts[t.line_id]) for t in selected)
    if held:
        raise InputError(f"refused: {held} selected takes of run {run_id} speak an ear-test line")
    unscored = sum(not teacher_cmd.is_scored(t) for t in records)
    return RunTakes(run_id, plan, lines_file, texts, selected, unscored)


# The score plan keys that name the run itself, so runs made the same way differ on them.
OWN_KEYS = ("run", "lines_file")
_ABSENT = object()


def _differing_keys(
    first: Mapping[str, object], other: Mapping[str, object], prefix: str = ""
) -> list[str]:
    """Return the dotted keys on which two plans differ, descending into nested mappings."""
    found: list[str] = []
    for key in sorted(set(first) | set(other)):
        if not prefix and key in OWN_KEYS:
            continue
        left, right = first.get(key, _ABSENT), other.get(key, _ABSENT)
        if left == right:
            continue
        if isinstance(left, Mapping) and isinstance(right, Mapping):
            found += _differing_keys(left, right, f"{prefix}{key}.")
        else:
            found.append(f"{prefix}{key}")
    return found


def _take_n(root: root_mod.Root, run_id: str) -> object:
    path = root.path / teacher_cmd.run_rel(run_id) / teacher_cmd.PLAN_NAME
    if not path.is_file():
        raise InputError(f"refused: run {run_id} has no generation plan")
    plan = read_json(path)
    if not isinstance(plan, dict) or "take_n" not in plan:
        raise InputError(f"refused: run {run_id}'s generation plan records no take_n")
    return plan["take_n"]


def _count(number: int, noun: str) -> str:
    return f"{number} {noun}" if number == 1 else f"{number} {noun}s"


def _check_same_way(root: root_mod.Root, runs: Sequence[RunTakes]) -> None:
    """Refuse runs not made the same way; a refusal names runs and keys or counts, no values.

    Raises:
        InputError: The runs' score plans differ on a key but `run` and `lines_file`; more than
            one run is given and a run's floor is not from the seed target; their generation
            plans differ on `take_n`; or two runs' lines files are the same file, or share a
            line id or a normalised text.
    """
    first = runs[0]
    if len(runs) > 1:
        names = ", ".join(r.run for r in runs)
        for run in runs:
            floor = run.score_plan.get("floor")
            if not isinstance(floor, Mapping) or floor.get("from") != teacher_cmd.FLOOR_SEED:
                raise InputError(
                    f"refused: runs {names} combine only with floors from the seed target, and "
                    f"run {run.run}'s is not"
                )
    for other in runs[1:]:
        keys = _differing_keys(first.score_plan, other.score_plan)
        if keys:
            raise InputError(
                f"refused: runs {first.run} and {other.run} were scored differently, on "
                f"{', '.join(keys)}"
            )
    if len(runs) > 1:
        take_n = _take_n(root, first.run)
        for other in runs[1:]:
            if _take_n(root, other.run) != take_n:
                raise InputError(
                    f"refused: runs {first.run} and {other.run}'s generation plans differ on "
                    "take_n"
                )
    for index, one in enumerate(runs):
        for two in runs[index + 1 :]:
            pair = f"runs {one.run} and {two.run}"
            if one.lines_file == two.lines_file:
                raise InputError(f"refused: {pair} read the same lines file")
            shared = len(set(one.texts) & set(two.texts))
            if shared:
                raise InputError(f"refused: {pair}'s lines files share {_count(shared, 'line id')}")
            said = {text_corpus.normalise(t) for t in one.texts.values()}
            same = len(said & {text_corpus.normalise(t) for t in two.texts.values()})
            if same:
                raise InputError(
                    f"refused: {pair}'s lines files share {_count(same, 'line')} by normalised "
                    "text"
                )


def teacher_runs(manifest: object) -> tuple[str, ...]:
    """Return a corpus manifest's teacher runs, in the order the build read them.

    A schema-2 manifest lists them in `takes_runs`; a schema-1 manifest, built from one run
    before corpora combined runs, names it in `takes_run`, so corpora built then stay readable.

    Raises:
        InputError: The manifest names no teacher run for its schema, names one twice, or
            names an id that is not one folder name.
    """
    doc = manifest if isinstance(manifest, Mapping) else {}
    schema, found = doc.get("schema"), None
    if schema == SCHEMA:
        found = doc.get("takes_runs")
    elif schema == 1 and isinstance(doc.get("takes_run"), str):
        found = [doc["takes_run"]]
    if not isinstance(found, list) or not found or not all(isinstance(r, str) for r in found):
        raise InputError("refused: the corpus manifest names no teacher run")
    for run in found:
        _check_id(run, "teacher run id")
    if len(set(found)) != len(found):
        raise InputError("refused: the corpus manifest names a teacher run twice")
    return tuple(found)


def _seed(
    root: root_mod.Root, score_plan: Mapping[str, object]
) -> tuple[str | None, list[tuple[str, str, str]]]:
    """Return the seed run behind the takes' seed target and its confirmed segments.

    Each segment is `(segment_id, source_id, audio)`: the job holds out whole sources for
    validation, so the plan carries each segment's source. Returns `(None, [])` when the takes
    were scored without a seed target.
    """
    floor = score_plan.get("floor")
    doc = floor if isinstance(floor, Mapping) else {}
    if doc.get("from") != teacher_cmd.FLOOR_SEED:
        return None, []
    target_id = doc.get("target")
    if not isinstance(target_id, str):
        raise InputError("refused: the takes' score plan names no seed target")
    record = target_mod.read_target(root, target_id)
    if record.seed_run is None:
        raise InputError(f"refused: target {target_id} has no seed run")
    seed = target_mod.read_seed(root, record.seed_run, target_id)
    return record.seed_run, [(s.segment_id, s.source_id, s.audio) for s in seed.segments]


def run_build(args: argparse.Namespace) -> int:
    """Plan the corpus, run the alignment job, gate its teacher hours and write the manifest."""
    root = root_mod.open_root(args.root)
    given: list[str] = list(args.takes)
    for run_id in given:
        _check_id(run_id, "run id")
    _check_id(args.version, "corpus version")
    twice = next((r for i, r in enumerate(given) if r in given[:i]), None)
    if twice is not None:
        raise InputError(f"refused: run {twice} is named twice")
    out_dir = corpus_rel(args.version)
    if (root.path / out_dir / MANIFEST_NAME).exists():
        raise InputError(f"refused: corpus {args.version} exists; choose a new version")
    hold_out = text_corpus.HoldOut(text_corpus.load_ear_test())
    runs = [_read_run(root, run_id, hold_out) for run_id in given]
    _check_same_way(root, runs)
    # The runs' score plans agree on the floor, so the first names the seed target they share.
    seed_run, segments = _seed(root, runs[0].score_plan)
    gates = gates_mod.load(root.path)
    families = jobs.load_families()
    aligner = families[FAMILY]["models"][ALIGNER_MODEL]  # type: ignore[index]
    plan = {
        "schema": SCHEMA,
        "version": args.version,
        "takes_runs": [{"run": r.run, "lines_file": r.lines_file} for r in runs],
        "takes": [
            {"take_id": t.take_id, "line_id": t.line_id, "audio": t.audio, "run": r.run}
            for r in runs
            for t in r.selected
        ],
        "seed_run": seed_run,
        "segments": [
            {"segment_id": segment_id, "source_id": source_id, "audio": audio}
            for segment_id, source_id, audio in segments
        ],
        "random_seed": args.seed,
        "oversample": OVERSAMPLE,
        "val_fraction": float(gates.value("corpus.val_fraction")),
        "seed_val_share": float(gates.value("corpus.seed_val_share")),
        "seed_share_max": float(gates.value("corpus.seed_share_max")),
        "teacher_hours_min": float(gates.value("corpus.teacher_hours_min")),
        "aligner": {
            "repository": aligner["repository"],
            "revision": aligner["revision"],
            "device": DEVICE,
            "batch_size": BATCH_SIZE,
        },
    }
    write_json(root, f"{out_dir}/{PLAN_NAME}", plan)
    inputs: list[JobInput] = []
    for run in runs:
        run_dir = teacher_cmd.run_rel(run.run)
        take_parts = [sha256_file(root.path / run_dir / teacher_cmd.TAKES_NAME)]
        take_parts += [sha256_file(root.path / t.audio) for t in run.selected]
        inputs += [
            JobInput(rel=run.lines_file, sha256=sha256_file(root.path / run.lines_file)),
            JobInput(rel=run_dir, sha256=_digest(take_parts)),
        ]
    if seed_run is not None:
        seed_dir = clean_cmd.run_rel(seed_run)
        seed_parts = [sha256_file(root.path / seed_dir / clean_cmd.SEGMENTS_NAME)]
        seed_parts += [sha256_file(root.path / audio) for _, _, audio in segments]
        inputs.append(JobInput(rel=seed_dir, sha256=_digest(seed_parts)))
    job = jobs.JobSpec(
        job_id=_job_id(root),
        family=FAMILY,
        image=image_tag(FAMILY),
        module=MODULE,
        args=("--version", args.version, "--device", DEVICE, "--batch-size", str(BATCH_SIZE)),
        inputs=tuple(inputs),
        outputs=out_dir,
        gpu=True,
        network=False,
        vram_gb=VRAM_GB,
    )
    jobs.executor_for("local", root=root, docker=make_docker()).run(job)
    summary = _summary(root, job)
    if summary.get("ear_test_lines") != 0:
        raise JobError(f"job {job.job_id}: the corpus holds ear-test lines")
    for line in build_lines(args.version, summary) + run_lines(runs, summary):
        print(line)
    teacher_hours = float(summary["teacher_hours"])  # type: ignore[arg-type]
    minimum = float(plan["teacher_hours_min"])  # type: ignore[arg-type]
    passed = teacher_hours >= minimum
    report = {
        "schema": SCHEMA,
        "version": args.version,
        "teacher_hours": teacher_hours,
        "teacher_hours_by_run": summary["teacher_hours_by_run"],
        "teacher_hours_min": minimum,
        "passed": passed,
    }
    write_json(root, f"{out_dir}/{REPORT_NAME}", report)
    if not passed:
        raise GateError(
            f"corpus {args.version}: {teacher_hours:.3f} h of aligned teacher takes, below "
            f"corpus.teacher_hours_min {minimum:g}"
        )
    files = [
        {"rel": rel, "sha256": sha256_file(root.path / rel)}
        for rel in (f"{out_dir}/{TRAIN_NAME}", f"{out_dir}/{VALID_NAME}")
    ]
    manifest = {
        "schema": SCHEMA,
        "version": args.version,
        "takes_runs": [r.run for r in runs],
        "seed_run": seed_run,
        **{key: summary[key] for key in MANIFEST_KEYS},
        "aligner": plan["aligner"],
        "job": job.job_id,
        "files": files,
    }
    write_json(root, f"{out_dir}/{MANIFEST_NAME}", manifest)
    print(f"  manifest {out_dir}/{MANIFEST_NAME}")
    return 0


# The stage's summary keys the manifest copies.
MANIFEST_KEYS = (
    "random_seed",
    "teacher_hours",
    "teacher_hours_by_run",
    "teacher_train_hours",
    "valid_hours",
    "seed_hours",
    "seed_share",
    "seed_share_max",
    "oversample",
    "val_fraction",
    "teacher_takes",
    "teacher_lines",
    "valid_lines",
    "seed_segments",
    "rows",
    "unaligned",
    "seed_left_out",
    "ear_test_lines",
    "seed_val_share",
    "seed_val_sources",
    "seed_val_segments",
    "seed_val_hours",
    "seed_val_reason",
)


def _summary(root: root_mod.Root, job: jobs.JobSpec) -> Mapping[str, object]:
    rel = f"{job.outputs}/{SUMMARY_NAME}"
    try:
        summary = read_json(root.path / rel)
    except InputError as exc:
        raise JobError(f"job {job.job_id}: {rel} is missing") from exc
    if not isinstance(summary, dict) or any(key not in summary for key in MANIFEST_KEYS):
        raise JobError(f"job {job.job_id}: {rel} is not the corpus summary")
    return summary


def _num(value: object) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return "?"
    if isinstance(value, int):
        return str(value)
    return f"{value:.3f}" if math.isfinite(value) else "?"


def build_lines(version: str, summary: Mapping[str, object]) -> list[str]:
    """Return what `corpus build` prints: hours, counts and shares, never text or paths."""
    unaligned = summary.get("unaligned")
    lost = unaligned if isinstance(unaligned, Mapping) else {}
    left = summary.get("seed_left_out")
    out = left if isinstance(left, Mapping) else {}
    return [
        f"corpus build {version}: teacher {_num(summary.get('teacher_hours'))} h aligned "
        f"({_num(summary.get('teacher_takes'))} takes, {_num(summary.get('valid_lines'))} of "
        f"{_num(summary.get('teacher_lines'))} lines held out)",
        f"  seed {_num(summary.get('seed_hours'))} h after {_num(summary.get('oversample'))}x "
        f"({_num(summary.get('seed_segments'))} segments), share {_num(summary.get('seed_share'))}"
        f" of at most {_num(summary.get('seed_share_max'))}",
        f"  unaligned: teacher {_num(lost.get(TEACHER))}, seed {_num(lost.get(SEED))}",
        f"  seed left out: untrusted text {_num(out.get('untrusted_text'))}, ear-test "
        f"{_num(out.get('ear_test'))}, over the share cap {_num(out.get('share_cap'))}",
        _seed_val_line(summary),
    ]


def run_lines(runs: Sequence[RunTakes], summary: Mapping[str, object]) -> list[str]:
    """Return what `corpus build` prints per run: counts and hours, never text or paths."""
    found = summary.get("teacher_hours_by_run")
    hours = found if isinstance(found, Mapping) else {}
    return [
        f"  run {run.run}: {len(run.selected)} selected takes, {run.unscored} not yet scored, "
        f"{_num(hours.get(run.run))} h aligned"
        for run in runs
    ]


def _seed_val_line(summary: Mapping[str, object]) -> str:
    # Counts and hours only: a source id names a recording, so no report carries one.
    reason = summary.get("seed_val_reason")
    if reason is not None:
        said = reason if reason in SEED_VAL_REASONS else "?"
        return f"  seed validation: none held out ({said})"
    return (
        f"  seed validation: {_num(summary.get('seed_val_sources'))} sources held out whole, "
        f"{_num(summary.get('seed_val_segments'))} segments, "
        f"{_num(summary.get('seed_val_hours'))} h (share {_num(summary.get('seed_val_share'))})"
    )
