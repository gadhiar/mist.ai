"""`calibrate run|apply|check|lower`: the six per-embedder retrieval values, set on public speakers.

All four actions run on a root of kind `rehearsal` only. The values are set by the calibration
job (`rehearsal/calibrate.py`) on public speakers that no rehearsal episode uses, so the
rehearsal (H6) tests them rather than fitting them; gate values change only through this command.

The job's threshold rule (A22): of the pairs of values, WavLM-SV's and ECAPA's, that hold the
joint false-accept rate at or below `ALPHA` in every impostor form of each dataset, each gate
takes the one that misses the fewest target trials of its `MATCHED` forms (the largest miss
first, then the mean), with ties to the lower largest FAR and then the higher WavLM-SV value.
v5's equal split of alpha between the embedders is written beside it only for comparison.

The job also measures each gate's miss out of sample, by leave one speaker out: for each
speaker it solves the rule without every trial that speaker takes part in, and scores the
speaker's own trials there. Each gate has a ceiling on that left-out miss (`CEILINGS`, Decision
17); a gate strictly over its ceiling makes the job write a calibration with `status`
`refused`, every number and no `keys`, print one `error:` line per such gate and exit 1, so
`run` exits 4 and applies nothing. Beside the values the job reports an operating curve: the
pair, its misses in sample and left out, and its largest false-accept rate at each alpha of
`CURVE_ALPHAS`, which holds `ALPHA`, so Raj can see what each budget costs on public speakers.

- `run --id C` writes `rehearsal/calibration/<C>/calibration-plan.json`, with `ALPHA`, `SEED`,
  `TRIALS`, `MATCHED`, `CEILINGS` and `CURVE_ALPHAS`, runs the job in the `score` image on the
  CPU with no network and `public/` mounted read-only, prints the job's lines and then applies
  its values. The plan lists every LibriTTS-R speaker of the build plan's
  sources that `rehearsal/build.json` does not name, with each file's sha256; every speaker the
  build names (the target, the impostor and each episode's other speakers) is listed as
  excluded. The job's own rule decides which listed speakers are eligible. RAVDESS's speech
  archive is named by its pinned sha256 and read in place by the job.
- `apply --id C` merges `calibration.json`'s six `keys` into `config/gates.json`, keeping every
  other key and creating the file when it is missing.
- `check --id C` exits 0 when `config/gates.json` holds the six values exactly, and 3 naming each
  key that differs.
- `apply`, `check` and `lower` exit 2 with "calibration C was refused" on a refused
  calibration. A `calibration.json` with no `status`, as every one before the ceilings is,
  reads as calibrated.
- `lower --score R --id C` sets each `retrieval.anchor_floor.<e>` to the lower of its current
  value and score run R's `retrieval.flag_floor` for e, and touches no other key, so a floor the
  impostor's cluster already reaches is never raised. It refuses a null flag floor, a run whose
  `retrieval.thresholds` are not C's six values, and a flag floor at or above both floors.

This module is host side: it imports the standard library and `scripts.` only, so it restates
the job's file names and gate names rather than importing the job module, which needs numpy.
Output names ids, counts, values and root-relative layout paths; never a public file name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from .. import gates as gates_mod
from .. import jobs
from .. import rehearsal as pkg
from .. import root as root_mod
from ..errors import GateError, InputError, JobError
from ..records import JobInput, read_json, write_json
from ..target import EMBEDDERS
from . import extract as extract_cmd
from . import ingest as ingest_cmd
from .rehearsal import open_rehearsal

FAMILY = "score"
MODULE = "scripts.voice_data.rehearsal.calibrate"
CALIBRATION_REL = f"{pkg.REHEARSAL_REL}/calibration"
PLAN_NAME = "calibration-plan.json"
RESULT_NAME = "calibration.json"
PLAN_KIND = "calibration-plan"
MAX_ID = 64
"""The job accepts an id of at most 64 characters (`rehearsal/calibrate.py` `_ID`)."""

ALPHA = 0.01
"""The false-accept rate each calibrated gate holds per trial (Decision 15)."""
SEED = 20261116
TRIALS: Mapping[str, object] = {
    "piece_s": [3.0, 1.5, 6.0],
    "max_pieces": 100,
    "model_pieces": 30,
    "rival_pieces": 20,
    "cluster_n": [10, 20, 30],
    "reported_n": [5],
    "draws": 10,
    "anchor_k": [1, 2, 4],
    "bootstrap": 1000,
    "anchor_sets": 10,
}
"""The Design's trial parameters, in the plan's `trials` schema; `anchor_sets` is v6's (A23)."""
MATCHED: Mapping[str, list[str]] = {
    "anchors.sim_min": ["sets"],
    "retrieval.anchor_floor": ["centroid/n10", "centroid/n20", "centroid/n30"],
    "retrieval.margin": ["rival"],
}
"""The target forms whose miss each gate's values minimise (Decision 18): the trials R2's
anchors produce, a whole anchor set, a cluster against the five-anchor centroid, and a turn
against the target's model and three rivals."""
CEILINGS: Mapping[str, float] = {
    "anchors.sim_min": 0.05,
    "retrieval.anchor_floor": 0.05,
    "retrieval.margin": 0.10,
}
"""Each gate's ceiling on its largest matched left-out miss (Decision 17): 5% of anchor sets,
5% of target clusters, 10% of target turns. Strictly over it, the calibration is refused."""
CURVE_ALPHAS: list[float] = [0.005, 0.01, 0.02, 0.05]
"""The alphas of the operating curve; `ALPHA` is one of them, so its row is the chosen pair."""
REFUSED = "refused"
"""`calibration.json`'s `status` when a gate is over its ceiling."""

LIBRITTSR = "librittsr"
RAVDESS = "ravdess"
RAVDESS_REL = "public/ravdess/Audio_Speech_Actors_01-24.zip"
RAVDESS_ARCHIVE_SHA256 = "5d208e01632cc3e5242106fa2af3273e6dc5239fb8143131979ac74c4aa40657"
"""H22's sha256 of the archive, whose md5 matched the published record."""
RAVDESS_LICENCE = "CC BY-NC-SA 4.0"
RAVDESS_SUBSET = "audio-only speech, actors 01-24"

FLOOR = "retrieval.anchor_floor"
GATES = ("anchors.sim_min", FLOOR, "retrieval.margin")
KEYS = tuple(f"{gate}.{name}" for gate in GATES for name in EMBEDDERS)
"""The six keys `calibration.json` sets and `apply` writes."""

# The reason each build role gives an excluded speaker, in the order a speaker's role is chosen.
TARGET_REASON = "build target"
IMPOSTOR_REASON = "build impostor"
OTHER_REASON = "build other speaker"


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the `calibrate` command and its four actions."""
    parser = subparsers.add_parser(
        "calibrate",
        help="set the six retrieval gate values on public speakers (rehearsal roots only)",
        description=(
            "Plan and run the gate calibration on public speakers no rehearsal episode uses, "
            "apply its values to the gate overlay, check the overlay against them, or lower the "
            "anchor floors to a scored run's flag floor. Runs on a root of kind rehearsal."
        ),
    )
    actions = parser.add_subparsers(dest="calibrate_action", metavar="ACTION", required=True)
    run = actions.add_parser("run", help="write the plan, run the job and apply its values")
    run.add_argument("--id", required=True, help="the calibration id, a folder name")
    run.set_defaults(func=run_run)
    apply = actions.add_parser("apply", help="merge the calibration's six values into the overlay")
    apply.add_argument("--id", required=True, help="the calibration id")
    apply.set_defaults(func=run_apply)
    check = actions.add_parser("check", help="exit 3 unless the overlay holds the six values")
    check.add_argument("--id", required=True, help="the calibration id")
    check.set_defaults(func=run_check)
    lower = actions.add_parser("lower", help="lower the anchor floors to a run's flag floor")
    lower.add_argument("--score", required=True, help="the scored extract run")
    lower.add_argument("--id", required=True, help="the calibration id the run used")
    lower.set_defaults(func=run_lower)


def make_docker() -> jobs.DockerOps:
    """Return the docker CLI; tests replace this function."""
    return jobs.HostDocker()


def image_tag(family: str) -> str:
    """Return the local image of `family` at HEAD, as `images build` tags it."""
    return ingest_cmd.image_tag(family)


def _check_id(value: str, what: str) -> str:
    # extract's run id rule (one plain folder name), with the job's length bound.
    if not extract_cmd.RUN_ID.fullmatch(value) or ".." in value or len(value) > MAX_ID:
        raise InputError(f"refused: {what} is one folder name of at most {MAX_ID} characters")
    return value


def folder_rel(cal_id: str) -> str:
    """Return calibration `cal_id`'s root-relative folder."""
    return f"{CALIBRATION_REL}/{cal_id}"


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------


def _revisions() -> dict[str, dict[str, str]]:
    models = jobs.load_families().get(FAMILY, {}).get("models", {})
    found = {}
    for name in EMBEDDERS:
        model = models.get(name) if isinstance(models, Mapping) else None
        revision = model.get("revision") if isinstance(model, Mapping) else None
        if not isinstance(revision, str) or not revision:
            raise InputError(f"refused: families.json pins no {name} revision in family {FAMILY}")
        found[name] = {"revision": revision}
    return found


def excluded_speakers(build: object, sexes: Mapping[str, str]) -> list[dict[str, str]]:
    """Return a row for every speaker the build record names, with its sex and role.

    `sexes` maps each LibriTTS-R speaker of the build plan's sources to its sex; the target's
    sex comes from the build record.

    Raises:
        InputError: The record lacks the target, the impostor or the episodes, or names a
            speaker whose sex neither it nor the sources give.
    """
    if not isinstance(build, Mapping):
        raise InputError(f"refused: {pkg.BUILD_REL} is not a build record")
    target, impostor = build.get("target"), build.get("impostor")
    episodes = build.get("episodes")
    if (
        not isinstance(target, Mapping)
        or not isinstance(impostor, Mapping)
        or not isinstance(episodes, list)
        or not all(isinstance(ep, Mapping) for ep in episodes)
    ):
        raise InputError(f"refused: {pkg.BUILD_REL} lacks the target, impostor or episodes")
    named = [(target.get("speaker"), TARGET_REASON, target.get("sex"))]
    named.append((impostor.get("speaker"), IMPOSTOR_REASON, None))
    for ep in episodes:
        others = ep.get("other_speakers")
        if not isinstance(others, list):
            raise InputError(f"refused: an episode of {pkg.BUILD_REL} has no other_speakers")
        named += [(speaker, OTHER_REASON, None) for speaker in others]
    rows: dict[str, dict[str, str]] = {}
    for speaker, reason, sex in named:
        speaker = str(speaker)
        if speaker in rows:
            continue
        sex = sex or sexes.get(speaker)
        if sex not in pkg.SEXES:
            raise InputError(
                f"refused: {pkg.BUILD_REL} names speaker {speaker}, whose sex the build plan's "
                "sources do not give"
            )
        rows[speaker] = {"speaker": speaker, "sex": str(sex), "reason": reason}
    return list(rows.values())


def calibration_plan(root: root_mod.Root, cal_id: str) -> dict[str, object]:
    """Return calibration `cal_id`'s plan from the build plan's sources and the build record.

    Raises:
        InputError: The build plan or record is missing or malformed, or its sources are not
            LibriTTS-R.
    """
    plan_doc = read_json(root.path / pkg.PLAN_REL)
    raw = plan_doc.get("sources") if isinstance(plan_doc, Mapping) else None
    sources = pkg.parse_sources(raw)
    others = sources.raw["others"]
    if not isinstance(others, Mapping) or others.get("dataset") != LIBRITTSR:
        raise InputError(f"refused: the build plan's other speakers are not {LIBRITTSR}")
    excluded = excluded_speakers(
        read_json(root.path / pkg.BUILD_REL), {s.speaker: s.sex for s in sources.others}
    )
    left_out = {row["speaker"] for row in excluded}
    speakers = [
        {"speaker": s.speaker, "sex": s.sex, "files": [list(f) for f in sorted(s.files)]}
        for s in sources.others
        if s.speaker not in left_out
    ]
    if not speakers:
        raise InputError("refused: the build uses every LibriTTS-R speaker of its sources")
    return {
        "schema": pkg.SCHEMA,
        "kind": PLAN_KIND,
        "id": cal_id,
        "alpha": ALPHA,
        "seed": SEED,
        "embedders": _revisions(),
        "trials": json.loads(json.dumps(dict(TRIALS))),
        "matched": json.loads(json.dumps(dict(MATCHED))),
        "ceilings": dict(CEILINGS),
        "curve_alphas": list(CURVE_ALPHAS),
        "datasets": [
            {
                "name": LIBRITTSR,
                "licence": str(others.get("licence")),
                "subset": str(others.get("subset")),
                "speakers": speakers,
                "excluded": excluded,
            },
            {
                "name": RAVDESS,
                "licence": RAVDESS_LICENCE,
                "subset": RAVDESS_SUBSET,
                "archive": [RAVDESS_REL, RAVDESS_ARCHIVE_SHA256],
            },
        ],
    }


def _plan_files(plan: Mapping[str, object]) -> dict[str, str]:
    files: dict[str, str] = {}
    for dataset in plan["datasets"]:  # type: ignore[attr-defined]
        for speaker in dataset.get("speakers", []):
            files.update((rel, sha) for rel, sha in speaker["files"])
        if "archive" in dataset:
            rel, sha = dataset["archive"]
            files[rel] = sha
    return files


def _public_input(files: Mapping[str, str]) -> JobInput:
    # The mount's sha256 is a digest of the files' paths and sha256s, as `rehearsal` digests it.
    manifest = json.dumps(sorted(files.items())).encode("utf-8")
    return JobInput(rel="public", sha256=hashlib.sha256(manifest).hexdigest())


def _job_id(root: root_mod.Root) -> str:
    now = datetime.now(UTC)
    number = 1
    while (root.path / "runs" / root_mod.job_id(now, number)).exists():
        number += 1
    return root_mod.job_id(now, number)


def _job_lines(log: Path) -> list[str]:
    # Only the job's own lines, which name ids and counts; a library's warning could name a file.
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    return [line for line in text.splitlines() if line.startswith(("calibrate: ", "error: "))]


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------


def run_run(args: argparse.Namespace) -> int:
    """Write the plan, run the calibration job, print its lines and apply its values."""
    root = open_rehearsal(args.root)
    cal_id = _check_id(args.id, "a calibration id")
    folder = folder_rel(cal_id)
    if (root.path / folder).exists():
        raise InputError(f"refused: {folder} exists; choose another --id")
    plan = calibration_plan(root, cal_id)
    if not (root.path / RAVDESS_REL).is_file():
        raise InputError("refused: the RAVDESS speech archive is missing; place H22's download")
    plan_rel = f"{folder}/{PLAN_NAME}"
    write_json(root, plan_rel, plan)
    job = jobs.JobSpec(
        job_id=_job_id(root),
        family=FAMILY,
        image=image_tag(FAMILY),
        module=MODULE,
        args=("--plan", plan_rel),
        inputs=(_public_input(_plan_files(plan)),),
        outputs=folder,
        gpu=False,
        network=False,
    )
    try:
        result = jobs.executor_for("local", root=root, docker=make_docker()).run(job)
    except JobError:
        for line in _job_lines(root.path / "runs" / job.job_id / "log.txt"):
            print(line)
        raise
    for line in _job_lines(root.path / result.log):
        print(line)
    _apply(root, cal_id)
    return 0


# ---------------------------------------------------------------------------
# apply, check and lower
# ---------------------------------------------------------------------------


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)


def read_keys(root: root_mod.Root, cal_id: str) -> dict[str, float]:
    """Return calibration `cal_id`'s six values, in `KEYS` order.

    Raises:
        InputError: The calibration is missing or was refused, or its `keys` are not the six
            numbers.
    """
    path = root.path / folder_rel(cal_id) / RESULT_NAME
    if not path.is_file():
        raise InputError(f"refused: calibration {cal_id} has no {RESULT_NAME}; run it first")
    data = read_json(path)
    if isinstance(data, Mapping) and data.get("status") == REFUSED:
        raise InputError(
            f"refused: calibration {cal_id} was refused: a gate's left-out miss is over its "
            "ceiling, so it sets no values"
        )
    keys = data.get("keys") if isinstance(data, Mapping) else None
    if (
        not isinstance(keys, Mapping)
        or set(keys) != set(KEYS)
        or not all(_is_number(v) for v in keys.values())
    ):
        raise InputError(f"refused: calibration {cal_id}'s keys are not the six gate values")
    return {key: keys[key] for key in KEYS}


def _overlay(root: root_mod.Root) -> dict[str, object]:
    path = root.path / gates_mod.OVERLAY_REL
    if not path.exists():
        return {}
    data = read_json(path)
    if not isinstance(data, dict):
        raise InputError(f"refused: {gates_mod.OVERLAY_REL} is not a JSON object")
    return data


def _write_overlay(root: root_mod.Root, values: Mapping[str, float]) -> None:
    """Set `values` in the overlay, keeping every other key, and check that `gates.load` reads it.

    Raises:
        InputError: The overlay is not an object, or the result names an unknown gate or a
            value that is not a number.
    """
    defaults = gates_mod.load_defaults()
    unknown = [key for key in values if key not in defaults]
    if unknown:
        raise InputError(f"refused: the gate defaults lack {', '.join(unknown)}")
    merged = {**_overlay(root), **values}
    write_json(root, gates_mod.OVERLAY_REL, merged)
    gates_mod.load(root.path)


def _apply(root: root_mod.Root, cal_id: str) -> None:
    keys = read_keys(root, cal_id)
    kept = len(set(_overlay(root)) - set(KEYS))
    _write_overlay(root, keys)
    for key, value in keys.items():
        print(f"calibrate apply: {key} = {value}")
    print(
        f"calibrate apply: calibration {cal_id}'s {len(keys)} values written to "
        f"{gates_mod.OVERLAY_REL}; {kept} other keys kept"
    )


def run_apply(args: argparse.Namespace) -> int:
    """Merge the calibration's six values into the gate overlay."""
    root = open_rehearsal(args.root)
    _apply(root, _check_id(args.id, "a calibration id"))
    return 0


def run_check(args: argparse.Namespace) -> int:
    """Exit 0 when the overlay holds the calibration's six values exactly.

    Raises:
        GateError: Naming each key whose overlay value differs or is unset (exit 3).
    """
    root = open_rehearsal(args.root)
    cal_id = _check_id(args.id, "a calibration id")
    keys = read_keys(root, cal_id)
    overlay = gates_mod.load(root.path).overlay
    differ = [key for key in KEYS if key not in overlay or overlay[key] != keys[key]]
    if differ:
        raise GateError(
            f"{gates_mod.OVERLAY_REL} differs from calibration {cal_id} in {len(differ)} keys: "
            + "; ".join(f"{k} is {overlay.get(k, 'unset')}, calibrated {keys[k]}" for k in differ)
        )
    print(f"calibrate check: {gates_mod.OVERLAY_REL} holds calibration {cal_id}'s six values")
    return 0


def lowered_floors(
    current: Mapping[str, float], flag_floor: Mapping[str, float]
) -> dict[str, float]:
    """Return each embedder's floor as the lower of its current value and the flag floor.

    Raises:
        InputError: Neither floor would fall, so lowering would change nothing.
    """
    lowered = {name: min(current[name], flag_floor[name]) for name in EMBEDDERS}
    if all(lowered[name] >= current[name] for name in EMBEDDERS):
        raise InputError(
            "refused: neither anchor floor would fall; the flag floor is at or above both"
        )
    return lowered


def run_lower(args: argparse.Namespace) -> int:
    """Lower each anchor floor to score run R's flag floor, and touch no other key."""
    root = open_rehearsal(args.root)
    run = _check_id(args.score, "a run id")
    cal_id = _check_id(args.id, "a calibration id")
    keys = read_keys(root, cal_id)
    path = root.path / pkg.SCORES_REL / run / pkg.SCORE_NAME
    if not path.is_file():
        raise InputError(f"refused: run {run} has no score; run rehearsal score --run {run}")
    score = read_json(path)
    retrieval = score.get("retrieval") if isinstance(score, Mapping) else None
    if not isinstance(retrieval, Mapping):
        raise InputError(f"refused: run {run}'s score has no retrieval summary")
    thresholds = retrieval.get("thresholds")
    if not isinstance(thresholds, Mapping) or any(
        thresholds.get(key) != value for key, value in keys.items()
    ):
        raise InputError(f"refused: run {run} was not made at calibration {cal_id}'s six values")
    flag = retrieval.get("flag_floor")
    if flag is None:
        raise InputError(
            f"refused: run {run}'s flag floor is null; no episode named the impostor's cluster"
        )
    if not isinstance(flag, Mapping) or not all(_is_number(flag.get(n)) for n in EMBEDDERS):
        raise InputError(f"refused: run {run}'s flag floor does not give each embedder a value")
    gates = gates_mod.load(root.path)
    current = {name: gates.value(f"{FLOOR}.{name}") for name in EMBEDDERS}
    lowered = lowered_floors(current, flag)
    _write_overlay(root, {f"{FLOOR}.{name}": value for name, value in lowered.items()})
    for name in EMBEDDERS:
        verb = "lowered" if lowered[name] < current[name] else "kept"
        print(
            f"calibrate lower: {FLOOR}.{name} {verb}: {current[name]} -> {lowered[name]} "
            f"(run {run}'s flag floor {flag[name]})"
        )
    return 0
