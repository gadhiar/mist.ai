"""`target build --target T [--seed-run R] [--identity VOICE_ID]`: the target voice (Design 6).

Reads clean run `R` (its confirmed segments and the references `clean --target T` placed under
`targets/<T>/refs/`) and the identity `voices/<VOICE_ID>/` (checked by v6's rules), runs the
`score` family's job for the numbers, then places the references the mode calls for and writes
`targets/<T>/target.json`. The mode is decided by SIM, not assumed, and printed with each
identity clip's two SIM values against its state's floor, for Raj to confirm.

Seed stages stay on this host, so there is no executor option. Output names ids, modes and
numbers only; a reference's transcript is copied and hashed as bytes and never printed.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import UTC, datetime

from .. import gates as gates_mod
from .. import identity as identity_mod
from .. import jobs
from .. import root as root_mod
from .. import target as target_mod
from ..errors import InputError
from ..records import JobInput, sha256_file, write_json
from . import clean as clean_cmd
from . import extract as extract_cmd

DEVICES = extract_cmd.DEVICES


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the `target` command and its `build` action."""
    parser = subparsers.add_parser(
        "target",
        help="the target voice: centroids, distributions, per-state floors, references",
        description="Build targets/<T>/target.json from a clean run, an identity, or both.",
    )
    actions = parser.add_subparsers(dest="target_action", metavar="ACTION", required=True)
    build = actions.add_parser(
        "build",
        help="build target.json; the mode is decided by SIM (local jobs only)",
        description=(
            "Embed the clean run's confirmed segments and the identity's clips, decide the mode "
            "(seed, identity or seed+identity) by each clip's SIM against its state's floor, and "
            "write targets/<T>/target.json with its references. Runs on this host only."
        ),
    )
    build.add_argument("--target", required=True, help="the target id, a folder under targets")
    build.add_argument("--seed-run", help="the clean run under seed/runs the seed comes from")
    build.add_argument("--identity", metavar="VOICE_ID", help="an identity under voices/")
    build.add_argument("--device", choices=DEVICES, default="cpu", help="where models run")
    build.set_defaults(func=run_build)


def make_docker() -> jobs.DockerOps:
    """Return the docker CLI; tests replace this function."""
    return jobs.HostDocker()


def image_tag(family: str) -> str:
    """Return the local image of `family` at HEAD, as `images build` tags it."""
    return extract_cmd.image_tag(family)


def _job_id(root: root_mod.Root) -> str:
    now = datetime.now(UTC)
    number = 1
    while (root.path / "runs" / root_mod.job_id(now, number)).exists():
        number += 1
    return root_mod.job_id(now, number)


def _digest(parts: list[str]) -> str:
    return hashlib.sha256("\n".join(parts).encode("utf-8")).hexdigest()


def _check_ids(args: argparse.Namespace) -> None:
    for name, value in (("target id", args.target), ("run id", args.seed_run)):
        if value is not None and (not extract_cmd.RUN_ID.fullmatch(value) or ".." in value):
            raise InputError(f"refused: a {name} is one folder name")
    if args.seed_run is None and args.identity is None:
        raise InputError("refused: target build needs --seed-run, --identity or both")
    if args.identity is None and not identity_mod.VOICE_ID.fullmatch(args.target):
        raise InputError(
            "refused: a seed target's id is its voice id, so it must be [a-z0-9][a-z0-9-]*"
        )


def run_build(args: argparse.Namespace) -> int:
    """Read the inputs, run the job, place the references, write and print the target."""
    root = root_mod.open_root(args.root)
    _check_ids(args)
    target_dir = target_mod.target_rel(args.target)
    if (root.path / target_dir / target_mod.TARGET_NAME).exists():
        raise InputError(
            f"refused: {target_dir}/{target_mod.TARGET_NAME} already exists; choose a new target"
        )
    thresholds = gates_mod.load(root.path).effective()
    pins = target_mod.embedder_pins(jobs.load_families())
    identity = identity_mod.check_in_root(root, args.identity) if args.identity else None
    seed = target_mod.read_seed(root, args.seed_run, args.target) if args.seed_run else None

    inputs: list[JobInput] = []
    if seed is not None:
        run_dir = clean_cmd.run_rel(seed.run_id)
        parts = [sha256_file(root.path / run_dir / clean_cmd.SEGMENTS_NAME)]
        parts += [sha256_file(root.path / r.audio) for r in seed.segments]
        inputs.append(JobInput(rel=run_dir, sha256=_digest(parts)))
    if identity is not None:
        voice_dir = identity_mod.voice_rel(identity.voice_id)
        parts = [sha256_file(identity.path / identity_mod.MANIFEST)]
        parts += [c.audio_sha256 for c in identity.clips]
        inputs.append(JobInput(rel=voice_dir, sha256=_digest(parts)))

    plan = target_mod.plan(
        target_id=args.target, seed=seed, identity=identity, thresholds=thresholds
    )
    write_json(root, f"{target_dir}/{target_mod.PLAN_NAME}", plan)
    job = jobs.JobSpec(
        job_id=_job_id(root),
        family=target_mod.FAMILY,
        image=image_tag(target_mod.FAMILY),
        module=target_mod.MODULE,
        args=("--target", args.target, "--device", args.device),
        inputs=tuple(inputs),
        outputs=target_dir,
        gpu=args.device == "cuda",
        network=False,
    )
    jobs.executor_for("local", root=root, docker=make_docker()).run(job)
    build = target_mod.read_build(root, args.target)
    record = target_mod.assemble(
        root, target_id=args.target, build=build, seed=seed, identity=identity, pins=pins
    )
    write_json(root, f"{target_dir}/{target_mod.TARGET_NAME}", record)
    for line in target_mod.report_lines(record):
        print(line)
    return 0
