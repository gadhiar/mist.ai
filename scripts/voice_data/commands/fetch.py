"""`fetch --family F [--executor local|remote:ALIAS]`: pinned model weights into the cache.

The `fetch` job is the only job with the network and the only one that may see `HF_TOKEN`: the
token is an optional env name, passed by name only when it is set on the host, so an ungated
family fetches without one and a gated model fails inside the job. The job mounts the cache
read-write (every other job mounts it read-only and runs with `HF_HUB_OFFLINE=1`), downloads each
model of the family by its `kind`, and writes a manifest of every file with its sha256 into
`runs/<job_id>/fetch/`. The host prints model ids, statuses and file counts from that manifest,
and the sha256 a `url` model whose `families.json` entry says `"unverified"` was given at its
first download, for Raj to record.

The checks below run on the host before any container starts, and the stage runs them again, so
a hand-edited `families.json` is refused in the same words on both sides.
"""

from __future__ import annotations

import argparse
import re
import urllib.parse
from collections.abc import Mapping
from datetime import UTC, datetime

from .. import jobs
from .. import root as root_mod
from ..errors import InputError, JobError
from ..records import read_json
from . import extract as extract_cmd

MODULE = jobs.FETCH_MODULE
TOKEN_ENV = "HF_TOKEN"
OUT_DIR = "fetch"
MANIFEST_NAME = "manifest.json"
UNVERIFIED = "unverified"
PINNED = re.compile(r"[0-9a-f]{40}")
KINDS = ("hf", "url", "hosted")


def register(subparsers: argparse._SubParsersAction) -> None:
    """Add the `fetch` command."""
    parser = subparsers.add_parser(
        "fetch",
        help="download a family's pinned model weights into the cache",
        description=(
            "Download every model of family F in families.json into the root's cache: Hugging "
            "Face repositories at their pinned revision, url files checked against their sha256. "
            "The only job with the network; HF_TOKEN is passed by name when set."
        ),
    )
    parser.add_argument("--family", required=True, help="a family key of families.json")
    parser.add_argument("--executor", default="local", help="local (default) or remote:<ssh alias>")
    parser.set_defaults(func=run)


def make_docker() -> jobs.DockerOps:
    """Return the docker CLI; tests replace this function."""
    return jobs.HostDocker()


def image_tag(family: str) -> str:
    """Return the local image of `family` at HEAD, as `images build` tags it."""
    return extract_cmd.image_tag(family)


def pinned(revision: object) -> bool:
    """Return whether a revision is a full commit sha; a branch or tag name moves."""
    return isinstance(revision, str) and PINNED.fullmatch(revision) is not None


def problems(models: Mapping[str, Mapping[str, object]]) -> list[str]:
    """Return why the family's models cannot be fetched as written, one reason per model.

    A Hugging Face model needs a pinned revision, and a `url` model an HTTPS URL, so a fetch
    can never take whatever a branch points at today or a file over plain HTTP.
    """
    found: list[str] = []
    for model_id, model in models.items():
        kind = model.get("kind")
        if kind not in KINDS:
            found.append(f"model {model_id} has an unknown kind")
        elif kind == "hf" and not pinned(model.get("revision")):
            found.append(f"model {model_id} has no pinned revision")
        elif kind == "url" and urllib.parse.urlparse(str(model.get("url"))).scheme != "https":
            found.append(f"model {model_id} is not an HTTPS url")
    return found


def family_models(
    families: Mapping[str, Mapping[str, object]], name: str
) -> Mapping[str, Mapping[str, object]]:
    """Return the family's models after the checks every fetch makes.

    Raises:
        InputError: The family is unknown or hosted, or a model fails `problems`.
    """
    family = families.get(name)
    if family is None:
        raise InputError(f"refused: unknown family {name}")
    if family.get("hosted") or not family.get("image"):
        raise InputError(f"refused: family {name} is hosted; nothing to fetch")
    models = family.get("models")
    if not isinstance(models, dict):
        raise InputError(f"refused: family {name} has no models map")
    found = problems(models)
    if found:
        raise InputError(f"refused: {'; '.join(found)}")
    return models


def _job_id(root: root_mod.Root) -> str:
    now = datetime.now(UTC)
    number = 1
    while (root.path / "runs" / root_mod.job_id(now, number)).exists():
        number += 1
    return root_mod.job_id(now, number)


def run(args: argparse.Namespace) -> int:
    """Check the family, run the fetch job and print its manifest's summary."""
    root = root_mod.open_root(args.root)
    family_models(jobs.load_families(), args.family)
    job_id = _job_id(root)
    out = f"runs/{job_id}/{OUT_DIR}"
    job = jobs.JobSpec(
        job_id=job_id,
        family=args.family,
        image=image_tag(args.family),
        module=MODULE,
        args=("--family", args.family, "--out", out),
        inputs=(),
        outputs=out,
        gpu=False,
        network=True,
        optional_env_names=(TOKEN_ENV,),
    )
    jobs.executor_for(args.executor, root=root, docker=make_docker()).run(job)
    manifest = read_json(root.path / out / MANIFEST_NAME)
    if not isinstance(manifest, dict) or not isinstance(manifest.get("models"), dict):
        raise JobError(f"job {job_id}: {out}/{MANIFEST_NAME} has no models map")
    for line in summary_lines(args.family, manifest["models"]):
        print(line)
    print(f"  manifest {out}/{MANIFEST_NAME}")
    return 0


def summary_lines(family: str, models: Mapping[str, Mapping[str, object]]) -> list[str]:
    """Return the console summary: counts, then one line per model with ids and numbers only."""
    statuses = [str(m.get("status")) for m in models.values()]
    lines = [
        f"fetch {family}: {len(models)} models, {statuses.count('fetched')} fetched, "
        f"{statuses.count('present')} present, {statuses.count('skipped')} skipped"
    ]
    for model_id, model in sorted(models.items()):
        files = model.get("files")
        count = len(files) if isinstance(files, list) else 0
        lines.append(f"  {model_id}: {model.get('status')}, {count} file{'s' * (count != 1)}")
        if model.get("sha256_unverified"):
            lines.append(
                f"  {model_id}: sha256 {model.get('sha256')} (unverified in families.json)"
            )
    return lines
