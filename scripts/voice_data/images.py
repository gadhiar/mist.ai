"""Stage images built from a committed tree, never from the working tree.

`git archive` exports the code and Dockerfiles at a commit into a temporary folder, and that folder
is the build context, so the root `.dockerignore` (which excludes `docker/`) and any uncommitted
or ignored file never shape an image. Each image is tagged with the commit, and the images bake
the code at that commit, so local and rented runs execute the same code.
"""

from __future__ import annotations

import io
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path

from .errors import InputError
from .jobs import DockerOps, load_families

COMPONENTS = ("scripts/voice_data", "scripts/model_bench", "docker/voice-data")
IMAGE_PREFIX = "mist-voice-"
BASE_IMAGE = "mist-voice-base"
# The code-free `runtime` stage of base.Dockerfile. The family images build on it rather than on
# the base image, whose last layers change with every commit, so their venv layers stay cached.
RUNTIME_IMAGE = "mist-voice-runtime"
RUNTIME_TARGET = "runtime"


def resolve_commit(checkout: Path, sha: str) -> str:
    """Return the full commit id `sha` names in the checkout.

    Raises:
        InputError: `sha` does not resolve to a commit there.
    """
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "--quiet", f"{sha}^{{commit}}"],
        cwd=checkout,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise InputError(f"refused: {sha} is not a commit in the checkout")
    return proc.stdout.strip()


def export_tree(checkout: Path, commit: str, dest: Path) -> None:
    """Extract the three component paths at `commit` into `dest`.

    The archive is read as bytes from `subprocess`, never through a shell pipe, which on Windows
    PowerShell would re-encode the tar stream.

    Raises:
        InputError: `git archive` failed, for example a component path absent at that commit.
    """
    proc = subprocess.run(
        ["git", "archive", "--format=tar", commit, *COMPONENTS],
        cwd=checkout,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise InputError(f"refused: git archive of {commit[:12]} failed")
    with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tar:
        tar.extractall(dest, filter="data")


def images_for(families: Mapping[str, Mapping[str, object]], family: str) -> list[str]:
    """Return the distinct images to build for `family` (or `all`), base first.

    Families share images (the `score` family is the base image), and a hosted family has no
    image, so the list is built from the `image` fields and each image appears once.

    Raises:
        InputError: `family` is neither `all` nor a family in `families.json`.
    """
    if family == "all":
        names = [entry.get("image") for entry in families.values()]
    elif family in families:
        names = [families[family].get("image")]
    else:
        raise InputError(f"refused: unknown family {family}")
    rest = [name for name in dict.fromkeys(names) if name and name != BASE_IMAGE]
    return [BASE_IMAGE, *rest]  # type: ignore[list-item]


def _dockerfile(context: Path, image: str) -> Path:
    return context / "docker" / "voice-data" / f"{image.removeprefix(IMAGE_PREFIX)}.Dockerfile"


def runtime_argv(context: Path, commit: str) -> list[str]:
    """Return the `docker build` argv for the `runtime` stage, which takes no build arg."""
    return [
        "docker",
        "build",
        "-f",
        str(_dockerfile(context, BASE_IMAGE)),
        "--target",
        RUNTIME_TARGET,
        "-t",
        f"{RUNTIME_IMAGE}:{commit[:12]}",
        str(context),
    ]


def build_argv(context: Path, image: str, commit: str) -> list[str]:
    """Return the `docker build` argv for one image from the exported context.

    A family image also gets `BASE_IMAGE`, the runtime tag of the same commit; its Dockerfile
    declares the arg without a default, so it cannot build on a tag from another commit.
    """
    argv = ["docker", "build", "-f", str(_dockerfile(context, image))]
    argv += ["--build-arg", f"GIT_SHA={commit}"]
    if image != BASE_IMAGE:
        argv += ["--build-arg", f"BASE_IMAGE={RUNTIME_IMAGE}:{commit[:12]}"]
    return [*argv, "-t", f"{image}:{commit[:12]}", str(context)]


def build(checkout: Path, sha: str, family: str, docker: DockerOps) -> list[str]:
    """Build the runtime stage, then the images for `family` (or `all`) at `sha`; return the tags.

    The family list comes from the exported tree's `families.json`, so it is the one at that
    commit. The temporary context is removed afterwards, whether the builds pass or fail.

    Raises:
        InputError: `sha` is not a commit, the export failed, or the family is unknown.
        JobError: A `docker build` exited non-zero.
    """
    commit = resolve_commit(checkout, sha)
    context = Path(tempfile.mkdtemp(prefix="mist-voice-image-"))
    try:
        export_tree(checkout, commit, context)
        families = load_families(context / "scripts" / "voice_data" / "families.json")
        images = images_for(families, family)
        docker.check(runtime_argv(context, commit))
        tags = [f"{RUNTIME_IMAGE}:{commit[:12]}"]
        for image in images:
            docker.check(build_argv(context, image, commit))
            tags.append(f"{image}:{commit[:12]}")
        return tags
    finally:
        shutil.rmtree(context, ignore_errors=True)
