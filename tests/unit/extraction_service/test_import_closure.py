"""Guards the extraction service's import closure.

Two tests, both in a fresh subprocess so a module an earlier test imported in the
same session cannot mask a real regression.

The denylist test: T1b builds the deploy image from whatever
`import backend.extraction_service.app` pulls in. `torch` and
`sentence_transformers` must never be part of that closure -- the service is
meant to run on a GTX 1070 host with only a llama-server dependency, not the
full backend's embedding/voice stack.

`neo4j`, `pandas`, and `pyarrow` must never be part of it either (goal
mist-two-loop v2-lazy-imports, MIS-171): the stateless extraction service
carries no storage dependency, but before
`backend/knowledge/extraction/__init__.py` and
`backend/knowledge/storage/__init__.py` became PEP 562 lazy, importing any
submodule of either package (which the service does, e.g.
`backend.knowledge.extraction.internal_derivation`) ran the package
`__init__` first, which eagerly imported `GraphStore`/`Neo4jConnection` ->
`neo4j` -> pandas (neo4j's own optional-deps probe) -> pyarrow (pandas's own
compat probe). See docker/extraction/requirements.txt's header comment for
the full chain.

The allowlist test: a denylist names only the packages someone thought of, and
numpy reached the closure through `backend.memory.config` importing the need
gate without being on it, so the image exited at import. The allowlist test
blocks every top-level module the image does not install: anything outside the
standard library, `backend`, and the distributions `docker/extraction/
requirements.txt` lists together with their transitive requirements, read from
the installed metadata at test time. The test environment's versions can differ
from the image's pins, so its transitive set is close to the image's, not equal.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path

from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[3]
EXTRACTION_REQUIREMENTS = REPO_ROOT / "docker" / "extraction" / "requirements.txt"

_PROBE_CODE = (
    "import sys\n"
    "import backend.extraction_service.app\n"
    "mods = {m.split('.')[0] for m in sys.modules}\n"
    "assert 'torch' not in mods, sorted(mods)\n"
    "assert 'sentence_transformers' not in mods, sorted(mods)\n"
    "assert 'neo4j' not in mods, sorted(mods)\n"
    "assert 'pandas' not in mods, sorted(mods)\n"
    "assert 'pyarrow' not in mods, sorted(mods)\n"
    "print('IMPORT_CLOSURE_OK')\n"
)


def test_extraction_service_app_import_excludes_gpu_and_storage_stack_deps():
    result = subprocess.run(
        [sys.executable, "-c", _PROBE_CODE],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "IMPORT_CLOSURE_OK" in result.stdout


# The child installs a finder ahead of every other one. A top-level import outside the allowlist
# raises ModuleNotFoundError naming the module and the frame that imported it, so a library's
# guarded optional import falls back as it would in the image, and an unguarded one fails the run
# with that message in its traceback.
_ALLOWLIST_PROBE_CODE = """
import importlib
import json
import sys

ALLOWED = frozenset(json.loads(sys.argv[1]))


def _allowed(top):
    return (
        top in ALLOWED
        or top in sys.stdlib_module_names
        or top in sys.builtin_module_names
        or top in ("__main__", "backend")
        or top.startswith("_sysconfigdata")
    )


def _importer():
    frame = sys._getframe(2)
    while frame is not None and frame.f_code.co_filename.startswith("<frozen"):
        frame = frame.f_back
    if frame is None:
        return "an unknown importer"
    name = frame.f_globals.get("__name__", "?")
    return f"{name} ({frame.f_code.co_filename}:{frame.f_lineno})"


class AllowlistFinder:
    @classmethod
    def find_spec(cls, fullname, path=None, target=None):
        if "." in fullname or _allowed(fullname):
            return None
        raise ModuleNotFoundError(
            f"{fullname!r} is outside the extraction image's import allowlist; "
            f"imported by {_importer()}",
            name=fullname,
        )


sys.meta_path.insert(0, AllowlistFinder)

for module in (
    "backend.extraction_service",
    "backend.extraction_service.__main__",
    "backend.extraction_service.adapters",
    "backend.extraction_service.app",
    "backend.extraction_service.engine",
    "backend.extraction_service.schemas",
    "backend.extraction_service.settings",
):
    importlib.import_module(module)

from fastapi import FastAPI

from backend.extraction_service.app import build_app_from_env

app = build_app_from_env()
assert isinstance(app, FastAPI), type(app)

try:
    import numpy
except ModuleNotFoundError as error:
    # Only the finder's own refusal counts: an environment without numpy would raise this
    # error even with a finder that blocks nothing.
    if "outside the extraction image's import allowlist" not in str(error):
        raise
else:
    raise AssertionError("numpy imported, so the allowlist blocks nothing")

print("ALLOWLIST_CLOSURE_OK")
"""

# What a compose file gives the service: the three required `EXTRACTION_SERVER_*` declarations,
# and an LLM URL that is never called, since building the app opens no connection.
_SERVICE_ENV = {
    "EXTRACTION_LLM_BASE_URL": "http://extraction-llm.invalid:8080",
    "EXTRACTION_SERVER_CTX_SIZE": "8192",
    "EXTRACTION_SERVER_DEVICE": "test-gpu",
    "EXTRACTION_SERVER_IMAGE": "test-llama-server@sha256:" + "0" * 64,
}


def _normalise(name: str) -> str:
    return re.sub(r"[-_.]+", "_", name).lower()


def _listed_requirements() -> list[Requirement]:
    requirements = []
    for line in EXTRACTION_REQUIREMENTS.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            requirements.append(Requirement(line))
    return requirements


def _allowed_import_names() -> set[str]:
    """Import names of the listed distributions and their requirements, as installed here.

    A requirement's markers are evaluated for no extra, so an optional dependency the image
    does not install stays blocked. A listed distribution missing from this environment adds
    its normalised name, which is its import name for `distro`.
    """
    closure: set[str] = set()
    missing: set[str] = set()
    pending = _listed_requirements()
    while pending:
        requirement = pending.pop()
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        name = _normalise(requirement.name)
        if name in closure or name in missing:
            continue
        try:
            requires = metadata.distribution(requirement.name).requires or []
        except metadata.PackageNotFoundError:
            missing.add(name)
            continue
        closure.add(name)
        pending.extend(Requirement(line) for line in requires)
    allowed = {
        top
        for top, dists in metadata.packages_distributions().items()
        if any(_normalise(dist) in closure for dist in dists)
    }
    return allowed | missing


def test_extraction_service_imports_only_what_its_image_installs():
    allowed = _allowed_import_names()
    env = {key: value for key, value in os.environ.items() if not key.startswith("EXTRACTION_")}
    env.update(_SERVICE_ENV)
    result = subprocess.run(
        [sys.executable, "-c", _ALLOWLIST_PROBE_CODE, json.dumps(sorted(allowed))],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=REPO_ROOT,
        env=env,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "ALLOWLIST_CLOSURE_OK" in result.stdout
