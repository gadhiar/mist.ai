"""Guard: one source file must never be loaded under two module names.

`backend/` is a PEP 420 namespace package -- there is no `backend/__init__.py`
-- and `backend/server.py`, the production entry point (the image runs
`CMD ["python", "backend/server.py"]`), puts BOTH the repository root and
`backend/` itself on `sys.path` (server.py:33-35). Under that path layout a
single file such as `backend/request_context.py` is importable under two
different dotted names, `request_context` and `backend.request_context`, and
Python will execute it twice and keep two independent module objects.

That is not a style complaint. Module-level state belongs to the module
*object*, so a `ContextVar` defined in such a file exists twice over: a `.set()`
reached through one name is invisible to a `.get()` reached through the other.
A session-id propagation fix written on 2026-08-02 landed on exactly this seam
and would have been a silent no-op. Every test on either side still passed,
because each side was internally consistent. No behavioural test could see the
problem, because the problem is *identity*, not behaviour -- which is why the
guard has to look at `sys.modules` directly.

Both tests below run the import in a subprocess. That is deliberate: importing
the production graph into the pytest interpreter would permanently rewrite that
interpreter's `sys.path` and `sys.modules` for every test that runs afterwards,
and a pristine child is also the only way to observe a `sys.modules` that is
production's rather than production's-plus-whatever-pytest-imported.
"""

import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Probe program. Populates sys.modules the way production does, then dumps
# {module name -> resolved __file__} as JSON for the parent to analyse.
_PROBE = r'''
import ast
import importlib
import json
import logging
import os
import pathlib
import sys

mode, repo_root, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
backend_dir = os.path.join(repo_root, "backend")

# Mirror backend/server.py:33-35. The production entry point puts the repo root
# AND backend/ on sys.path, and that path layout is precisely what makes one
# file importable under two names. Without reproducing it the check cannot see
# the defect it exists to catch.
sys.path.insert(0, repo_root)
sys.path.insert(0, backend_dir)


class _NoFileHandler(logging.Handler):
    """Swallow file logging inside the probe.

    server.py installs a FileHandler on /app/logs/mist-backend.log at module
    scope. This subprocess only inspects sys.modules; it has no business
    appending to the live production log. Accepts any signature because it
    stands in for FileHandler(path, mode=..., encoding=...).
    """

    def __init__(self, *args, **kwargs):
        super().__init__()

    def emit(self, record):
        pass


logging.FileHandler = _NoFileHandler

if mode == "entrypoint":
    # Execute exactly the top-level import statements of server.py, parsed out
    # of the file itself rather than copied into this probe. A hand-copied
    # import list would go stale the first time someone edits server.py -- the
    # same "built, tested, and wired to nothing" failure this file exists to
    # prevent. Only the imports are executed, not the module body, so the
    # probe does not construct the FastAPI app or touch the log.
    source = pathlib.Path(backend_dir, "server.py").read_text(encoding="utf-8")
    namespace = {"__name__": "__mist_entrypoint_probe__"}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            compiled = compile(
                ast.Module(body=[node], type_ignores=[]), "<server.py imports>", "exec"
            )
            exec(compiled, namespace)
else:
    root = pathlib.Path(backend_dir)
    for path in sorted(root.rglob("*.py")):
        parts = list(path.relative_to(root.parent).with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
        try:
            importlib.import_module(".".join(parts))
        except BaseException:
            # A module that will not import is simply not covered by this run.
            # That is not a module-identity failure and must not be reported as
            # one. The parent's coverage floor catches wholesale import
            # collapse, so this cannot silently hollow the check out.
            pass

loaded = {}
for name, module in list(sys.modules.items()):
    file = getattr(module, "__file__", None)
    if file:
        # realpath, not abspath: a symlinked source file reached by two routes
        # is one module identity, and comparing unresolved paths would both
        # miss real duplicates and invent fake ones.
        loaded[name] = os.path.realpath(file)

pathlib.Path(out_path).write_text(json.dumps(loaded), encoding="utf-8")
'''

# Installed distributions legitimately alias themselves (setuptools.extern.*,
# requests.packages.urllib3.*, scipy's compiled .so files registered under both
# a private and a package-qualified name, and CPython's own os.path/posixpath
# and _frozen_importlib/importlib._bootstrap). There were 40+ such pairs in a
# single probe run. None are this repository's defects and none are fixable
# here, so the check is scoped to files the repo owns rather than carrying a
# name allowlist that would need constant feeding.
_INSTALLED_PACKAGE_MARKERS = ("site-packages", "dist-packages")

# `__main__` is always an alias: `python backend/server.py` binds server.py to
# `__main__`, and any later import of it by name yields a second object. That
# aliasing is a property of executing a file as a script, not a naming defect in
# the module graph, and no import rewrite can remove it -- so it is excluded
# here rather than reported as a duplicate this repo could fix.
_SCRIPT_ALIAS_NAMES = frozenset({"__main__"})


def _is_first_party(path: str) -> bool:
    """True for source files this repository owns, and could therefore fix."""
    candidate = Path(path)
    if not candidate.is_relative_to(REPO_ROOT):
        return False
    return not any(marker in candidate.parts for marker in _INSTALLED_PACKAGE_MARKERS)


def _run_probe(mode: str) -> dict[str, str]:
    """Import the backend in a clean interpreter; return {module name -> file}."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "loaded.json"
        result = subprocess.run(
            [sys.executable, "-c", _PROBE, mode, str(REPO_ROOT), str(out_path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )
        # A probe that crashed would report zero duplicates and pass. Fail loudly
        # instead -- a guard that goes vacuous on error is worse than no guard.
        assert result.returncode == 0, (
            f"module-identity probe ({mode}) failed with exit {result.returncode}. "
            f"This test cannot report duplicates it never got to look for.\n"
            f"--- stderr ---\n{result.stderr[-4000:]}"
        )
        return json.loads(out_path.read_text(encoding="utf-8"))


def _first_party_duplicates(loaded: dict[str, str]) -> dict[str, list[str]]:
    """Map each first-party file loaded under 2+ names to those names."""
    by_file: dict[str, list[str]] = defaultdict(list)
    for name, path in loaded.items():
        if name in _SCRIPT_ALIAS_NAMES or not _is_first_party(path):
            continue
        by_file[path].append(name)
    return {
        str(Path(path).relative_to(REPO_ROOT).as_posix()): sorted(names)
        for path, names in sorted(by_file.items())
        if len(names) > 1
    }


def _describe(duplicates: dict[str, list[str]]) -> str:
    return "\n".join(
        f"  {path}\n      loaded as: {' AND '.join(n)}" for path, n in duplicates.items()
    )


# Known-broken baseline, NOT an exemption list. Each entry is a live defect:
# these files really are executed twice in the running server, so their
# module-level state really is duplicated. The tests assert the found set
# EQUALS this baseline, which means a new duplicate fails immediately and
# fixing one of these also fails until the entry is deleted here. The baseline
# can only shrink, and `test_no_backend_source_file_is_ever_loaded_twice`
# below stays red until it reaches zero.
#
# Found 2026-08-03 while building this guard. Root cause: server.py imports
# `config`, `voice_processor`, `factories`, `knowledge.config` and `log_handler`
# by bare name, while 341 import sites inside backend/ spell the same modules
# `backend.*`.
ENTRY_POINT_BASELINE = {
    "backend/factories.py": ["backend.factories", "factories"],
    "backend/knowledge/__init__.py": ["backend.knowledge", "knowledge"],
    "backend/knowledge/config.py": ["backend.knowledge.config", "knowledge.config"],
}

# The whole-tree walk reaches modules the entry point does not, so it sees a
# superset: the six extra pairs are collision-capable today and become live the
# moment any reachable module imports them under the other spelling.
WHOLE_TREE_BASELINE = {
    **ENTRY_POINT_BASELINE,
    "backend/audio_protocol.py": ["audio_protocol", "backend.audio_protocol"],
    "backend/config.py": ["backend.config", "config"],
    "backend/log_handler.py": ["backend.log_handler", "log_handler"],
    "backend/voice_models/__init__.py": ["backend.voice_models", "voice_models"],
    "backend/voice_models/model_manager.py": [
        "backend.voice_models.model_manager",
        "voice_models.model_manager",
    ],
    "backend/voice_processor.py": ["backend.voice_processor", "voice_processor"],
}

# Coverage floors. A probe that imported almost nothing would find no
# duplicates and pass, which is the exact shape of vacuous green this guard
# exists to make impossible. Set well under the real counts (entry point ~60,
# whole tree ~120) so ordinary growth never trips them.
_ENTRY_POINT_MODULE_FLOOR = 40
_WHOLE_TREE_MODULE_FLOOR = 100


def _assert_matches_baseline(found: dict[str, list[str]], baseline: dict[str, list[str]]) -> None:
    new = {p: n for p, n in found.items() if p not in baseline}
    assert not new, (
        "A source file is now loaded under two module names.\n\n"
        f"{_describe(new)}\n\n"
        "Those names are two different module objects built from one file, so "
        "module-level state (ContextVars, caches, singletons, registries) is "
        "duplicated: a write through one name is invisible through the other. "
        "Fix by making every import of the file agree on one spelling -- "
        "`backend.<module>` is this repo's convention."
    )

    fixed = sorted(set(baseline) - set(found))
    assert not fixed, (
        "Good news, and this test needs updating: "
        f"{', '.join(fixed)} no longer loads under two names. "
        "Delete the corresponding entries from the baseline in this file so "
        "the fix is locked in and cannot regress."
    )

    drifted = {p: n for p, n in found.items() if p in baseline and n != baseline[p]}
    assert not drifted, (
        f"The names a known-duplicated file loads under changed: {drifted}. "
        "The duplication moved rather than being fixed; update the baseline "
        "deliberately after confirming what changed."
    )


class TestProductionEntryPointModuleIdentity:
    """What the running server actually loads, via server.py's own imports."""

    def test_the_production_entry_point_loads_no_new_source_file_under_two_names(self):
        # Arrange / Act
        loaded = _run_probe("entrypoint")

        # Assert -- coverage floor first, so a hollow probe cannot pass quietly.
        first_party = [n for n, p in loaded.items() if _is_first_party(p)]
        assert len(first_party) >= _ENTRY_POINT_MODULE_FLOOR, (
            f"probe loaded only {len(first_party)} first-party modules "
            f"(floor {_ENTRY_POINT_MODULE_FLOOR}) -- it is not exercising the "
            "production graph, so its 'no duplicates' result means nothing"
        )

        _assert_matches_baseline(_first_party_duplicates(loaded), ENTRY_POINT_BASELINE)


class TestWholeBackendTreeModuleIdentity:
    """Every module under backend/, including those the entry point misses."""

    def test_importing_every_backend_module_loads_no_new_source_file_under_two_names(self):
        # Arrange / Act
        loaded = _run_probe("whole-tree")

        # Assert
        first_party = [n for n, p in loaded.items() if _is_first_party(p)]
        assert len(first_party) >= _WHOLE_TREE_MODULE_FLOOR, (
            f"probe loaded only {len(first_party)} first-party modules "
            f"(floor {_WHOLE_TREE_MODULE_FLOOR}) -- the tree walk is broken, so "
            "its 'no duplicates' result means nothing"
        )

        _assert_matches_baseline(_first_party_duplicates(loaded), WHOLE_TREE_BASELINE)


@pytest.mark.xfail(
    strict=True,
    reason=(
        "Known live defect found 2026-08-03: backend/factories.py, "
        "backend/knowledge/__init__.py and backend/knowledge/config.py are each "
        "executed twice in the running server because server.py imports them by "
        "bare name while the rest of backend/ imports them as backend.*. This "
        "marker is strict, so the moment the imports are unified this test "
        "XPASSes and fails the suite, forcing the marker and the baselines above "
        "to be removed. It is the reminder that the baseline is a defect list, "
        "not an allowlist."
    ),
)
def test_no_backend_source_file_is_ever_loaded_twice():
    duplicates = _first_party_duplicates(_run_probe("entrypoint"))

    assert not duplicates, (
        "the production entry point loads these files under two module "
        f"names:\n{_describe(duplicates)}"
    )
