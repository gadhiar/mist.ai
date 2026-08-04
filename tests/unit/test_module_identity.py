"""Guard: one source file must never be loaded under two module names.

`backend/` is a PEP 420 namespace package -- there is no `backend/__init__.py`
-- and the production entry point runs `CMD ["python", "backend/server.py"]`,
which makes CPython put the SCRIPT's directory, `backend/`, at `sys.path[0]`.
`server.py:39` then adds the repository root so the `backend.*` spelling
resolves. Both directories are therefore on `sys.path`, and under that layout a
single file such as `backend/request_context.py` is importable under two
different dotted names, `request_context` and `backend.request_context`, and
Python will execute it twice and keep two independent module objects.

The 2026-08-03 fix unified every first-party import on the `backend.` spelling,
so nothing loads a bare name any more and both baselines below are empty. It did
NOT -- and could not -- take `backend/` off `sys.path`: that placement comes
from script-directory semantics, not from an insert that could be deleted. The
bare spelling remains *importable*, so this guard remains load-bearing rather
than historical, and the probe below still reproduces both path entries.

That is not a style complaint. Module-level state belongs to the module
*object*, so a `ContextVar` defined in such a file exists twice over: a `.set()`
reached through one name is invisible to a `.get()` reached through the other.
A session-id propagation fix written on 2026-08-02 landed on exactly this seam
and would have been a silent no-op. Every test on either side still passed,
because each side was internally consistent. No behavioural test could see the
problem, because the problem is *identity*, not behaviour -- which is why the
guard has to look at `sys.modules` directly.

Both probes run the import in a subprocess. That is deliberate: importing the
production graph into the pytest interpreter would permanently rewrite that
interpreter's `sys.path` and `sys.modules` for every test that runs afterwards,
and a pristine child is also the only way to observe a `sys.modules` that is
production's rather than production's-plus-whatever-pytest-imported. Each probe
runs ONCE per module (module-scoped fixtures) and every test reads its result.

Three things here are asserted as strict-equality baselines rather than as
thresholds: the duplicate module pairs, and the set of modules that fail to
import. A count or a floor would let the checked surface shrink quietly while
still reporting green, which is the same "emptiness is a universal alibi"
failure this guard exists to remove. Strict equality means both a regression
AND a fix break the build until someone edits the baseline deliberately, so
every list below can only shrink.
"""

import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

# Roots walked by the "all-roots" probe. `dependencies/` is deliberately NOT
# walked: it is vendored Apache-2.0 CSM code that the project treats as legacy
# and inactive, it is not ours to re-spell, and importing it would drag in a
# dormant torch model stack. It remains COVERED anyway -- the duplicate scan is
# scoped by path, not by walked root, so if anything ever loads a
# dependencies/csm module under two names the check still fires. See the
# `model_manager.py:13` note on _ROOT_MODULE_FLOORS below.
WALKED_ROOTS = ("backend", "src", "scripts")

# Probe program. Populates sys.modules the way production does, then dumps
# what loaded, what failed to load, and how many modules each root offered.
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
walked_roots = ("backend", "src", "scripts")

# Reproduce the production path layout. Both entries are real: `python
# backend/server.py` puts backend/ at sys.path[0] via script-directory
# semantics, and server.py:39 adds the repo root. That layout is precisely what
# makes one file importable under two names, so the probe must keep BOTH -- the
# 2026-08-03 import unification removed the redundant explicit insert of
# backend/, not backend/ itself. Dropping backend_dir here would make the check
# pass for a reason that does not hold in production, which is a weaker guard
# wearing a green badge.
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

attempted = {}
failed = {}

if mode == "entrypoint":
    # Execute exactly the top-level import statements of server.py, parsed out
    # of the file itself rather than copied into this probe. A hand-copied
    # import list would go stale the first time someone edits server.py -- the
    # same "built, tested, and wired to nothing" failure this file exists to
    # prevent. Only the imports are executed, not the module body, so the probe
    # does not construct the FastAPI app or touch the log.
    source = pathlib.Path(backend_dir, "server.py").read_text(encoding="utf-8")
    namespace = {"__name__": "__mist_entrypoint_probe__"}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            compiled = compile(
                ast.Module(body=[node], type_ignores=[]), "<server.py imports>", "exec"
            )
            exec(compiled, namespace)
else:
    for root_name in walked_roots:
        root = pathlib.Path(repo_root, root_name)
        count = 0
        for path in sorted(root.rglob("*.py")):
            parts = list(path.relative_to(root.parent).with_suffix("").parts)
            if parts[-1] == "__init__":
                parts = parts[:-1]
            module_name = ".".join(parts)
            count += 1
            try:
                importlib.import_module(module_name)
            except BaseException as exc:
                # Recorded, never swallowed. A module that silently stopped
                # importing would shrink the surface this guard examines while
                # it kept reporting green -- the exact defect class the guard
                # exists to remove. BaseException because a script that calls
                # sys.exit() at import raises SystemExit, which is a failure to
                # import like any other.
                failed[module_name] = "{0}: {1}".format(type(exc).__name__, exc)
        attempted[root_name] = count

loaded = {}
for name, module in list(sys.modules.items()):
    file = getattr(module, "__file__", None)
    if file:
        # realpath, not abspath: a symlinked source file reached by two routes
        # is one module identity, and comparing unresolved paths would both
        # miss real duplicates and invent fake ones.
        loaded[name] = os.path.realpath(file)

payload = {"loaded": loaded, "failed": failed, "attempted": attempted}
pathlib.Path(out_path).write_text(json.dumps(payload), encoding="utf-8")
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


def _run_probe(mode: str) -> dict:
    """Import in a clean interpreter; return {loaded, failed, attempted}."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "probe.json"
        result = subprocess.run(
            [sys.executable, "-c", _PROBE, mode, str(REPO_ROOT), str(out_path)],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
        # A probe that crashed would report zero duplicates and pass. Fail loudly
        # instead -- a guard that goes vacuous on error is worse than no guard.
        assert result.returncode == 0, (
            f"module-identity probe ({mode}) failed with exit {result.returncode}. "
            f"This test cannot report duplicates it never got to look for.\n"
            f"--- stderr ---\n{result.stderr[-4000:]}"
        )
        return json.loads(out_path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def entrypoint_probe() -> dict:
    """One subprocess, shared by every test that reads the entry-point graph."""
    return _run_probe("entrypoint")


@pytest.fixture(scope="module")
def all_roots_probe() -> dict:
    """One subprocess covering backend/, src/ and scripts/ together.

    A single interpreter for all three roots is not just cheaper than three --
    it is more sensitive. Duplicate identity only shows up when both spellings
    are resident at once, so co-loading the roots can reveal cross-root
    collisions that per-root probes would each individually miss.
    """
    return _run_probe("all-roots")


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


# Known-broken baseline, NOT an exemption list. Each entry would be a live
# defect: a file really executed twice in the running server, with its
# module-level state really duplicated. The tests assert the found set EQUALS
# this baseline, so a new duplicate fails immediately and fixing one also fails
# until its entry is deleted here. The baseline can only shrink.
#
# EMPTY is the correct state and the state as of 2026-08-03.
#
# History, kept because it is the whole reason this guard exists. When the guard
# was built on 2026-08-03 it recorded three live duplicates here --
# backend/factories.py, backend/knowledge/__init__.py and
# backend/knowledge/config.py -- because server.py imported `config`,
# `voice_processor`, `factories`, `knowledge.config` and `log_handler` by bare
# name while 341 import sites inside backend/ spelled the same modules
# `backend.*`. factories.py is the dependency-injection wiring module, so the
# DI graph was built twice in production. Those imports were unified onto the
# `backend.` spelling later the same day (server.py:50-59,
# voice_processor.py:31/40), which emptied this baseline and the one below.
ENTRY_POINT_BASELINE: dict[str, list[str]] = {}

# The all-roots walk reaches modules the entry point does not, so it sees a
# superset. It once carried six extra pairs -- audio_protocol, config,
# log_handler, voice_models, voice_models.model_manager and voice_processor --
# which were collision-capable rather than live: they would have duplicated the
# moment any reachable module imported them the other way. Two of them were in
# fact already live through voice_processor.py's bare `audio_protocol` and
# `voice_models.model_manager` imports. All six were resolved by the same
# unification, so this baseline is empty too.
#
# Walking src/ and scripts/ on 2026-08-03 added 44 modules and found NO
# duplicate of its own -- those two trees were clean then and are clean now.
ALL_ROOTS_BASELINE: dict[str, list[str]] = {
    **ENTRY_POINT_BASELINE,
}

# Modules that do not import, as a strict-equality baseline. EMPTY is the
# correct state and the state as of 2026-08-03: all 158 modules across the
# three roots import cleanly.
#
# This replaced a silent `except BaseException: pass`, which let a module drop
# out of the examined surface with no signal at all -- a guard quietly checking
# less than it did yesterday while still reporting green.
#
# Format if an entry is ever genuinely warranted: module name -> exception type
# name, with an inline comment giving the WHY (optional dependency absent in the
# test image, hardware/CUDA requirement, and so on). "It started failing" is not
# a justification; an entry without a stated reason is rot. Comparison is on the
# exception TYPE only, because messages carry absolute paths and versions that
# would churn the baseline.
IMPORT_FAILURE_BASELINE: dict[str, str] = {}

# Per-root floors on how many modules the walk OFFERED (not how many loaded).
# These catch a whole root silently dropping out of the walk -- a bad rglob, a
# renamed directory, a bad relative_to -- which would otherwise shrink the
# surface without failing anything. Set below the real counts (backend 114,
# src 14, scripts 30) so ordinary growth never trips them.
#
# Related live hazard these floors do NOT cover, recorded here because it is
# the reason src/ was worth walking: backend/voice_models/model_manager.py:12-13
# inserts the repo root and then the RELATIVE path "dependencies/csm" onto
# sys.path, so it resolves only when CWD is the repo root. Combined with the
# bare `from generator import Segment` at model_manager.py:240/736 and
# src/multimodal/tts.py:38/103/199, dependencies/csm/generator.py is
# double-identity-capable. It stays latent only because CSM is dormant behind
# Chatterbox and those imports sit inside functions, where a static walk cannot
# reach them. If CSM is ever reactivated the duplicate scan will catch it.
_ROOT_MODULE_FLOORS = {"backend": 90, "src": 10, "scripts": 20}

# Floors on first-party modules actually resident in sys.modules. A probe that
# imported almost nothing would find no duplicates and pass, which is the exact
# shape of vacuous green this guard exists to make impossible. Real counts are
# ~80 and ~150.
_ENTRY_POINT_MODULE_FLOOR = 40
_ALL_ROOTS_MODULE_FLOOR = 100


def _assert_no_hollow_probe(probe: dict, floor: int) -> None:
    """Fail unless the probe actually loaded a production-sized graph."""
    resident = [n for n, p in probe["loaded"].items() if _is_first_party(p)]
    assert len(resident) >= floor, (
        f"probe loaded only {len(resident)} first-party modules (floor {floor}) "
        "-- it is not exercising the real graph, so its 'no duplicates' result "
        "means nothing"
    )


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

    def test_the_production_entry_point_loads_no_new_source_file_under_two_names(
        self, entrypoint_probe
    ):
        # Arrange / Act
        duplicates = _first_party_duplicates(entrypoint_probe["loaded"])

        # Assert -- coverage floor first, so a hollow probe cannot pass quietly.
        _assert_no_hollow_probe(entrypoint_probe, _ENTRY_POINT_MODULE_FLOOR)
        _assert_matches_baseline(duplicates, ENTRY_POINT_BASELINE)


class TestAllRootsModuleIdentity:
    """Every module under backend/, src/ and scripts/, co-loaded in one run."""

    def test_walking_every_root_loads_no_new_source_file_under_two_names(self, all_roots_probe):
        # Arrange / Act
        duplicates = _first_party_duplicates(all_roots_probe["loaded"])

        # Assert
        _assert_no_hollow_probe(all_roots_probe, _ALL_ROOTS_MODULE_FLOOR)
        _assert_matches_baseline(duplicates, ALL_ROOTS_BASELINE)

    def test_every_root_still_offers_the_walk_a_full_set_of_modules(self, all_roots_probe):
        # Arrange / Act
        attempted = all_roots_probe["attempted"]

        # Assert
        assert set(attempted) == set(WALKED_ROOTS), (
            f"the walk covered {sorted(attempted)} but should cover "
            f"{sorted(WALKED_ROOTS)} -- a root stopped being examined"
        )

        thin = {r: n for r, n in attempted.items() if n < _ROOT_MODULE_FLOORS[r]}
        assert not thin, (
            f"a root offered fewer modules than its floor: {thin} "
            f"(floors {_ROOT_MODULE_FLOORS}). The walk is finding less than it "
            "used to, so any clean result from it covers less than it claims."
        )


class TestEveryWalkedModuleStillImports:
    """The surface this guard examines must not shrink without saying so."""

    def test_no_module_under_the_walked_roots_has_stopped_importing(self, all_roots_probe):
        # Arrange / Act
        failed: dict[str, str] = all_roots_probe["failed"]

        # Assert
        new = {n: d for n, d in failed.items() if n not in IMPORT_FAILURE_BASELINE}
        assert not new, (
            "A module under the walked roots no longer imports:\n\n"
            + "\n".join(f"  {name}\n      {detail}" for name, detail in sorted(new.items()))
            + "\n\nUntil this is fixed that module is invisible to the "
            "module-identity check, which will keep reporting green over a "
            "surface that just shrank. Fix the import, or add a baseline entry "
            "with a written justification if it genuinely cannot import here."
        )

        recovered = sorted(set(IMPORT_FAILURE_BASELINE) - set(failed))
        assert not recovered, (
            f"Good news, and this test needs updating: {', '.join(recovered)} "
            "imports again. Delete the entry from IMPORT_FAILURE_BASELINE so "
            "the recovery is locked in."
        )

        drifted = {
            name: detail
            for name, detail in failed.items()
            if name in IMPORT_FAILURE_BASELINE
            and detail.split(":", 1)[0] != IMPORT_FAILURE_BASELINE[name]
        }
        assert not drifted, (
            f"a known-failing module now fails differently: {drifted}. The "
            "recorded justification may no longer describe the real cause."
        )


def test_no_backend_source_file_is_ever_loaded_twice(entrypoint_probe):
    """The absolute statement, independent of any baseline.

    Carried `@pytest.mark.xfail(strict=True)` from 2026-08-03 until the imports
    were unified the same day; the marker then XPASSed and failed the suite,
    which is exactly what it was for. It is a plain assertion now. Unlike the
    baseline-comparing tests above, this one cannot be satisfied by editing a
    dict -- it stays honest even if someone adds an entry to a baseline instead
    of fixing the duplicate.
    """
    # Arrange / Act
    duplicates = _first_party_duplicates(entrypoint_probe["loaded"])

    # Assert -- coverage floor first, so a hollow probe cannot pass quietly.
    _assert_no_hollow_probe(entrypoint_probe, _ENTRY_POINT_MODULE_FLOOR)
    assert not duplicates, (
        "the production entry point loads these files under two module "
        f"names:\n{_describe(duplicates)}"
    )
