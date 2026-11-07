"""The run identity: what it covers, what it ignores, and the drift guard.

Every extraction result carries a run identity, a hash of everything that
shapes extraction output: the prompt, the code that renders and parses it,
the request settings, the ontology and extraction versions, the adapter, the
model and the llama.cpp build (MIS-188). A component that could change
without moving the identity would make two different setups look the same in
every report, so each one is shown to move it.

The code side is hashed from source with comments, docstrings, blank lines and
trailing whitespace removed. The pinned sample below fixes that normalisation:
a Python upgrade whose tokenizer splits source differently (3.12 tokenizes
f-strings into parts) fails here instead of moving every identity silently.

The drift guard at the bottom makes each change to the covered code a dated,
deliberate entry in `version_stamps.CODE_REVISIONS`.
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import re
import uuid
from pathlib import Path
from types import ModuleType

import pytest

from backend.extraction_service import engine
from backend.extraction_service.adapters import get_adapter
from backend.extraction_service.app import run_identity_components
from backend.extraction_service.settings import ServiceSettings
from backend.knowledge import version_stamps
from backend.knowledge.extraction import run_identity
from backend.knowledge.extraction.run_identity import (
    CODE_IDENTITY_FIELDS,
    RunIdentityComponents,
    code_identity,
    code_sha256,
    compute_run_identity,
    normalized_source,
)
from backend.knowledge.version_stamps import CODE_REVISIONS, EXTRACTION_VERSION, CodeRevision

_BACKEND_ROOT = Path(version_stamps.__file__).resolve().parents[1]
_REPO_ROOT = _BACKEND_ROOT.parent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _module_from_source(tmp_path: Path, name: str, source: str) -> ModuleType:
    """A module object named `name` whose source file holds `source`.

    `inspect.getsource` reads a module through its `__file__`, so this stands
    in for an edited copy of a real module without executing it. Each call
    writes a fresh file, so `linecache` never serves an earlier variant.
    """
    path = tmp_path / f"m_{uuid.uuid4().hex}.py"
    path.write_text(source, encoding="utf-8")
    module = ModuleType(name)
    module.__file__ = str(path)
    return module


def _components(**overrides) -> RunIdentityComponents:
    values = {
        "prompt_sha256": "p" * 64,
        "code_sha256": "c" * 64,
        "ontology_version": "1.4.0",
        "extraction_version": "2026-06-14-r5",
        "adapter": "gptoss",
        "model_hash": "gpt-oss-20b",
        "llama_cpp_build": "b11151",
        "constrained_mode": "schema",
        "reasoning_effort": "low",
        "reasoning_budget_tokens": None,
        "temperature": 0.0,
        "max_attempts": 2,
        "scope_enabled": True,
        "scope_max_tokens": 96,
        "extract_max_tokens": 2048,
        "derive_max_tokens": 400,
        "list_batch_max_units": None,
        "list_context_max_units": None,
    }
    values.update(overrides)
    return RunIdentityComponents(**values)


def _changed(value: object) -> object:
    """A different value of the same kind, for any component value."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 0.5
    if isinstance(value, str):
        return value + "-changed"
    if value is None:
        return 7
    raise AssertionError(f"no change rule for {value!r}")


_FIELD_NAMES = [f.name for f in dataclasses.fields(RunIdentityComponents)]


# ---------------------------------------------------------------------------
# The components and the identity
# ---------------------------------------------------------------------------


class TestComponents:
    def test_the_eighteen_components_in_order(self):
        field_names = [f.name for f in dataclasses.fields(RunIdentityComponents)]

        assert field_names == [
            "prompt_sha256",
            "code_sha256",
            "ontology_version",
            "extraction_version",
            "adapter",
            "model_hash",
            "llama_cpp_build",
            "constrained_mode",
            "reasoning_effort",
            "reasoning_budget_tokens",
            "temperature",
            "max_attempts",
            "scope_enabled",
            "scope_max_tokens",
            "extract_max_tokens",
            "derive_max_tokens",
            "list_batch_max_units",
            "list_context_max_units",
        ]

    def test_frozen_with_slots(self):
        components = _components()

        with pytest.raises(dataclasses.FrozenInstanceError):
            components.model_hash = "other"  # type: ignore[misc]
        assert not hasattr(components, "__dict__")

    def test_the_code_side_subset(self):
        assert CODE_IDENTITY_FIELDS == (
            "prompt_sha256",
            "code_sha256",
            "ontology_version",
            "scope_max_tokens",
            "extract_max_tokens",
            "derive_max_tokens",
            "list_batch_max_units",
            "list_context_max_units",
        )


class TestComputeRunIdentity:
    def test_format_is_ri1_and_a_sha256(self):
        assert re.fullmatch(r"ri1-[0-9a-f]{64}", compute_run_identity(_components()))

    def test_deterministic(self):
        assert compute_run_identity(_components()) == compute_run_identity(_components())

    def test_hashes_the_canonical_json_of_the_components(self):
        components = _components()
        canonical = json.dumps(
            dataclasses.asdict(components), sort_keys=True, separators=(",", ":")
        )

        assert compute_run_identity(components) == (
            "ri1-" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        )

    @pytest.mark.parametrize("name", _FIELD_NAMES)
    def test_each_component_moves_the_identity(self, name):
        base = _components()
        moved = dataclasses.replace(base, **{name: _changed(getattr(base, name))})

        message = f"changing {name} left the run identity unchanged"
        assert compute_run_identity(moved) != compute_run_identity(base), message

    def test_a_none_and_a_zero_are_different_identities(self):
        assert compute_run_identity(_components(reasoning_budget_tokens=None)) != (
            compute_run_identity(_components(reasoning_budget_tokens=0))
        )


class TestCodeIdentity:
    def test_format_is_ci1_and_a_sha256(self):
        assert re.fullmatch(r"ci1-[0-9a-f]{64}", code_identity(_components()))

    def test_differs_from_the_run_identity(self):
        components = _components()

        assert code_identity(components)[4:] != compute_run_identity(components)[4:]

    @pytest.mark.parametrize("name", _FIELD_NAMES)
    def test_moves_for_code_fields_and_only_those(self, name):
        base = _components()
        moved = dataclasses.replace(base, **{name: _changed(getattr(base, name))})

        if name in CODE_IDENTITY_FIELDS:
            assert code_identity(moved) != code_identity(base), name
        else:
            message = f"{name} is deployment configuration, not code; it must not move it"
            assert code_identity(moved) == code_identity(base), message


# ---------------------------------------------------------------------------
# Source normalisation and the code hash
# ---------------------------------------------------------------------------

# A fixed sample, f-strings included. Line 11 ends in three spaces of trailing
# whitespace, and line 21 in two that sit inside a string literal (content, so
# kept); both are written as escapes so no editor or hook strips them.
_SAMPLE_SOURCE = "\n".join(
    [
        '"""Module docstring."""',
        "",
        "# a comment line",
        "import os",
        "",
        "",
        "def greet(name: str) -> str:  # trailing comment",
        '    """Say hello.',
        "",
        '    Two lines."""',
        '    greeting = f"hello {name!r} # not a comment"\x20\x20\x20',
        "    return greeting + f\"{'#'}\" + os.sep",
        "",
        "",
        "class Box:",
        '    """A box."""',
        "",
        "    size = 3",
        "",
        "    def area(self):",
        '        text = """keep\x20\x20',
        "",
        '   this blank line and these spaces   """',
        "        return self.size**2, text",
        "",
    ]
)

_SAMPLE_NORMALIZED = "\n".join(
    [
        "import os",
        "def greet(name: str) -> str:",
        '    greeting = f"hello {name!r} # not a comment"',
        "    return greeting + f\"{'#'}\" + os.sep",
        "class Box:",
        "    size = 3",
        "    def area(self):",
        '        text = """keep\x20\x20',
        "",
        '   this blank line and these spaces   """',
        "        return self.size**2, text",
    ]
)

# sha256 of the sample, as `code_sha256` hashes it under the module name
# `run_identity_sample`. If this moves, the hashing changed: every recorded
# run identity moves with it, so re-pin only on purpose, together with a new
# CODE_REVISIONS entry.
_SAMPLE_PINNED_SHA256 = "1fdca017c7dbc89560ce150cd2109aed17d01aeb6832369231c6dee50410ff92"


class TestNormalizedSource:
    def test_the_pinned_sample_normalises_to_the_expected_text(self):
        assert normalized_source(_SAMPLE_SOURCE) == _SAMPLE_NORMALIZED

    def test_the_pinned_sample_hash(self, tmp_path):
        module = _module_from_source(tmp_path, "run_identity_sample", _SAMPLE_SOURCE)

        assert code_sha256([module]) == _SAMPLE_PINNED_SHA256

    def test_a_comment_inside_a_string_is_kept(self):
        assert "# not a comment" in normalized_source(_SAMPLE_SOURCE)

    def test_crlf_line_endings_normalise_like_lf(self):
        assert normalized_source(_SAMPLE_SOURCE.replace("\n", "\r\n")) == _SAMPLE_NORMALIZED

    def test_a_non_ascii_docstring_is_removed_at_the_right_columns(self):
        source = 'def f():\n    """Café — déjà vu."""; x = 1\n    return 2\n'

        assert normalized_source(source) == "def f():\n    ; x = 1\n    return 2"


def _engine_source() -> str:
    return Path(engine.__file__).read_text(encoding="utf-8")


class TestCodeSha256:
    """Edits to a covered module, made on the real engine source."""

    def _hash(self, tmp_path: Path, source: str) -> str:
        return code_sha256([_module_from_source(tmp_path, engine.__name__, source)])

    def test_the_real_engine_hashes_as_its_source(self, tmp_path):
        assert code_sha256([engine]) == self._hash(tmp_path, _engine_source())

    def test_a_comment_only_edit_does_not_move_it(self, tmp_path):
        source = _engine_source()
        edited = source.replace(
            "class ExtractionEngine:",
            "# A comment added above the class.\nclass ExtractionEngine:  # and one here",
            1,
        )
        assert edited != source

        assert self._hash(tmp_path, edited) == self._hash(tmp_path, source)

    def test_a_docstring_only_edit_does_not_move_it(self, tmp_path):
        source = _engine_source()
        edited = source.replace(
            '"""Initialize the engine.', '"""Initialize the engine, reworded.\n\n        More.', 1
        )
        assert edited != source

        assert self._hash(tmp_path, edited) == self._hash(tmp_path, source)

    def test_blank_lines_and_trailing_whitespace_do_not_move_it(self, tmp_path):
        source = _engine_source()
        edited = source.replace("\n\n", "\n\n\n").replace(
            "logger = logging.getLogger(__name__)",
            "logger = logging.getLogger(__name__)\x20\x20",
            1,
        )
        assert edited != source

        assert self._hash(tmp_path, edited) == self._hash(tmp_path, source)

    @pytest.mark.parametrize(
        ("old", "new"),
        [
            pytest.param("SCOPE_MAX_TOKENS = 96", "SCOPE_MAX_TOKENS = 97", id="a-constant"),
            pytest.param(
                "for attempt in range(1, self._settings.max_attempts + 1):",
                "for attempt in range(0, self._settings.max_attempts + 1):",
                id="a-loop-bound",
            ),
            pytest.param("JSON object", "JSON  object", id="text-inside-a-string"),
            pytest.param(
                "logger = logging.getLogger(__name__)",
                "logger = logging.getLogger(__name__)\nEXTRA = 1",
                id="an-added-statement",
            ),
        ],
    )
    def test_a_code_edit_moves_it(self, tmp_path, old, new):
        source = _engine_source()
        assert old in source

        assert self._hash(tmp_path, source.replace(old, new, 1)) != self._hash(tmp_path, source)

    def test_order_and_names_are_part_of_the_hash(self, tmp_path):
        a = _module_from_source(tmp_path, "mod_a", "x = 1\n")
        b = _module_from_source(tmp_path, "mod_b", "y = 2\n")
        renamed = _module_from_source(tmp_path, "mod_c", "x = 1\n")

        assert code_sha256([a, b]) != code_sha256([b, a])
        assert code_sha256([a]) != code_sha256([renamed])

    def test_a_function_is_hashed_from_its_own_source(self):
        before = code_sha256([engine.parse_recorded_at])

        assert re.fullmatch(r"[0-9a-f]{64}", before)
        assert before != code_sha256([engine._json_type])


# ---------------------------------------------------------------------------
# Coverage: every backend module the service's engine and adapters import from
# ---------------------------------------------------------------------------

_SERVICE_SOURCES = (
    _BACKEND_ROOT / "extraction_service" / "engine.py",
    _BACKEND_ROOT / "extraction_service" / "adapters.py",
)


def _is_backend_module(dotted: str) -> bool:
    path = _REPO_ROOT.joinpath(*dotted.split("."))
    return path.with_suffix(".py").is_file() or path.is_dir()


def _backend_imports(source: str) -> set[str]:
    """The `backend.*` modules `source` imports, or imports names from.

    `from backend.pkg import mod` counts as `backend.pkg.mod` when `mod` is a
    module, so a submodule imported by name is not mistaken for its package.
    """
    modules: set[str] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            modules.update(a.name for a in node.names if a.name.startswith("backend."))
        elif isinstance(node, ast.ImportFrom):
            assert node.level == 0, "relative imports are not resolved by this guard"
            if node.module is None or not node.module.startswith("backend."):
                continue
            for alias in node.names:
                candidate = f"{node.module}.{alias.name}"
                modules.add(candidate if _is_backend_module(candidate) else node.module)
    return modules


def _listed() -> set[str]:
    covered = {module.__name__ for module in engine.RUN_IDENTITY_CODE}
    excluded = {name for name, _reason in engine.RUN_IDENTITY_EXCLUDED}
    return covered | excluded


def _unlisted(source: str) -> list[str]:
    return sorted(_backend_imports(source) - _listed())


def _service_source() -> str:
    return "\n".join(path.read_text(encoding="utf-8") for path in _SERVICE_SOURCES)


class TestRunIdentityCoverage:
    def test_every_backend_import_is_covered_or_excluded(self):
        assert _unlisted(_service_source()) == [], (
            "engine.py or adapters.py imports from a backend module that is in neither "
            "RUN_IDENTITY_CODE nor RUN_IDENTITY_EXCLUDED (backend/extraction_service/engine.py). "
            "Cover it if it can change what the service returns; exclude it, with the reason, "
            "if it cannot."
        )

    def test_the_guard_fails_on_an_unlisted_import(self):
        fake_import = "from backend.knowledge.storage.graph_store import GraphStore"
        source = _service_source() + "\n" + fake_import + "\n"

        assert _unlisted(source) == ["backend.knowledge.storage.graph_store"]

    def test_the_guard_sees_a_plain_import_and_a_submodule_import(self):
        source = (
            "import backend.knowledge.curation.graph_writer\n"
            "from backend.knowledge import config\n"
        )

        assert _unlisted(source) == [
            "backend.knowledge.config",
            "backend.knowledge.curation.graph_writer",
        ]

    def test_the_covered_modules(self):
        assert [module.__name__ for module in engine.RUN_IDENTITY_CODE] == [
            "backend.extraction_service.engine",
            "backend.extraction_service.adapters",
            "backend.extraction_service.schemas",
            "backend.knowledge.extraction.internal_derivation",
            "backend.knowledge.extraction.internal_prompts",
            "backend.knowledge.extraction.ontology_extractor",
            "backend.knowledge.extraction.preprocessor",
            "backend.knowledge.extraction.prompts",
            "backend.knowledge.extraction.scope_classifier",
            "backend.knowledge.extraction.structure_splitter",
            "backend.llm.llama_server_provider",
        ]

    def test_the_excluded_modules_each_carry_a_reason(self):
        excluded = dict(engine.RUN_IDENTITY_EXCLUDED)

        assert sorted(excluded) == [
            "backend.errors",
            "backend.extraction_contract.models",
            "backend.extraction_service.settings",
            "backend.knowledge.version_stamps",
            "backend.llm.models",
            "backend.llm.provider",
        ]
        assert all(reason.strip() for reason in excluded.values())

    def test_no_exclusion_is_stale(self):
        imported = _backend_imports(_service_source())

        assert [name for name, _ in engine.RUN_IDENTITY_EXCLUDED if name not in imported] == []

    def test_version_stamps_is_never_covered(self):
        """Covering it would make the drift guard unsatisfiable.

        `version_stamps` holds CODE_REVISIONS: adding the entry the guard asks
        for would move the code identity that entry has to name.
        """
        assert version_stamps not in engine.RUN_IDENTITY_CODE

    def test_run_identity_module_imports_only_the_standard_library(self):
        tree = ast.parse(Path(run_identity.__file__).read_text(encoding="utf-8"))
        imported = {
            (node.module if isinstance(node, ast.ImportFrom) else alias.name).split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import | ast.ImportFrom)
            for alias in node.names
        }

        assert imported <= {
            "__future__",
            "ast",
            "collections",
            "dataclasses",
            "hashlib",
            "inspect",
            "io",
            "json",
            "textwrap",
            "tokenize",
            "types",
        }, imported


# ---------------------------------------------------------------------------
# The drift guard
# ---------------------------------------------------------------------------


def _production_code_identity() -> str:
    """The code identity of the service as it ships: default settings, default adapter."""
    settings = ServiceSettings(llm_base_url="http://unused", model_hash="unused", model_file="u")
    adapter = get_adapter(
        settings.adapter_name,
        reasoning_effort=settings.reasoning_effort,
        reasoning_budget_tokens=settings.reasoning_budget_tokens,
    )
    return code_identity(run_identity_components(settings, adapter))


class TestCodeRevisions:
    def test_code_revision_is_frozen_with_slots(self):
        revision = CODE_REVISIONS[-1]

        assert isinstance(revision, CodeRevision)
        with pytest.raises(dataclasses.FrozenInstanceError):
            revision.reason = "edited"  # type: ignore[misc]
        assert not hasattr(revision, "__dict__")

    def test_entries_are_dated_oldest_first_with_a_reason(self):
        assert CODE_REVISIONS, "CODE_REVISIONS is empty"
        dates = [revision.date for revision in CODE_REVISIONS]
        assert all(re.fullmatch(r"\d{4}-\d{2}-\d{2}", date) for date in dates), dates
        assert dates == sorted(dates)
        for revision in CODE_REVISIONS:
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}-r\d+", revision.extraction_version)
            assert re.fullmatch(r"ci1-[0-9a-f]{64}", revision.code_identity)
            assert revision.reason.strip()

    def test_no_code_identity_is_recorded_twice(self):
        identities = [revision.code_identity for revision in CODE_REVISIONS]

        assert len(identities) == len(set(identities))

    def test_the_newest_revision_names_the_current_version_and_code(self):
        newest = CODE_REVISIONS[-1]
        current = _production_code_identity()

        assert (newest.extraction_version, newest.code_identity) == (EXTRACTION_VERSION, current), (
            "The extraction code changed without a CODE_REVISIONS entry.\n"
            f"  current:  EXTRACTION_VERSION={EXTRACTION_VERSION!r} code_identity={current!r}\n"
            f"  recorded: extraction_version={newest.extraction_version!r} "
            f"code_identity={newest.code_identity!r}\n"
            "A prompt, a per-stage token limit, the ontology version or a module in "
            "RUN_IDENTITY_CODE moved. Record the change in backend/knowledge/version_stamps.py, "
            "one of two ways:\n"
            "  1. New turns only: append a CodeRevision under the same extraction_version "
            f"({EXTRACTION_VERSION!r}) with code_identity={current!r}, today's date and the "
            "reason. Turns already extracted keep their results.\n"
            "  2. Re-extract history at the next cutover: bump EXTRACTION_VERSION (with its "
            "history line), then append a CodeRevision under the new version with "
            f"code_identity={current!r}."
        )
