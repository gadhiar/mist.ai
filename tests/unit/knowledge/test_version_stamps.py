"""Structural guard: the version stamps have exactly one authority each.

Six sites used to restate `ontology_version` / `extraction_version`, and they
disagreed (1.1.0, 1.2.1, 1.4.0 live at once). The stamps are descriptive, so
the disagreement was invisible -- until `extraction_cache.cache_key`, which
hashes `event_id|ontology_version|extraction_version|model_hash` and turns a
mislabel into a hard cache miss that makes a deterministic rebuild impossible.

The collapse: the ontology stamp is DERIVED from the ontology object itself,
the extraction stamp lives in `backend.knowledge.version_stamps`, and neither is
env-configurable. These tests pin that shape so a future literal cannot creep
back in unnoticed -- `TestNoBackendModuleRestatesAStamp` scans the source, which
is what catches a reintroduction that an equality assertion would happily pass.
"""

from __future__ import annotations

import ast
import os
import re
from contextlib import contextmanager
from pathlib import Path

from backend.knowledge import version_stamps
from backend.knowledge.config import KnowledgeConfig
from backend.knowledge.ontologies.v1_0_0 import ONTOLOGY_V1_0_0
from backend.knowledge.version_stamps import EXTRACTION_VERSION, ONTOLOGY_VERSION

# `backend` is a namespace package (no __init__.py), so `backend.__file__` is
# None -- anchor on a module that has a real file instead.
_VERSION_STAMPS_MODULE = Path(version_stamps.__file__).resolve()
_BACKEND_ROOT = _VERSION_STAMPS_MODULE.parents[1]

# Matches an ontology-version stamp assigned a quoted literal, in Python source
# or inside an embedded Cypher string (`e.ontology_version = '1.0.0'`).
_CYPHER_STAMP_LITERAL_RE = re.compile(r"ontology_version\s*=\s*['\"]\d")

# Stamp-shaped literals. Bound names alone are too coarse -- a value subtree
# legitimately contains the string "ontology_version" (a dict key, a row
# lookup), so only version-SHAPED strings count as a restatement.
_STAMP_SHAPES = {
    "ontology_version": re.compile(r"^\d+\.\d+\.\d+"),
    "extraction_version": re.compile(r"^\d{4}-\d{2}-\d{2}-r\d+"),
}


@contextmanager
def _env(**values):
    original = {k: os.environ.get(k) for k in values}
    try:
        for k, v in values.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        yield
    finally:
        for k, v in original.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _backend_sources() -> list[Path]:
    return sorted(p for p in _BACKEND_ROOT.rglob("*.py") if "__pycache__" not in p.parts)


def _string_literals_under(node: ast.AST) -> list[str]:
    """Every string constant reachable from `node`, including inside a ternary."""
    return [
        child.value
        for child in ast.walk(node)
        if isinstance(child, ast.Constant) and isinstance(child.value, str)
    ]


def _named_bindings(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    """Every (bound name, value node) pair a module establishes.

    Covers module/class attributes, keyword arguments, dict entries keyed by a
    string, and function parameter defaults -- the four shapes a restated stamp
    took across the six original authorities.
    """
    bindings: list[tuple[str, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign) and node.value is not None:
            if isinstance(node.target, ast.Name):
                bindings.append((node.target.id, node.value))
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    bindings.append((target.id, node.value))
        elif isinstance(node, ast.keyword) and node.arg is not None:
            bindings.append((node.arg, node.value))
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values):
                if isinstance(key, ast.Constant) and isinstance(key.value, str):
                    bindings.append((key.value, value))
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            args = node.args
            positional = args.posonlyargs + args.args
            for arg, default in zip(
                positional[len(positional) - len(args.defaults) :], args.defaults
            ):
                bindings.append((arg.arg, default))
            for arg, default in zip(args.kwonlyargs, args.kw_defaults):
                if default is not None:
                    bindings.append((arg.arg, default))
    return bindings


def _literal_stamps(path: Path, name_fragment: str) -> list[tuple[str, str]]:
    """Return (bound name, literal) for every stamp-shaped literal bound to the stamp."""
    shape = _STAMP_SHAPES[name_fragment]
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        (name, literal)
        for name, value in _named_bindings(tree)
        if name_fragment in name.lower()
        for literal in _string_literals_under(value)
        if shape.match(literal)
    ]


class TestOntologyStampIsDerived:
    """The ontology stamp reads the ontology object, so it cannot disagree."""

    def test_module_constant_is_the_ontology_objects_own_version(self):
        assert ONTOLOGY_VERSION is ONTOLOGY_V1_0_0.version

    def test_fresh_config_stamps_the_active_ontology_version(self):
        config = KnowledgeConfig.from_env()

        assert config.ontology_version == ONTOLOGY_V1_0_0.version

    def test_dataclass_default_stamps_the_active_ontology_version(self):
        assert KnowledgeConfig.ontology_version == ONTOLOGY_V1_0_0.version


class TestStampsAreNotEnvConfigurable:
    """A version that describes CODE cannot be pinned per deployment.

    The live drift was a `.env` pinning ONTOLOGY_VERSION=1.2.1 and
    EXTRACTION_VERSION=2026-06-12-r1 against code running 1.4.0 / r5. Removing
    the reads closes that hole structurally rather than by convention.
    """

    def test_ontology_version_env_var_is_ignored(self):
        with _env(ONTOLOGY_VERSION="9.9.9"):
            config = KnowledgeConfig.from_env()

        assert config.ontology_version == ONTOLOGY_V1_0_0.version

    def test_extraction_version_env_var_is_ignored(self):
        with _env(EXTRACTION_VERSION="2027-01-01-r3"):
            config = KnowledgeConfig.from_env()

        assert config.extraction_version == EXTRACTION_VERSION

    def test_model_hash_remains_env_configurable(self):
        """model_hash names a deployed model file, which genuinely varies."""
        with _env(MIST_MODEL_HASH="custom-llama-7b-v2"):
            config = KnowledgeConfig.from_env()

        assert config.model_hash == "custom-llama-7b-v2"


class TestNoBackendModuleRestatesAStamp:
    """Source-level guard: reintroducing a literal fails here.

    An equality assertion cannot catch this -- re-adding `ontology_version =
    "1.4.0"` still equals the derived value today and only diverges at the next
    bump, which is exactly when it does damage. Scanning the source catches it
    on the commit that adds it.
    """

    def test_no_module_binds_a_literal_ontology_version(self):
        offenders = [
            (path.relative_to(_BACKEND_ROOT).as_posix(), name, literal)
            for path in _backend_sources()
            for name, literal in _literal_stamps(path, "ontology_version")
        ]

        assert offenders == [], (
            "Ontology version restated as a literal. It must be derived from "
            "ONTOLOGY_V1_0_0.version via backend/knowledge/version_stamps.py, "
            f"so the stamp cannot disagree with the ontology in use: {offenders}"
        )

    def test_only_version_stamps_binds_a_literal_extraction_version(self):
        offenders = [
            (path.relative_to(_BACKEND_ROOT).as_posix(), name, literal)
            for path in _backend_sources()
            if path != _VERSION_STAMPS_MODULE
            for name, literal in _literal_stamps(path, "extraction_version")
        ]

        assert offenders == [], (
            "Extraction version restated outside its single home. Import "
            "EXTRACTION_VERSION from backend/knowledge/version_stamps.py "
            f"instead: {offenders}"
        )

    def test_no_embedded_cypher_hardcodes_an_ontology_version(self):
        offenders = [
            f"{path.relative_to(_BACKEND_ROOT).as_posix()}:{lineno}"
            for path in _backend_sources()
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if _CYPHER_STAMP_LITERAL_RE.search(line)
        ]

        assert offenders == [], (
            "Cypher stamps an ontology-version literal. Bind it as a query "
            f"parameter sourced from version_stamps instead: {offenders}"
        )
