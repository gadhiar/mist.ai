"""Structural guard: the version stamps have exactly one authority each.

Six sites used to restate `ontology_version` / `extraction_version`, and they
disagreed (1.1.0, 1.2.1, 1.4.0 live at once). The stamps are descriptive, so a
disagreement is invisible to every pipeline branch. For `extraction_version` it
is still caught downstream anyway: `extraction_cache.cache_key` hashes it
(currently `event_id|extraction_version|model_hash`;
verified via `grep -n 'raw = "|".join' backend/knowledge/extraction_cache.py`),
turning a mislabel into a hard cache miss that makes a deterministic rebuild
impossible. As of `extraction-cache-phase-1` spec D3, `ontology_version` is
deliberately OUT of that key, so a disagreement there is a silent mislabel
with no such downstream catch -- this scan is now the ONLY thing that catches
it.

The collapse: the ontology stamp is DERIVED from the ontology object itself,
the extraction stamp lives in `backend.knowledge.version_stamps`, and neither is
env-configurable. These tests pin that shape so a future literal is caught at the
commit that adds it -- `TestNoBackendModuleRestatesAStamp` scans the source,
which catches a reintroduction that an equality assertion would happily pass.

The scan is a sieve, not a proof. It recognises the binding shapes enumerated on
`_named_bindings`, and `TestGuardSeesEveryBindingShape` pins each of them against
a source snippet, because a scan that has gone blind reports the same clean
result as a codebase that is actually clean. What the sieve does not catch is
recorded on `_named_bindings` too -- read that list before trusting a green run.
"""

from __future__ import annotations

import ast
import os
import re
from contextlib import contextmanager
from pathlib import Path

import pytest

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


def _folded_string(node: ast.AST) -> str | None:
    """The string `node` evaluates to, when that is decidable from source alone.

    A bare walk for `ast.Constant` sees `"1.4" + ".0"` as two fragments, neither
    of which matches the version shape, so splitting a literal was enough to
    walk past the guard. Folding the PURE cases -- concatenation, an f-string of
    constants, `.format()` with constant arguments -- reassembles the value and
    puts it back in front of the shape check.

    Anything with a runtime component folds to None deliberately. A value the
    guard cannot decide statically is a value read from somewhere else, which is
    the DERIVED shape the guard exists to permit: `f"{ONTOLOGY_VERSION}"` must
    stay legal.
    """
    if isinstance(node, ast.Constant):
        return node.value if isinstance(node.value, str) else None
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _folded_string(node.left)
        right = _folded_string(node.right)
        return None if left is None or right is None else left + right
    if isinstance(node, ast.JoinedStr):
        parts: list[str] = []
        for part in node.values:
            if isinstance(part, ast.FormattedValue):
                # `!r` / `!s` and a format spec are foldable in principle, but
                # both are noise on a version stamp; treating them as undecidable
                # keeps this helper free of format-language reimplementation.
                if part.conversion != -1 or part.format_spec is not None:
                    return None
                if not isinstance(part.value, ast.Constant):
                    return None
                parts.append(str(part.value.value))
                continue
            folded = _folded_string(part)
            if folded is None:
                return None
            parts.append(folded)
        return "".join(parts)
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        if node.func.attr != "format":
            return None
        template = _folded_string(node.func.value)
        if template is None:
            return None
        if not all(isinstance(arg, ast.Constant) for arg in node.args):
            return None
        if not all(kw.arg and isinstance(kw.value, ast.Constant) for kw in node.keywords):
            return None
        try:
            return template.format(
                *(arg.value for arg in node.args),
                **{kw.arg: kw.value.value for kw in node.keywords},  # type: ignore[misc]
            )
        except (IndexError, KeyError, ValueError):
            return None
    return None


def _candidate_strings(node: ast.AST) -> list[str]:
    """Every string `node` could evaluate to, by reachability and by folding.

    Walks rather than folding only the root so a literal buried in a ternary
    branch still counts, which is how the original six authorities hid one.
    """
    candidates: list[str] = []
    for child in ast.walk(node):
        folded = _folded_string(child)
        if folded is not None and folded not in candidates:
            candidates.append(folded)
    return candidates


def _target_names(target: ast.AST) -> list[str]:
    """The stamp-relevant name a single assignment target binds, if any.

    `ast.Subscript` covers `params["ontology_version"] = ...`, the house idiom
    for stamping Cypher params -- live at `admin.py` and `graph_writer.py` (x3),
    two of them lines the version-stamp collapse itself rewrote, so it is the
    likeliest route for a literal to return. A non-constant key (`params[KEY]`)
    yields nothing: resolving it needs constant propagation, which this guard
    does not do.
    """
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Attribute):
        return [target.attr]
    if isinstance(target, ast.Subscript):
        key = target.slice
        if isinstance(key, ast.Constant) and isinstance(key.value, str):
            return [key.value]
        return []
    if isinstance(target, ast.Starred):
        return _target_names(target.value)
    return []


def _bind_target(target: ast.AST, value: ast.AST, bindings: list[tuple[str, ast.AST]]) -> None:
    """Record every (name, value) pair `target = value` establishes.

    Unpacking pairs element-wise when both sides are sequences of equal length,
    so `self.ontology_version, _ = "1.4.0", None` attributes the literal to the
    stamp and not to `_`. Otherwise every name binds the whole right-hand side,
    which over-attributes rather than under-attributes -- the safe direction for
    a guard.
    """
    if isinstance(target, ast.Tuple | ast.List):
        if isinstance(value, ast.Tuple | ast.List) and len(value.elts) == len(target.elts):
            for element, element_value in zip(target.elts, value.elts):
                _bind_target(element, element_value, bindings)
        else:
            for element in target.elts:
                _bind_target(element, value, bindings)
        return
    for name in _target_names(target):
        bindings.append((name, value))


def _named_bindings(tree: ast.AST) -> list[tuple[str, ast.AST]]:
    """Every (bound name, value node) pair a module establishes.

    Covered, each pinned by a case in `TestGuardSeesEveryBindingShape`:

    - assignment targets -- plain, annotated, augmented, and walrus -- where the
      target is a bare name, an attribute (`self.ontology_version`), a
      string-keyed subscript (`params["ontology_version"]`), or a tuple/list
      unpacking of those
    - keyword arguments and function parameter defaults, positional and
      keyword-only
    - dict entries keyed by a string constant

    NOT covered, and left uncovered on purpose (see the report on the commit
    that added this list): any binding whose NAME or VALUE needs constant
    propagation to resolve -- `params[KEY] = "1.4.0"`, or a literal parked in
    `VERSIONS = ["1.4.0"]` and read back as `VERSIONS[0]`. Both need a symbol
    table this guard does not build. `TestNoBackendModuleRestatesAStamp` is
    therefore a guard against a literal creeping back, not a proof that none can.
    """
    bindings: list[tuple[str, ast.AST]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.AnnAssign | ast.AugAssign | ast.NamedExpr):
            if node.value is not None:
                _bind_target(node.target, node.value, bindings)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                _bind_target(target, node.value, bindings)
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


def _literal_stamps_in_source(source: str, name_fragment: str) -> list[tuple[str, str]]:
    """Return (bound name, literal) for every stamp-shaped literal bound to the stamp.

    Split out from `_literal_stamps` so `TestGuardSeesEveryBindingShape` can drive
    the guard from a source string. The synthetic cases must exercise THIS
    function rather than a reimplementation of it -- a shape test that parses its
    own way would keep passing while the real scan went blind, which is the
    failure this whole module exists to prevent.
    """
    shape = _STAMP_SHAPES[name_fragment]
    tree = ast.parse(source)
    return [
        (name, literal)
        for name, value in _named_bindings(tree)
        if name_fragment in name.lower()
        for literal in _candidate_strings(value)
        if shape.match(literal)
    ]


def _literal_stamps(path: Path, name_fragment: str) -> list[tuple[str, str]]:
    """Return (bound name, literal) for every stamp-shaped literal bound to the stamp."""
    return _literal_stamps_in_source(path.read_text(encoding="utf-8"), name_fragment)


class TestGuardSeesEveryBindingShape:
    """The guard's own coverage, pinned shape by shape.

    Without this class the guard's reach was itself untested. `_named_bindings`
    grew `ast.Subscript` and `ast.Attribute` targets on 2026-08-03 and shipped
    with no test: deleting BOTH branches again left the file at 9 passed, so the
    extension was a green mutation -- present in the source, absent from the
    suite, and free to be refactored away by anyone who read the branches as
    dead. The scan below (`TestNoBackendModuleRestatesAStamp`) cannot cover this
    gap, because it only proves that CLEAN source produces no offenders, which a
    guard that sees nothing at all also does.

    Each case is a source string driven through `_literal_stamps_in_source`, the
    same function the real scan calls.
    """

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param('ontology_version = "1.4.0"', id="name-target"),
            pytest.param('ontology_version: str = "1.4.0"', id="annotated-name-target"),
            pytest.param('params["ontology_version"] = "1.4.0"', id="subscript-target"),
            pytest.param('self.ontology_version = "1.4.0"', id="attribute-target"),
            pytest.param('self.ontology_version: str = "1.4.0"', id="annotated-attribute-target"),
            pytest.param('write(ontology_version="1.4.0")', id="keyword-argument"),
            pytest.param('params = {"ontology_version": "1.4.0"}', id="dict-entry"),
            pytest.param('def write(ontology_version="1.4.0"): pass', id="parameter-default"),
            pytest.param('def write(*, ontology_version="1.4.0"): pass', id="kwonly-default"),
            pytest.param('ontology_version = other if other else "1.4.0"', id="ternary-branch"),
            pytest.param('if (ontology_version := "1.4.0"): pass', id="walrus-target"),
            pytest.param('self.ontology_version, _ = "1.4.0", None', id="tuple-unpack-target"),
            pytest.param('params["ontology_version"] += "1.4.0"', id="augmented-subscript-target"),
            pytest.param('params["ontology_version"] = "1.4" + ".0"', id="split-concatenation"),
            pytest.param('params["ontology_version"] = f"1.{4}.0"', id="split-fstring"),
            pytest.param(
                'params["ontology_version"] = "{}.{}.{}".format(1, 4, 0)',
                id="split-format-call",
            ),
        ],
    )
    def test_rejects_a_hardcoded_stamp_bound_by(self, source: str):
        assert _literal_stamps_in_source(source, "ontology_version") != [], (
            f"The guard is blind to this binding shape, so a hardcoded stamp "
            f"written this way reaches main unnoticed: {source!r}"
        )

    @pytest.mark.parametrize(
        "source",
        [
            pytest.param('params["ontology_version"] = ontology_version', id="local-passthrough"),
            pytest.param(
                'params["ontology_version"] = self._rebuild_stamps.ontology_version',
                id="rebuild-stamps-passthrough",
            ),
            pytest.param('params["ontology_version"] = ONTOLOGY_VERSION', id="authority-import"),
            pytest.param(
                'params["ontology_version"] = f"{ONTOLOGY_VERSION}"',
                id="authority-through-fstring",
            ),
            pytest.param("self._ontology_version = ontology_version", id="attribute-passthrough"),
            pytest.param(
                'self.ontology_version, other = ontology_version, "1.4.0"',
                id="tuple-unpack-attributes-element-wise",
            ),
            pytest.param('params["ontology_version"] = "seed"', id="non-version-shaped-literal"),
            pytest.param(
                'query = "SET e.ontology_version = $ontology_version"',
                id="cypher-parameter-reference",
            ),
        ],
    )
    def test_allows_a_stamp_sourced_from_the_authority(self, source: str):
        assert _literal_stamps_in_source(source, "ontology_version") == [], (
            f"The guard flagged a stamp that is DERIVED, not restated. Widening "
            f"it this far would make the real scan fail on correct code: {source!r}"
        )

    def test_the_path_entry_point_sees_a_subscript_literal(self, tmp_path: Path):
        """The file-reading entry point, not just the source-string one.

        `_literal_stamps` is what the backend scan actually calls. Proving the
        shape only through `_literal_stamps_in_source` would leave the read-and-
        parse step unproven.
        """
        module = tmp_path / "writer.py"
        module.write_text('params["ontology_version"] = "1.4.0"\n', encoding="utf-8")

        assert _literal_stamps(module, "ontology_version") == [("ontology_version", "1.4.0")]


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
