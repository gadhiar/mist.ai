"""Mechanism A: composition-root completeness check for `backend.server.lifespan`.

The defect class this exists to catch: a factory grows an optional parameter
whose default is falsy, the factory itself is correct and tested, and the ONE
call site in the composition root never passes it. Nothing errors, nothing
warns, every test stays green, and a feature is dead in production.

The live instance that motivated this guard (2026-08-03) was
`build_curation_scheduler(knowledge_config)` at `backend/server.py:444`,
which left `event_store`, `tracker` and `llm_provider` at their `None`
defaults. `SelfReflectionJob` returned zeros on its first line for the life
of every process, and `SkillDerivationJob` ran `detect_patterns()` against a
freshly-built `ToolUsageTracker` while `.record()` calls landed on the
DIFFERENT instance `build_conversation_handler` wired into
`ConversationHandler`. Two of the scheduler's nine jobs were structurally
incapable of producing output.

WHY THE ASSERTION POINTS AT THE CALL SITE, NOT THE FACTORY
----------------------------------------------------------
`backend/factories.py` was not wrong. `build_curation_scheduler` accepted
every dependency it needed and wired each one correctly. A guard aimed at the
factories -- signature checks, "every job gets its collaborator" checks --
would have passed on the defective build. The only place the defect was
visible was the argument list in `lifespan()`, so that is what this file
parses.

WHAT THIS GUARD COVERS
----------------------
Exactly one thing: `ast.Call` nodes lexically inside the `lifespan` function
in `backend/server.py`, whose callee is a bare `ast.Name` beginning with
`build_` or `_build_`. For each, every parameter of the resolved callee whose
default is falsy (`None`, `False`, `0`, `""`, an empty collection) must be
either supplied at that call site -- positionally or by keyword -- or listed
in `EXEMPTIONS` with a written justification.

WHAT THIS GUARD DOES NOT COVER (read this before trusting it)
-------------------------------------------------------------
- Builder calls ANYWHERE ELSE. Not elsewhere in `server.py`, and not in any
  other module. Notably `build_conversation_handler` is invoked from the
  VoiceProcessor -> ModelManager -> KnowledgeIntegration chain, not from
  `lifespan`, and is therefore invisible here.
- Attribute-form calls (`factories.build_x(...)`) and any call reached
  through indirection: a builder bound to a local variable, `getattr`, a
  dispatch dict. Only bare-name calls are matched.
- Parameters with non-falsy defaults. A factory that defaults to a real but
  WRONG collaborator passes this guard.
- RUNTIME VALUES. This is the important one. `f(event_store=x)` satisfies the
  guard whether `x` holds a live object or `None`. The guard proves the call
  site MENTIONS the parameter; it cannot prove a live object arrives, and it
  cannot prove the object is the RIGHT one (the `tracker` half of the 2026-08-03
  defect was a wrong-instance bug, which a supplied-ness check can never see).
  Those two properties are covered separately, by asserting on the constructed
  object graph, in `tests/unit/test_curation_scheduler_wiring.py`.
- `*args` / `**kwargs` unpacking at a call site. These make supplied-ness
  statically undecidable; the guard FAILS rather than guessing.

Signature resolution is done by importing `backend.server` and looking each
callee name up in that module's namespace, then `inspect.signature`. That
resolves the exact object the call site binds at runtime -- through the
`from factories import ...` re-export, through any aliasing, and through
decorators -- which an AST-only resolver would have to reimplement import
machinery to match. `from backend import server` is already established
practice in this suite (`tests/unit/test_server_catchup_lifecycle.py`).
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass
from pathlib import Path

import pytest

from backend import server

# Minimum length of an exemption justification. Long enough that "n/a",
# "ok", or a bare ticket id cannot satisfy it.
_MIN_REASON_CHARS = 60

_LIFESPAN_NAME = "lifespan"
_BUILDER_PREFIXES = ("build_", "_build_")

_SERVER_PATH = Path(inspect.getsourcefile(server))
_SERVER_REL = "backend/server.py"


@dataclass(frozen=True, slots=True)
class Exemption:
    """One (callee, parameter) pair the completeness check may skip.

    `reason` is validated at construction, so `EXEMPTIONS` cannot be
    populated with a reasonless entry: a bad entry raises `ValueError` at
    module import and every test in this file errors out. That is
    deliberate -- an allowlist whose entries carry no justification stops
    being a decision record and becomes rot, which is how the defect this
    guard exists to catch survives contact with a guard.
    """

    callee: str
    parameter: str
    reason: str

    def __post_init__(self) -> None:
        if not self.callee or not self.parameter:
            raise ValueError("Exemption requires both a callee name and a parameter name")
        reason = self.reason.strip()
        if len(reason) < _MIN_REASON_CHARS:
            raise ValueError(
                f"Exemption({self.callee}.{self.parameter}) needs a written justification of "
                f"at least {_MIN_REASON_CHARS} characters explaining why the composition root "
                f"deliberately lets the factory substitute its own value. Got {reason!r} "
                f"({len(reason)} chars)."
            )


EXEMPTIONS: tuple[Exemption, ...] = (
    Exemption(
        callee="build_sidecar_index",
        parameter="embedding_provider",
        reason=(
            "Deliberate. At this point in lifespan() the server owns no embedding "
            "provider to share -- the graph store (and with it the ModelManager-warmed "
            "EmbeddingGenerator) is built later, inside VoiceProcessor.initialize(). "
            "The factory's fallback builds EmbeddingGenerator(config.embedding.model_name), "
            "which is lazy: no model is loaded at construction, and lifespan warms it "
            "explicitly via vault_sidecar.warmup() on the next lines. Passing None here "
            "is the wiring, not an omission."
        ),
    ),
)

_EXEMPTED: frozenset[tuple[str, str]] = frozenset((e.callee, e.parameter) for e in EXEMPTIONS)

# Anti-vacuity anchor. If a refactor moves one of these calls out of
# lifespan(), the guard stops watching it -- silently, and with every test
# still green. That is the same failure mode the guard exists to catch, so
# the set of watched call sites is itself asserted.
EXPECTED_BUILDER_CALL_SITES: frozenset[str] = frozenset(
    {
        "build_vault_writer",
        "build_sidecar_index",
        "build_phase3_components",
        "build_curation_scheduler",
        "_build_session_note_catchup",
    }
)


@dataclass(frozen=True, slots=True)
class CallSite:
    """A builder invocation found in `lifespan`'s source."""

    callee: str
    lineno: int
    positional_count: int
    keywords: frozenset[str]
    has_star_args: bool
    has_star_kwargs: bool


def _is_falsy_default(value: object) -> bool:
    """True when `value` is a falsy parameter default.

    Deliberately enumerates types rather than calling `bool(value)`: an
    arbitrary default object may define `__bool__` or `__len__` with side
    effects, or raise. Sentinels and real objects are correctly treated as
    non-falsy.
    """
    if value is inspect.Parameter.empty:
        return False
    if value is None or value is False:
        return True
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float) and value == 0:
        return True
    if isinstance(value, str | bytes | list | tuple | set | frozenset | dict):
        return len(value) == 0
    return False


def _lifespan_node() -> ast.AsyncFunctionDef | ast.FunctionDef:
    """Parse `backend/server.py` and return the `lifespan` function node."""
    tree = ast.parse(_SERVER_PATH.read_text(encoding="utf-8"), filename=str(_SERVER_PATH))
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef | ast.FunctionDef) and node.name == _LIFESPAN_NAME:
            return node
    raise AssertionError(
        f"No function named {_LIFESPAN_NAME!r} found in {_SERVER_REL}. The composition-root "
        "completeness check has lost its subject and is now watching nothing."
    )


def _builder_call_sites() -> list[CallSite]:
    """Collect every bare-name `build_*` / `_build_*` call inside `lifespan`."""
    sites: list[CallSite] = []
    for node in ast.walk(_lifespan_node()):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        if not node.func.id.startswith(_BUILDER_PREFIXES):
            continue
        sites.append(
            CallSite(
                callee=node.func.id,
                lineno=node.lineno,
                positional_count=sum(1 for a in node.args if not isinstance(a, ast.Starred)),
                keywords=frozenset(kw.arg for kw in node.keywords if kw.arg is not None),
                has_star_args=any(isinstance(a, ast.Starred) for a in node.args),
                has_star_kwargs=any(kw.arg is None for kw in node.keywords),
            )
        )
    return sites


def _unsupplied_falsy_defaults(site: CallSite, signature: inspect.Signature) -> list[str]:
    """Names of falsy-default parameters this call site leaves to the factory."""
    positional_slots = [
        p.name
        for p in signature.parameters.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    supplied = set(positional_slots[: site.positional_count]) | set(site.keywords)

    missing: list[str] = []
    for param in signature.parameters.values():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            continue
        if not _is_falsy_default(param.default):
            continue
        if param.name in supplied:
            continue
        missing.append(param.name)
    return missing


def _resolve_callee(name: str):
    """Resolve a callee name against `backend.server`'s module namespace."""
    fn = getattr(server, name, None)
    return fn if callable(fn) else None


class TestCompositionRootCompleteness:
    def test_lifespan_supplies_every_falsy_default_or_exempts_it(self):
        """Every optional dependency of every builder called in `lifespan`
        must be an explicit decision at the call site -- either supplied, or
        exempted in writing. Silence is not a decision.
        """
        # Arrange
        sites = _builder_call_sites()
        assert sites, (
            f"Found no bare-name build_* calls inside {_SERVER_REL}:{_LIFESPAN_NAME}. "
            "The guard is vacuous."
        )

        # Act
        violations: list[str] = []
        for site in sites:
            location = f"{_SERVER_REL}:{site.lineno}  {site.callee}(...)"

            if site.has_star_args or site.has_star_kwargs:
                violations.append(
                    f"{location} uses * / ** unpacking, which makes supplied-ness "
                    "statically undecidable. Spell the parameters out at the call site."
                )
                continue

            fn = _resolve_callee(site.callee)
            if fn is None:
                violations.append(
                    f"{location} could not be resolved in backend.server's namespace, so its "
                    "signature could not be checked. The guard cannot vouch for this call."
                )
                continue

            signature = inspect.signature(fn)
            for param_name in _unsupplied_falsy_defaults(site, signature):
                if (site.callee, param_name) in _EXEMPTED:
                    continue
                default = signature.parameters[param_name].default
                violations.append(
                    f"{location} does not supply `{param_name}` (default {default!r}). "
                    f"The factory will silently substitute its own value, and nothing at "
                    f"runtime will report that the composition root declined to wire it."
                )

        # Assert
        assert not violations, "\n".join(
            [
                "Composition-root completeness check FAILED.",
                "",
                *(f"  [FAIL] {v}" for v in violations),
                "",
                "Fix by supplying the dependency at the call site in backend/server.py, or -- "
                "if the factory's own default IS the intended production wiring -- by adding "
                f"an Exemption to {__name__}.EXEMPTIONS with a written justification.",
            ]
        )

    def test_guard_still_watches_every_known_builder_call_site(self):
        """Anti-vacuity: a refactor that moves a builder call out of
        `lifespan` silently narrows this guard's scope while leaving every
        test green. Pin the watched set so that narrowing is loud.
        """
        found = frozenset(site.callee for site in _builder_call_sites())

        assert found == EXPECTED_BUILDER_CALL_SITES, (
            "The set of builder calls inside lifespan() changed.\n"
            f"  no longer watched: {sorted(EXPECTED_BUILDER_CALL_SITES - found)}\n"
            f"  newly watched:     {sorted(found - EXPECTED_BUILDER_CALL_SITES)}\n"
            "If the change is intended, update EXPECTED_BUILDER_CALL_SITES. If a call moved "
            "out of lifespan, this guard no longer covers it -- say so explicitly."
        )


class TestExemptionsCannotRot:
    @pytest.mark.parametrize(
        "reason",
        [
            pytest.param("", id="empty"),
            pytest.param("   ", id="whitespace-only"),
            pytest.param("n/a", id="dismissive"),
            pytest.param("see MIS-999", id="bare-ticket-reference"),
        ],
    )
    def test_exemption_without_a_written_justification_is_unconstructible(self, reason):
        """The allowlist must make a reasonless entry impossible to write,
        not merely discouraged by convention.
        """
        with pytest.raises(ValueError, match="written justification"):
            Exemption(callee="build_x", parameter="y", reason=reason)

    def test_every_exemption_is_still_live(self):
        """Allowlists rot in both directions. An exemption whose call site
        now DOES supply the parameter (or whose callee no longer takes it)
        is a stale claim about the codebase and must be deleted.
        """
        sites_by_callee: dict[str, list[CallSite]] = {}
        for site in _builder_call_sites():
            sites_by_callee.setdefault(site.callee, []).append(site)

        stale: list[str] = []
        for exemption in EXEMPTIONS:
            sites = sites_by_callee.get(exemption.callee)
            if not sites:
                stale.append(
                    f"{exemption.callee}.{exemption.parameter}: no such call inside lifespan()"
                )
                continue

            fn = _resolve_callee(exemption.callee)
            assert fn is not None, f"{exemption.callee} is exempted but is not resolvable"
            signature = inspect.signature(fn)

            if exemption.parameter not in signature.parameters:
                stale.append(
                    f"{exemption.callee}.{exemption.parameter}: callee has no such parameter"
                )
                continue

            if not any(
                exemption.parameter in _unsupplied_falsy_defaults(site, signature) for site in sites
            ):
                stale.append(
                    f"{exemption.callee}.{exemption.parameter}: the call site now supplies it "
                    "(or its default is no longer falsy) -- delete the exemption"
                )

        assert not stale, "\n".join(["Stale EXEMPTIONS entries:", "", *(f"  {s}" for s in stale)])
