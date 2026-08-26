"""R1.2 rebuild gates: determinism (rebuild-twice) + live divergence report."""

from __future__ import annotations

import difflib
import json

from backend.errors import MistError


class RebuildDeterminismError(MistError):
    """Raised when two rebuilds of the same log produce different canonical forms."""


class RebuildVacuityError(MistError):
    """Raised when a canonical form describes too small a graph to prove anything."""


def assert_canonical_form_non_vacuous(form: str, *, minimum_nodes: int = 1) -> None:
    r"""Hard gate: a canonical form must describe at least `minimum_nodes` nodes.

    Every gate in this module compares canonical forms for EQUALITY, and two
    empty graphs are byte-identical. So an equality gate over an empty rebuild
    is green and meaningless. This is the guard that makes such a run fail
    instead, and it must be called BEFORE the equality assertion.

    Do NOT reimplement this as a truthiness check on the string.
    `canonical_graph_form` ends in `json.dumps(...) + "\n"`
    (`canonical_serialize.py`, grep `return json.dumps`), so a graph with no
    nodes serialises to a non-empty JSON envelope and `form.strip()` is ALWAYS
    truthy. `test_golden_log_rebuild.py` carried exactly that expression under a
    comment claiming it failed closed on vacuity; it could not fail at all.
    `test_rebuild_gate_vacuity.py` pins both that fact and this fix.

    `minimum_nodes` is a floor, not an equality check -- a caller that knows how
    many nodes its corpus must yield should pass it. Note the floor counts ALL
    nodes in the form, including seed-derived ones, so on a seeded graph it does
    NOT establish that the REPLAY produced anything. A gate needing that must
    additionally bound the replay-derived subset (edges carrying
    `source_utterance_id` / `version_key`); this function does not do it for you.

    Raises:
        RebuildVacuityError: if the form is not a canonical graph form, or
            describes fewer than `minimum_nodes` nodes.
    """
    try:
        nodes = json.loads(form)["nodes"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RebuildVacuityError(
            f"not a canonical graph form, so its node count cannot be checked: {exc}"
        ) from exc

    if not isinstance(nodes, list):
        raise RebuildVacuityError(f"canonical form's 'nodes' is {type(nodes).__name__}, not a list")

    count = len(nodes)
    if count < minimum_nodes:
        raise RebuildVacuityError(
            f"non-vacuity gate FAILED: canonical form describes {count} node(s), "
            f"but at least {minimum_nodes} are required. An equality gate over a "
            f"graph this small proves nothing -- two empty graphs are byte-identical."
        )


def assert_rebuild_twice_identical(build_a: str, build_b: str) -> None:
    """Hard gate: two rebuilds of the same epoch+log must be byte-identical."""
    if build_a != build_b:
        diff = "\n".join(
            difflib.unified_diff(
                build_a.splitlines(),
                build_b.splitlines(),
                fromfile="rebuild-1",
                tofile="rebuild-2",
                lineterm="",
            )
        )
        raise RebuildDeterminismError(
            "rebuild-twice determinism gate FAILED: two rebuilds differ.\n" + diff
        )


def live_vs_rebuilt_report(live_form: str, rebuilt_form: str) -> str:
    """Diagnostic (NOT a gate in R1.2): describe live vs rebuilt divergence.

    `live == rebuilt` green closure is deferred to R1.6 (after vault->graph
    retirement + seed migration make the live graph purely log-derived).
    """
    if live_form == rebuilt_form:
        return "live == rebuilt: no divergence (entity subgraph canonical forms match)."
    diff = "\n".join(
        difflib.unified_diff(
            live_form.splitlines(),
            rebuilt_form.splitlines(),
            fromfile="live",
            tofile="rebuilt",
            lineterm="",
        )
    )
    n = sum(
        1
        for line in diff.splitlines()
        if line and line[0] in "+-" and not line.startswith(("+++", "---"))
    )
    return f"live != rebuilt: {n} differing canonical lines (expected pre-R1.3/R1.4).\n" + diff
