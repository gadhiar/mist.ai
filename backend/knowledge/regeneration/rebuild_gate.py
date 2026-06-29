"""R1.2 rebuild gates: determinism (rebuild-twice) + live divergence report."""

from __future__ import annotations

import difflib

from backend.errors import MistError


class RebuildDeterminismError(MistError):
    """Raised when two rebuilds of the same log produce different canonical forms."""


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
