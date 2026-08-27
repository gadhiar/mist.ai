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


# The property pair that distinguishes a reconciliation-written edge from a
# seed-written one. From the R1.4.6 hydration design's T5 acceptance test:
# hydrated edges must be "structurally indistinguishable from usage edges" --
# carrying `version_key`, `source_utterance_id`, `recorded_at` and a currency
# triple -- "not the two-property seed shape".
#
# BOTH are required, never either. `canonical_serialize._rel_key` already sorts
# on both, so both survive into the canonical form, and a seed edge that
# happens to carry one should not be counted as replay output.
REPLAY_EDGE_MARKERS = frozenset({"source_utterance_id", "version_key"})


def _relationships(form: str) -> list[dict]:
    """Parse a canonical form's relationships, refusing anything that is not one."""
    try:
        rels = json.loads(form)["relationships"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RebuildVacuityError(
            f"not a canonical graph form, so its edges cannot be counted: {exc}"
        ) from exc
    if not isinstance(rels, list):
        raise RebuildVacuityError(
            f"canonical form's 'relationships' is {type(rels).__name__}, not a list"
        )
    return rels


def count_replay_derived_edges(form: str) -> int:
    """Count edges in `form` carrying the full replay marker set."""
    return sum(
        1
        for rel in _relationships(form)
        if set((rel.get("properties") or {}).keys()) >= REPLAY_EDGE_MARKERS
    )


def assert_replay_derived_non_vacuous(form: str, *, minimum_edges: int) -> None:
    """Hard gate: the REPLAY must have produced edges, not just the seed.

    `assert_canonical_form_non_vacuous` counts every node including seed-derived
    ones, and its docstring says so. That is not a floor on the replay: 100% of
    today's live graph is seed content (32 nodes, 0 conversation turns), so a
    whole-graph floor of any size up to 32 is satisfied with the replay having
    produced nothing at all, and the gate would compare two identical
    seed-shaped graphs and pass.

    This bounds the subset that only a replay can create.

    Args:
        form: A canonical graph form.
        minimum_edges: How many replay-derived edges the corpus must yield.
            Must be >= 1 -- a floor of zero is satisfied by anything, including
            exactly the emptiness this gate exists to refuse.

    Raises:
        ValueError: for `minimum_edges < 1`, which is a caller bug, not a gate
            failure -- surfaced as a different type so it cannot be mistaken
            for one.
        RebuildVacuityError: when the form is unparsable or carries too few
            replay-derived edges.
    """
    if minimum_edges < 1:
        raise ValueError(
            f"minimum_edges={minimum_edges} would be satisfied by an empty replay; "
            "pass at least 1."
        )
    count = count_replay_derived_edges(form)
    if count < minimum_edges:
        raise RebuildVacuityError(
            f"replay non-vacuity gate FAILED: {count} replay-derived edge(s) "
            f"(carrying {sorted(REPLAY_EDGE_MARKERS)}), but at least "
            f"{minimum_edges} are required. Seed-written edges do not count: a "
            "graph that is entirely seed proves nothing about the replay, and "
            "the whole-graph node floor cannot tell the two apart."
        )


def assert_turns_processed(*, processed: int, expected: int) -> None:
    """Hard gate: the replay consumed exactly the corpus, no more and no less.

    Equality rather than a floor, in both directions. Fewer turns means a
    partial graph. MORE turns means the event store was not empty when the run
    started, which shifts every hydration-clock key -- the same divergence
    `scripts/hydration/preflight.assert_sessions_unused` checks for up front,
    caught here from the other end in case the run bypassed preflight.

    Raises:
        ValueError: when `expected < 1`; a corpus of zero turns cannot support
            a gate.
        RebuildVacuityError: on any mismatch.
    """
    if expected < 1:
        raise ValueError(f"expected={expected} turns cannot support a gate; pass at least 1.")
    if processed != expected:
        direction = "short" if processed < expected else "over"
        raise RebuildVacuityError(
            f"turns gate FAILED ({direction}): {processed} turns processed, {expected} "
            "expected. Fewer means a partial graph; more means the event store was "
            "not empty at the start, which shifts every hydration-clock key."
        )


def assert_extraction_cache_non_vacuous(rows, *, minimum: int) -> None:
    """Hard gate: the cache holds real extraction output, not successful nothings.

    Counts only rows whose outcome is `extracted` AND whose payload is
    non-empty. Both conditions matter and for different reasons: a `skipped`
    row is a recorded decision rather than output, and an `extracted` row with
    an empty entity and relationship list is what a truncated or refused model
    response records. A count of rows alone reports a healthy cache built
    entirely from the latter.

    Args:
        rows: Cache rows, each with `outcome` and optional `entities` /
            `relationships` payload lists.
        minimum: How many substantive rows the corpus must yield. Must be >= 1.

    Raises:
        ValueError: for `minimum < 1`.
        RebuildVacuityError: when too few rows carry real payloads.
    """
    if minimum < 1:
        raise ValueError(
            f"minimum={minimum} would be satisfied by an empty cache; pass at least 1."
        )
    substantive = sum(
        1
        for row in rows
        if row.get("outcome") == "extracted"
        and ((row.get("entities") or []) or (row.get("relationships") or []))
    )
    if substantive < minimum:
        raise RebuildVacuityError(
            f"extraction-cache non-vacuity gate FAILED: {substantive} row(s) with "
            f"outcome='extracted' and a non-empty payload, but at least {minimum} "
            f"are required (of {len(rows)} row(s) total). Skipped rows are recorded "
            "decisions, not output, and an extracted row with an empty payload is "
            "what a truncated model response records."
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
