"""R1.2 rebuild gates: determinism (rebuild-twice) + live divergence report."""

from __future__ import annotations

import difflib
import json

from backend.errors import MistError


class RebuildDeterminismError(MistError):
    """Raised when two rebuilds of the same log produce different canonical forms."""


class RebuildDivergenceError(MistError):
    """The rebuild is deterministic and disagrees with the live graph.

    Separate from `RebuildDeterminismError` on purpose -- see
    `assert_live_equals_rebuilt`. Non-determinism and a derivation gap are
    different findings with different remedies, and collapsing them costs the
    operator the diagnosis.
    """


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
    # Materialise first: `rows` is the natural shape for a cursor or generator,
    # and the failure branch below calls len(rows). Measuring after consuming
    # would replace this gate's diagnosis with a TypeError from inside its own
    # error path -- a guard that crashes instead of explaining.
    rows = list(rows)
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


def _self_model_node_count(form: str, *, side: str) -> int:
    """Read the self-model node count from a canonical form, or refuse."""
    try:
        payload = json.loads(form)
    except json.JSONDecodeError as exc:
        raise RebuildVacuityError(f"{side} form is not a canonical graph form: {exc}") from exc
    if "self_model" not in payload:
        raise RebuildVacuityError(
            f"{side} form carries no 'self_model' key, so it was produced WITHOUT "
            "include_self_model=True. Comparing two such forms would report the "
            "self-model verified while never having looked at it -- the most "
            "dangerous pass available here. Rebuild the form with "
            "include_self_model=True."
        )
    nodes = payload["self_model"].get("nodes")
    if not isinstance(nodes, list):
        raise RebuildVacuityError(f"{side} form's self_model.nodes is not a list")
    return len(nodes)


def assert_self_model_applied(live_form: str, rebuilt_form: str) -> None:
    """Hard gate: the self-model must be present on both sides AND equal.

    Non-zero AND equal, never merely equal, and the distinction is the whole
    reason this exists. The closure design's sequencing error was exactly this:
    delete copy-forward, wire a seed-apply that writes zero nodes, and every
    existing gate stays green -- determinism passes because two empty
    self-models are byte-identical, and `live == rebuilt` passes because it
    never looked at that partition at all. The retirement would have been
    proven by nothing.

    So "applied nothing" and "applied correctly" have to be different
    observables. An equality check alone cannot tell them apart; a count floor
    alone cannot catch a partial apply. Both, together, can.

    Args:
        live_form: Canonical form of the source graph, built with
            `include_self_model=True`.
        rebuilt_form: Canonical form of the staging graph, same setting.

    Raises:
        RebuildVacuityError: when either form lacks the partition (a caller
            bug that would otherwise pass silently), when both sides are empty,
            when the rebuild applied nothing, or when the counts disagree.
    """
    live = _self_model_node_count(live_form, side="live")
    rebuilt = _self_model_node_count(rebuilt_form, side="rebuilt")

    if live == 0 and rebuilt == 0:
        raise RebuildVacuityError(
            "self-model gate FAILED: both sides carry 0 self-model nodes. Two empty "
            "partitions are byte-identical, so an equality check over them is green "
            "and meaningless. Either the seed was never applied or the partition was "
            "never populated; neither is a passing state."
        )
    if rebuilt == 0:
        raise RebuildVacuityError(
            f"self-model gate FAILED: the rebuild applied nothing -- live carries "
            f"{live} self-model node(s), the rebuild carries 0. This is the "
            "seed-apply-writes-zero case the copy-forward retirement must not be "
            "able to hide."
        )
    if live != rebuilt:
        raise RebuildVacuityError(
            f"self-model gate FAILED: live carries {live} self-model node(s), the "
            f"rebuild carries {rebuilt}. A partial apply is not a pass."
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


def assert_live_equals_rebuilt(live_form: str, rebuilt_form: str) -> None:
    """Hard gate: the rebuilt entity subgraph must equal the live one.

    Promoted from `live_vs_rebuilt_report` (below), which computed exactly this
    diff and returned it as a STRING. The only runnable comparison in the repo
    therefore printed its own red result and exited 0 -- `mist_admin.py:1125-1126`
    -- so `hydrate && rebuild && echo GREEN` printed GREEN on any divergence, and
    ADR-023's reserved evidence field "Mutation results proving the gate can fail"
    had no assertion to mutate against. MIS-137.

    Deliberately raises `RebuildDivergenceError` and NOT
    `RebuildDeterminismError`. `assert_rebuild_twice_identical` failing means
    NON-DETERMINISM: identical inputs produced two different graphs.
    This failing means a DERIVATION GAP: the rebuild is perfectly deterministic
    and disagrees with live. Different causes, different remedies, and a caller
    mapping both onto one exit code cannot tell an operator which happened.

    Both forms must be built with the SAME `include_provenance` /
    `include_self_model` switches. Comparing across switch sets compares two
    different surfaces, and the diff would be dominated by the surface difference
    rather than by content.

    Caveat this assertion cannot enforce, recorded because a green result here
    will be cited: while `rebuild()` still populates `:__SelfModel__` by copying
    from `source_conn` (the live store, i.e. this function's own left-hand side),
    a form built with `include_self_model=True` compares that partition against a
    copy of itself and is green by construction. Keep that switch OFF until the
    copy-forward retirement lands (MIS-130), or report the partition as copied
    rather than reproduced.

    Raises:
        RebuildDivergenceError: when the two canonical forms differ.
    """
    if live_form == rebuilt_form:
        return
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
    raise RebuildDivergenceError(
        f"live != rebuilt: {n} differing canonical line(s). The rebuild is not a "
        "reproduction of the live entity subgraph.\n" + diff
    )


def live_vs_rebuilt_report(live_form: str, rebuilt_form: str) -> str:
    """Diagnostic, NOT a gate -- use `assert_live_equals_rebuilt` for that.

    Retained for the explicit `--diagnostic` mode and for callers that want the
    divergence described without failing. Every gate path uses the assertion.
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


def assert_seed_applied(*, nodes_written: int, minimum: int) -> None:
    """Hard gate: the seed-apply wrote real nodes, BEFORE the replay loop ran.

    This is the floor MIS-130's sequencing paragraph asks for, and it reads a
    COUNT at the apply site rather than a canonical form. That distinction is
    what lets the seed-apply land before the comparison surface is extended:
    "applied nothing" and "applied something" become different observables
    without any comparison surface existing. Delete the copy-forward, wire a
    seed-apply that writes zero nodes, and every OTHER gate stays green --
    determinism passes because two empty self-models are byte-identical, and
    `live == rebuilt` passes because the compared surface never looks at that
    partition. This is the assertion that does not.

    `minimum` is the caller's floor and is deliberately not defaulted anywhere
    up the stack, for the reason MIS-137 gave: a defaulted floor is a number
    nobody chose that the run then reports as passed.

    Args:
        nodes_written: Nodes the seed-apply reports having written to staging.
        minimum: The floor this run was configured with. Must be at least 1.

    Raises:
        ValueError: when `minimum < 1`; a floor of zero is not a gate.
        RebuildVacuityError: when fewer than `minimum` nodes were written.
    """
    if minimum < 1:
        raise ValueError(f"minimum={minimum} seed nodes cannot support a gate; pass at least 1.")
    if nodes_written < minimum:
        raise RebuildVacuityError(
            f"seed gate FAILED: the seed-apply wrote {nodes_written} node(s), floor is "
            f"{minimum}. A rebuild whose seed step wrote nothing produces an empty "
            "self-model partition that every other gate certifies as correct: two "
            "empty partitions are byte-identical and the compared surface does not "
            "read them. Nothing downstream can catch this, which is why it is caught "
            "here."
        )


def assert_seed_embeddings_present(gate_result) -> None:
    """Hard gate: the seeded nodes carry usable embeddings.

    A second floor rather than a clause of `assert_seed_applied`, because a node
    count cannot see this failure and the canonical form is blind to it BY
    DESIGN: `canonical_serialize` excludes `embedding` (`seed/gates.py:264-268`
    -- "byte-identical whether embeddings are present, absent, or all-zero").
    `apply_seed_documents` never writes `embedding` either; only the backfill
    does. So a seed-apply that skips or fails the backfill yields a graph
    nothing can retrieve from, and it certifies as identical to one that did
    not. Embeddings have already been silently lost on live data twice for
    exactly this reason (`mist_admin.py:257-270`).

    Takes a `GateResult` from `check_embeddings` rather than the backfill's own
    count on purpose. The backfill runs after the seed writes have committed, so
    a failure inside it leaves a fully-seeded, fully-unembedded graph; and its
    returned count is rows it THOUGHT it wrote, reported by the same code that
    failed to write them. `check_embeddings` re-reads the graph.

    A pass that examined nothing is refused too. `check_embeddings` is the only
    seed gate that populates `examined` (`seed/gates.py:62-88`), so the check is
    meaningful here and would be meaningless against the other four.

    Raises:
        RebuildVacuityError: when the gate failed, or passed vacuously.
    """
    if not gate_result.passed:
        detail = "; ".join(gate_result.failures) or "no detail reported"
        raise RebuildVacuityError(
            f"seed embedding gate FAILED: {detail}. The canonical form cannot catch "
            "this -- it excludes `embedding` -- so an unembedded rebuild would compare "
            "byte-identical to an embedded one while retrieving nothing."
        )
    if gate_result.examined == 0:
        raise RebuildVacuityError(
            "seed embedding gate FAILED: it passed having examined 0 nodes. A gate "
            "that looked at nothing reports the same `passed=True` as one that "
            "verified everything, and this is the shape of both historical live "
            "embedding losses."
        )
