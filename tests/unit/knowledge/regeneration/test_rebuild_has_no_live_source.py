"""`rebuild()` must have no route to the live graph, and no copy-forward.

## What this closes

Before MIS-130, `rebuild()` took `source_conn` -- a handle on the LIVE graph --
and used it AFTER the replay loop to `copy_self_model_partition` and
`rederive_self_model_cross_layer_edges`. The rebuilt `:__SelfModel__` partition
was therefore a verbatim photocopy of live (`MERGE ... SET x = $props`), not a
derivation, so the shipped function was `graph = f(log, epoch, live_graph)` where
the R1 spec claims `graph = f(seed, log, epoch)`.

The copy is invisible to every gate that runs today: the compared surface is
`:__Entity__`-only (`canonical_graph_form`'s `include_self_model` defaults False
and `mist_admin.py:1172,1181` both omit it), and `_dump_subgraph` requires BOTH
endpoints to carry the label it is given. That invisibility is precisely why the
retirement has to come FIRST. Extend the comparison surface while the copy still
stands and the gate compares the self-model partition against a copy of the
comparison's own left-hand side -- guaranteed green, presented as new coverage.
`assert_self_model_applied` cannot fail in that world either: its "equal" arm is
guaranteed by `SET x = $props` and its "non-zero" arm holds whenever live is
non-zero.

`copy_self_model_partition` also carried a latent contamination route into the
compared surface, which is a second and independent reason to delete rather than
defer it. It MERGEs `labels(n)` read verbatim from the source, so a self-model
node that also carried `:__Entity__` would put live content directly into the
`:__Entity__` partition the gate DOES compare, with no symptom -- a false GREEN
manufactured by the copy itself. Latent only: a live label census on 2026-09-14
returned 21 `:__SelfModel__` and 11 `:__Entity__` nodes with no node carrying
both.

## Sequencing (the ruling, 2026-09-14)

MIS-130's own constraint ("retire the copies only AFTER the comparison surface
can observe it") and the ADR-023 reviewers' ("retire copy-forward BEFORE
extending the surface") read as a contradiction only because MIS-130 bundles two
separable changes under one ticket. Split, each constraint governs one half and
the contradiction dissolves:

  A. retire the copies (THIS COMMIT) -- must precede the surface extension
  B. add the seed-apply + embedding backfill, BEFORE the replay loop -- its
     non-vacuity is proven by a node COUNT at the apply site, not by the compared
     surface, so it does not need the extension either
  C. extend the surface, fold in MIS-139, wire `assert_self_model_applied`

The branch does not merge until C. Between A and C the rebuild is knowingly
incomplete in a way no gate reports -- which is also true of the tree before A,
so the window introduces no new false claim, but it is not a state to ship.

## What survives from the two retired test files

Both are deleted in this commit along with their subject, not worked around:

- `tests/unit/knowledge/regeneration/test_cross_layer_edge_coverage.py` (9 tests)
- `tests/integration/knowledge/test_log_regenerator_selfmodel.py` (2 tests)

Their PREMISE is not deleted, and it is owed by C rather than discharged here:
whatever replaces the copy must either produce every ontology-permitted
cross-layer type, or ADR-023 must exclude the pair BY NAME. C is the first point
at which the surface can observe which of those two happened -- and MIS-139's
self-model <-> provenance clause is part of why, since `LEARNED_SELF` is
compared by no gate key at all until it lands.

Deleting a test whose subject is gone is not the same as weakening coverage,
but it does move an obligation. Recording where it moved to is the difference.

## Why a signature test rather than a behavioural one

"Never writes to live" is a negative claim over an unbounded space. The honest
enforceable form is that no live handle is reachable from the function at all: a
caller cannot pass what the signature will not take. The behavioural half --
that the rebuilt self-model is DERIVED -- is not assertable until B exists, and
asserting it now would be an assertion with no production caller, which is the
defect MIS-137 existed to fix.

## A parameter deliberately NOT pinned here

`staging_conn` also leaves `rebuild()` in this commit, because the two copies
were its only readers. It is NOT pinned absent, because B re-introduces it as a
REQUIRED, undefaulted parameter for the seed-apply -- a rebuild that cannot seed
is refused before it starts, the same fail-closed shape as MIS-137's
`--expect-turns`. Pinning its absence here would be a test written to be deleted
next commit. `source_conn` is different in kind: it is the live handle, nothing
re-introduces it, and the pin is permanent.
"""

import inspect

import pytest

from backend.knowledge.regeneration.log_regenerator import LogRegenerator
from scripts import mist_admin


class TestNoLiveHandleReachesRebuild:
    """`source_conn` is gone from the signature and stays gone."""

    def test_rebuild_accepts_no_source_conn_parameter(self):
        params = inspect.signature(LogRegenerator.rebuild).parameters
        assert "source_conn" not in params, (
            "rebuild() still accepts `source_conn`, a handle on the LIVE graph. "
            "Replay must derive from the log and the seed only; a live handle in "
            "the signature is the route by which `graph = f(log, epoch, "
            "live_graph)` was true."
        )

    def test_rebuild_signature_names_no_live_connection_at_all(self):
        """Catches a rename that reintroduces the same hazard under a new word."""
        params = inspect.signature(LogRegenerator.rebuild).parameters
        offenders = [
            name
            for name in params
            if name != "live_uri" and ("live" in name.lower() or "source" in name.lower())
        ]
        assert offenders == [], (
            f"rebuild() takes connection-shaped parameter(s) {offenders} naming live "
            "or source. `live_uri` is the sole permitted mention: it is a guard "
            "value compared by `assert_rebuild_target_not_live`, never connected to."
        )


class TestCopyForwardIsRetired:
    """Both copy methods are deleted, not merely unreferenced."""

    @pytest.mark.parametrize(
        "method",
        ["copy_self_model_partition", "rederive_self_model_cross_layer_edges"],
    )
    def test_method_no_longer_exists(self, method):
        assert not hasattr(LogRegenerator, method), (
            f"LogRegenerator.{method} still exists. An unreferenced copy-forward is "
            "one call site away from returning, and the surface extension in C is "
            "only safe while it is absent."
        )

    @pytest.mark.parametrize(
        "constant",
        ["_INTRA_SELF_MODEL_EDGES", "_PARTITION_DIRECTIONS"],
    )
    def test_copy_only_constant_no_longer_exists(self, constant):
        assert not hasattr(LogRegenerator, constant), (
            f"LogRegenerator.{constant} existed solely to drive the retired copy "
            "methods. Leaving it behind invites a future reader to rebuild the "
            "copy around it."
        )


class TestCliPassesNoLiveConnectionIntoRebuild:
    """The one production caller stops handing `rebuild()` the live graph."""

    def test_rebuild_call_site_does_not_pass_source_conn(self):
        source = inspect.getsource(mist_admin.cmd_graph_rebuild_from_log)
        assert "source_conn" not in source, (
            "cmd_graph_rebuild_from_log still passes `source_conn` into rebuild(). "
            "This is the single production caller and it passed `live_conn` "
            "(mist_admin.py:1164), which is what made the rebuilt self-model a "
            "photocopy of the comparison's own left-hand side."
        )
