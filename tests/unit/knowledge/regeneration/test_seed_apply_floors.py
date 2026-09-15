"""The seed-apply must prove it wrote something, and that what it wrote is usable.

## Why these two floors and not one

MIS-130's hazard, stated in the ticket and re-derived independently by two
ADR-023 reviewers: delete the copy-forward, wire a seed-apply that writes zero
nodes, and every existing gate stays green. Determinism passes because two empty
self-models are byte-identical, and `live == rebuilt` passes because the compared
surface never looks at that partition. The retirement would be proven by nothing.

`assert_seed_applied` is the floor that makes "applied nothing" and "applied
something" different observables, and it reads a COUNT at the apply site rather
than a canonical form. That is what lets step B land before the surface extension
in step C: the count is a direct observation on the staging connection and needs
no comparison surface at all. The sequencing constraint MIS-130 states ("must
land AFTER the comparison surface can observe it") is satisfied here, by this
assertion, rather than by waiting for C.

`assert_seed_embeddings_present` is a SECOND floor because a node count cannot
see the failure that has actually cost this project live data twice.
`apply_seed_documents` never writes `embedding` -- only the backfill does
(`mist_admin.py:180`) -- and `canonical_serialize` EXCLUDES `embedding`
outright (`seed/gates.py:264-268`: "byte-identical whether embeddings are
present, absent, or all-zero"). So a seed-apply that skips or fails the backfill
produces a graph nothing can retrieve from, and it certifies as identical to one
that did not. Node count green, determinism green, equality green, graph dead.

The two floors fail for different reasons and neither subsumes the other:
a partial apply has nodes and embeddings, a failed backfill has nodes and no
embeddings, and only the pair distinguishes both from a correct run.

## Why the count is taken at the apply site, not inferred later

The backfill runs AFTER the seed writes have already committed, so a failure
inside it (model load, cache miss, OOM) leaves a fully-seeded, fully-unembedded
graph that every downstream gate passes. Reporting the backfill's own count
proves nothing either -- it counts rows the backfill THOUGHT it wrote, from the
same code that failed to write them. `check_embeddings` re-reads the graph,
which is why the embedding floor takes its `GateResult` rather than the
backfill's integer.
"""

import pytest

from backend.knowledge.regeneration.rebuild_gate import (
    RebuildVacuityError,
    assert_seed_applied,
    assert_seed_embeddings_present,
)
from backend.knowledge.seed.gates import GateResult


class TestSeedAppliedFloor:
    def test_zero_nodes_written_is_refused(self):
        with pytest.raises(RebuildVacuityError, match="seed gate FAILED"):
            assert_seed_applied(nodes_written=0, minimum=1)

    def test_below_the_floor_is_refused(self):
        with pytest.raises(RebuildVacuityError, match="21"):
            assert_seed_applied(nodes_written=3, minimum=21)

    def test_at_the_floor_passes(self):
        assert_seed_applied(nodes_written=21, minimum=21)

    def test_above_the_floor_passes(self):
        assert_seed_applied(nodes_written=22, minimum=21)

    def test_a_floor_below_one_is_a_caller_error(self):
        """Matches `assert_turns_processed`: a floor of 0 is not a gate."""
        with pytest.raises(ValueError, match="at least 1"):
            assert_seed_applied(nodes_written=5, minimum=0)

    def test_the_refusal_names_both_numbers(self):
        """An operator must be able to tell a partial apply from a dead one."""
        with pytest.raises(RebuildVacuityError) as exc:
            assert_seed_applied(nodes_written=3, minimum=21)
        assert "3" in str(exc.value) and "21" in str(exc.value)


class TestSeedEmbeddingFloor:
    def test_a_failed_embedding_gate_is_refused(self):
        result = GateResult(passed=False, failures=["mist-identity has no embedding"], examined=21)
        with pytest.raises(RebuildVacuityError, match="embedding gate FAILED"):
            assert_seed_embeddings_present(result)

    def test_the_refusal_carries_the_underlying_failures(self):
        result = GateResult(passed=False, failures=["mist-identity has no embedding"], examined=21)
        with pytest.raises(RebuildVacuityError, match="mist-identity"):
            assert_seed_embeddings_present(result)

    def test_a_passing_gate_that_examined_nothing_is_refused(self):
        """The vacuous pass is the dangerous one, and it reports `passed=True`.

        `check_embeddings` is the only seed gate that populates `examined`
        (`seed/gates.py:62-88`), so this check is meaningful here and would be
        meaningless against any of the other four.
        """
        result = GateResult(passed=True, failures=[], examined=0)
        with pytest.raises(RebuildVacuityError, match="examined 0"):
            assert_seed_embeddings_present(result)

    def test_a_passing_gate_that_examined_nodes_passes(self):
        assert_seed_embeddings_present(GateResult(passed=True, failures=[], examined=21))
