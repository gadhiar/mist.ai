"""Test double for `StagingSeeder`, the seed-apply step `rebuild()` requires.

MIS-130 step B made `staging_seeder` a REQUIRED constructor argument on
`LogRegenerator`, for the reason the constructor comment gives: a defaulted
seeder makes the empty-self-model state reachable and silent, and that is the
exact state MIS-130's hazard paragraph is about -- an empty partition passes
determinism (two empty partitions are byte-identical) and passes
`live == rebuilt` (the compared surface does not read it).

Required means every construction site states what it seeds. Most of them are
testing the REPLAY and not the seed, so they want a seeder that reports a
healthy apply and gets out of the way; `FakeStagingSeeder()` with no arguments
is that. Sites that care about the seed's effect on `rebuild()` -- ordering
relative to the replay loop, the floors, the timestamp -- drive it explicitly.

A fake rather than the real `StagingSeeder` because the real one needs a graph
connection, an embedding generator and a valid seed corpus, none of which most
of these worlds have. What `rebuild()` owes the seeder is narrow and fully
observable through this fake: call it once, before the replay loop, with an
epoch-derived `now_iso`, and act on what it returns. `StagingSeeder`'s own
three-step behaviour -- apply, backfill, then RE-READ -- is pinned separately in
tests/unit/knowledge/regeneration/test_staging_seeder.py against a recording
connection, which is where it belongs: this fake asserts nothing about it and
must not be read as covering it.

## A gap this fake creates, stated rather than left to be discovered

The INTEGRATION tests under tests/integration/knowledge/ also take this fake, so
as of MIS-130 step B nothing exercises the real `StagingSeeder` against a real
Neo4j. `apply_seed_documents`, `_backfill_embeddings_for_seed` and
`check_embeddings` each have their own integration coverage from R1.4; what has
none is the three composed, in this order, by this class. It was not written
here because the eval and staging Neo4j instances are both down, and an
integration test that has never been executed is a claim rather than evidence.

That coverage belongs with step C, which needs those instances anyway, and where
the comparison surface can finally observe the partition the seeder populates.
Until it exists, "the seed-apply works" rests on unit tests over a recording
connection plus the two floors in `rebuild()` -- which is enough to catch a seed
that wrote nothing, and NOT enough to catch one that wrote the wrong thing.
"""

from __future__ import annotations

from backend.knowledge.regeneration.staging_seeder import SeedApplyResult
from backend.knowledge.seed.gates import GateResult

# The live self-model is 21 nodes (`mist-memory/seed/mist.md`, verified against
# the graph on 2026-09-14). Defaulting to it keeps a fake that "seeded normally"
# recognisable as such, rather than an arbitrary number a reader has to chase.
DEFAULT_SEED_NODES = 21


class FakeStagingSeeder:
    """Records each `apply` and returns a chosen outcome.

    Defaults describe a healthy seed: 21 nodes written, embeddings present, the
    embedding gate having actually examined them. Override to drive a refusal.
    """

    def __init__(
        self,
        *,
        order_sink: list[str] | None = None,
        stamps: list[str] | None = None,
        nodes: int = DEFAULT_SEED_NODES,
        gate_passed: bool = True,
        examined: int = DEFAULT_SEED_NODES,
    ) -> None:
        self._order_sink = order_sink
        self._stamps = stamps
        self._nodes = nodes
        self._gate_passed = gate_passed
        self._examined = examined
        self.calls = 0

    def apply(self, *, now_iso: str) -> SeedApplyResult:
        self.calls += 1
        if self._order_sink is not None:
            self._order_sink.append("seed")
        if self._stamps is not None:
            self._stamps.append(now_iso)
        return SeedApplyResult(
            nodes_written=self._nodes,
            facts_written=self._nodes,
            embedded=self._nodes,
            embedding_gate=GateResult(
                passed=self._gate_passed,
                failures=[] if self._gate_passed else ["mist-identity has no embedding"],
                examined=self._examined,
            ),
        )
