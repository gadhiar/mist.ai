"""Apply the seed corpus to a staging graph, and re-read what it wrote.

MIS-130 step B. This is what `rebuild()` uses in place of the retired
`copy_self_model_partition`, and the difference is the whole point of the
ticket: the copy read the LIVE graph, so the shipped function was
`graph = f(log, epoch, live_graph)`, where the R1 spec claims
`graph = f(seed, log, epoch)`. A seed-apply makes `seed` a real input.

## Three steps, not one

The live seed path is three steps (`mist_admin.py:155-200`) and skipping any of
them yields a graph that passes every gate:

1. `apply_seed_documents` writes nodes and facts stamped with `seed_version`.
   It NEVER writes `embedding` -- verified, not assumed: the applier's writes
   set `created_at`/`updated_at`/`seed_version` and the authored properties.
2. `_backfill_embeddings_for_seed` computes and SETs the vectors.
3. `check_embeddings` RE-READS the graph and compares each stored vector to its
   authored source text.

Step 3 is not redundant with step 2. The backfill runs after the seed writes
have already committed, so a failure inside it (model load, cache miss, OOM)
leaves a fully-seeded, fully-unembedded graph that every other gate passes; and
its return value counts rows it THOUGHT it wrote, reported by the same code that
failed to write them. Embeddings have been silently lost on live data twice for
exactly this reason.

The three live behind one object so they cannot drift apart at the one call site
that matters. `seed/gates.py:264-268` states the cost of drift outright:
`canonical_serialize` excludes `embedding`, so a graph with no vectors is
byte-identical to one with correct vectors, and nothing downstream of here can
tell them apart.

## What this object deliberately does NOT do

It does not decide whether the result is acceptable. `apply` returns what
happened; `assert_seed_applied` and `assert_seed_embeddings_present` in
`rebuild_gate` decide. Keeping the measurement separate from the judgement is
what lets the floor be a number the operator CHOSE rather than one this class
picked -- the same reasoning MIS-137 applied to `--expect-turns`.
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.knowledge.admin import _backfill_embeddings_for_seed
from backend.knowledge.seed.applier import apply_seed_documents
from backend.knowledge.seed.gates import GateResult, check_embeddings
from backend.knowledge.seed.models import SeedDocument


@dataclass(frozen=True, slots=True)
class SeedApplyResult:
    """What one seed-apply actually did. Judged by the gates, not by this class."""

    nodes_written: int
    facts_written: int
    embedded: int
    embedding_gate: GateResult


class StagingSeeder:
    """Applies a seed corpus to ONE graph connection and reports the outcome.

    Dependencies are injected and required (the DI rule): the connection to
    write to, the documents to write, the version to stamp, and the embedding
    generator the backfill and the gate both need. A seeder that cannot seed
    cannot be constructed.

    The live-target refusal is NOT implemented here. It lives at the write site,
    in `_assert_seed_target_permitted` (`seed/applier.py:91`), default-CLOSED and
    reached through `apply_seed_documents`'s `allow_live` parameter, which this
    class never sets. A guard at the call site is absent exactly when it matters;
    a guard at the write site cannot be bypassed by forgetting it here.
    """

    def __init__(
        self,
        *,
        connection,
        documents: list[SeedDocument],
        seed_version: str,
        embedding_generator,
        expected_dimension: int,
    ) -> None:
        if not documents:
            raise ValueError(
                "StagingSeeder was given no seed documents. A rebuild whose seed step "
                "has nothing to apply produces an empty self-model partition that "
                "every other gate certifies as correct, so this is refused at "
                "construction rather than left for the floor to catch after the "
                "replay has already run."
            )
        if not seed_version:
            raise ValueError(
                "StagingSeeder requires a seed_version. It is the key the backfill "
                "and the embedding gate both match on; an empty one silently matches "
                "no node and turns both into no-ops."
            )
        self._connection = connection
        self._documents = documents
        self._seed_version = seed_version
        self._embedding_generator = embedding_generator
        self._expected_dimension = expected_dimension

    def apply(self, *, now_iso: str) -> SeedApplyResult:
        """Apply the seed, backfill embeddings, then verify by re-reading.

        Args:
            now_iso: The timestamp every seeded node and edge is stamped with.
                Injected rather than read from the clock so that application is
                byte-reproducible: two rebuilds of one epoch must stamp
                identical `created_at`/`updated_at`, or
                `assert_rebuild_twice_identical` fails on wall-clock noise
                rather than on content.

        Returns:
            SeedApplyResult carrying the counts and the re-read embedding gate.

        Raises:
            EvalIsolationError: via `apply_seed_documents`, before any write,
                when `connection` names the live graph.
        """
        counts = apply_seed_documents(
            self._connection,
            self._documents,
            seed_version=self._seed_version,
            now_iso=now_iso,
        )
        embedded = _backfill_embeddings_for_seed(
            self._connection, self._embedding_generator, self._seed_version
        )
        gate = check_embeddings(
            self._connection,
            self._documents,
            seed_version=self._seed_version,
            embedding_generator=self._embedding_generator,
            expected_dimension=self._expected_dimension,
        )
        return SeedApplyResult(
            nodes_written=counts["nodes"],
            facts_written=counts["facts"],
            embedded=embedded,
            embedding_gate=gate,
        )
