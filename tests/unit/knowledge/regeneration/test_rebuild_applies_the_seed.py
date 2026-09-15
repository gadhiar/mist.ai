"""`rebuild()` seeds staging before it replays, and refuses if the seed wrote nothing.

## The three things this pins

**1. ORDER.** The seed-apply runs BEFORE the replay loop, and the two writers do
not commute. Seed does `ON MATCH SET n += $properties` (`seed/applier.py:62`) --
an unconditional clobber of every authored property. Extraction does
`display_name = CASE WHEN size(e.display_name) < size($display_name) ...`
(`curation/graph_writer.py:251-256`) -- longest-wins. Live applies seed first
(`mist_admin seed` then `mist_admin hydrate`), so replayed facts reconcile ONTO
seeded nodes. Seeding after the loop would let `n += $properties` overwrite
values the replay resolved by the longest-wins rule, producing a different graph
from identical inputs. `display_name` and `description` are in no exclusion
frozenset, so the gate SEES the difference -- and reads it as non-determinism,
which is the wrong diagnosis for an ordering bug.

The retired copy-forward sat AFTER the loop. Its replacement does not go where
it was, and that is the single most likely mistake here: a post-loop composition
step is the natural place to add "one more thing".

**2. NON-VACUITY, before the replay rather than after it.** MIS-130's hazard:
delete the copies, wire a seed-apply that writes zero nodes, and every existing
gate stays green -- determinism passes because two empty self-models are
byte-identical, and `live == rebuilt` passes because the compared surface never
reads that partition. `assert_seed_applied` is the observable that distinguishes
"applied nothing" from "applied something", and it reads a COUNT at the apply
site. That is what lets this land before the surface extension in step C
rather than after it.

**3. DETERMINISM OF THE STAMP.** The seeder is handed an epoch-derived
`now_iso`, never the wall clock. Two rebuilds of one epoch must stamp identical
`created_at`/`updated_at` or `assert_rebuild_twice_identical` fails on
wall-clock noise rather than on content -- a RED that says nothing about
determinism while looking exactly like one that does.

## Why the floor is a required argument

Same reasoning MIS-137 applied to `--expect-turns`: a defaulted floor is a
number nobody chose that the run then reports as passed. `min_seed_nodes` is
keyword-only and undefaulted, so a caller that has not thought about the floor
cannot start a rebuild at all.
"""

import pytest

from backend.knowledge.regeneration.log_regenerator import LogRegenerator
from backend.knowledge.regeneration.rebuild_gate import RebuildVacuityError

REBUILD_ARGS = {
    "staging_uri": "bolt://mist-neo4j-staging:7687",
    "live_uri": "bolt://mist-neo4j:7687",
    "epoch": {
        "epoch_id": 1,
        "ontology_version": "1.4.0",
        "extraction_version": "2026-06-14-r5",
        "model_hash": "test-model-hash",
        "activated_at": "2026-07-01T08:00:00+00:00",
    },
}


class TestSeedRunsBeforeTheReplayLoop:
    @pytest.mark.asyncio
    async def test_the_seed_is_applied_before_the_first_turn_is_curated(self, regenerator_factory):
        order: list[str] = []
        regen = regenerator_factory(order_sink=order)

        await regen.rebuild(**REBUILD_ARGS, min_seed_nodes=1)

        assert "seed" in order, "the rebuild never applied the seed at all"
        assert "curate" in order, "the rebuild never reached the replay loop"
        assert order.index("seed") < order.index("curate"), (
            f"seed-apply ran AFTER the replay loop (order: {order}). Seed's "
            "`n += $properties` would overwrite what the replay resolved by "
            "longest-wins, producing a different graph from identical inputs -- and "
            "the determinism gate would report it as non-determinism."
        )

    @pytest.mark.asyncio
    async def test_the_seeder_is_stamped_with_the_epoch_not_the_clock(self, regenerator_factory):
        stamps: list[str] = []
        regen = regenerator_factory(seed_stamps=stamps)

        await regen.rebuild(**REBUILD_ARGS, min_seed_nodes=1)

        assert stamps == [REBUILD_ARGS["epoch"]["activated_at"]], (
            f"the seeder was stamped {stamps}, expected the epoch's activated_at. A "
            "wall-clock stamp makes two rebuilds of one epoch differ on timestamps "
            "alone."
        )


class TestSeedFloorsAreEnforced:
    @pytest.mark.asyncio
    async def test_a_seed_that_wrote_nothing_refuses_the_rebuild(self, regenerator_factory):
        regen = regenerator_factory(seed_nodes=0)

        with pytest.raises(RebuildVacuityError, match="seed gate FAILED"):
            await regen.rebuild(**REBUILD_ARGS, min_seed_nodes=1)

    @pytest.mark.asyncio
    async def test_the_refusal_happens_before_any_turn_is_curated(self, regenerator_factory):
        """Refuse before doing work, not after. MIS-137's shape."""
        order: list[str] = []
        regen = regenerator_factory(order_sink=order, seed_nodes=0)

        with pytest.raises(RebuildVacuityError):
            await regen.rebuild(**REBUILD_ARGS, min_seed_nodes=1)

        assert "curate" not in order, (
            "the rebuild replayed turns after its seed floor had already failed. A "
            "gate that fires only once the expensive work is done is a report, not a "
            "gate."
        )

    @pytest.mark.asyncio
    async def test_a_partial_seed_below_the_floor_refuses(self, regenerator_factory):
        regen = regenerator_factory(seed_nodes=3)

        with pytest.raises(RebuildVacuityError, match="21"):
            await regen.rebuild(**REBUILD_ARGS, min_seed_nodes=21)

    @pytest.mark.asyncio
    async def test_a_failed_embedding_gate_refuses_the_rebuild(self, regenerator_factory):
        regen = regenerator_factory(embedding_gate_passed=False)

        with pytest.raises(RebuildVacuityError, match="embedding gate FAILED"):
            await regen.rebuild(**REBUILD_ARGS, min_seed_nodes=1)

    @pytest.mark.asyncio
    async def test_a_vacuous_embedding_gate_refuses_the_rebuild(self, regenerator_factory):
        """Passed, having examined nothing -- the shape of both live losses."""
        regen = regenerator_factory(embedding_examined=0)

        with pytest.raises(RebuildVacuityError, match="examined 0"):
            await regen.rebuild(**REBUILD_ARGS, min_seed_nodes=1)


class TestTheFloorMustBeChosen:
    @pytest.mark.asyncio
    async def test_rebuild_refuses_to_run_without_a_seed_floor(self, regenerator_factory):
        """Undefaulted on purpose: an unchosen floor is not a floor."""
        regen = regenerator_factory()

        with pytest.raises(TypeError, match="min_seed_nodes"):
            await regen.rebuild(**REBUILD_ARGS)


class TestTheSeederIsRequired:
    def test_a_regenerator_cannot_be_constructed_without_a_seeder(self):
        with pytest.raises(TypeError, match="staging_seeder"):
            LogRegenerator(
                event_store=object(),
                extraction_cache=object(),
                staging_curation_pipeline=object(),
                journal=object(),
                confidence_scorer=object(),
                temporal_resolver=object(),
                normalizer=object(),
                validator=object(),
            )


class TestTheRunRecordsItsSeed:
    @pytest.mark.asyncio
    async def test_the_report_states_how_many_seed_nodes_were_written(self, regenerator_factory):
        """The domain a run took, recorded rather than asserted by a plan. MIS-138."""
        regen = regenerator_factory(seed_nodes=21)

        report = await regen.rebuild(**REBUILD_ARGS, min_seed_nodes=21)

        assert report.seed_nodes_written == 21
