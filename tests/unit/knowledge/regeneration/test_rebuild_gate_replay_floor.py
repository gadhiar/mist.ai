"""MIS-131: a floor that bounds the REPLAY-derived subset, not the whole graph.

`assert_canonical_form_non_vacuous` counts every node in the canonical form,
and says so in its own docstring: "the floor counts ALL nodes in the form,
including seed-derived ones, so on a seeded graph it does NOT establish that
the REPLAY produced anything."

That caveat is not hypothetical here. 100% of today's live graph is seed
content -- 32 nodes, 0 conversation turns -- so a whole-graph floor of any size
up to 32 is satisfied by seed alone, with the replay having produced literally
nothing. The gate would compare two identical seed-shaped graphs and pass.

The discriminator comes from the R1.4.6 hydration design (T5): a
reconciliation-written edge carries `version_key`, `source_utterance_id`,
`recorded_at` and a currency triple, where a seed-written edge carries the
two-property seed shape. So "did the replay produce anything" is answerable
from the canonical form itself, by SHAPE rather than by count.

Both markers are required, not either: `canonical_serialize`'s `_rel_key`
already sorts on both, so both survive into the form, and a seed edge carrying
one by accident should not count as replay output.
"""

from __future__ import annotations

import json

import pytest

from backend.knowledge.regeneration.rebuild_gate import (
    REPLAY_EDGE_MARKERS,
    RebuildVacuityError,
    assert_extraction_cache_non_vacuous,
    assert_replay_derived_non_vacuous,
    assert_turns_processed,
    count_replay_derived_edges,
)


def _form(*relationships, nodes=None):
    return json.dumps(
        {
            "nodes": nodes if nodes is not None else [{"id": "n1"}],
            "relationships": list(relationships),
        }
    )


def _replay_edge(source="a", target="b"):
    return {
        "source": source,
        "type": "USES",
        "target": target,
        "properties": {
            "source_utterance_id": "evt-1",
            "version_key": "v1",
            "recorded_at": "2025-09-02T08:00:00+00:00",
        },
    }


def _seed_edge(source="s", target="t"):
    """The two-property seed shape the T5 acceptance test contrasts against."""
    return {
        "source": source,
        "type": "USES",
        "target": target,
        "properties": {"seed_version": "profile-v1", "valid_from": "2025-01-01"},
    }


class TestCountReplayDerivedEdges:
    def test_counts_only_edges_carrying_both_markers(self):
        form = _form(_replay_edge(), _seed_edge(), _replay_edge("c", "d"))
        assert count_replay_derived_edges(form) == 2

    def test_a_seed_shaped_graph_counts_zero(self):
        """Today's live graph, exactly: all seed, no replay."""
        assert count_replay_derived_edges(_form(_seed_edge(), _seed_edge("u", "v"))) == 0

    @pytest.mark.parametrize("marker", sorted(REPLAY_EDGE_MARKERS))
    def test_one_marker_alone_does_not_count(self, marker):
        """Both required. A seed edge that happens to carry one is not replay output."""
        edge = _seed_edge()
        edge["properties"][marker] = "x"
        assert count_replay_derived_edges(_form(edge)) == 0

    def test_an_empty_graph_counts_zero(self):
        assert count_replay_derived_edges(_form()) == 0


class TestAssertReplayDerivedNonVacuous:
    def test_passes_at_the_floor(self):
        assert_replay_derived_non_vacuous(_form(_replay_edge()), minimum_edges=1)

    def test_refuses_below_the_floor(self):
        with pytest.raises(RebuildVacuityError, match="1 replay-derived"):
            assert_replay_derived_non_vacuous(_form(_replay_edge()), minimum_edges=2)

    def test_a_fully_seeded_graph_refuses_however_many_nodes_it_has(self):
        """The whole point, stated as a test.

        Thirty-two nodes and zero replay edges is precisely today's live graph.
        A whole-graph node floor passes it; this must not.
        """
        nodes = [{"id": f"n{i}"} for i in range(32)]
        form = _form(_seed_edge(), nodes=nodes)
        with pytest.raises(RebuildVacuityError, match="0 replay-derived"):
            assert_replay_derived_non_vacuous(form, minimum_edges=1)

    def test_a_malformed_form_refuses_rather_than_counting_zero(self):
        """Unparsable is not the same as empty, and must not read as either."""
        with pytest.raises(RebuildVacuityError, match="not a canonical graph form"):
            assert_replay_derived_non_vacuous("{not json}", minimum_edges=1)

    def test_a_zero_floor_is_rejected_as_meaningless(self):
        """A floor of zero is satisfied by anything, including nothing.

        Accepting it would let a caller "satisfy" this gate while proving
        exactly what the gate exists to disprove.
        """
        with pytest.raises(ValueError, match="at least 1"):
            assert_replay_derived_non_vacuous(_form(_replay_edge()), minimum_edges=0)


class TestAssertTurnsProcessed:
    def test_exact_match_passes(self):
        assert_turns_processed(processed=87, expected=87)

    def test_a_short_run_refuses(self):
        with pytest.raises(RebuildVacuityError, match="87"):
            assert_turns_processed(processed=60, expected=87)

    def test_an_over_run_also_refuses(self):
        """Equality, not a floor.

        More turns than the corpus means the store was not empty, which shifts
        every hydration-clock key -- the same divergence preflight checks for,
        caught here from the other end.
        """
        with pytest.raises(RebuildVacuityError):
            assert_turns_processed(processed=90, expected=87)

    def test_zero_expected_is_rejected(self):
        with pytest.raises(ValueError):
            assert_turns_processed(processed=0, expected=0)


class TestAssertExtractionCacheNonVacuous:
    def test_passes_when_enough_rows_carry_payloads(self):
        rows = [{"outcome": "extracted", "entities": [{"id": "e"}]} for _ in range(3)]
        assert_extraction_cache_non_vacuous(rows, minimum=3)

    def test_skipped_rows_do_not_count(self):
        """A recorded decision to skip is not extraction output."""
        rows = [{"outcome": "skipped", "entities": []} for _ in range(5)]
        with pytest.raises(RebuildVacuityError, match="0 "):
            assert_extraction_cache_non_vacuous(rows, minimum=1)

    def test_extracted_rows_with_empty_payloads_do_not_count(self):
        """The failure this catches is a cache full of successful nothings.

        `outcome='extracted'` with an empty entity list is what a truncated or
        refused model response records. Counting rows alone would report a
        healthy cache built entirely from them.
        """
        rows = [{"outcome": "extracted", "entities": [], "relationships": []} for _ in range(9)]
        with pytest.raises(RebuildVacuityError):
            assert_extraction_cache_non_vacuous(rows, minimum=1)

    def test_relationships_alone_are_a_payload(self):
        """An utterance can yield a relationship between known entities."""
        rows = [{"outcome": "extracted", "entities": [], "relationships": [{"type": "USES"}]}]
        assert_extraction_cache_non_vacuous(rows, minimum=1)


class TestCacheRowsMayBeAnIterable:
    """Cloud/code review: the gate called len(rows) after consuming rows.

    `rows` is untyped and a cursor or generator is the natural shape for
    "cache rows". The failure branch calls `len(rows)` for its message, so a
    generator turned this gate's diagnosis into a TypeError raised from inside
    its own error path -- a guard that crashes instead of explaining.
    """

    def test_a_generator_of_substantive_rows_passes(self):
        rows = ({"outcome": "extracted", "entities": [{"id": "e"}]} for _ in range(3))
        assert_extraction_cache_non_vacuous(rows, minimum=3)

    def test_a_generator_that_fails_still_produces_the_diagnosis(self):
        """The actual bug: the error path, not the happy path."""
        rows = ({"outcome": "skipped", "entities": []} for _ in range(4))
        with pytest.raises(RebuildVacuityError, match="of 4 row"):
            assert_extraction_cache_non_vacuous(rows, minimum=1)
