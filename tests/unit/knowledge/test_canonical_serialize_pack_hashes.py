"""The canonical form excludes the two pack hashes with the other epoch stamps.

An edge is stamped with the pack hashes that were active when it was written
(`RebuildStamps`). A rebuild re-stamps every edge with today's, so a graph
written under one rails hash and its rebuild under another must still serialize
equal when their facts are equal: the proof is "same log + same epoch => same
facts", not "same stamps".
"""

from __future__ import annotations

import pytest

from backend.knowledge import canonical_serialize as cs
from backend.knowledge.canonical_serialize import _canon_props, canonical_graph_form
from tests.mocks.neo4j import FakeNeo4jConnection

BOTH_FIELDS = pytest.mark.parametrize("field", ["pack_prompt_hash", "pack_rails_hash"])


def _conn(nodes, rels):
    return FakeNeo4jConnection(
        query_responses={
            "RETURN n.id AS id": nodes,
            "RETURN s.id AS source": rels,
        }
    )


def _graph(*, prompt_hash: str | None, rails_hash: str | None, target: str = "rust"):
    stamps = {
        "ontology_version": "1.4.0",
        "extraction_version": "2026-11-11-r9",
        "model_hash": "model-x",
        "pack_prompt_hash": prompt_hash,
        "pack_rails_hash": rails_hash,
    }
    nodes = [
        {
            "id": "rust",
            "labels": ["Technology"],
            "properties": {"display_name": "Rust", "source_utterance_id": "e1", **stamps},
        }
    ]
    rels = [
        {
            "source": "user",
            "type": "USES",
            "target": target,
            "properties": {"source_utterance_id": "e1", "confidence": 0.9, **stamps},
        }
    ]
    return _conn(nodes, rels)


class TestPackHashesAreEpochStamps:
    @BOTH_FIELDS
    def test_each_hash_is_in_the_epoch_stamp_set(self, field):
        assert field in cs.EPOCH_STAMP_FIELDS

    @BOTH_FIELDS
    def test_each_hash_is_dropped_from_nodes_and_edges(self, field):
        props = {"display_name": "Rust", field: "a" * 64}

        assert _canon_props(props, is_node=True) == {"display_name": "Rust"}
        assert _canon_props(props, is_node=False) == {"display_name": "Rust"}

    def test_every_stamp_a_writer_carries_is_excluded(self):
        """T45 Acceptance 3: `canonical_serialize` still excludes every stamp.

        Derived from `RebuildStamps`, so a stamp field added to the writer later
        fails here until the exclusion follows it.
        """
        import dataclasses

        from backend.knowledge.curation.graph_writer import RebuildStamps

        stamp_fields = {f.name for f in dataclasses.fields(RebuildStamps)}

        assert stamp_fields <= cs.EPOCH_STAMP_FIELDS
        assert _canon_props(dict.fromkeys(stamp_fields, "x"), is_node=False) == {}


class TestTwoRailsHashesSerializeEqual:
    def test_a_graph_written_under_two_rails_hashes_serializes_equal(self):
        written = _graph(prompt_hash="a" * 64, rails_hash="b" * 64)
        rebuilt = _graph(prompt_hash="a" * 64, rails_hash="c" * 64)

        assert canonical_graph_form(written) == canonical_graph_form(rebuilt)

    def test_a_graph_written_before_the_hashes_equals_its_stamped_rebuild(self):
        legacy = _graph(prompt_hash=None, rails_hash=None)
        rebuilt = _graph(prompt_hash="a" * 64, rails_hash="b" * 64)

        assert canonical_graph_form(legacy) == canonical_graph_form(rebuilt)

    def test_a_different_fact_under_the_same_hashes_still_differs(self):
        """The exclusion hides the stamps only: a changed fact is still seen."""
        one = _graph(prompt_hash="a" * 64, rails_hash="b" * 64, target="rust")
        other = _graph(prompt_hash="a" * 64, rails_hash="b" * 64, target="zig")

        assert canonical_graph_form(one) != canonical_graph_form(other)
