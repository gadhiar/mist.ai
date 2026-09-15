"""MIS-139: a whole partition PAIR was invisible to the gate, under every switch.

## The hole

`backend/knowledge/admin.py` had exactly three relationship-collection shapes:

- `_dump_subgraph(conn, label)` -- `MATCH (s:{label})-[r]->(t:{label})`, which
  requires BOTH endpoints to carry the same label, so it is intra-partition only.
- `cross_layer_edges` -- entity <-> provenance, both directions.
- `self_model_cross_layer_edges` -- entity <-> self-model, both directions.

There was no fourth. A self-model <-> provenance edge was therefore absent from
the canonical form with `include_provenance=True` AND `include_self_model=True`
together -- not merely off by default, but unreachable by any combination.

## What lives in that pair

The ontology permits three, all with a `MistIdentity` source: `LEARNED_SELF` ->
`LearningEvent` (written `MERGE (le:__Provenance__:LearningEvent ...)`,
`curation/graph_writer.py:456,489`), `DERIVED_FROM` -> VectorChunk /
ExternalSource / VaultNote, and `RELATED_TO` to any provenance target.
`LEARNED_SELF` is emitted on a belief change on the live path. A rebuild drops
it and no canonical form could report the drop.

## Why this belongs with the surface extension, not with the regenerator

Design section 8.2 named `LEARNED_SELF` as missing from the regenerator's
cross-layer edge list. Fixing the REGENERATOR would not have made the fix
observable: the gate had no term to compare it against. This is a
comparison-surface defect, and it lands in the same commit as the surface
extension because that is the first commit in which the answer is observable at
all.

## The asymmetry that made it easy to miss

`count_relationships_by_partition` (`admin.py:886-915`) DOES bucket all nine
ordered pairs, so `graph-stats` would show a `__SelfModel__ -> __Provenance__`
count. The census could see the pair the gate could not -- so a divergence was
diagnosable after the fact and never gate-visible.

## Urgency, stated honestly

Zero such edges exist on live today (read-only census 2026-09-14: 10
`:__Entity__` and 20 `:__SelfModel__` intra-partition edges, no cross-layer edge
of any kind). This is latent. It stops being latent the moment a corpus produces
one, which the 87-turn hydration may well do.
"""

from __future__ import annotations

import json

from backend.knowledge import admin
from backend.knowledge.canonical_serialize import canonical_graph_form
from tests.mocks.neo4j import FakeNeo4jConnection

_SM_PROV_QUERY = "WHERE (s:__SelfModel__ AND t:__Provenance__)"


def _connection(sm_prov_edges=()):
    return FakeNeo4jConnection(
        query_responses={
            "MATCH (n:__Entity__)": [],
            "MATCH (s:__Entity__)-[r]->(t:__Entity__)": [],
            "MATCH (n:__SelfModel__)": [],
            "MATCH (s:__SelfModel__)-[r]->(t:__SelfModel__)": [],
            "MATCH (n:__Provenance__)": [],
            "MATCH (s:__Provenance__)-[r]->(t:__Provenance__)": [],
            "WHERE (s:__Entity__ AND t:__Provenance__)": [],
            "WHERE (s:__Entity__ AND t:__SelfModel__)": [],
            _SM_PROV_QUERY: list(sm_prov_edges),
        }
    )


def _edge(source="mist-identity", type_="LEARNED_SELF", target="le-1"):
    return {"source": source, "type": type_, "target": target, "properties": {}}


class TestTheFourthClauseExists:
    def test_both_switches_together_emit_the_pair(self):
        """The combination the ticket names: previously unreachable."""
        payload = admin.dump_graph_json(
            _connection([_edge()]), include_provenance=True, include_self_model=True
        )

        assert "self_model_provenance_edges" in payload, (
            "no key for self-model <-> provenance even with BOTH switches on. A "
            "dropped LEARNED_SELF edge is unobservable to every gate."
        )
        assert payload["self_model_provenance_edges"] == [_edge()]

    def test_the_default_surface_is_unchanged(self):
        payload = admin.dump_graph_json(_connection([_edge()]))
        assert "self_model_provenance_edges" not in payload

    def test_provenance_alone_does_not_emit_the_pair(self):
        """One partition is not enough: the edge spans two."""
        payload = admin.dump_graph_json(_connection([_edge()]), include_provenance=True)
        assert "self_model_provenance_edges" not in payload

    def test_self_model_alone_does_not_emit_the_pair(self):
        payload = admin.dump_graph_json(_connection([_edge()]), include_self_model=True)
        assert "self_model_provenance_edges" not in payload


class TestCanonicalForm:
    def test_the_pair_is_canonicalised_into_the_form(self):
        form = json.loads(
            canonical_graph_form(
                _connection([_edge()]), include_provenance=True, include_self_model=True
            )
        )
        assert "self_model_provenance_edges" in form
        assert form["self_model_provenance_edges"][0]["target"] == "le-1"

    def test_ordering_is_deterministic_regardless_of_query_order(self):
        """Two graphs with identical content must serialise identically."""
        a = _edge(type_="LEARNED_SELF", target="le-1")
        b = _edge(type_="DERIVED_FROM", target="vc-9")

        forward = canonical_graph_form(
            _connection([a, b]), include_provenance=True, include_self_model=True
        )
        reverse = canonical_graph_form(
            _connection([b, a]), include_provenance=True, include_self_model=True
        )

        assert forward == reverse, (
            "the new key is not sorted, so two identical graphs produce different "
            "forms and the equality gate reports a diff caused by query order."
        )

    def test_a_dropped_edge_changes_the_form(self):
        """The whole point: the gate can now SEE the drop it previously could not."""
        with_edge = canonical_graph_form(
            _connection([_edge()]), include_provenance=True, include_self_model=True
        )
        without = canonical_graph_form(
            _connection([]), include_provenance=True, include_self_model=True
        )

        assert with_edge != without, (
            "dropping a self-model <-> provenance edge left the canonical form "
            "byte-identical, which is exactly the blindness MIS-139 reports."
        )
