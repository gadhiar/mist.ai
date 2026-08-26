"""Partition-aware backup and census.

Subject: `dump_full_graph_json` and `count_nodes_by_partition` in
`backend/knowledge/admin.py`.

WHY THIS EXISTS. Every routine graph tool in this repo was scoped to
`:__Entity__`, and the live graph is 32 nodes of which only 11 carry that
label -- the other 21 are `:__SelfModel__`.

  - `dump_graph_json` -> `_dump_subgraph(connection, "__Entity__")`, plus
    `:__Provenance__` under a flag. So `mist_admin graph-dump`, the obvious
    thing to reach for before a risky operation, captured 11 of 32 nodes and
    dropped every embedding via `_strip_embedding`.
  - `count_nodes_by_type` is `MATCH (n:__Entity__)`, so `graph-stats` reported
    "11 total" for a 32-node graph -- and would report an unchanged 11 after
    the entire self-model partition was destroyed.

The one tool with the right shape, `scripts/hydration/snapshot.py`, calls
`assert_neo4j_dev_isolated` before building its config and so structurally
refuses to read the live graph. The backup that made the 2026-07-31 recovery
possible was label-agnostic, which is why it worked.

These two functions are DELIBERATELY SEPARATE from `dump_graph_json` rather
than a flag on it: `canonical_graph_form` consumes that function's default
output, so widening its scope would silently change what the rebuild
determinism gate compares.
"""

from __future__ import annotations

from backend.knowledge.admin import (
    count_nodes_by_partition,
    count_relationships_by_partition,
    dump_full_graph_json,
)
from tests.mocks.neo4j import FakeNeo4jConnection


def _rel_census_conn(rows):
    """Connection for the relationship census only.

    Keyed on `labels(s) AS source_labels`, which appears in no other query under
    test -- kept in its own helper rather than added to `_full_conn`, because
    `MATCH (s)-[r]->(t)` is a substring of the census query too and
    first-match-wins would otherwise decide which response is returned.
    """
    return FakeNeo4jConnection(query_responses={"labels(s) AS source_labels": rows})


_EMBEDDING = [0.1, 0.2, 0.3]


def _full_conn(nodes, rels):
    """FakeNeo4jConnection matches by query SUBSTRING (`if pattern in query`).

    The three keys below are chosen to be mutually disjoint across the three
    reads under test, so first-match-wins ordering cannot matter:

      - `properties(n) AS properties` -> only the full-dump NODE query
      - `MATCH (s)-[r]->(t)`          -> only the full-dump RELATIONSHIP query
      - `MATCH (n) RETURN`            -> only the census query, which is a
        single line; the dump's node query has a newline after `MATCH (n)`
        and so does not contain this substring.
    """
    return FakeNeo4jConnection(
        query_responses={
            "properties(n) AS properties": nodes,
            "MATCH (s)-[r]->(t)": rels,
            "MATCH (n) RETURN": nodes,
        }
    )


def _node(node_id: str, labels: list[str], **props) -> dict:
    return {
        "id": node_id,
        "labels": labels,
        "properties": {"id": node_id, **props},
    }


class TestDumpFullGraphJson:
    def test_captures_nodes_from_every_partition(self):
        """The defect this closes: a dump that saw only :__Entity__."""
        conn = _full_conn(
            [
                _node("rust", ["__Entity__", "Technology"]),
                _node("mist-identity", ["__SelfModel__", "MistIdentity"]),
                _node("turn-1", ["__Provenance__", "ConversationContext"]),
            ],
            [],
        )

        payload = dump_full_graph_json(conn)

        ids = {n["id"] for n in payload["nodes"]}
        assert ids == {"rust", "mist-identity", "turn-1"}

    def test_retains_embeddings(self):
        """A backup that drops vectors cannot restore a retrievable graph.

        `_strip_embedding` is correct for the analysis dump and wrong here --
        `canonical_serialize` excludes `embedding`, so a vectorless restore is
        invisible to every determinism gate in the repo.
        """
        conn = _full_conn([_node("rust", ["__Entity__"], embedding=_EMBEDDING)], [])

        payload = dump_full_graph_json(conn)

        assert payload["nodes"][0]["properties"]["embedding"] == _EMBEDDING

    def test_captures_relationships_spanning_partitions(self):
        """Cross-layer and intra-self-model edges must survive a backup.

        20 of the live graph's 30 relationships are intra-self-model, and
        `_dump_subgraph`'s `MATCH (s:L)-[r]->(t:L)` requires BOTH endpoints to
        carry the same partition label, so it captured none of them.
        """
        conn = _full_conn(
            [],
            [
                {
                    "source": "mist-identity",
                    "type": "HAS_TRAIT",
                    "target": "curious",
                    "properties": {},
                }
            ],
        )

        payload = dump_full_graph_json(conn)

        assert payload["relationships"][0]["type"] == "HAS_TRAIT"

    def test_reports_counts_alongside_the_payload(self):
        """A backup artifact must state its own size so a truncated one is detectable."""
        conn = _full_conn(
            [_node("a", ["__Entity__"]), _node("b", ["__SelfModel__"])],
            [{"source": "a", "type": "USES", "target": "b", "properties": {}}],
        )

        payload = dump_full_graph_json(conn)

        assert payload["node_count"] == 2
        assert payload["rel_count"] == 1

    def test_is_deterministic_in_node_ordering(self):
        """Two dumps of one graph must not differ by row order."""
        nodes = [_node("z", ["__Entity__"]), _node("a", ["__SelfModel__"])]

        payload = dump_full_graph_json(_full_conn(nodes, []))

        assert [n["id"] for n in payload["nodes"]] == ["a", "z"]


class TestCountNodesByPartition:
    def test_counts_every_partition_not_just_entity(self):
        """The census that makes a destroyed self-model visible."""
        conn = _full_conn(
            [
                _node("rust", ["__Entity__", "Technology"]),
                _node("mist-identity", ["__SelfModel__", "MistIdentity"]),
                _node("curious", ["__SelfModel__", "MistTrait"]),
                _node("turn-1", ["__Provenance__"]),
            ],
            [],
        )

        counts = {row["partition"]: row["count"] for row in count_nodes_by_partition(conn)}

        assert counts["__Entity__"] == 1
        assert counts["__SelfModel__"] == 2
        assert counts["__Provenance__"] == 1

    def test_reports_nodes_carrying_no_partition_label(self):
        """An unpartitioned node is invisible to every partition-scoped query.

        R1.4 T10 stripped live nodes to a bare partition label; the inverse --
        a node with NO partition label -- would be equally invisible, so the
        census must not silently drop it.
        """
        conn = _full_conn([_node("orphan", ["Technology"])], [])

        counts = {row["partition"]: row["count"] for row in count_nodes_by_partition(conn)}

        assert counts["(unpartitioned)"] == 1

    def test_total_matches_a_label_agnostic_node_count(self):
        """The census total must equal `MATCH (n)`, or it has its own blind spot."""
        nodes = [
            _node("a", ["__Entity__"]),
            _node("b", ["__SelfModel__"]),
            _node("c", []),
        ]

        rows = count_nodes_by_partition(_full_conn(nodes, []))

        assert sum(row["count"] for row in rows) == len(nodes)


def _rel(source_labels: list[str], rel_type: str, target_labels: list[str]) -> dict:
    return {
        "source_labels": source_labels,
        "type": rel_type,
        "target_labels": target_labels,
    }


class TestCountRelationshipsByPartition:
    def test_counts_intra_self_model_edges(self):
        """20 of the live graph's 30 relationships are intra-self-model.

        `count_relationships_by_type` only sees edges whose endpoints are both
        `:__Entity__`, so `graph-stats` reported 10 of 30 -- the edge-side twin
        of the node blindness this module closes.
        """
        conn = _rel_census_conn(
            [
                _rel(["__SelfModel__", "MistIdentity"], "HAS_TRAIT", ["__SelfModel__"]),
                _rel(["__SelfModel__", "MistIdentity"], "HAS_TRAIT", ["__SelfModel__"]),
                _rel(["__Entity__"], "USES", ["__Entity__"]),
            ]
        )

        counts = {row["partitions"]: row["count"] for row in count_relationships_by_partition(conn)}

        assert counts["__SelfModel__ -> __SelfModel__"] == 2
        assert counts["__Entity__ -> __Entity__"] == 1

    def test_counts_cross_layer_edges_distinctly(self):
        """A cross-partition edge must not be folded into either endpoint's bucket."""
        conn = _rel_census_conn([_rel(["__SelfModel__"], "ADAPTED_FOR", ["__Entity__"])])

        counts = {row["partitions"]: row["count"] for row in count_relationships_by_partition(conn)}

        assert counts["__SelfModel__ -> __Entity__"] == 1

    def test_total_matches_a_label_agnostic_relationship_count(self):
        """Every edge lands in exactly one bucket, including unpartitioned endpoints."""
        rows = [
            _rel(["__SelfModel__"], "HAS_TRAIT", ["__SelfModel__"]),
            _rel(["__Entity__"], "USES", ["__Entity__"]),
            _rel(["Technology"], "DANGLING", ["__Entity__"]),
        ]

        result = count_relationships_by_partition(_rel_census_conn(rows))

        assert sum(row["count"] for row in result) == len(rows)
