"""The memory trace's refs join the tools' refs, whichever way a tool returned the edge.

A graph candidate carries `ref = graph_ref(subject_id, rel_type, object_id)` from its `FactRow`,
which is directed (`_CURRENT_FACTS_CYPHER` matches `(s)-[r]->(o)`), and
`alt_ref = graph_ref(object_id, rel_type, subject_id)`. The tools' refs come from `fact_ref` over
the facts `KnowledgeRetriever._gather_facts` builds from two queries that do not keep the stored
direction:

- `get_user_relationships_to_entities` matches `(user)-[r]-(entity)` and the retriever always
  makes `user_id` the subject (knowledge_retriever.py, `subject=user_id`), so an edge stored
  into the user comes back reversed;
- `get_entity_neighborhood` matches the path undirected and returns each hop in path order
  (`nodes[idx]` as source, `nodes[idx+1]` as target), so a hop walked against the stored
  direction comes back reversed.

The rows below are shaped like each query's RETURN clause and run through the retriever's own
mapping; the FactRows are built by the fact source's own record mapping from rows shaped like
`_CURRENT_FACTS_CYPHER`.
"""

from __future__ import annotations

import asyncio
import hashlib
import re

import pytest

from backend.knowledge.models import RetrievedFact, fact_ref, graph_ref, vault_ref
from backend.knowledge.retrieval.knowledge_retriever import KnowledgeRetriever
from backend.knowledge.storage.partitions import USER_ENTITY_ID
from backend.memory.fact_source import _fact_row
from backend.memory.models import FactRow, MemoryCandidate
from tests.mocks.config import build_test_config

_REF = re.compile(r"^(graph|vault):[0-9a-f]{20}$")

# Stored edges, as `(s)-[r]->(o)` holds them. RELATED_TO is undirected, so reconciliation stores
# it with the lexically smaller id first: "acme" < "user" puts the user at the target.
_STORED = [
    ("user", "USES", "python"),
    ("acme", "RELATED_TO", "user"),
    ("mist", "ADAPTED_FOR", "user"),
    ("python", "USED_FOR", "garden_planner"),
]


def _fact_rows() -> list[FactRow]:
    return [
        _fact_row(
            {
                "page_key": f"e{index}",
                "subject_id": subject_id,
                "subject_name": subject_id.title(),
                "rel_type": rel_type,
                "object_id": object_id,
                "object_name": object_id.title(),
                "version_key": f"v{index}",
                "valid_from": "-inf",
                "valid_to": None,
                "recorded_at": "2026-11-01T10:00:00+00:00",
                "confidence": 0.9,
                "source_utterance_id": f"utt-{index}",
            }
        )
        for index, (subject_id, rel_type, object_id) in enumerate(_STORED)
    ]


def _graph_candidate(row: FactRow, rank: int) -> MemoryCandidate:
    return MemoryCandidate(
        source="graph",
        ref=graph_ref(row.subject_id, row.rel_type, row.object_id),
        alt_ref=graph_ref(row.object_id, row.rel_type, row.subject_id),
        rank=rank,
        score=0.5,
        item=None,
        disposition="dropped_budget",
    )


class _ToolGraphStore:
    """Returns rows shaped like the two tool queries' RETURN clauses, for the stored edges."""

    def get_user_relationships_to_entities(self, user_id, entity_ids, relationship_types=None):
        # RETURN user.id, entity.id, entity.entity_type, type(r), properties(r), direction.
        rows = []
        for subject_id, rel_type, object_id in _STORED:
            if user_id not in (subject_id, object_id):
                continue
            entity_id = object_id if subject_id == user_id else subject_id
            if entity_id not in entity_ids:
                continue
            rows.append(
                {
                    "user_id": user_id,
                    "entity_id": entity_id,
                    "entity_type": "Technology",
                    "relationship_type": rel_type,
                    "properties": {},
                    "direction": "outgoing" if subject_id == user_id else "incoming",
                }
            )
        return rows

    def get_entity_neighborhood(self, entity_id, max_hops=2, relationship_types=None):
        # One row per hop of each path, in path order: nodes[idx] -> nodes[idx + 1].
        paths = {
            # python -[USES]- user walks USES against its stored direction.
            "python": [[("python", "USES", "user")], [("python", "USED_FOR", "garden_planner")]],
            # garden_planner -[USED_FOR]- python -[USES]- user walks both hops backwards.
            "garden_planner": [
                [("garden_planner", "USED_FOR", "python"), ("python", "USES", "user")]
            ],
        }
        rows = []
        for path in paths.get(entity_id, []):
            for source, rel_type, target in path:
                rows.append(
                    {
                        "path_length": len(path),
                        "source": source,
                        "source_type": "Technology",
                        "relationship": rel_type,
                        "target": target,
                        "target_type": "Technology",
                        "properties": {},
                    }
                )
        return rows


def _tool_facts() -> list[RetrievedFact]:
    retriever = KnowledgeRetriever(config=build_test_config(), graph_store=_ToolGraphStore())
    similar = [
        {"entity_id": entity_id, "similarity": 0.8}
        for entity_id in ("python", "acme", "mist", "garden_planner")
    ]
    return asyncio.run(
        retriever._gather_facts(
            user_id=USER_ENTITY_ID, similar_entities=similar, max_hops=3, filters=None
        )
    )


def _orientation(fact: RetrievedFact, candidates: list[MemoryCandidate]) -> list[str]:
    tool_ref = fact_ref(fact)
    found = []
    for candidate in candidates:
        if tool_ref == candidate.ref:
            found.append("ref")
        elif tool_ref == candidate.alt_ref:
            found.append("alt_ref")
    return found


def test_every_tool_ref_matches_exactly_one_candidate_ref_or_alt_ref():
    candidates = [_graph_candidate(row, rank) for rank, row in enumerate(_fact_rows(), start=1)]
    facts = _tool_facts()

    shapes = {(fact.subject, fact.predicate, fact.object): fact for fact in facts}
    # Both queries' shapes are present, so the test covers each one.
    assert set(shapes) == {
        ("user", "USES", "python"),
        ("user", "RELATED_TO", "acme"),
        ("user", "ADAPTED_FOR", "mist"),
        ("python", "USES", "user"),
        ("python", "USED_FOR", "garden_planner"),
        ("garden_planner", "USED_FOR", "python"),
    }
    for shape, fact in shapes.items():
        assert len(_orientation(fact, candidates)) == 1, f"no single candidate for {shape}"


@pytest.mark.parametrize(
    ("shape", "expected"),
    [
        # The user query, edge stored out of the user: path order equals the stored order.
        (("user", "USES", "python"), "ref"),
        # The user query, edges stored into the user: the user is made the subject.
        (("user", "RELATED_TO", "acme"), "alt_ref"),
        (("user", "ADAPTED_FOR", "mist"), "alt_ref"),
        # The neighbourhood query, hops walked with and against the stored direction.
        (("python", "USED_FOR", "garden_planner"), "ref"),
        (("python", "USES", "user"), "alt_ref"),
        (("garden_planner", "USED_FOR", "python"), "alt_ref"),
    ],
)
def test_a_reversed_tool_edge_matches_the_alt_ref(shape, expected):
    candidates = [_graph_candidate(row, rank) for rank, row in enumerate(_fact_rows(), start=1)]
    fact = next(f for f in _tool_facts() if (f.subject, f.predicate, f.object) == shape)

    assert _orientation(fact, candidates) == [expected]


def test_the_user_subject_is_the_fact_rows_subject_id():
    """The tools' `user_id` and FactRow's `subject_id` are the same stored id, `user`."""
    user_rows = [row for row in _fact_rows() if row.subject_id == USER_ENTITY_ID]

    assert [row.rel_type for row in user_rows] == ["USES"]
    assert USER_ENTITY_ID == "user"


def test_a_different_predicate_matches_no_candidate():
    candidates = [_graph_candidate(row, rank) for rank, row in enumerate(_fact_rows(), start=1)]
    stray = RetrievedFact(
        subject="user",
        subject_type="Person",
        predicate="DISLIKES",
        object="python",
        object_type="Technology",
        properties={},
        similarity_score=0.5,
        graph_distance=0,
    )

    assert _orientation(stray, candidates) == []


class TestGraphRef:
    def test_is_the_hash_fact_ref_gives_a_graph_fact(self):
        fact = RetrievedFact(
            subject="user",
            subject_type="Person",
            predicate="USES",
            object="python",
            object_type="Technology",
            properties={"confidence": 0.9},
            similarity_score=0.7,
            graph_distance=0,
        )

        assert graph_ref("user", "USES", "python") == fact_ref(fact)

    def test_hashes_the_pipe_joined_triple(self):
        digest = hashlib.sha256(b"user|USES|python").hexdigest()[:20]

        assert graph_ref("user", "USES", "python") == f"graph:{digest}"

    def test_keeps_the_stated_order(self):
        assert graph_ref("user", "USES", "python") != graph_ref("python", "USES", "user")
        assert _REF.match(graph_ref("python", "USES", "user"))


class TestVaultRef:
    @pytest.mark.parametrize("heading", ["Plans", "(file)"])
    def test_is_the_hash_fact_ref_gives_a_vault_fact(self, heading):
        """The retriever keys a file-level chunk with the heading `(file)`."""
        fact = RetrievedFact(
            subject="VaultNote",
            subject_type="VaultSession",
            predicate="MENTIONS",
            object=heading,
            object_type="VaultChunk",
            properties={"path": "notes/garden.md", "text": "Tomatoes go in after the frost."},
            similarity_score=0.6,
            graph_distance=99,
        )

        ref = vault_ref("notes/garden.md", heading, "Tomatoes go in after the frost.")

        assert ref == fact_ref(fact)
        assert _REF.match(ref)

    def test_two_sections_under_one_heading_differ_by_text(self):
        first = vault_ref("notes/garden.md", "Plans", "Tomatoes.")
        second = vault_ref("notes/garden.md", "Plans", "Beans.")

        assert first != second
