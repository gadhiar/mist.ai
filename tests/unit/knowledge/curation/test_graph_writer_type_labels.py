"""The graph writer stores an entity's ontology type as a label beside `__Entity__`.

The `entity_type` property stays the record; the label mirrors it, set on
create only. `LabelGraphConnection` applies the label clauses of the writer's
entity upsert to an in-memory node table, so a test can assert the labels a
node ends with rather than only the statement text.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from backend.knowledge.curation.confidence import ConfidenceManager
from backend.knowledge.curation.deduplication import MergeAction
from backend.knowledge.curation.graph_writer import CurationGraphWriter
from backend.knowledge.storage.partitions import ENTITY_LABEL
from backend.knowledge.storage.type_labels import ENTITY_TYPE_LABELS
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection
from tests.unit.knowledge.curation._graph_writer_fakes import (
    ENTITY_MERGE,
    TEST_REBUILD_STAMPS,
    UPSERT_REPLAY_GUARD_FRAGMENTS,
    make_writer,
    writes_matching,
)
from tests.unit.knowledge.curation.conftest import make_entity_dict

SESSION = "sess-labels"

# `e:<Label>` items in a SET clause. `\b` keeps `le:` (the LearningEvent
# variable) and `r:` (relationship types) out.
_NODE_LABEL_ITEM = re.compile(r"\be:([A-Za-z_][A-Za-z0-9_]*)")

# The properties of the entity upsert's ON CREATE SET that the canonical form
# compares (audit fields, embeddings, confidence and epoch stamps are excluded
# there, so they are not modelled here).
_CREATE_PROPERTIES = (
    ("entity_type", "entity_type"),
    ("display_name", "display_name"),
    ("knowledge_domain", "domain"),
    ("source_type", "source_type"),
    ("description", "description"),
    ("aliases", "aliases"),
)


def _sections(query: str) -> tuple[str, str]:
    """Split the entity MERGE into its ON CREATE SET and ON MATCH SET text.

    The ON MATCH text ends at the first ` WITH e`, where the conversational
    path goes on to the LearningEvent and EXTRACTED_FROM clauses.
    """
    before, sep, rest = query.partition("ON MATCH SET")
    assert sep, query
    _, sep, on_create = before.partition("ON CREATE SET")
    assert sep, query
    on_match = rest.split(" WITH e", 1)[0]
    return on_create, on_match


class LabelGraphConnection(FakeNeo4jConnection):
    """A fake that applies the entity upsert's label and create-property clauses.

    On CREATE a node gets `__Entity__`, every `e:<Label>` item of the ON CREATE
    SET clause and every `SET e:<Label>` after ON MATCH (which Cypher runs on
    both branches). On MATCH it gets only the latter. It answers the two
    `_dump_subgraph` queries from the node table, so `canonical_graph_form`
    runs over what the writer wrote. It does not evaluate any other Cypher.
    """

    def __init__(self) -> None:
        super().__init__()
        self.nodes: dict[str, dict[str, Any]] = {}

    def execute_write(self, query, params=None):
        super().execute_write(query, params)
        if ENTITY_MERGE in query:
            self._upsert(query, params or {})
        return []

    def execute_query(self, query, params=None):
        self.queries.append((query, params))
        if "RETURN n.id AS id" in query:
            return [
                {"id": node_id, "labels": list(node["labels"]), "properties": node["properties"]}
                for node_id, node in sorted(self.nodes.items())
            ]
        return []

    def _upsert(self, query: str, params: dict[str, Any]) -> None:
        on_create, on_match = _sections(query)
        every_write = re.findall(r"\bSET e:([A-Za-z_][A-Za-z0-9_]*)", on_match)
        node = self.nodes.get(params["entity_id"])
        if node is None:
            labels = [ENTITY_LABEL]
            for label in _NODE_LABEL_ITEM.findall(on_create) + every_write:
                if label not in labels:
                    labels.append(label)
            self.nodes[params["entity_id"]] = {
                "labels": labels,
                "properties": {
                    "id": params["entity_id"],
                    **{prop: params[key] for prop, key in _CREATE_PROPERTIES},
                },
            }
            return
        for label in every_write:
            if label not in node["labels"]:
                node["labels"].append(label)

    def labels_of(self, node_id: str) -> set[str]:
        return set(self.nodes[node_id]["labels"])


def make_label_writer(conn: LabelGraphConnection) -> CurationGraphWriter:
    return CurationGraphWriter(
        executor=FakeGraphExecutor(connection=conn),
        embedding_provider=FakeEmbeddingGenerator(),
        confidence_manager=ConfidenceManager(),
        rebuild_stamps=TEST_REBUILD_STAMPS,
    )


def _matched(entity_id: str) -> MergeAction:
    """The MergeAction dedup emits when an incoming entity maps onto an existing node."""
    return MergeAction(existing_entity_id=entity_id, incoming_entity={}, merge_instructions={})


async def _write(writer, entities, merge_actions=None, event_id="evt-1") -> None:
    await writer.write(
        entities=entities,
        merge_actions=merge_actions or [],
        event_id=event_id,
        session_id=SESSION,
    )


async def _upsert_query(entity_type: str, entity_id: str = "thing") -> str:
    writer, conn = make_writer()
    await _write(writer, [make_entity_dict(entity_id=entity_id, entity_type=entity_type)])
    [(query, _params)] = writes_matching(conn, ENTITY_MERGE)
    return query


class TestTypeLabelOnCreate:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("entity_type", sorted(ENTITY_TYPE_LABELS - {"User"}))
    async def test_new_node_has_exactly_entity_and_its_type_label(self, entity_type):
        conn = LabelGraphConnection()
        await _write(make_label_writer(conn), [make_entity_dict(entity_type=entity_type)])

        assert conn.labels_of("python") == {ENTITY_LABEL, entity_type}

    @pytest.mark.asyncio
    async def test_label_is_an_item_of_the_on_create_set_clause(self):
        query = await _upsert_query("Person", entity_id="alice")

        on_create, on_match = _sections(query)
        assert _NODE_LABEL_ITEM.findall(on_create) == ["Person"]
        assert "e.provenance = 'extraction', e:Person ON MATCH SET" in query
        assert _NODE_LABEL_ITEM.findall(on_match) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "entity_type",
        ["MistIdentity", "Unknown", "Topic", "Milestone", "", "person", "Person) DETACH DELETE e"],
    )
    async def test_type_outside_the_label_set_gets_no_label(self, entity_type):
        query = await _upsert_query(entity_type)

        assert _NODE_LABEL_ITEM.findall(query) == [ENTITY_LABEL]
        conn = LabelGraphConnection()
        await _write(make_label_writer(conn), [make_entity_dict(entity_type=entity_type)])
        assert conn.labels_of("python") == {ENTITY_LABEL}

    @pytest.mark.asyncio
    async def test_statement_fragments_the_replay_fakes_match_are_unchanged(self):
        query = await _upsert_query("Technology")

        assert ENTITY_MERGE in query
        for fragment in UPSERT_REPLAY_GUARD_FRAGMENTS:
            assert fragment in query


class TestOnMatchNeverRelabels:
    @pytest.mark.asyncio
    async def test_on_match_sets_no_type_and_no_type_label(self):
        query = await _upsert_query("Organization", entity_id="acme")

        _on_create, on_match = _sections(query)
        assert "entity_type" not in on_match
        assert _NODE_LABEL_ITEM.findall(on_match) == []

    @pytest.mark.asyncio
    async def test_a_later_write_with_another_type_keeps_the_first_label(self):
        conn = LabelGraphConnection()
        writer = make_label_writer(conn)
        await _write(writer, [make_entity_dict(entity_id="acme", entity_type="Organization")])

        await _write(
            writer,
            [make_entity_dict(entity_id="acme", entity_type="Project")],
            merge_actions=[_matched("acme")],
            event_id="evt-2",
        )

        assert conn.labels_of("acme") == {ENTITY_LABEL, "Organization"}
        assert conn.nodes["acme"]["properties"]["entity_type"] == "Organization"


class TestUserNode:
    @pytest.mark.asyncio
    async def test_user_node_keeps_the_user_label_on_every_write(self):
        query = await _upsert_query("User", entity_id="user")

        on_create, on_match = _sections(query)
        assert _NODE_LABEL_ITEM.findall(on_create) == ["User"]
        assert on_match.endswith(" SET e:User")

    @pytest.mark.asyncio
    async def test_user_node_carries_user_after_create_and_after_match(self):
        conn = LabelGraphConnection()
        writer = make_label_writer(conn)
        user = make_entity_dict(entity_id="user", entity_type="User", display_name="Raj")

        await _write(writer, [user])
        assert conn.labels_of("user") == {ENTITY_LABEL, "User"}

        await _write(
            writer,
            [user],
            merge_actions=[_matched("user")],
            event_id="evt-2",
        )
        assert conn.labels_of("user") == {ENTITY_LABEL, "User"}

    @pytest.mark.asyncio
    async def test_a_user_node_already_in_the_graph_gains_the_user_label(self):
        """A user node an older writer left bare gets `:User` on its next write."""
        conn = LabelGraphConnection()
        conn.nodes["user"] = {"labels": [ENTITY_LABEL], "properties": {"id": "user"}}

        await _write(
            make_label_writer(conn),
            [make_entity_dict(entity_id="user", entity_type="User")],
            merge_actions=[_matched("user")],
        )

        assert conn.labels_of("user") == {ENTITY_LABEL, "User"}
