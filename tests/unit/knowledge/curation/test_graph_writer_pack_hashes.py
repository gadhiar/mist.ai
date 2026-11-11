"""A written EXTRACTED_FROM edge carries both pack hashes with its epoch stamps.

`RebuildStamps` is what the writer stamps every provenance edge with. The two
pack hashes ride it beside the triple: `pack_prompt_hash` for what the ontology
pack fed the model, `pack_rails_hash` for the rules that ran on the payload
before the write.
"""

from __future__ import annotations

import pytest

from backend.knowledge.curation.graph_writer import RebuildStamps
from tests.unit.knowledge.curation._graph_writer_fakes import (
    PROVENANCE_EDGE_MERGE,
    TEST_REBUILD_STAMPS,
    make_writer,
    writes_matching,
)
from tests.unit.knowledge.curation.conftest import make_entity_dict

PROMPT = "a" * 64
RAILS = "b" * 64

STAMPS_WITH_HASHES = RebuildStamps(
    ontology_version="1.4.0-test",
    extraction_version="2026-06-14-r5-test",
    model_hash="test-model-hash",
    pack_prompt_hash=PROMPT,
    pack_rails_hash=RAILS,
)


async def _edge_write(stamps: RebuildStamps) -> tuple[str, dict]:
    writer, conn = make_writer(stamps)
    await writer.write(
        entities=[make_entity_dict(entity_id="python")],
        merge_actions=[],
        event_id="evt-001",
        session_id="sess-001",
    )
    writes = writes_matching(conn, PROVENANCE_EDGE_MERGE)
    assert len(writes) == 1, "expected one statement that writes the EXTRACTED_FROM edge"
    return writes[0]


class TestTheStampsObject:
    def test_both_hashes_default_to_none(self):
        """A writer built without a pack stamps no hash: the fields are optional."""
        assert (TEST_REBUILD_STAMPS.pack_prompt_hash, TEST_REBUILD_STAMPS.pack_rails_hash) == (
            None,
            None,
        )

    @pytest.mark.parametrize("field", ["pack_prompt_hash", "pack_rails_hash"])
    def test_each_hash_takes_part_in_equality(self, field):
        """The cross-factory agreement test compares stamps with `==`; a hash must count."""
        import dataclasses

        other = dataclasses.replace(STAMPS_WITH_HASHES, **{field: "f" * 64})

        assert other != STAMPS_WITH_HASHES


class TestTheWrittenEdge:
    @pytest.mark.asyncio
    async def test_the_edge_is_given_both_hashes(self):
        """T45 Acceptance 3: a written edge carries both hashes."""
        _, params = await _edge_write(STAMPS_WITH_HASHES)

        assert (params["pack_prompt_hash"], params["pack_rails_hash"]) == (PROMPT, RAILS)

    @pytest.mark.asyncio
    @pytest.mark.parametrize("field", ["pack_prompt_hash", "pack_rails_hash"])
    async def test_each_hash_is_set_on_create_and_on_match(self, field):
        """A re-extraction overwrites the triple on the edge, and the hashes with it."""
        query, _ = await _edge_write(STAMPS_WITH_HASHES)

        assignment = f"r.{field} = ${field}"
        edge_clause = query.split(PROVENANCE_EDGE_MERGE, 1)[1]
        on_create, on_match = edge_clause.split("ON MATCH SET", 1)

        assert assignment in on_create
        assert assignment in on_match

    @pytest.mark.asyncio
    async def test_the_triple_still_rides_the_edge(self):
        _, params = await _edge_write(STAMPS_WITH_HASHES)

        assert (
            params["ontology_version"],
            params["extraction_version"],
            params["model_hash"],
        ) == ("1.4.0-test", "2026-06-14-r5-test", "test-model-hash")

    @pytest.mark.asyncio
    async def test_a_writer_without_hashes_stamps_null(self):
        """Null is the legacy value: an edge written with no pack named holds no hash."""
        _, params = await _edge_write(TEST_REBUILD_STAMPS)

        assert (params["pack_prompt_hash"], params["pack_rails_hash"]) == (None, None)
