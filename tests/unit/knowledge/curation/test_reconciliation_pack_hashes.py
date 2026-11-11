"""The two pack hashes on the relation edges reconciliation writes.

Every graph write that stamps the triple (`ontology_version`,
`extraction_version`, `model_hash`) stamps `pack_prompt_hash` and
`pack_rails_hash` beside it, from the same `RebuildStamps`: the hashes of the
pack the WRITER ran, not of a cached row's service. There are three such
statements, pinned by name below: the fact edge (`_apply_append`), the
structural edge (`_apply_structural`) and the EXTRACTED_FROM edge
(`CurationGraphWriter._extracted_from_clause`).

The canonical form excludes both hashes by key, so stamping them on a relation
edge does not move it.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from backend.knowledge.canonical_serialize import canonical_graph_form
from backend.knowledge.curation.graph_writer import CurationGraphWriter, RebuildStamps
from backend.knowledge.curation.reconciliation import ReconciliationEngine
from tests.mocks.neo4j import FakeGraphExecutor, FakeNeo4jConnection

PROMPT = "a" * 64
RAILS = "b" * 64
RECORDED_AT = "2026-11-11T12:00:00+00:00"

TRIPLE = {"ontology_version": "1.4.0", "extraction_version": "v-test", "model_hash": "m-test"}
WITH_HASHES = RebuildStamps(**TRIPLE, pack_prompt_hash=PROMPT, pack_rails_hash=RAILS)
WITHOUT_HASHES = RebuildStamps(**TRIPLE)

HASH_SETS = ("r.pack_prompt_hash = $pack_prompt_hash", "r.pack_rails_hash = $pack_rails_hash")
# The line `grep -rn --include=*.py "model_hash = \$model_hash" backend scripts` looks for.
TRIPLE_SET = "model_hash = $model_hash"

FACT = {"source": "user", "target": "rust", "type": "USES", "confidence": 0.9, "properties": {}}
STRUCTURAL = {
    "source": "python",
    "target": "programming-language",
    "type": "IS_A",
    "confidence": 0.9,
    "properties": {},
}


async def _write(relationship: dict, stamps: RebuildStamps) -> tuple[str, dict]:
    """Reconcile one relationship against an empty graph; return its one stamped write."""
    conn = FakeNeo4jConnection()  # every read returns []: no prior belief
    engine = ReconciliationEngine(executor=FakeGraphExecutor(conn), rebuild_stamps=stamps)
    await engine.reconcile_turn(
        [relationship], recorded_at=RECORDED_AT, event_id="e1", session_id="s1"
    )
    stamped = [(query, params) for query, params in conn.writes if TRIPLE_SET in query]
    assert len(stamped) == 1, [query for query, _ in conn.writes]
    return stamped[0]


def _on_create_and_on_match(query: str) -> tuple[str, str]:
    """A MERGE statement's ON CREATE SET text and its ON MATCH SET text ('' when it has none)."""
    after_create = query.split("ON CREATE SET ", 1)[1]
    create, _, match = after_create.partition("ON MATCH SET ")
    return create, match


class TestFactEdge:
    """`_apply_append`: `MERGE ... {version_key: $vk}`, written once, on create."""

    @pytest.mark.asyncio
    async def test_the_statement_sets_both_hashes_beside_the_triple(self):
        query, _ = await _write(FACT, WITH_HASHES)

        assert "MERGE (s)-[r:USES {version_key: $vk}]->(t)" in query
        create, match = _on_create_and_on_match(query)
        assert TRIPLE_SET in create
        for assignment in HASH_SETS:
            assert assignment in create
        assert match == ""

    @pytest.mark.asyncio
    async def test_the_values_are_the_engines_rebuild_stamps(self):
        _, params = await _write(FACT, WITH_HASHES)

        assert (params["pack_prompt_hash"], params["pack_rails_hash"]) == (PROMPT, RAILS)
        assert {name: params[name] for name in TRIPLE} == TRIPLE

    @pytest.mark.asyncio
    async def test_an_engine_whose_stamps_hold_no_hash_passes_none(self):
        """A null SET writes no property, so the edge then carries neither hash."""
        query, params = await _write(FACT, WITHOUT_HASHES)

        assert (params["pack_prompt_hash"], params["pack_rails_hash"]) == (None, None)
        for assignment in HASH_SETS:
            assert assignment in query


class TestStructuralEdge:
    """`_apply_structural`: `MERGE (s)-[r:TYPE]->(t)`; the stamps are set on create alone."""

    @pytest.mark.asyncio
    async def test_the_statement_sets_both_hashes_on_create_beside_the_triple(self):
        query, _ = await _write(STRUCTURAL, WITH_HASHES)

        assert "MERGE (s)-[r:IS_A]->(t)" in query
        create, _ = _on_create_and_on_match(query)
        assert TRIPLE_SET in create
        for assignment in HASH_SETS:
            assert assignment in create

    @pytest.mark.asyncio
    async def test_a_match_rewrites_no_stamp(self):
        """An edge that exists keeps the stamps it was created with: no row is rewritten."""
        query, _ = await _write(STRUCTURAL, WITH_HASHES)

        _, match = _on_create_and_on_match(query)
        assert match, "the structural upsert has an ON MATCH SET"
        for name in (*TRIPLE, "pack_prompt_hash", "pack_rails_hash"):
            assert f"r.{name} " not in match

    @pytest.mark.asyncio
    async def test_the_values_are_the_engines_rebuild_stamps(self):
        _, params = await _write(STRUCTURAL, WITH_HASHES)

        assert (params["pack_prompt_hash"], params["pack_rails_hash"]) == (PROMPT, RAILS)
        assert {name: params[name] for name in TRIPLE} == TRIPLE

    @pytest.mark.asyncio
    async def test_an_engine_whose_stamps_hold_no_hash_passes_none(self):
        _, params = await _write(STRUCTURAL, WITHOUT_HASHES)

        assert (params["pack_prompt_hash"], params["pack_rails_hash"]) == (None, None)


class TestEveryStatementThatStampsTheTriple:
    """The three sites the grep prints, and nothing else, stamp the triple."""

    def test_the_grep_finds_exactly_the_three_known_sites(self):
        """A fourth stamped write fails here until it is pinned to set both hashes."""
        sites: dict[str, int] = {}
        for root in ("backend", "scripts"):
            for path in sorted(Path(root).rglob("*.py")):
                count = sum(
                    TRIPLE_SET in line for line in path.read_text(encoding="utf-8").splitlines()
                )
                if count:
                    sites[path.as_posix()] = count

        assert sites == {
            "backend/knowledge/curation/graph_writer.py": 1,
            "backend/knowledge/curation/reconciliation.py": 2,
        }

    @pytest.mark.asyncio
    async def test_each_of_the_three_statements_sets_both_hashes(self):
        fact, _ = await _write(FACT, WITH_HASHES)
        structural, _ = await _write(STRUCTURAL, WITH_HASHES)
        extracted_from, params = CurationGraphWriter._extracted_from_clause(
            SimpleNamespace(_rebuild_stamps=WITH_HASHES)  # type: ignore[arg-type]
        )

        for statement in (fact, structural, extracted_from):
            assert TRIPLE_SET in statement
            for assignment in HASH_SETS:
                assert assignment in statement
        assert (params["pack_prompt_hash"], params["pack_rails_hash"]) == (PROMPT, RAILS)


def _graph_with_relation_stamps(**relation_stamps) -> FakeNeo4jConnection:
    """One node and one USES edge; `relation_stamps` are extra properties on the edge."""
    nodes = [
        {
            "id": "rust",
            "labels": ["Technology"],
            "properties": {"display_name": "Rust", "source_utterance_id": "e1"},
        }
    ]
    rels = [
        {
            "source": "user",
            "type": "USES",
            "target": "rust",
            "properties": {
                "source_utterance_id": "e1",
                "confidence": 0.9,
                **TRIPLE,
                **relation_stamps,
            },
        }
    ]
    return FakeNeo4jConnection(
        query_responses={"RETURN n.id AS id": nodes, "RETURN s.id AS source": rels}
    )


class TestCanonicalFormDoesNotMove:
    """`canonical_serialize` excludes both hashes by key, on a relation edge too."""

    def test_a_relation_edge_serializes_the_same_with_and_without_the_hashes(self):
        without = _graph_with_relation_stamps()
        stamped = _graph_with_relation_stamps(pack_prompt_hash=PROMPT, pack_rails_hash=RAILS)

        assert canonical_graph_form(stamped) == canonical_graph_form(without)
        assert "pack_prompt_hash" not in canonical_graph_form(stamped)
        assert "pack_rails_hash" not in canonical_graph_form(stamped)

    def test_another_fact_on_the_edge_still_moves_it(self):
        """The exclusion is by key: a compared property that differs is still seen."""
        one = _graph_with_relation_stamps(pack_prompt_hash=PROMPT, pack_rails_hash=RAILS)
        other = _graph_with_relation_stamps(
            pack_prompt_hash=PROMPT, pack_rails_hash=RAILS, context="at work"
        )

        assert canonical_graph_form(one) != canonical_graph_form(other)
