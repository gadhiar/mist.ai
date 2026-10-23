"""Retrieval candidates and opaque fact refs.

The turn trace (plan item 10) needs to know every memory a retrieval leg saw,
not just the facts that reached the prompt, and it must name them without
writing their text. These tests pin both halves: `retrieve()` reports every
leg row as a `RetrievalCandidate`, and `fact_ref` hashes each fact's natural
key into a ref that leaks none of it. The snapshot class pins that adding
candidates left `facts`, `formatted_context` and the counts byte-identical.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from typing import Any

import pytest

from backend.knowledge.models import (
    RetrievalCandidate,
    RetrievalFilters,
    RetrievalResult,
    RetrievedFact,
    VectorSearchResult,
    fact_ref,
)
from tests.unit.knowledge.conftest import FakeVectorStore
from tests.unit.knowledge.retrieval.test_knowledge_retriever import (
    _make_retriever,
    build_vector_search_result,
)
from tests.unit.knowledge.retrieval.test_knowledge_retriever_vault_sidecar import (
    FakeSidecar,
    _make_retriever_with_sidecar,
)

# ---------------------------------------------------------------------------
# Scenario fixtures
# ---------------------------------------------------------------------------


class _CannedVectorStore(FakeVectorStore):
    """Returns a fixed ranked result list, truncated to the requested limit."""

    def __init__(self, results: list[VectorSearchResult]) -> None:
        super().__init__()
        self._results = results

    def search(self, query_embedding, limit, filters=None):
        self._calls.append(("search", (query_embedding, limit, filters)))
        return list(self._results[:limit])


def _vault_row(path: str, heading: str | None, content: str, score: float) -> dict[str, Any]:
    return {
        "path": path,
        "heading": heading,
        "content": content,
        "score": score,
        "vector_rank": 1,
        "fts_rank": 1,
        "sources": ["vector", "fts"],
        "display_similarity": score,
    }


_SIMILAR_ENTITIES = [
    {"entity_id": "python", "entity_type": "Technology", "similarity": 0.9},
    {"entity_id": "rust", "entity_type": "Technology", "similarity": 0.7},
]
_USER_RELS = [
    {
        "relationship_type": "USES",
        "entity_id": "python",
        "entity_type": "Technology",
        "properties": {"proficiency": "expert"},
    },
    {
        "relationship_type": "LEARNING",
        "entity_id": "rust",
        "entity_type": "Technology",
        "properties": {},
    },
]
_NEIGHBORHOOD = [
    {
        "source": "python",
        "source_type": "Technology",
        "relationship": "USED_FOR",
        "target": "data-science",
        "target_type": "Topic",
        "properties": {},
        "path_length": 1,
    },
    {
        "source": "rust",
        "source_type": "Technology",
        "relationship": "RELATED_TO",
        "target": "systems",
        "target_type": "Topic",
        "properties": {},
        "path_length": 1,
    },
]
_VECTOR_RESULTS = [
    build_vector_search_result(
        chunk_id="chunk-a",
        text="Sourdough needs a starter.",
        similarity=0.92,
        source_id="src-bread",
        metadata={"title": "Bread"},
    ),
    build_vector_search_result(
        chunk_id="chunk-b", text="Proof overnight.", similarity=0.81, source_id="src-proof"
    ),
    build_vector_search_result(
        chunk_id="chunk-c",
        text="Bake at high heat.",
        similarity=0.64,
        source_id="src-oven",
        metadata={"title": "Oven"},
    ),
]
_HISTORICAL_ROWS = [
    _vault_row("/v/s/garden.md", "Tomatoes", "Planted tomatoes in May.", 0.9),
    _vault_row("/v/s/empty.md", "Empty", "", 0.8),
    _vault_row("/v/s/garden.md", "Tomatoes", "Harvested tomatoes in August.", 0.7),
    _vault_row("/v/s/blank.md", "Blank", "   \n\n  ", 0.6),
    _vault_row("/v/s/travel.md", "Lisbon", "Trams climb the hills.", 0.5),
    _vault_row("/v/s/books.md", None, "Finished the novel.", 0.4),
]
_HYBRID_ROWS = [
    _vault_row("/v/s/a.md", "Notes", "Notes from the first file.", 0.6),
    _vault_row("/v/s/hollow.md", "Hollow", "", 0.5),
    _vault_row("/v/s/b.md", "Notes", "Notes from the second file.", 0.4),
]


async def _relational_result():
    retriever, _ = _make_retriever(
        intent="relational",
        search_similar_results=_SIMILAR_ENTITIES,
        user_rels_results=_USER_RELS,
        neighborhood_results=_NEIGHBORHOOD,
    )
    return await retriever.retrieve(query="What languages do I use?", user_id="User", limit=3)


async def _factual_result():
    retriever, _ = _make_retriever(
        intent="factual", vector_store=_CannedVectorStore(_VECTOR_RESULTS)
    )
    return await retriever.retrieve(query="How do I bake bread?", user_id="User", limit=2)


async def _historical_result():
    retriever, _ = _make_retriever_with_sidecar(
        intent="historical", sidecar=FakeSidecar(rows=_HISTORICAL_ROWS)
    )
    return await retriever.retrieve(query="What happened in the garden?", limit=3)


async def _hybrid_result():
    retriever, _ = _make_retriever_with_sidecar(
        intent="hybrid",
        sidecar=FakeSidecar(rows=_HYBRID_ROWS),
        vector_store=_CannedVectorStore(_VECTOR_RESULTS[:2]),
        search_similar_results=_SIMILAR_ENTITIES[:1],
    )
    retriever.graph_store.get_user_relationships_to_entities = (  # type: ignore[method-assign]
        lambda **kw: _USER_RELS[:1]
    )
    return await retriever.retrieve(query="Tell me about my notes", limit=3)


_SCENARIOS = {
    "relational": _relational_result,
    "factual": _factual_result,
    "historical": _historical_result,
    "hybrid": _hybrid_result,
}


def _snapshot(result) -> dict[str, Any]:
    """Everything a caller could read off a result, minus timings and candidates."""
    return {
        "facts_repr_sha256": hashlib.sha256(repr(result.facts).encode("utf-8")).hexdigest(),
        "formatted_context": result.formatted_context,
        "entities_found": result.entities_found,
        "total_facts": result.total_facts,
        "document_chunks_used": result.document_chunks_used,
        "intent": result.intent,
    }


# Captured from `retrieve()` at the integration base (666eff0), before
# candidates existed. A drift here means the candidate bookkeeping changed what
# the retriever returns, which this task must not do.
_BASE_SNAPSHOTS: dict[str, dict[str, Any]] = {
    "relational": {
        "facts_repr_sha256": "e78ff958b685362c0300b1141e589cee5ea1c416dc7fff4076210595a704cfd9",
        "formatted_context": (
            "Relevant knowledge from your graph (query: 'What languages do I use?'):\n\n"
            "### User\n"
            "  - uses python (Technology) [proficiency=expert]\n"
            "  - learning rust (Technology)\n\n"
            "### python\n"
            "  - used for data-science (Topic)\n\n"
            "---\nTotal facts: 3"
        ),
        "entities_found": 5,
        "total_facts": 3,
        "document_chunks_used": 0,
        "intent": "relational",
    },
    "factual": {
        "facts_repr_sha256": "7bb2a2ddc335917b8dc0ad2d717c78483da15f084aa0f5076a274c644c830f17",
        "formatted_context": (
            "Relevant knowledge from your graph (query: 'How do I bake bread?'):\n\n"
            "### Relevant Documents\n"
            "[doc-1] Source: Bread (similarity: 0.92)\n"
            "    Sourdough needs a starter.\n"
            "[doc-2] Source: src-proof (similarity: 0.81)\n"
            "    Proof overnight.\n\n"
            "---\nTotal facts: 2"
        ),
        "entities_found": 0,
        "total_facts": 2,
        "document_chunks_used": 2,
        "intent": "factual",
    },
    "historical": {
        "facts_repr_sha256": "199f588949fea5b5bf58c1cbce4725b0c34c18f616255fc9004b3436bc81cf85",
        "formatted_context": (
            "Relevant prose from your vault (query: 'What happened in the garden?'):\n\n"
            "Planted tomatoes in May.\n\n"
            "Harvested tomatoes in August.\n\n"
            "Trams climb the hills.\n"
        ),
        "entities_found": 0,
        "total_facts": 3,
        "document_chunks_used": 0,
        "intent": "historical",
    },
    "hybrid": {
        "facts_repr_sha256": "2911c355909f7ead56846c2376b22e31a4935e0e53bfe87a00f791ff3eaaf8de",
        "formatted_context": (
            "Relevant knowledge from your graph (query: 'Tell me about my notes'):\n\n"
            "### user\n"
            "  - uses python (Technology) [proficiency=expert]\n\n"
            "### Relevant Documents\n"
            "[doc-1] Source: Notes (similarity: 0.60)\n"
            "    Notes from the first file.\n"
            "[doc-2] Source: Bread (similarity: 0.92)\n"
            "    Sourdough needs a starter.\n\n"
            "---\nTotal facts: 3"
        ),
        "entities_found": 2,
        "total_facts": 3,
        "document_chunks_used": 2,
        "intent": "hybrid",
    },
}


# ---------------------------------------------------------------------------
# Independent natural-key hashes (the contract mist-read-path re-derives)
# ---------------------------------------------------------------------------


def _h(key: str) -> str:
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:20]


def _vault_ref(path: str, heading: str, text: str) -> str:
    text_digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return "vault:" + _h(f"{path}\n{heading}\n{text_digest}")


def _graph_ref(subject: str, predicate: str, obj: str) -> str:
    return "graph:" + _h(f"{subject}|{predicate}|{obj}")


def _vector_ref(chunk_id: str) -> str:
    return "vector:" + _h(chunk_id)


def _row_vault_ref(row: dict[str, Any]) -> str:
    return _vault_ref(row["path"], row["heading"] or "(file)", row["content"])


def _selected_refs(result) -> list[str]:
    return sorted(c.ref for c in result.candidates if c.selected)


def _fact_refs(result) -> list[str]:
    return sorted(fact_ref(f) for f in result.facts)


# ---------------------------------------------------------------------------
# Contract shape
# ---------------------------------------------------------------------------


class TestRetrievalCandidateContract:
    def test_candidate_fields_match_the_trace_contract(self):
        names = [f.name for f in dataclasses.fields(RetrievalCandidate)]

        assert names == ["source", "ref", "rank", "score", "selected", "empty"]

    def test_candidate_is_frozen_and_slotted(self):
        candidate = RetrievalCandidate(
            source="graph", ref="graph:" + "0" * 20, rank=1, score=None, selected=False, empty=False
        )

        assert not hasattr(candidate, "__dict__")
        with pytest.raises(dataclasses.FrozenInstanceError):
            candidate.rank = 2  # type: ignore[misc]

    def test_result_built_without_candidates_defaults_to_empty_tuple(self):
        result = RetrievalResult(
            query="q",
            user_id="u",
            facts=[],
            entities_found=0,
            total_facts=0,
            formatted_context="",
            retrieval_time_ms=0.0,
            vector_search_time_ms=0.0,
            graph_traversal_time_ms=0.0,
            config_used={},
        )

        assert result.candidates == ()


# ---------------------------------------------------------------------------
# retrieve() fills candidates
# ---------------------------------------------------------------------------


class TestRetrieveCandidates:
    @pytest.mark.asyncio
    async def test_historical_lists_every_sidecar_row_including_empty_ones(self):
        result = await _historical_result()

        observed = [(c.source, c.rank, c.score, c.selected, c.empty) for c in result.candidates]

        assert observed == [
            ("vault", 1, 0.9, True, False),
            ("vault", 2, 0.8, False, True),
            ("vault", 3, 0.7, True, False),
            ("vault", 4, 0.6, False, True),
            ("vault", 5, 0.5, True, False),
            ("vault", 6, 0.4, False, False),
        ]

    @pytest.mark.asyncio
    async def test_historical_refs_hash_each_row_natural_key(self):
        result = await _historical_result()

        assert [c.ref for c in result.candidates] == [_row_vault_ref(r) for r in _HISTORICAL_ROWS]

    @pytest.mark.asyncio
    async def test_relational_lists_every_graph_fact_before_the_limit(self):
        result = await _relational_result()

        observed = [(c.ref, c.rank, c.score, c.selected, c.empty) for c in result.candidates]

        assert observed == [
            (_graph_ref("User", "USES", "python"), 1, 0.9, True, False),
            (_graph_ref("User", "LEARNING", "rust"), 2, 0.7, True, False),
            (_graph_ref("python", "USED_FOR", "data-science"), 3, 0.9, True, False),
            (_graph_ref("rust", "RELATED_TO", "systems"), 4, 0.7, False, False),
        ]

    @pytest.mark.asyncio
    async def test_factual_lists_every_vector_result_the_leg_returned(self):
        result = await _factual_result()

        observed = [(c.ref, c.rank, c.score, c.selected) for c in result.candidates]

        assert observed == [
            (_vector_ref("chunk-a"), 1, 0.92, True),
            (_vector_ref("chunk-b"), 2, 0.81, True),
        ]

    @pytest.mark.asyncio
    async def test_hybrid_lists_legs_in_order_graph_vector_vault(self):
        result = await _hybrid_result()

        observed = [(c.source, c.rank, c.selected, c.empty) for c in result.candidates]

        # Vector: src-proof loses the cut to limit=3. Vault: b.md shares the
        # (VaultNote, MENTIONS, Notes) merge key with a.md, so the merge keeps
        # a.md only.
        assert observed == [
            ("graph", 1, True, False),
            ("vector", 1, True, False),
            ("vector", 2, False, False),
            ("vault", 1, True, False),
            ("vault", 2, False, True),
            ("vault", 3, False, False),
        ]
        assert [c.ref for c in result.candidates if c.source == "vault"] == [
            _row_vault_ref(r) for r in _HYBRID_ROWS
        ]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", list(_SCENARIOS))
    async def test_selected_is_exactly_the_returned_facts(self, name):
        result = await _SCENARIOS[name]()

        assert _selected_refs(result) == _fact_refs(result)

    @pytest.mark.asyncio
    async def test_live_intent_has_no_candidates(self):
        retriever, _ = _make_retriever(intent="live")

        result = await retriever.retrieve(query="show my linear tickets")

        assert result.candidates == ()


async def _graph_leg_result(filters: RetrievalFilters | None):
    """Five graph rows, one a repeat of the `USED_FOR data-science` triple."""
    retriever, _ = _make_retriever(
        intent="relational",
        search_similar_results=_SIMILAR_ENTITIES[:1],
        user_rels_results=_USER_RELS,
        neighborhood_results=[*_NEIGHBORHOOD, _NEIGHBORHOOD[0]],
    )
    return await retriever.retrieve(
        query="What languages do I use?", user_id="User", limit=20, filters=filters
    )


class TestGraphLegCandidatesAfterFiltersAndDedup:
    """The graph leg reports its facts after `RetrievalFilters` and triple
    dedup: the turn trace has no disposition for a row the caller's filters
    excluded, and a repeated triple has the same ref as the fact it repeats.
    """

    @pytest.mark.asyncio
    async def test_duplicate_triple_yields_one_candidate(self):
        result = await _graph_leg_result(filters=None)

        assert [c.ref for c in result.candidates] == [
            _graph_ref("User", "USES", "python"),
            _graph_ref("User", "LEARNING", "rust"),
            _graph_ref("python", "USED_FOR", "data-science"),
            _graph_ref("rust", "RELATED_TO", "systems"),
        ]

    @pytest.mark.asyncio
    async def test_rows_excluded_by_entity_type_filter_are_not_candidates(self):
        result = await _graph_leg_result(filters=RetrievalFilters(entity_types=["Topic"]))

        assert [c.ref for c in result.candidates] == [
            _graph_ref("python", "USED_FOR", "data-science"),
            _graph_ref("rust", "RELATED_TO", "systems"),
        ]


class TestRetrieveOutputUnchanged:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("name", list(_SCENARIOS))
    async def test_result_matches_the_base_snapshot(self, name):
        result = await _SCENARIOS[name]()

        assert _snapshot(result) == _BASE_SNAPSHOTS[name]


# ---------------------------------------------------------------------------
# fact_ref
# ---------------------------------------------------------------------------

# Synthetic fixtures with no run of three lowercase hex characters, so a correct
# digest can never match one of their substrings by chance, while any leaked
# natural-key text (which always carries non-hex characters) is caught.
_GRAPH_FACT = RetrievedFact(
    subject="Hiking-Group",
    subject_type="Person",
    predicate="JOINS",
    object="Mountain Trails",
    object_type="Concept",
    properties={},
    similarity_score=0.5,
    graph_distance=0,
)
_VAULT_PATH = "/journal/spring/notes.md"
_VAULT_HEADING = "Morning Walk"


def _vault_fact(text: str) -> RetrievedFact:
    return RetrievedFact(
        subject="VaultNote",
        subject_type="VaultSession",
        predicate="MENTIONS",
        object=_VAULT_HEADING,
        object_type="VaultChunk",
        properties={"path": _VAULT_PATH, "text": text, "content": text},
        similarity_score=0.5,
        graph_distance=999,
    )


_VAULT_FACT = _vault_fact("Saw two herons by the pond.")
_VECTOR_FACT = RetrievedFact(
    subject="Document",
    subject_type="DocumentChunk",
    predicate="CONTAINS",
    object="Trail Guide",
    object_type="markdown",
    properties={"chunk_id": "chunk-north-trail", "text": "Pack water.", "source_id": "src-guide"},
    similarity_score=0.5,
    graph_distance=999,
)
_REF_CASES = [
    pytest.param(_GRAPH_FACT, "graph", id="graph"),
    pytest.param(_VAULT_FACT, "vault", id="vault"),
    pytest.param(_VECTOR_FACT, "vector", id="vector"),
]
_REF_PATTERN = re.compile(r"^(vault|graph|vector):[0-9a-f]{20}$")


def _substrings(text: str, minimum: int = 3) -> set[str]:
    return {text[i:j] for i in range(len(text)) for j in range(i + minimum, len(text) + 1)}


class TestFactRef:
    @pytest.mark.parametrize(("fact", "source"), _REF_CASES)
    def test_ref_matches_the_pattern_with_its_source_prefix(self, fact, source):
        ref = fact_ref(fact)

        assert _REF_PATTERN.match(ref)
        assert ref.startswith(f"{source}:")

    @pytest.mark.parametrize(("fact", "source"), _REF_CASES)
    def test_ref_is_stable_across_calls_and_equal_copies(self, fact, source):
        copy = dataclasses.replace(fact, properties=dict(fact.properties))

        assert fact_ref(fact) == fact_ref(fact) == fact_ref(copy)

    def test_graph_ref_hashes_the_triple(self):
        assert fact_ref(_GRAPH_FACT) == _graph_ref("Hiking-Group", "JOINS", "Mountain Trails")

    def test_vault_ref_hashes_path_heading_and_text_digest(self):
        assert fact_ref(_VAULT_FACT) == _vault_ref(
            _VAULT_PATH, _VAULT_HEADING, "Saw two herons by the pond."
        )

    def test_vector_ref_hashes_the_chunk_id(self):
        assert fact_ref(_VECTOR_FACT) == _vector_ref("chunk-north-trail")

    @pytest.mark.parametrize(("fact", "source"), _REF_CASES)
    def test_ref_contains_no_fixture_substring(self, fact, source):
        fields = [fact.subject, fact.predicate, fact.object]
        fields += [str(fact.properties.get(k, "")) for k in ("path", "chunk_id", "text")]
        assert not [f for f in fields if re.search(r"[0-9a-f]{3}", f)], "fixture precondition"
        # The source prefix is fixed by the contract, and the vault subject
        # `VaultNote` shares "aul"/"ult" with it; only the prefix is exempt.
        prefix = f"{source}:"
        leaked = {
            s
            for field in fields
            for s in _substrings(field)
            if s in fact_ref(fact) and s not in prefix
        }

        assert leaked == set()

    def test_vault_sections_sharing_path_and_heading_get_distinct_refs(self):
        first = _vault_fact("Planted the beds.")
        second = _vault_fact("Harvested the beds.")

        assert fact_ref(first) != fact_ref(second)
