"""The normalizer and the validator take a pack.

The fixture packs name invented types, so a result that follows them shows the injected
pack was read and today's ontology was not.
"""

from __future__ import annotations

import copy

import pytest

from backend.knowledge.extraction.normalizer import EntityNormalizer, rail_inputs
from backend.knowledge.extraction.ontology_extractor import ExtractionResult
from backend.knowledge.extraction.validator import ExtractionValidator
from backend.knowledge.ontologies import registry
from backend.knowledge.ontologies.pack import load_pack
from tests.mocks.embeddings import FakeEmbeddingGenerator
from tests.unit.knowledge.ontologies.fixtures.pet_pack.conventions import PET_ROSTER_CASE

BASE = load_pack("tests.unit.knowledge.ontologies.fixtures.base_pack")
PET = load_pack("tests.unit.knowledge.ontologies.fixtures.pet_pack")


def _extraction(entities, relationships=()) -> ExtractionResult:
    return ExtractionResult(
        entities=copy.deepcopy(list(entities)),
        relationships=copy.deepcopy(list(relationships)),
        source_utterance="",
    )


# --- the normalizer ---------------------------------------------------------------------


def test_the_normalizer_defaults_to_the_registrys_active_pack():
    normalizer = EntityNormalizer(FakeEmbeddingGenerator())

    assert normalizer._pack is registry.active_pack()


@pytest.mark.asyncio
async def test_the_normalizer_runs_the_rails_of_the_pack_it_is_given():
    normalizer = EntityNormalizer(FakeEmbeddingGenerator(), pack=BASE)

    out = await normalizer.normalize(
        _extraction(
            [
                {"id": "the-north-mooring", "name": "North Mooring No. 7", "type": "Mooring"},
                {"id": "the-flotsam", "name": "Flotsam", "type": "Driftwood"},
            ]
        )
    )

    assert [(e["id"], e["type"]) for e in out.entities] == [
        ("mooring-7", "Mooring"),
        ("flotsam", "Deckhand"),
    ]


@pytest.mark.asyncio
async def test_the_normalizer_builds_the_rail_row_from_what_it_is_handed():
    normalizer = EntityNormalizer(FakeEmbeddingGenerator(), pack=PET)
    case = PET_ROSTER_CASE

    with_row = await normalizer.normalize(
        _extraction(case.entities, case.relationships),
        scope=case.scope,
        batch_record=copy.deepcopy(case.batch_record),
        turn_plan=copy.deepcopy(case.turn_plan),
    )
    without_row = await normalizer.normalize(_extraction(case.entities, case.relationships))

    assert with_row.relationships == list(case.expected_relationships)
    assert without_row.relationships == list(case.relationships)


@pytest.mark.asyncio
async def test_the_normalizer_logs_a_reserved_name_of_its_own_pack(caplog):
    normalizer = EntityNormalizer(FakeEmbeddingGenerator(), pack=BASE)

    with caplog.at_level("WARNING"):
        await normalizer.normalize(
            _extraction([{"id": "hm", "name": "The Harbourmaster", "type": "Mooring"}])
        )

    assert "id='harbourmaster' type='Deckhand'" in caplog.text


def test_rail_inputs_are_the_fields_a_cache_row_carries():
    row = {"entities": [], "batch_record": [{"units": []}], "turn_plan": {}}

    assert rail_inputs(row) == {"batch_record": [{"units": []}], "turn_plan": {}}


def test_a_row_written_before_the_columns_existed_hands_over_nothing():
    assert rail_inputs({"entities": [], "scope": None}) == {}
    assert rail_inputs({"entities": [], "scope": None, "batch_record": None}) == {}


# --- the validator -----------------------------------------------------------------------


def _validated(pack, entities, relationships):
    result = ExtractionValidator(pack=pack).validate(_extraction(entities, relationships))
    return [e["id"] for e in result.entities], [r["type"] for r in result.relationships]


def test_the_validator_reads_types_edges_and_pairs_from_the_pack_it_is_given():
    entities = [
        {"id": "ada", "type": "Deckhand"},
        {"id": "bo", "type": "Deckhand"},
        {"id": "mooring-2", "type": "Mooring"},
        {"id": "rust", "type": "Technology"},
    ]
    relationships = [
        {"source": "ada", "target": "mooring-2", "type": "MOORED_AT"},
        {"source": "mooring-2", "target": "ada", "type": "MOORED_AT"},
        {"source": "ada", "target": "bo", "type": "USES"},
        {"source": "mooring-2", "target": "ada", "type": "ALONGSIDE"},
    ]

    assert _validated(BASE, entities, relationships) == (
        ["ada", "bo", "mooring-2"],
        ["MOORED_AT", "ALONGSIDE"],
    )


def test_a_type_and_an_edge_one_pack_adds_are_unknown_to_the_other():
    entities = [{"id": "ada", "type": "Deckhand"}, {"id": "pip", "type": "Pet"}]
    owned = [{"source": "ada", "target": "pip", "type": "HAS_PET"}]
    reversed_pair = [{"source": "pip", "target": "ada", "type": "HAS_PET"}]

    assert _validated(PET, entities, owned) == (["ada", "pip"], ["HAS_PET"])
    assert _validated(PET, entities, reversed_pair) == (["ada", "pip"], [])
    assert _validated(BASE, entities, owned) == (["ada"], [])


def test_the_validator_without_a_pack_enforces_the_active_packs_pair_table():
    validator = ExtractionValidator()

    assert validator._pack is registry.active_pack()
    assert validator._constraints.keys() == registry.active_pack().pair_table.keys()
