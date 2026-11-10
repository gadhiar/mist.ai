"""The id rails build the stored id from the payload's `id` field (plan v11, D14).

The normalizer and the eval scorer both call these functions, so each step is pinned here
once, in ADR-054 Decision 2's order, and the normalizer is pinned to equal
`normalize_payload` on the same payload.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from backend.knowledge.extraction import id_rails
from backend.knowledge.extraction.id_rails import (
    n1_rail,
    normalize_entity,
    normalize_payload,
    stored_entity_id,
    version_rail,
)
from backend.knowledge.extraction.normalizer import EntityNormalizer
from backend.knowledge.extraction.ontology_extractor import ExtractionResult
from tests.mocks.embeddings import FakeEmbeddingGenerator

_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEV_LISTS = _REPO_ROOT / "data" / "gold" / "extraction" / "dev-lists-2026-11.jsonl"
_SIXTY = _REPO_ROOT / "data" / "ingest" / "extraction-gold-2026-06-14.jsonl"


def _normalized(**entity) -> dict:
    normalize_entity(entity)
    return entity


# --- step 1: the user ---------------------------------------------------------------


@pytest.mark.parametrize("raw_id", ["user", "User", "USER"])
def test_the_user_stays_user(raw_id):
    assert _normalized(id=raw_id, name="Raj", type="User")["id"] == "user"


# --- step 2: reserved names -----------------------------------------------------------


def test_a_reserved_display_name_sets_mist_identity_and_its_type():
    entity = _normalized(id="assistant-company", name="MIST", type="Organization")

    assert (entity["id"], entity["type"]) == ("mist-identity", "MistIdentity")


def test_a_reserved_slugged_id_sets_mist_identity_when_the_name_is_not_reserved():
    entity = _normalized(id="The AI", name="the helper", type="Concept")

    assert (entity["id"], entity["type"]) == ("mist-identity", "MistIdentity")


def test_a_reserved_name_ends_the_steps():
    # The N1 rail would otherwise drop `the` from `the-assistant`.
    entity = _normalized(id="the-assistant", type="Concept")

    assert entity["id"] == "mist-identity"


# --- step 3: the id field, or the pre-canonicalization aliases on the name ---------------


def test_the_stored_id_is_the_slug_of_the_id_field_not_of_the_name():
    assert _normalized(id="Rust Lang", name="Rust", type="Technology")["id"] == "rust-lang"


@pytest.mark.parametrize(("name", "expected"), [("C++", "c-plus-plus"), ("c#", "c-sharp")])
def test_a_pre_canonicalization_alias_on_the_name_sets_the_id(name, expected):
    assert _normalized(id="cpp-lang", name=name, type="Technology")["id"] == expected


def test_the_name_stands_in_for_an_empty_or_missing_id():
    assert _normalized(id="", name="Duck DB", type="Technology")["id"] == "duck-db"
    assert _normalized(name="Duck DB", type="Technology")["id"] == "duck-db"


# --- step 4: the version rail -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_id", "name", "expected"),
    [
        ("python-3-11", "Python 3.11", "python"),
        ("python-311", "Python 3.11", "python"),
        ("react-v18", "React v18", "react"),
        ("python", "Python 3.11", "python"),
        ("web3", "web3", "web3"),
        ("oauth2", "OAuth2", "oauth2"),
    ],
)
def test_the_version_rail_drops_a_technology_version_named_in_its_name(raw_id, name, expected):
    assert _normalized(id=raw_id, name=name, type="Technology")["id"] == expected


def test_the_version_rail_leaves_an_event_date_in_place():
    entity = _normalized(
        id="team-offsite-on-2026-05-30", name="Team offsite on 2026-05-30", type="Event"
    )

    assert entity["id"] == "team-offsite-on-2026-05-30"


def test_the_version_rail_keeps_an_id_that_does_not_end_with_the_version():
    assert version_rail("python-3", "Python 3.11", "Technology") == "python-3"
    assert version_rail("3-11", "Python 3.11", "Technology") == "3-11"
    assert version_rail("python-3-11", None, "Technology") == "python-3-11"


def test_the_node_version_suffix_then_takes_the_static_alias():
    assert _normalized(id="node-18-x", name="Node 18.x", type="Technology")["id"] == "nodejs"


# --- step 5: static aliases --------------------------------------------------------------


def test_a_static_alias_applies_after_the_slug():
    assert _normalized(id="VS Code", name="VS Code", type="Technology")["id"] == (
        "visual-studio-code"
    )
    assert _normalized(id="js", name="JavaScript", type="Technology")["id"] == "javascript"


# --- step 6: the registry ----------------------------------------------------------------


def test_a_registry_hit_sets_id_and_type_and_ends_the_steps(monkeypatch):
    monkeypatch.setitem(id_rails.CANONICAL_REGISTRY, "the-graph", ("the-graph-db", "Technology"))

    entity = _normalized(id="the-graph", name="the graph", type="Concept")

    assert (entity["id"], entity["type"]) == ("the-graph-db", "Technology")


# --- step 7: resolver passes 3 to 5 --------------------------------------------------------


def test_a_milestone_becomes_an_event_with_its_event_type():
    entity = _normalized(id="house-closing", type="Milestone", properties={})

    assert entity["type"] == "Event"
    assert entity["properties"]["event_type"] == "milestone"


def test_a_metric_id_is_built_from_its_properties_with_one_spelling():
    entity = _normalized(
        id="cold-start",
        type="Metric",
        properties={"value": 2.4, "unit": "s", "quantity": "cold start"},
    )

    assert entity["id"] == "2-4-seconds-cold-start"


def test_a_metric_without_rebuildable_properties_reads_its_raw_id_so_the_decimal_survives():
    assert _normalized(id="tool-selection-0.94", type="Metric")["id"] == "0-94-tool-selection"


def test_an_unknown_type_falls_back_to_abstraction():
    assert _normalized(id="thing", type="NotARealType")["type"] == "Abstraction"


# --- step 8: the N1 rail ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw_id", "expected"),
    [
        ("the-acme-corp", "acme-corp"),
        ("my-sister", "sister"),
        ("our-the-band", "band"),
        ("the", "the"),
        ("my-the", "the"),
        ("theatre-club", "theatre-club"),
        ("acme-the-corp", "acme-the-corp"),
    ],
)
def test_the_n1_rail_drops_leading_articles_and_possessives(raw_id, expected):
    assert n1_rail(raw_id) == expected


def test_the_n1_rail_runs_last_on_every_type_but_the_exempt_ones():
    assert _normalized(id="The Acme Corp", type="Organization")["id"] == "acme-corp"
    assert _normalized(id="my-sister", type="Person")["id"] == "sister"
    assert _normalized(id="the-thing", type="NotARealType")["id"] == "thing"
    assert _normalized(id="the-3-runs", type="Metric")["id"] == "3-the-runs"


# --- stored_entity_id and idempotence --------------------------------------------------------


def test_stored_entity_id_does_not_change_the_callers_properties():
    properties: dict = {}

    stored_entity_id("launch", "Milestone", properties)

    assert properties == {}


def _gold_ids() -> list[tuple[str, str]]:
    ids: list[tuple[str, str]] = []
    for path in (_DEV_LISTS, _SIXTY):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.startswith("#"):
                continue
            row = json.loads(line)
            for entity in row["expected_entities"]:
                ids.append((entity["id"], entity["type"]))
                ids += [(alt, entity["type"]) for alt in entity.get("alternatives", ())]
            for rel in row["expected_relationships"]:
                ids.append((rel["source"], rel["source_type"]))
                ids.append((rel["target"], rel["target_type"]))
    return ids


_PRODUCED = [
    {"id": "Python 3.11", "name": "Python 3.11", "type": "Technology"},
    {"id": "react-v18", "name": "React v18", "type": "Technology"},
    {"id": "The Acme Corp", "name": "The Acme Corp", "type": "Organization"},
    {"id": "vs-code", "name": "VS Code", "type": "Technology"},
    {"id": "mist", "name": "MIST", "type": "Organization"},
    {"id": "c++", "name": "C++", "type": "Technology"},
    {"id": "tool-selection-0.94", "name": "tool selection", "type": "Metric"},
    {
        "id": "p95",
        "name": "p95 latency",
        "type": "Metric",
        "properties": {"value": "120", "unit": "msec", "quantity": "p95 latency"},
    },
    {
        "id": "rps",
        "name": "throughput",
        "type": "Metric",
        "properties": {"value": "9,500", "unit": "req/s"},
    },
    {"id": "launch", "name": "launch", "type": "Milestone"},
    {"id": "team-offsite-2026-05-30", "name": "Team offsite", "type": "Event"},
    {"id": "2026-05-30", "name": "2026-05-30", "type": "Date"},
    {"id": "my-old-notes", "name": "my old notes", "type": "Topic"},
    {"id": "the", "name": "the", "type": "Concept"},
]


def test_stored_entity_id_is_idempotent_on_every_gold_id():
    ids = _gold_ids()
    moved = [
        (raw, etype)
        for raw, etype in ids
        if stored_entity_id(stored_entity_id(raw, etype), etype) != stored_entity_id(raw, etype)
    ]

    assert len(ids) > 500
    assert moved == []


@pytest.mark.parametrize("entity", _PRODUCED, ids=[e["id"] for e in _PRODUCED])
def test_stored_entity_id_is_idempotent_on_a_produced_entity(entity):
    stored = copy.deepcopy(entity)
    normalize_entity(stored)

    assert stored_entity_id(stored["id"], stored["type"]) == stored["id"]


# --- normalize_payload --------------------------------------------------------------------


def test_normalize_payload_remaps_endpoints_and_deduplicates():
    entities = [
        {"id": "The Acme Corp", "name": "The Acme Corp", "type": "Organization"},
        {"id": "acme-corp", "name": "Acme Corp", "type": "Organization"},
        {"id": "user", "name": "user", "type": "User"},
    ]
    relationships = [
        {"source": "user", "target": "The Acme Corp", "type": "WORKS_AT"},
        {"source": "user", "target": "acme-corp", "type": "WORKS_AT"},
    ]

    out_entities, out_relationships = normalize_payload(entities, relationships)

    assert [e["id"] for e in out_entities] == ["acme-corp", "user"]
    assert out_entities[0]["name"] == "The Acme Corp"
    assert out_relationships == [{"source": "user", "target": "acme-corp", "type": "WORKS_AT"}]


def _payload() -> tuple[list[dict], list[dict]]:
    entities = copy.deepcopy(_PRODUCED) + [{"id": "user", "name": "user", "type": "User"}]
    relationships = [
        {"source": "user", "target": "Python 3.11", "type": "USES"},
        {"source": "user", "target": "react-v18", "type": "USES"},
        {"source": "user", "target": "The Acme Corp", "type": "WORKS_AT"},
        {"source": "mist", "target": "c++", "type": "IMPLEMENTED_WITH"},
        {"source": "user", "target": "team-offsite-2026-05-30", "type": "EXPERIENCED"},
        {"source": "team-offsite-2026-05-30", "target": "2026-05-30", "type": "OCCURRED_ON"},
        {"source": "user", "target": "launch", "type": "EXPERIENCED"},
        {"source": "user", "target": "Unlisted Thing", "type": "USES"},
        {"source": "user", "target": "the-acme-corp", "type": "WORKS_AT"},
    ]
    return entities, relationships


@pytest.mark.asyncio
async def test_normalize_payload_equals_the_normalizer_on_the_same_payload():
    entities, relationships = _payload()
    rail_entities, rail_relationships = normalize_payload(
        copy.deepcopy(entities), copy.deepcopy(relationships)
    )
    normalizer = EntityNormalizer(embedding_generator=FakeEmbeddingGenerator(), executor=None)

    out = await normalizer.normalize(
        ExtractionResult(entities=entities, relationships=relationships)
    )

    assert out.entities == rail_entities
    assert out.relationships == rail_relationships


def test_id_rails_imports_nothing_from_scripts_and_no_io_dependency():
    source = Path(id_rails.__file__).read_text(encoding="utf-8")
    imports = [line for line in source.splitlines() if line.startswith(("import ", "from "))]

    assert imports == [
        "from __future__ import annotations",
        "import re",
        "from collections.abc import Mapping",
        "from backend.knowledge.extraction.canonical_id import (",
        "from backend.knowledge.ontologies.hierarchy import RETIRED_TYPE_MAP",
        "from backend.knowledge.ontologies.v1_0_0 import ALL_NODE_TYPE_NAMES",
    ]
