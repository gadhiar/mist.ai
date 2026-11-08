"""The ontology type labels an `:__Entity__` node may carry beside its partition label."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from backend.knowledge.canonical_serialize import canonical_graph_form
from backend.knowledge.ontologies.hierarchy import RETIRED_TYPE_MAP
from backend.knowledge.ontologies.v1_0_0 import _EXTERNAL_NODE_TYPES, ALL_NODE_TYPE_NAMES
from backend.knowledge.storage.partitions import SELF_MODEL_TYPES
from backend.knowledge.storage.type_labels import ENTITY_TYPE_LABELS, entity_type_label
from scripts.migrations.ontology_v1_4_0 import CYPHER as RETYPE_CYPHER
from tests.unit.knowledge.curation.conftest import make_entity_dict
from tests.unit.knowledge.curation.test_graph_writer_type_labels import (
    LabelGraphConnection,
    make_label_writer,
)

_REPO_ROOT = Path(__file__).resolve().parents[4]


class TestLabelSet:
    def test_is_the_external_node_types_without_mist_identity(self):
        external = {nt.type_name for nt in _EXTERNAL_NODE_TYPES}

        assert external - {"MistIdentity"} == ENTITY_TYPE_LABELS
        assert len(ENTITY_TYPE_LABELS) == 20
        assert "User" in ENTITY_TYPE_LABELS

    def test_excludes_self_model_retired_and_unknown_types(self):
        assert not ENTITY_TYPE_LABELS & SELF_MODEL_TYPES
        assert not ENTITY_TYPE_LABELS & set(RETIRED_TYPE_MAP)
        assert "Unknown" not in ENTITY_TYPE_LABELS

    def test_every_label_is_an_ontology_type_and_a_plain_identifier(self):
        assert set(ALL_NODE_TYPE_NAMES) >= ENTITY_TYPE_LABELS
        for label in ENTITY_TYPE_LABELS:
            assert re.fullmatch(r"[A-Z][A-Za-z]*", label), label


class TestEntityTypeLabel:
    @pytest.mark.parametrize("entity_type", sorted(ENTITY_TYPE_LABELS))
    def test_a_type_in_the_set_is_its_own_label(self, entity_type):
        assert entity_type_label(entity_type) == entity_type

    @pytest.mark.parametrize(
        "entity_type",
        [
            "MistIdentity",
            "MistTrait",
            "Unknown",
            "Topic",
            "Milestone",
            "ConversationContext",
            "",
            "person",
            " Person",
            "Person`) DETACH DELETE n //",
        ],
    )
    def test_any_other_string_has_no_label(self, entity_type):
        assert entity_type_label(entity_type) is None


class TestNoDynamicLabels:
    def test_no_backend_module_writes_a_dynamic_label(self):
        """Every type label is a literal that came through `entity_type_label`."""
        offenders = [
            f"{path.relative_to(_REPO_ROOT)}:{lineno}"
            for path in sorted((_REPO_ROOT / "backend").rglob("*.py"))
            for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if ":$(" in line
        ]

        assert offenders == []


class TestRetypeMigrationSwapsTheLabel:
    """`scripts/migrations/ontology_v1_4_0.py` retypes a node and swaps its label
    in the same statement, so the label keeps mirroring `entity_type`.
    """

    @pytest.mark.parametrize(
        ("index", "old", "new"), [(0, "Topic", "Concept"), (1, "Milestone", "Event")]
    )
    def test_each_retype_sets_the_new_label_and_removes_the_old(self, index, old, new):
        query = RETYPE_CYPHER[index]
        match, sep, rest = query.partition(" SET ")
        assert sep, query
        assert match == f"MATCH (e:__Entity__) WHERE e.entity_type = '{old}'"
        set_items, sep, remove = rest.partition(" REMOVE ")
        assert sep, query
        assert set_items.startswith(f"e.entity_type = '{new}', ")
        assert set_items.endswith(f", e:{entity_type_label(new)}")
        assert remove == f"e:{old}"
        assert entity_type_label(old) is None

    def test_the_migration_keeps_two_statements(self):
        assert len(RETYPE_CYPHER) == 2


class TestRebuildDeterminism:
    @staticmethod
    async def _build(entities) -> str:
        conn = LabelGraphConnection()
        writer = make_label_writer(conn)
        for index, entity in enumerate(entities):
            await writer.write(
                entities=[entity], merge_actions=[], event_id=f"evt-{index}", session_id="s-1"
            )
        return canonical_graph_form(conn)

    @pytest.mark.asyncio
    async def test_same_nodes_built_twice_give_identical_canonical_output(self):
        entities = [
            make_entity_dict(entity_id="user", entity_type="User", display_name="Raj"),
            make_entity_dict(entity_id="rust", entity_type="Technology", display_name="Rust"),
            make_entity_dict(entity_id="acme", entity_type="Organization", display_name="Acme"),
            make_entity_dict(entity_id="old-topic", entity_type="Topic", display_name="Old"),
        ]

        first = await self._build(entities)
        second = await self._build(list(reversed(entities)))

        assert first == second
        labels = {node["id"]: node["labels"] for node in json.loads(first)["nodes"]}
        assert labels == {
            "acme": ["Organization"],
            "old-topic": [],
            "rust": ["Technology"],
            "user": ["User"],
        }
