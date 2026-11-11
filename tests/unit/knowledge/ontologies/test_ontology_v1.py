"""Snapshot tests for the MIST.AI v1.0.0 ontology definition.

These tests pin the ontology structure so that accidental changes to entity
types, relationship types, confidence policies, or extraction rules cause
immediate, visible test failures.
"""

from __future__ import annotations

import ast
import copy
import dataclasses
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from backend.knowledge import version_stamps
from backend.knowledge.extraction import id_rails, rails
from backend.knowledge.ontologies import hierarchy, registry, v1_0_0
from backend.knowledge.ontologies.base import KnowledgeDomain
from backend.knowledge.ontologies.pack import POST_LLM_PHASES, Pack, load_pack
from backend.knowledge.ontologies.packs.v1_4_0 import conventions as pack_conventions
from backend.knowledge.ontologies.packs.v1_4_0 import manifest as pack_manifest
from backend.knowledge.ontologies.packs.v1_4_0 import plugins as pack_plugins
from backend.knowledge.ontologies.packs.v1_4_0 import types as pack_types
from backend.knowledge.ontologies.v1_0_0 import (
    ALL_EDGE_TYPES,
    ALL_NODE_TYPES,
    CONFIDENCE_BRIDGING,
    CONFIDENCE_EXTERNAL,
    CONFIDENCE_INTERNAL,
    EXTRACTABLE_NODE_TYPES,
    EXTRACTABLE_RELATIONSHIP_TYPES,
    ONTOLOGY_V1_0_0,
)

# -------------------------------------------------------------------
# Entity types
# -------------------------------------------------------------------


class TestEntityTypes:
    """Verify the complete set of entity types in the ontology."""

    def test_has_30_entity_types(self):
        # Cluster 8 Phase 6 (ADR-010) added VaultNote as a bridging/provenance type.
        # Post-MVP additive (2026-04-22) added Date, Metric, Document
        # as EXTERNAL extractable types (Milestone retired in v1.4.0).
        # v1.1.0 additive (2026-05-06) added Pattern, Convention, Mechanism,
        # Strategy, DataStructure as EXTERNAL extractable types.
        # v1.4.0 (2026-06-14): Topic and Milestone retired; Abstraction supertype
        # added. Net change: -2 retired, +1 Abstraction = -1 total (31 -> 30).
        assert len(ALL_NODE_TYPES) == pack_manifest.NODE_TYPE_COUNT

    @pytest.mark.parametrize(
        "type_name",
        [
            pytest.param("MistIdentity", id="MistIdentity"),
            pytest.param("MistTrait", id="MistTrait"),
            pytest.param("MistCapability", id="MistCapability"),
            pytest.param("MistPreference", id="MistPreference"),
            pytest.param("MistUncertainty", id="MistUncertainty"),
        ],
    )
    def test_internal_domain_has_5_types(self, type_name: str):
        internal_types = [
            nt for nt in ALL_NODE_TYPES if nt.knowledge_domain == KnowledgeDomain.INTERNAL
        ]

        internal_names = [nt.type_name for nt in internal_types]

        assert len(internal_types) == 5
        assert type_name in internal_names

    @pytest.mark.parametrize(
        "type_name",
        [
            pytest.param("User", id="User"),
            pytest.param("Person", id="Person"),
            pytest.param("Organization", id="Organization"),
            pytest.param("Technology", id="Technology"),
            pytest.param("Skill", id="Skill"),
            pytest.param("Project", id="Project"),
            pytest.param("Concept", id="Concept"),
            # Topic retired in v1.4.0 (merged to Concept); removed from this list.
            pytest.param("Event", id="Event"),
            pytest.param("Goal", id="Goal"),
            pytest.param("Preference", id="Preference"),
            pytest.param("Location", id="Location"),
            # Post-MVP additive (2026-04-22):
            pytest.param("Date", id="Date"),
            # Milestone retired in v1.4.0 (demoted to Event event_type=milestone);
            # removed from this list.
            pytest.param("Metric", id="Metric"),
            pytest.param("Document", id="Document"),
            # v1.1.0 additive (2026-05-06):
            pytest.param("Pattern", id="Pattern"),
            pytest.param("Convention", id="Convention"),
            pytest.param("Mechanism", id="Mechanism"),
            pytest.param("Strategy", id="Strategy"),
            pytest.param("DataStructure", id="DataStructure"),
            # v1.4.0 additive (2026-06-14): Abstraction supertype.
            pytest.param("Abstraction", id="Abstraction"),
        ],
    )
    def test_external_domain_has_20_types(self, type_name: str):
        # v1.4.0: Topic + Milestone retired (-2), Abstraction added (+1) = 20.
        external_types = [
            nt for nt in ALL_NODE_TYPES if nt.knowledge_domain == KnowledgeDomain.EXTERNAL
        ]

        external_names = [nt.type_name for nt in external_types]

        assert len(external_types) == 20
        assert type_name in external_names

    @pytest.mark.parametrize(
        "type_name",
        [
            pytest.param("LearningEvent", id="LearningEvent"),
            pytest.param("ConversationContext", id="ConversationContext"),
            pytest.param("ExternalSource", id="ExternalSource"),
            pytest.param("VectorChunk", id="VectorChunk"),
            pytest.param("VaultNote", id="VaultNote"),
        ],
    )
    def test_bridging_domain_has_5_types(self, type_name: str):
        # Cluster 8 Phase 6 (ADR-010) added VaultNote.
        bridging_types = [
            nt for nt in ALL_NODE_TYPES if nt.knowledge_domain == KnowledgeDomain.BRIDGING
        ]

        bridging_names = [nt.type_name for nt in bridging_types]

        assert len(bridging_types) == 5
        assert type_name in bridging_names

    def test_vault_note_required_property_is_path(self):
        # ADR-010 Phase 6: VaultNote nodes are MERGE-idempotent keyed by `path`.
        # The schema must declare `path` as a required property.
        from backend.knowledge.ontologies.v1_0_0 import VAULT_NOTE

        required_names = [p.name for p in VAULT_NOTE.required_properties]
        assert required_names == ["path"]
        assert VAULT_NOTE.knowledge_domain == KnowledgeDomain.BRIDGING

    def test_derived_from_allows_vault_note_target(self):
        # ADR-010 Phase 6: DERIVED_FROM extends to entity -> VaultNote for the
        # vault-rebuild contract. VectorChunk and ExternalSource targets remain.
        from backend.knowledge.ontologies.v1_0_0 import DERIVED_FROM

        assert "VaultNote" in DERIVED_FROM.allowed_target_types
        assert "VectorChunk" in DERIVED_FROM.allowed_target_types
        assert "ExternalSource" in DERIVED_FROM.allowed_target_types
        # MistIdentity must be a permitted source so MIST-scope entities can
        # carry vault-note provenance.
        assert "MistIdentity" in DERIVED_FROM.allowed_source_types

    def test_extractable_types_are_external_plus_mist_identity(self):
        # Cluster 1 (Bug J): MistIdentity is promoted from INTERNAL-only to
        # also extractable so that MIST-scope facts ("MIST uses LanceDB")
        # survive extraction. The extractable set is the full external
        # types plus MistIdentity.
        # Post-MVP additive (2026-04-22): 12 -> 16 external, 13 -> 17 extractable.
        # v1.1.0 additive (2026-05-06): 16 -> 21 external, 17 -> 22 extractable.
        # v1.4.0 (2026-06-14): Topic + Milestone retired (-2), Abstraction added
        # (+1): 21 -> 20 external, 22 -> 21 extractable.
        external_names = {
            nt.type_name for nt in ALL_NODE_TYPES if nt.knowledge_domain == KnowledgeDomain.EXTERNAL
        }
        expected = external_names | {"MistIdentity"}

        extractable_set = set(EXTRACTABLE_NODE_TYPES)

        assert len(EXTRACTABLE_NODE_TYPES) == pack_manifest.EXTRACTABLE_NODE_TYPE_COUNT
        assert extractable_set == expected


# -------------------------------------------------------------------
# Relationship types
# -------------------------------------------------------------------


class TestRelationshipTypes:
    """Verify the complete set of relationship types in the ontology."""

    def test_has_51_relationship_types(self):
        # Cluster 1 added 4 MIST-scope edges: IMPLEMENTED_WITH,
        # MIST_HAS_CAPABILITY, MIST_HAS_TRAIT, MIST_HAS_PREFERENCE.
        # Post-MVP additive (2026-04-22) added 4 temporal / quantified / document
        # edges: OCCURRED_ON, HAS_METRIC, REFERENCES_DOCUMENT, PRECEDED_BY.
        # v1.1.0 additive (2026-05-06) added 8 mechanism/pattern/strategy edges:
        # MECHANISM_OF, OPERATES_ON, INPUT_TO, IMPROVES, COMPRISES,
        # APPLICABLE_TO, STRATEGY_FOR, NAMING_CONVENTION_OF.
        # v1.3.0 additive (2026-06-13) added 2 user-centric edges: RECOMMENDS,
        # HAS_HABIT.
        assert len(ALL_EDGE_TYPES) == pack_manifest.EDGE_TYPE_COUNT

    def test_extractable_relationships_count(self):
        # 13 user-centric + 8 original structural (excludes LEARNED_FROM, ABOUT,
        # SUPERSEDES + provenance SOURCED_FROM/REFERENCES/DERIVED_FROM) + 4
        # MIST-scope (IMPLEMENTED_WITH, MIST_HAS_CAPABILITY, MIST_HAS_TRAIT,
        # MIST_HAS_PREFERENCE) + 4 post-MVP additive structural (OCCURRED_ON,
        # HAS_METRIC, REFERENCES_DOCUMENT, PRECEDED_BY) + 8 v1.1.0 additive
        # (MECHANISM_OF, OPERATES_ON, INPUT_TO, IMPROVES, COMPRISES,
        # APPLICABLE_TO, STRATEGY_FOR, NAMING_CONVENTION_OF) + 2 v1.3.0 additive
        # user-centric (RECOMMENDS, HAS_HABIT).
        assert len(EXTRACTABLE_RELATIONSHIP_TYPES) == pack_manifest.EXTRACTABLE_EDGE_TYPE_COUNT

    @pytest.mark.parametrize(
        "type_name",
        [
            pytest.param("IMPLEMENTED_WITH", id="IMPLEMENTED_WITH"),
            pytest.param("MIST_HAS_CAPABILITY", id="MIST_HAS_CAPABILITY"),
            pytest.param("MIST_HAS_TRAIT", id="MIST_HAS_TRAIT"),
            pytest.param("MIST_HAS_PREFERENCE", id="MIST_HAS_PREFERENCE"),
        ],
    )
    def test_mist_scope_edges_are_extractable(self, type_name: str):
        assert type_name in EXTRACTABLE_RELATIONSHIP_TYPES
        assert type_name in {et.type_name for et in ALL_EDGE_TYPES}


# -------------------------------------------------------------------
# Post-MVP additive: temporal / quantified / document invariants (2026-04-22)
# -------------------------------------------------------------------


class TestNewNodeTypeInvariants:
    """Each post-MVP additive node type must be EXTERNAL-domain, have the
    promised required-property contract, and be present in the extractable
    set so the LLM extractor sees it as a valid target.

    `expected_allowed_values` pins the enum vocabulary for properties that
    restrict inputs (Document.doc_type). Silent widening of these enums would
    break downstream normalization and retrieval filters, so this test catches it.

    v1.4.0 (2026-06-14): Milestone retired (demoted to Event event_type=milestone)
    and replaced in this fixture with Abstraction, the new EXTERNAL supertype.
    """

    @pytest.mark.parametrize(
        "type_const_name, expected_required, expected_optional, expected_allowed_values",
        [
            pytest.param(
                "DATE",
                ["iso_date"],
                [],
                {},
                id="Date",
            ),
            # Milestone retired in v1.4.0; replaced here by Abstraction.
            pytest.param(
                "ABSTRACTION",
                [],
                [],
                {},
                id="Abstraction",
            ),
            pytest.param(
                "METRIC",
                ["value", "unit"],
                ["as_of_date"],
                {},
                id="Metric",
            ),
            pytest.param(
                "DOCUMENT",
                ["title"],
                ["doc_type", "identifier"],
                {
                    "doc_type": (
                        "adr",
                        "spec",
                        "paper",
                        "book",
                        "article",
                        "other",
                    ),
                },
                id="Document",
            ),
        ],
    )
    def test_new_node_type_contract(
        self,
        type_const_name: str,
        expected_required: list[str],
        expected_optional: list[str],
        expected_allowed_values: dict[str, tuple[str, ...]],
    ):
        import backend.knowledge.ontologies.v1_0_0 as onto

        node = getattr(onto, type_const_name)
        assert node.knowledge_domain == KnowledgeDomain.EXTERNAL
        required_names = [p.name for p in node.required_properties]
        optional_names = [p.name for p in node.optional_properties]
        assert required_names == expected_required
        assert optional_names == expected_optional
        assert node.type_name in EXTRACTABLE_NODE_TYPES

        # Pin enum vocabularies for properties that use allowed_values.
        all_props = list(node.required_properties) + list(node.optional_properties)
        by_name = {p.name: p for p in all_props}
        for prop_name, expected_values in expected_allowed_values.items():
            actual = by_name[prop_name].allowed_values
            assert actual == expected_values, (
                f"{type_const_name}.{prop_name} allowed_values drifted: "
                f"expected {expected_values!r}, got {actual!r}"
            )


class TestNewEdgeTypeInvariants:
    """Each post-MVP additive edge must have the exact allowed source/target
    sets from the plan and be present in the extractable relationship set.
    """

    @pytest.mark.parametrize(
        "edge_const_name, expected_sources, expected_targets",
        [
            pytest.param(
                "OCCURRED_ON",
                ("Event", "Milestone"),
                ("Date",),
                id="OCCURRED_ON",
            ),
            pytest.param(
                "HAS_METRIC",
                ("User", "Project", "Technology", "Skill", "Concept", "Goal"),
                ("Metric",),
                id="HAS_METRIC",
            ),
            pytest.param(
                "REFERENCES_DOCUMENT",
                (
                    "User",
                    "MistIdentity",
                    "Project",
                    "Concept",
                    "Topic",
                    "Goal",
                    "Event",
                ),
                ("Document",),
                id="REFERENCES_DOCUMENT",
            ),
            pytest.param(
                "PRECEDED_BY",
                ("Event", "Milestone"),
                ("Event", "Milestone", "Date"),
                id="PRECEDED_BY",
            ),
        ],
    )
    def test_new_edge_type_contract(
        self,
        edge_const_name: str,
        expected_sources: tuple[str, ...],
        expected_targets: tuple[str, ...],
    ):
        import backend.knowledge.ontologies.v1_0_0 as onto

        edge = getattr(onto, edge_const_name)
        # Tuple equality (not set): catches accidental duplicates + ordering
        # drift. The constants are tuple-typed, so this is the precise check.
        assert edge.allowed_source_types == expected_sources
        assert edge.allowed_target_types == expected_targets
        assert edge.type_name in EXTRACTABLE_RELATIONSHIP_TYPES


# -------------------------------------------------------------------
# Confidence policies
# -------------------------------------------------------------------


class TestConfidencePolicies:
    """Verify confidence decay configuration per knowledge domain."""

    def test_internal_policy_no_decay(self):
        assert CONFIDENCE_INTERNAL.decay_enabled is False
        assert CONFIDENCE_INTERNAL.decay_half_life_days is None

    def test_external_policy_has_180_day_half_life(self):
        assert CONFIDENCE_EXTERNAL.decay_enabled is True
        assert CONFIDENCE_EXTERNAL.decay_half_life_days == 180

    def test_bridging_policy_has_365_day_half_life(self):
        assert CONFIDENCE_BRIDGING.decay_enabled is True
        assert CONFIDENCE_BRIDGING.decay_half_life_days == 365


# -------------------------------------------------------------------
# Ontology version
# -------------------------------------------------------------------


class TestOntologyVersion:
    """Verify the top-level ontology version metadata."""

    def test_version_is_1_4_0(self):
        # v1.4.0 (2026-06-14): MECE taxonomy redesign -- Topic and Milestone
        # retired, Abstraction supertype added with parent_type on seven abstract
        # leaves. Build-time edge allowed-set transform adds Abstraction wherever
        # Concept is allowed.
        assert ONTOLOGY_V1_0_0.version == "1.4.0"
        assert ONTOLOGY_V1_0_0.parent_version == "1.3.0"

    def test_vault_writer_ontology_version_paired(self):
        # The vault writer stamps the ontology version into session-note
        # frontmatter; it must track the active ontology version so rebuilt
        # notes record the schema they were written under. It used to mirror
        # the value as its own `_ONTOLOGY_VERSION` literal -- it now imports
        # the derived stamp, so the pairing holds by construction.
        from backend.vault import writer

        assert ONTOLOGY_V1_0_0.version == writer.ONTOLOGY_VERSION
        assert not hasattr(
            writer, "_ONTOLOGY_VERSION"
        ), "the mirrored literal is deleted, not merely unreferenced"


# -------------------------------------------------------------------
# The pack's files: the types live in `packs/v1_4_0/types.py`
# -------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[4]
_PACK_PACKAGE = "backend.knowledge.ontologies.packs.v1_4_0"
_PACK_TYPES = f"{_PACK_PACKAGE}.types"
_PACKS_DIR = Path(pack_types.__file__).resolve().parents[1]

# The names `types.py` binds for the pack loader alone, which the old path never had.
_LOADER_ONLY_NAMES = {"NODE_TYPES", "EDGE_TYPES", "EXTRACTABLE_EDGE_TYPES"}


def _bound_names(source: str) -> set[str]:
    """Every name a module's own top-level statements bind, imports left out."""
    names: set[str] = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.FunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return names


def _imported_modules(path: Path) -> set[str]:
    """Every module a file imports, at any depth, by its dotted name."""
    modules: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            modules.add(node.module)
            modules.update(f"{node.module}.{alias.name}" for alias in node.names)
    return modules


class TestTheOldPathIsAReExport:
    """`v1_0_0.py` is the import path older code uses, and holds nothing of its own."""

    def test_its_only_statements_are_a_docstring_and_imports_from_the_packs_types(self):
        body = ast.parse(Path(v1_0_0.__file__).read_text(encoding="utf-8")).body

        assert isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
        rest = body[1:]
        assert rest, "the module re-exports nothing"
        assert all(isinstance(node, ast.ImportFrom) for node in rest)
        assert {node.module for node in rest} == {"__future__", _PACK_TYPES}

    def test_every_name_the_packs_types_bind_is_the_same_object_under_the_old_path(self):
        bound = _bound_names(Path(pack_types.__file__).read_text(encoding="utf-8"))
        expected = bound - _LOADER_ONLY_NAMES

        # A scan that has gone blind would pass with nothing to compare.
        assert len(expected) > 100
        assert {"ONTOLOGY_V1_0_0", "_EXTERNAL_NODE_TYPES", "_apply_v1_4_0"} <= expected
        assert bound >= _LOADER_ONLY_NAMES
        missing = sorted(name for name in expected if not hasattr(v1_0_0, name))
        assert missing == []
        other = sorted(
            name for name in expected if getattr(v1_0_0, name) is not getattr(pack_types, name)
        )
        assert other == []

    def test_the_old_path_binds_nothing_the_packs_types_do_not(self):
        own = {name for name in vars(v1_0_0) if not name.startswith("__")} - {"annotations"}

        assert sorted(name for name in own if not hasattr(pack_types, name)) == []


class TestWhatTheLoaderReadsFromTheTypes:
    """`types.py` gives the pack loader the lists the old path always had."""

    def test_the_node_types_and_both_extractable_lists_are_the_old_lists(self):
        assert tuple(ALL_NODE_TYPES) == pack_types.NODE_TYPES
        assert pack_types.EXTRACTABLE_NODE_TYPES is EXTRACTABLE_NODE_TYPES
        assert tuple(EXTRACTABLE_RELATIONSHIP_TYPES) == pack_types.EXTRACTABLE_EDGE_TYPES

    def test_the_edges_are_the_old_ones_as_written(self):
        # As written, before the retired types and the supertypes are applied: the
        # loader applies the manifest to them.
        assert [edge.type_name for edge in pack_types.EDGE_TYPES] == [
            edge.type_name for edge in ALL_EDGE_TYPES
        ]
        assert pack_types.EDGE_TYPES is v1_0_0._RAW_EDGE_TYPES
        assert [pack_types._apply_v1_4_0(edge) for edge in pack_types.EDGE_TYPES] == ALL_EDGE_TYPES


class TestThePackImportsWithNoCycle:
    """The loader imports the version stamps, which import the types through the old path.

    So nothing the old path imports may reach the loader, whichever module a process
    imports first.
    """

    @pytest.mark.parametrize(
        "first",
        [
            "backend.knowledge.version_stamps",
            "backend.knowledge.ontologies.pack",
            "backend.knowledge.ontologies.v1_0_0",
            "backend.knowledge.ontologies.registry",
            "backend.knowledge.extraction.id_rails",
            _PACK_TYPES,
            f"{_PACK_PACKAGE}.manifest",
            f"{_PACK_PACKAGE}.conventions",
            f"{_PACK_PACKAGE}.plugins",
        ],
    )
    def test_a_fresh_interpreter_imports_the_chain_from_any_end(self, first: str):
        code = (
            f"import {first}\n"
            "from backend.knowledge import version_stamps\n"
            "from backend.knowledge.ontologies import pack, v1_0_0\n"
            f"from {_PACK_PACKAGE} import types\n"
            "assert v1_0_0.ONTOLOGY_V1_0_0 is types.ONTOLOGY_V1_0_0\n"
            "assert version_stamps.ONTOLOGY_VERSION is types.ONTOLOGY_V1_0_0.version\n"
            f"built = pack.load_pack('{_PACK_PACKAGE}')\n"
            "assert built.version == version_stamps.ONTOLOGY_VERSION\n"
        )

        done = subprocess.run(
            [sys.executable, "-B", "-c", code],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )

        assert done.returncode == 0, done.stderr[-2000:]

    @pytest.mark.parametrize(
        "path",
        [
            _PACKS_DIR / "__init__.py",
            _PACKS_DIR / "v1_4_0" / "__init__.py",
            _PACKS_DIR / "v1_4_0" / "types.py",
        ],
        ids=["packs", "packs.v1_4_0", "packs.v1_4_0.types"],
    )
    def test_what_the_old_path_loads_never_imports_the_loader(self, path: Path):
        reaching_the_loader = {
            name
            for name in _imported_modules(path)
            if name.startswith(
                (
                    "backend.knowledge.ontologies.pack",
                    "backend.knowledge.ontologies.registry",
                    "backend.knowledge.version_stamps",
                    "backend.knowledge.extraction",
                )
            )
        }

        # `packs` itself starts with `...ontologies.pack`, so the pack's own modules
        # are told apart from the loader by the dot that follows.
        assert {
            name
            for name in reaching_the_loader
            if not name.startswith("backend.knowledge.ontologies.packs")
        } == set()


# ---------------------------------------------------------------------------
# The labelling guide: the ontology's rows are the pack's
# ---------------------------------------------------------------------------

_GUIDE = _REPO_ROOT / "data" / "gold" / "extraction" / "LABELLING.md"
_PACK_GUIDE = _PACKS_DIR / "v1_4_0" / "labelling.md"

# The guide before the split, at 058e52c: its non-empty lines, sorted, then hashed. The
# split moves lines between the two files and adds or rewords none, headings included,
# so the two files together keep this value and the line count.
_GUIDE_LINE_COUNT = 471
_GUIDE_LINES_SHA256 = "5e428caba6e710eadab0b29c6e999056750ce4c97272f9ac79e9030f9c60d498"

# The rules that name this ontology's types and predicates, and the ones that hold for
# any ontology.
_ONTOLOGY_RULES = ("N2", "N3", "N5", "N7", "N8", "N10", "N11", "N12", "N13", "N14", "N16", "N17")
_GENERIC_RULES = ("N1", "N4", "N6", "N15")
_ONTOLOGY_HEADINGS = (
    "### Event and metric ids",
    "### Date entities",
    "### Habits and incidental scope",
)
_GENERIC_HEADINGS = (
    "### Ids",
    "### Assertion kinds",
    "### No-fact probes",
    "## List-shaped probes (new)",
)
_GOAL_LIST_LINE = "  except under a lead-in that names goals, where it is `HAS_GOAL`."


def _lines(path: Path) -> list[str]:
    return [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _rule_starts(lines: list[str], rule: str) -> int:
    """How many lines open the rule `rule`, as a bullet or as a paragraph."""
    return sum(1 for line in lines if line.startswith((f"**{rule} ", f"- **{rule} ")))


class TestTheGuideIsSplitBetweenTheGoldSetsAndThePack:
    """The gold guide keeps the rules of any ontology; the pack holds its own."""

    def test_the_two_files_hold_the_lines_of_the_guide_before_the_split(self):
        lines = sorted(_lines(_GUIDE) + _lines(_PACK_GUIDE))

        assert len(lines) == _GUIDE_LINE_COUNT
        digest = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
        assert digest == _GUIDE_LINES_SHA256

    @pytest.mark.parametrize("rule", _ONTOLOGY_RULES)
    def test_an_ontology_rule_is_in_the_pack_alone(self, rule: str):
        assert _rule_starts(_lines(_PACK_GUIDE), rule) == 1
        assert _rule_starts(_lines(_GUIDE), rule) == 0

    @pytest.mark.parametrize("rule", _GENERIC_RULES)
    def test_a_generic_rule_stays_in_the_gold_guide(self, rule: str):
        assert _rule_starts(_lines(_GUIDE), rule) == 1
        assert _rule_starts(_lines(_PACK_GUIDE), rule) == 0

    def test_the_sections_follow_their_rules(self):
        guide, pack_guide = _lines(_GUIDE), _lines(_PACK_GUIDE)

        assert [h for h in _ONTOLOGY_HEADINGS if h in pack_guide] == list(_ONTOLOGY_HEADINGS)
        assert [h for h in _ONTOLOGY_HEADINGS if h in guide] == []
        assert [h for h in _GENERIC_HEADINGS if h in guide] == list(_GENERIC_HEADINGS)
        assert [h for h in _GENERIC_HEADINGS if h in pack_guide] == []

    def test_the_goal_line_of_the_list_rules_is_the_packs(self):
        assert _GOAL_LIST_LINE in _lines(_PACK_GUIDE)
        assert _GOAL_LIST_LINE not in _lines(_GUIDE)


# ---------------------------------------------------------------------------
# The pack built from its files is the pack the registry serves
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def file_pack() -> Pack:
    """The v1.4.0 pack as the loader builds it from the pack's files."""
    return load_pack(_PACK_PACKAGE)


@pytest.fixture(scope="module")
def registry_pack() -> Pack:
    """The pack the registry serves for the version."""
    return registry.get_pack(ONTOLOGY_V1_0_0.version)


# The rules in the order the rails run them, as the adapter the registry held before
# the pack declared them.
_RULE_IDS = [
    "reserved-mist-identity",
    "n1-exempt",
    "pre-canon-aliases",
    "technology-version",
    "static-aliases",
    "metric-id",
    "event-date",
    "plan-vocabulary",
]


def _identity(rule) -> tuple:
    """What a rule does: everything the rails hash reads of it, so not its pinned case."""
    return (rule.id, rule.kind, rule.phase, rule.params, rule.prompt_rule)


def _without_cases(pack: Pack, *, dropped: str | None = None) -> Pack:
    """`pack` with no pinned case, and without the rule `dropped` when one is named."""
    rules = tuple(
        dataclasses.replace(rule, pinned_case=None) for rule in pack.rules if rule.id != dropped
    )
    return dataclasses.replace(pack, rules=rules)


def _run_case(case, pack: Pack) -> tuple[list[dict], list[dict]]:
    row = rails.rail_row(
        copy.deepcopy(list(case.entities)),
        copy.deepcopy(list(case.relationships)),
        scope=case.scope,
        batch_record=copy.deepcopy(case.batch_record),
        turn_plan=copy.deepcopy(case.turn_plan),
    )
    return rails.run_rails(row, pack)


class TestTheRegistryServesThePack:
    """The registry's pack is the one the loader builds from the pack's files."""

    def test_both_hashes_are_the_release_lines(self, file_pack: Pack):
        release = version_stamps.PACK_RELEASE

        assert file_pack.version == release.pack_version
        assert file_pack.pack_prompt_hash == release.pack_prompt_hash
        assert file_pack.pack_rails_hash == release.pack_rails_hash

    def test_the_types_are_the_ones_the_registry_serves(self, file_pack: Pack, registry_pack: Pack):
        assert file_pack.node_types == registry_pack.node_types
        assert file_pack.edge_types == registry_pack.edge_types
        assert file_pack.edge_types == tuple(ALL_EDGE_TYPES)
        assert file_pack.extractable_node_types == registry_pack.extractable_node_types
        assert file_pack.extractable_edge_types == registry_pack.extractable_edge_types

    def test_the_pair_table_the_prompt_and_the_schema_are_the_ones_the_registry_serves(
        self, file_pack: Pack, registry_pack: Pack
    ):
        assert file_pack.pair_table == registry_pack.pair_table
        assert file_pack.system_prompt == registry_pack.system_prompt
        assert file_pack.extraction_schema == registry_pack.extraction_schema

    def test_the_rules_are_the_ones_the_registry_serves_in_order_with_their_cases(
        self, file_pack: Pack, registry_pack: Pack
    ):
        assert [_identity(rule) for rule in file_pack.rules] == [
            _identity(rule) for rule in registry_pack.rules
        ]
        assert registry_pack.rules == pack_conventions.RULES
        assert [rule.id for rule in registry_pack.rules] == _RULE_IDS

    def test_the_manifest_is_the_one_the_registry_serves(
        self, file_pack: Pack, registry_pack: Pack
    ):
        assert file_pack.manifest == registry_pack.manifest
        assert registry_pack.manifest is pack_manifest.MANIFEST

    def test_the_plug_ins_are_the_ones_the_registry_serves(
        self, file_pack: Pack, registry_pack: Pack
    ):
        assert [(plugin.name, plugin.phase) for plugin in file_pack.plugins] == [
            (plugin.name, plugin.phase) for plugin in registry_pack.plugins
        ]
        for mine, theirs in zip(file_pack.plugins, registry_pack.plugins, strict=True):
            assert mine.fn is theirs.fn

    def test_the_pack_adds_no_prompt_fragment(self, file_pack: Pack):
        assert file_pack.rules_of("prompt_fragment") == ()

    def test_the_loader_reads_the_packs_own_files(self, file_pack: Pack):
        assert file_pack.manifest is pack_manifest.MANIFEST
        assert file_pack.rules == pack_conventions.RULES
        assert file_pack.plugins == pack_plugins.PLUGINS
        assert file_pack.labelling == _PACK_GUIDE.read_text(encoding="utf-8")


class TestThePackStatesWhatTheOldModulesHold:
    """The pack states its data itself, equal to the modules that still hold a copy.

    While the two are equal, removing the modules' copy is a deletion.
    """

    def test_the_alias_vocabularies(self):
        assert tuple(id_rails.STATIC_ALIASES.items()) == pack_conventions.STATIC_ALIASES
        assert tuple(id_rails.PRE_CANON_ALIASES.items()) == pack_conventions.PRE_CANON_ALIASES

    def test_the_reserved_names(self):
        identity = (pack_conventions.RESERVED_ID, pack_types.MIST_IDENTITY.type_name)

        assert tuple(id_rails.RESERVED_NAMES) == pack_conventions.RESERVED_NAMES
        assert set(id_rails.RESERVED_NAMES.values()) == {identity}

    def test_the_types_the_id_rail_leaves_alone(self):
        assert id_rails.ID_EXEMPT_TYPES == pack_conventions.ID_EXEMPT_TYPES

    def test_the_retired_types_and_what_they_gain(self):
        retired = pack_manifest.MANIFEST.retired_types

        assert {r.name: r.replacement for r in retired} == hierarchy.RETIRED_TYPE_MAP
        assert [r.name for r in retired] == list(hierarchy.RETIRED_TYPE_MAP)
        assert {r.name: r.property_defaults for r in retired} == id_rails.RETIRED_PROPERTY_DEFAULTS

    def test_the_supertype_the_fallback_and_the_user_anchor(self):
        manifest = pack_manifest.MANIFEST

        assert manifest.supertypes == tuple(hierarchy.ANCHOR_CHILD.items())
        assert manifest.fallback_type == id_rails.FALLBACK_TYPE
        assert manifest.user_id == id_rails.USER_ID


class TestThePacksCounts:
    """The four counts are stated once, in the manifest module, and hold for the pack."""

    def test_the_counts_are_the_packs_lists(self, file_pack: Pack):
        assert len(file_pack.node_types) == pack_manifest.NODE_TYPE_COUNT
        assert len(file_pack.extractable_node_types) == pack_manifest.EXTRACTABLE_NODE_TYPE_COUNT
        assert len(file_pack.edge_types) == pack_manifest.EDGE_TYPE_COUNT
        assert len(file_pack.extractable_edge_types) == pack_manifest.EXTRACTABLE_EDGE_TYPE_COUNT


_POST_LLM_RULES = [rule.id for rule in pack_conventions.RULES if rule.phase in POST_LLM_PHASES]


class TestThePinnedCases:
    """Each rule the rails run holds a row and the payload the pack turns it into."""

    def test_every_rule_the_rails_run_has_a_case(self):
        assert len(_POST_LLM_RULES) == 7
        assert [
            rule.id
            for rule in pack_conventions.RULES
            if rule.id in _POST_LLM_RULES and rule.pinned_case is None
        ] == []

    def test_every_case_holds(self, file_pack: Pack):
        assert rails.check_pinned_cases(file_pack) == []

    @pytest.mark.parametrize("rule_id", _POST_LLM_RULES)
    def test_the_registrys_pack_gives_the_cases_payload(self, rule_id: str, registry_pack: Pack):
        case = next(rule.pinned_case for rule in pack_conventions.RULES if rule.id == rule_id)

        entities, relationships = _run_case(case, registry_pack)

        assert entities == list(case.expected_entities)
        assert relationships == list(case.expected_relationships)

    @pytest.mark.parametrize("rule_id", _POST_LLM_RULES)
    def test_a_case_needs_its_rule(self, rule_id: str, file_pack: Pack):
        # A case its rule does not change would hold with the rule gone, and pin nothing.
        case = next(rule.pinned_case for rule in pack_conventions.RULES if rule.id == rule_id)

        with_rule = _run_case(case, _without_cases(file_pack))
        without_rule = _run_case(case, _without_cases(file_pack, dropped=rule_id))

        assert with_rule == (list(case.expected_entities), list(case.expected_relationships))
        assert without_rule != with_rule

    @pytest.mark.parametrize("rule_id", _POST_LLM_RULES)
    def test_every_edge_a_case_expects_is_one_the_pack_allows(self, rule_id: str, file_pack: Pack):
        case = next(rule.pinned_case for rule in pack_conventions.RULES if rule.id == rule_id)
        types = {entity["id"]: entity["type"] for entity in case.expected_entities}

        refused = []
        for edge in case.expected_relationships:
            sources, targets = file_pack.pair_table[edge["type"]]
            source, target = types[edge["source"]], types[edge["target"]]
            if (sources is not None and source not in sources) or (
                targets is not None and target not in targets
            ):
                refused.append((source, edge["type"], target))
        assert refused == []

    def test_the_tuning_runs_payloads_are_among_the_cases(self):
        # Three cases are payloads the service wrote for dev-lists probes: the tag of
        # each probe is named beside the rule its payload pins.
        assert sorted(pack_conventions.CASES_FROM_THE_TUNING_RUN) == [
            "event-date",
            "metric-id",
            "static-aliases",
        ]
        assert set(pack_conventions.CASES_FROM_THE_TUNING_RUN) <= set(_POST_LLM_RULES)
        for tag in pack_conventions.CASES_FROM_THE_TUNING_RUN.values():
            assert tag.startswith("dvl-")

    def test_one_case_carries_what_the_manifest_states_without_a_rule(self):
        # A retired type, an unknown type and the user anchor are manifest data, so no
        # rule of their own can hold a case for them: the first id rule's case does.
        case = pack_conventions.RULES[1].pinned_case
        manifest = pack_manifest.MANIFEST
        written = [entity["type"] for entity in case.entities]
        stored = {entity["id"]: entity for entity in case.expected_entities}

        assert [r.name for r in manifest.retired_types if r.name not in written] == []
        assert [t for t in written if t not in pack_types.ALL_NODE_TYPE_NAMES] == [
            "Milestone",
            "Topic",
            "Feeling",
        ]
        assert stored["first-release"]["type"] == "Event"
        assert stored["first-release"]["properties"] == {"event_type": "milestone"}
        assert stored["release-cadence"]["type"] == "Concept"
        assert stored["hunch"]["type"] == manifest.fallback_type
        assert [e["id"] for e in case.entities if e["id"].lower() == manifest.user_id] == ["User"]
        assert manifest.user_id in stored
