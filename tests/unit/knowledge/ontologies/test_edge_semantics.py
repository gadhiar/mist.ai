"""C1 reconciliation-semantics schema invariants (design 4.1)."""

from backend.knowledge.ontologies.base import (
    Cardinality,
    EdgeTypeDefinition,
    TemporalClass,
)


class TestSemanticsFieldsExist:
    def test_defaults_are_backward_compatible(self):
        # Existing defs constructed without the new kwargs must still build,
        # defaulting to the safest semantics: accumulate, stative, no links.
        d = EdgeTypeDefinition(
            type_name="X",
            description="d",
            allowed_source_types=("User",),
            allowed_target_types=("Technology",),
        )
        assert d.cardinality is Cardinality.MULTI
        assert d.temporal_class is TemporalClass.STATIVE
        assert d.contradicts == ()
        assert d.progression_supersedes == ()

    def test_enum_values(self):
        assert Cardinality.SINGLE.value == "single"
        assert Cardinality.MULTI.value == "multi"
        assert TemporalClass.STATIVE.value == "stative"
        assert TemporalClass.EVENT.value == "event"
        assert TemporalClass.DURABLE.value == "durable"


from backend.knowledge.ontologies.v1_0_0 import (  # noqa: E402
    EDGE_TYPES_BY_NAME,
    EXTRACTABLE_RELATIONSHIP_TYPES,
    ONTOLOGY_V1_0_0,
)

# Snapshot of the design 4.1 semantics table. Drift here is intentional-only.
EXPECTED_SINGLE = {"WORKS_AT", "OCCURRED_ON"}
EXPECTED_EVENT = {"DECIDED", "EXPERIENCED", "OCCURRED_ON", "HAS_METRIC"}
EXPECTED_DURABLE = {
    "IS_A",
    "PART_OF",
    "RELATED_TO",
    "PRECEDED_BY",
    "MECHANISM_OF",
    "OPERATES_ON",
    "INPUT_TO",
    "COMPRISES",
    "APPLICABLE_TO",
    "STRATEGY_FOR",
    "NAMING_CONVENTION_OF",
}
# USES carries no contradicts/progression: usage (behavior) and dislike or
# struggle (sentiment/competence) are orthogonal dimensions that co-occur
# ("I use Jira but I hate it"); pairing them erased co-true current beliefs.
EXPECTED_CONTRADICTS = {
    "PREFERS": {"DISLIKES"},
    "INTERESTED_IN": {"DISLIKES"},
    "DISLIKES": {"PREFERS", "INTERESTED_IN"},
    "EXPERT_IN": {"STRUGGLES_WITH"},
    "STRUGGLES_WITH": {"EXPERT_IN"},
}
EXPECTED_PROGRESSION = {
    "EXPERT_IN": {"LEARNING", "STRUGGLES_WITH"},
}


class TestSemanticsTable:
    def test_version_bumped(self):
        assert ONTOLOGY_V1_0_0.version == "1.2.1"

    def test_name_index_covers_all_edges(self):
        assert set(EDGE_TYPES_BY_NAME) == {e.type_name for e in ONTOLOGY_V1_0_0.edge_types}

    def test_single_cardinality_set(self):
        single = {
            n
            for n in EXTRACTABLE_RELATIONSHIP_TYPES
            if EDGE_TYPES_BY_NAME[n].cardinality is Cardinality.SINGLE
        }
        assert single == EXPECTED_SINGLE

    def test_temporal_class_sets(self):
        event = {
            n
            for n in EXTRACTABLE_RELATIONSHIP_TYPES
            if EDGE_TYPES_BY_NAME[n].temporal_class is TemporalClass.EVENT
        }
        durable = {
            n
            for n in EXTRACTABLE_RELATIONSHIP_TYPES
            if EDGE_TYPES_BY_NAME[n].temporal_class is TemporalClass.DURABLE
        }
        assert event == EXPECTED_EVENT
        assert durable == EXPECTED_DURABLE

    def test_contradicts_table(self):
        declared = {
            n: set(EDGE_TYPES_BY_NAME[n].contradicts)
            for n in EXTRACTABLE_RELATIONSHIP_TYPES
            if EDGE_TYPES_BY_NAME[n].contradicts
        }
        assert declared == EXPECTED_CONTRADICTS

    def test_contradicts_is_pairwise_symmetric(self):
        for name in EXTRACTABLE_RELATIONSHIP_TYPES:
            for other in EDGE_TYPES_BY_NAME[name].contradicts:
                assert (
                    name in EDGE_TYPES_BY_NAME[other].contradicts
                ), f"{name} contradicts {other} but not vice versa"

    def test_progression_table_and_targets_exist(self):
        declared = {
            n: set(EDGE_TYPES_BY_NAME[n].progression_supersedes)
            for n in EXTRACTABLE_RELATIONSHIP_TYPES
            if EDGE_TYPES_BY_NAME[n].progression_supersedes
        }
        assert declared == EXPECTED_PROGRESSION
        for _name, supers in declared.items():
            for s in supers:
                assert s in EDGE_TYPES_BY_NAME

    def test_undirected_edges_use_directional_flag(self):
        assert EDGE_TYPES_BY_NAME["WORKS_WITH"].directional is False
        assert EDGE_TYPES_BY_NAME["RELATED_TO"].directional is False
