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
