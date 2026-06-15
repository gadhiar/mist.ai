# tests/unit/knowledge/storage/test_partitions.py
"""Unit tests for the partition label constants."""

from backend.knowledge.storage.partitions import (
    ENTITY_LABEL,
    PROVENANCE_LABEL,
    SELF_MODEL_LABEL,
    SELF_MODEL_TYPES,
)


def test_partition_labels_are_the_canonical_strings():
    assert ENTITY_LABEL == "__Entity__"
    assert PROVENANCE_LABEL == "__Provenance__"
    assert SELF_MODEL_LABEL == "__SelfModel__"


def test_self_model_types_is_the_five_internal_entity_types():
    assert (
        frozenset(
            {
                "MistIdentity",
                "MistTrait",
                "MistCapability",
                "MistPreference",
                "MistUncertainty",
            }
        )
        == SELF_MODEL_TYPES
    )


def test_self_model_types_is_immutable():
    assert isinstance(SELF_MODEL_TYPES, frozenset)
