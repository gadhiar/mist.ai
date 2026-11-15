"""Tests for backend.memory.facets: grouping fact keys into facets a type-scoped probe targets."""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass

import pytest

from backend.memory.facets import FacetSpec, facet_key, facets_of


@dataclass(frozen=True, slots=True)
class _Row:
    fact_key: str
    subject_id: str = "s1"
    subject_name: str = "Wren"
    subject_type: str | None = "User"
    rel_type: str = "MEMBER_OF"
    object_type: str | None = "Organization"
    recorded_at: str | None = "2024-03-05T10:00:00+00:00"


def _only(facets: dict[str, FacetSpec]) -> FacetSpec:
    [facet] = facets.values()
    return facet


class TestFacetKey:
    def test_is_the_hex_sha256_of_the_three_joined_by_bars(self):
        expected = hashlib.sha256(b"s1|MEMBER_OF|Organization").hexdigest()
        assert facet_key("s1", "MEMBER_OF", "Organization") == expected

    def test_differs_when_any_part_differs(self):
        keys = {
            facet_key("s1", "MEMBER_OF", "Organization"),
            facet_key("s2", "MEMBER_OF", "Organization"),
            facet_key("s1", "WORKS_AT", "Organization"),
            facet_key("s1", "MEMBER_OF", "Project"),
        }
        assert len(keys) == 4


class TestFacetSpec:
    def test_is_frozen_with_slots(self):
        spec = FacetSpec(
            facet_key="f",
            subject_id="s1",
            subject_name="Wren",
            subject_type="User",
            rel_type="MEMBER_OF",
            object_type="Organization",
            member_keys=("k1", "k2"),
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            spec.rel_type = "USES"  # type: ignore[misc]
        assert not hasattr(spec, "__dict__")


class TestFacetsOf:
    def test_two_facts_of_one_subject_relation_and_object_type_make_one_facet(self):
        facets = facets_of([_Row("k1"), _Row("k2")])
        key = facet_key("s1", "MEMBER_OF", "Organization")
        assert list(facets) == [key]
        assert facets[key] == FacetSpec(
            facet_key=key,
            subject_id="s1",
            subject_name="Wren",
            subject_type="User",
            rel_type="MEMBER_OF",
            object_type="Organization",
            member_keys=("k1", "k2"),
        )

    def test_two_versions_of_one_fact_key_make_none(self):
        rows = [
            _Row("k1", recorded_at="2024-03-05T10:00:00+00:00"),
            _Row("k1", recorded_at="2025-01-01T10:00:00+00:00"),
        ]
        assert facets_of(rows) == {}

    @pytest.mark.parametrize("object_type", [None, "User", "MistTrait", ""])
    def test_an_object_type_none_user_or_outside_type_terms_makes_none(self, object_type):
        rows = [_Row("k1", object_type=object_type), _Row("k2", object_type=object_type)]
        assert facets_of(rows) == {}

    def test_a_single_fact_makes_none(self):
        assert facets_of([_Row("k1")]) == {}
        assert facets_of([]) == {}

    def test_subject_relation_and_object_type_each_split_the_group(self):
        rows = [
            _Row("k1"),
            _Row("k2", subject_id="s2"),
            _Row("k3", rel_type="WORKS_AT"),
            _Row("k4", object_type="Project"),
        ]
        assert facets_of(rows) == {}

    def test_separate_groups_make_separate_facets(self):
        rows = [
            _Row("k1"),
            _Row("k2"),
            _Row("k3", rel_type="USES", object_type="Technology"),
            _Row("k4", rel_type="USES", object_type="Technology"),
        ]
        facets = facets_of(rows)
        assert set(facets) == {
            facet_key("s1", "MEMBER_OF", "Organization"),
            facet_key("s1", "USES", "Technology"),
        }
        assert facets[facet_key("s1", "USES", "Technology")].member_keys == ("k3", "k4")

    def test_members_are_newest_recorded_first(self):
        rows = [
            _Row("k1", recorded_at="2024-01-01T00:00:00+00:00"),
            _Row("k2", recorded_at="2026-01-01T00:00:00+00:00"),
            _Row("k3", recorded_at="2025-01-01T00:00:00+00:00"),
        ]
        assert _only(facets_of(rows)).member_keys == ("k2", "k3", "k1")

    def test_a_key_is_placed_by_its_newest_version_and_listed_once(self):
        rows = [
            _Row("k1", recorded_at="2024-01-01T00:00:00+00:00"),
            _Row("k2", recorded_at="2025-01-01T00:00:00+00:00"),
            _Row("k1", recorded_at="2026-01-01T00:00:00+00:00"),
        ]
        assert _only(facets_of(rows)).member_keys == ("k1", "k2")

    def test_equal_recorded_at_ties_break_by_fact_key_and_none_sorts_oldest(self):
        rows = [
            _Row("kc", recorded_at=None),
            _Row("kb", recorded_at="2025-01-01T00:00:00+00:00"),
            _Row("ka", recorded_at="2025-01-01T00:00:00+00:00"),
            _Row("kd", recorded_at="2024-01-01T00:00:00+00:00"),
        ]
        assert _only(facets_of(rows)).member_keys == ("ka", "kb", "kd", "kc")
        assert _only(facets_of(list(reversed(rows)))).member_keys == ("ka", "kb", "kd", "kc")

    def test_subject_name_and_type_come_from_the_newest_recorded_row(self):
        rows = [
            _Row("k1", subject_name="Ana", subject_type="Person", recorded_at="2024-01-01"),
            _Row("k2", subject_name="Ana R.", subject_type=None, recorded_at="2026-01-01"),
            _Row("k1", subject_name="Ana", subject_type="Person", recorded_at="2025-01-01"),
        ]
        facet = _only(facets_of(rows))
        assert (facet.subject_name, facet.subject_type) == ("Ana R.", None)
        facet = _only(facets_of(list(reversed(rows))))
        assert (facet.subject_name, facet.subject_type) == ("Ana R.", None)

    def test_accepts_any_iterable_once(self):
        facets = facets_of(row for row in [_Row("k1"), _Row("k2")])
        assert _only(facets).member_keys == ("k1", "k2")
