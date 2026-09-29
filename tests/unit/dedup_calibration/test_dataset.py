"""Dataset validation for the dedup calibration tool."""

from __future__ import annotations

import json
import re
from dataclasses import replace

import pytest

from scripts.dedup_calibration.dataset import (
    DEFAULT_DATASET_PATH,
    LABEL_DISTINCT,
    LABEL_DUPLICATE,
    MIN_DISTINCT_PAIRS,
    MIN_DUPLICATE_PAIRS,
    MIN_HARD_DISTINCT_PAIRS,
    SLICE_EXTRACTED_VS_EXTRACTED,
    SLICE_EXTRACTED_VS_SEED,
    SLICES,
    DatasetError,
    LabelledPair,
    known_entity_types,
    load_dataset,
    parse_dataset,
    validate_pairs,
)


def _pair(**overrides) -> LabelledPair:
    base = LabelledPair(
        slice=SLICE_EXTRACTED_VS_EXTRACTED,
        entity_type="Technology",
        label=LABEL_DUPLICATE,
        a="Python programming language",
        b="Python",
    )
    return replace(base, **overrides)


def _raw(**overrides) -> dict:
    base = {
        "slice": SLICE_EXTRACTED_VS_EXTRACTED,
        "entity_type": "Technology",
        "label": "duplicate",
        "a": "Python programming language",
        "b": "Python",
    }
    base.update(overrides)
    return base


def _problems(pairs: list[LabelledPair]) -> list[str]:
    return validate_pairs(pairs, enforce_minimums=False)


class TestCommittedDataset:
    @pytest.fixture(scope="class")
    def pairs(self) -> list[LabelledPair]:
        return load_dataset(DEFAULT_DATASET_PATH)

    def test_meets_the_count_floors(self, pairs):
        duplicates = [p for p in pairs if p.label == LABEL_DUPLICATE]
        distinct = [p for p in pairs if p.label == LABEL_DISTINCT]
        assert len(duplicates) >= MIN_DUPLICATE_PAIRS
        assert len(distinct) >= MIN_DISTINCT_PAIRS

    def test_has_hard_negatives(self, pairs):
        hard = [p for p in pairs if p.label == LABEL_DISTINCT and p.hard]
        assert len(hard) >= MIN_HARD_DISTINCT_PAIRS

    def test_at_least_forty_hard_negatives_overlap_lexically(self, pairs):
        """Same-type near misses with overlapping tokens, not just any distinct pair.

        Overlap means a shared word, or a shared three-letter prefix (GitHub / GitLab).
        The remaining hard pairs are same-category near misses with no shared text
        (Podman / Docker).
        """

        def words(name: str) -> set[str]:
            return {w for w in re.split(r"[^a-z0-9]+", name.casefold()) if w}

        def overlaps(a: str, b: str) -> bool:
            return bool(words(a) & words(b)) or a.casefold()[:3] == b.casefold()[:3]

        hard = [p for p in pairs if p.label == LABEL_DISTINCT and p.hard]
        overlapping = [p for p in hard if overlaps(p.a, p.b)]
        assert len(overlapping) >= MIN_HARD_DISTINCT_PAIRS

    @pytest.mark.parametrize("slice_name", SLICES)
    def test_every_slice_has_both_labels(self, pairs, slice_name):
        in_slice = [p for p in pairs if p.slice == slice_name]
        assert any(p.label == LABEL_DUPLICATE for p in in_slice)
        assert any(p.label == LABEL_DISTINCT for p in in_slice)

    def test_every_pair_has_a_valid_slice_label_and_ontology_type(self, pairs):
        types = known_entity_types()
        for p in pairs:
            assert p.slice in SLICES
            assert p.label in (LABEL_DUPLICATE, LABEL_DISTINCT)
            assert p.entity_type in types

    def test_no_repeated_pairs(self, pairs):
        keys = [p.key for p in pairs]
        assert len(keys) == len(set(keys))

    def test_a_name_pair_is_never_labelled_both_ways(self, pairs):
        by_names: dict[frozenset[str], set[str]] = {}
        for p in pairs:
            names = frozenset((p.a.casefold(), p.b.casefold()))
            by_names.setdefault(names, set()).add(p.label)
        assert all(len(labels) == 1 for labels in by_names.values())

    def test_pair_names_are_stripped_and_non_empty(self, pairs):
        for p in pairs:
            assert p.a == p.a.strip() and p.b == p.b.strip()
            assert p.a and p.b

    def test_dataset_file_is_ascii(self):
        DEFAULT_DATASET_PATH.read_bytes().decode("ascii")


class TestValidatePairs:
    def test_clean_pair_has_no_problems(self):
        assert _problems([_pair()]) == []

    def test_unknown_slice(self):
        assert any("slice" in p for p in _problems([_pair(slice="extracted_vs_graph")]))

    def test_unknown_label(self):
        assert any("label" in p for p in _problems([_pair(label="maybe")]))

    def test_entity_type_must_be_an_ontology_type(self):
        assert any("ontology type" in p for p in _problems([_pair(entity_type="Gadget")]))

    def test_custom_allowed_types(self):
        problems = validate_pairs(
            [_pair(entity_type="Gadget")],
            allowed_types=frozenset({"Gadget"}),
            enforce_minimums=False,
        )
        assert problems == []

    def test_same_name_on_both_sides(self):
        assert any("same name" in p for p in _problems([_pair(a="Python", b="python")]))

    def test_repeated_pair_is_rejected(self):
        problems = _problems([_pair(), _pair()])
        assert any("duplicates pairs[0]" in p for p in problems)

    def test_swapped_and_recased_pair_counts_as_repeated(self):
        problems = _problems(
            [_pair(a="Postgres", b="PostgreSQL"), _pair(a="postgresql", b="POSTGRES")]
        )
        assert any("duplicates pairs[0]" in p for p in problems)

    def test_same_names_in_another_slice_or_type_are_not_repeats(self):
        pairs = [
            _pair(),
            _pair(slice=SLICE_EXTRACTED_VS_SEED),
            _pair(entity_type="Skill"),
        ]
        assert _problems(pairs) == []

    def test_same_names_with_conflicting_labels_are_rejected(self):
        problems = _problems([_pair(), _pair(label=LABEL_DISTINCT)])
        assert any("duplicates pairs[0]" in p for p in problems)

    def test_hard_only_on_distinct_pairs(self):
        assert any("hard" in p for p in _problems([_pair(hard=True)]))
        assert _problems([_pair(label=LABEL_DISTINCT, hard=True)]) == []

    def test_b_description_only_in_seed_slice(self):
        assert any("b_description" in p for p in _problems([_pair(b_description="x")]))
        seed = _pair(slice=SLICE_EXTRACTED_VS_SEED, b_description="x")
        assert _problems([seed]) == []

    def test_count_floors_are_enforced_by_default(self):
        problems = validate_pairs([_pair()])
        joined = " ".join(problems)
        assert "duplicate pairs; need at least" in joined
        assert "distinct pairs; need at least" in joined
        assert "hard distinct pairs" in joined
        assert "slice extracted_vs_seed" in joined

    def test_count_floors_can_be_relaxed(self):
        assert validate_pairs([_pair()], enforce_minimums=False) == []


class TestParseAndLoad:
    def test_parse_accepts_a_minimal_document(self):
        doc = {"schema_version": 1, "pairs": [_raw(hard=False)]}
        pairs = parse_dataset(doc, enforce_minimums=False)
        assert pairs == [_pair()]

    def test_wrong_schema_version(self):
        with pytest.raises(DatasetError, match="schema_version"):
            parse_dataset({"schema_version": 2, "pairs": [_raw()]}, enforce_minimums=False)

    def test_top_level_must_be_object(self):
        with pytest.raises(DatasetError, match="JSON object"):
            parse_dataset([_raw()])

    def test_pairs_must_be_a_list(self):
        with pytest.raises(DatasetError, match="list"):
            parse_dataset({"schema_version": 1, "pairs": {}})

    def test_missing_key(self):
        raw = _raw()
        del raw["label"]
        with pytest.raises(DatasetError, match="missing keys"):
            parse_dataset({"schema_version": 1, "pairs": [raw]}, enforce_minimums=False)

    def test_unknown_key(self):
        with pytest.raises(DatasetError, match="unknown keys"):
            parse_dataset({"schema_version": 1, "pairs": [_raw(extra=1)]}, enforce_minimums=False)

    @pytest.mark.parametrize("bad", ["", "   ", 5, None])
    def test_names_must_be_non_empty_strings(self, bad):
        with pytest.raises(DatasetError, match="non-empty string"):
            parse_dataset({"schema_version": 1, "pairs": [_raw(a=bad)]}, enforce_minimums=False)

    def test_hard_must_be_boolean(self):
        with pytest.raises(DatasetError, match="boolean"):
            parse_dataset(
                {"schema_version": 1, "pairs": [_raw(label="distinct", hard="yes")]},
                enforce_minimums=False,
            )

    def test_error_lists_every_problem(self):
        doc = {"schema_version": 1, "pairs": [_raw(a=""), _raw(b=3)]}
        with pytest.raises(DatasetError) as info:
            parse_dataset(doc, enforce_minimums=False)
        assert len(info.value.problems) == 2

    def test_load_missing_file(self, tmp_path):
        with pytest.raises(DatasetError, match="cannot read"):
            load_dataset(tmp_path / "nope.json")

    def test_load_invalid_json(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json", encoding="utf-8")
        with pytest.raises(DatasetError, match="not valid JSON"):
            load_dataset(path)

    def test_load_enforces_floors_on_a_small_file(self, tmp_path):
        path = tmp_path / "small.json"
        path.write_text(json.dumps({"schema_version": 1, "pairs": [_raw()]}), encoding="utf-8")
        with pytest.raises(DatasetError, match="need at least"):
            load_dataset(path)
        assert len(load_dataset(path, enforce_minimums=False)) == 1
