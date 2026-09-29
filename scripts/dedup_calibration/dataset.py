"""Labelled pair dataset for the dedup threshold calibration tool.

The dataset is a JSON file of entity-name pairs. Each pair carries a slice
(which dedup situation it models), an entity type, a label (duplicate or
distinct) and the two names. `validate_pairs` is strict on purpose: a
calibration built on a mislabelled or accidentally duplicated pair reports a
threshold nobody can trust.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SLICE_EXTRACTED_VS_EXTRACTED = "extracted_vs_extracted"
SLICE_EXTRACTED_VS_SEED = "extracted_vs_seed"
SLICES: tuple[str, ...] = (SLICE_EXTRACTED_VS_EXTRACTED, SLICE_EXTRACTED_VS_SEED)

LABEL_DUPLICATE = "duplicate"
LABEL_DISTINCT = "distinct"
LABELS: tuple[str, ...] = (LABEL_DUPLICATE, LABEL_DISTINCT)

# Floors for the committed dataset. Below these the per-slice distributions are
# too thin to read a threshold from.
MIN_DUPLICATE_PAIRS = 40
MIN_DISTINCT_PAIRS = 80
MIN_DUPLICATE_PAIRS_PER_SLICE = 20
MIN_DISTINCT_PAIRS_PER_SLICE = 40
MIN_HARD_DISTINCT_PAIRS = 40

DEFAULT_DATASET_PATH = Path(__file__).resolve().parent / "pairs.json"

_REQUIRED_KEYS = ("slice", "entity_type", "label", "a", "b")
_OPTIONAL_KEYS = ("hard", "b_description")
_SUPPORTED_SCHEMA_VERSION = 1


class DatasetError(ValueError):
    """Raised when the dataset file is unreadable or fails validation."""

    def __init__(self, problems: list[str]) -> None:
        self.problems = problems
        super().__init__("; ".join(problems))


@dataclass(frozen=True, slots=True)
class LabelledPair:
    """One labelled pair.

    Side `a` is the extraction probe. Side `b` is the stored node: an
    extraction-created node for `extracted_vs_extracted`, a seed node for
    `extracted_vs_seed`.
    """

    slice: str
    entity_type: str
    label: str
    a: str
    b: str
    hard: bool = False
    b_description: str | None = None

    @property
    def is_duplicate(self) -> bool:
        """True when the pair is labelled as one real-world entity."""
        return self.label == LABEL_DUPLICATE

    @property
    def key(self) -> tuple[str, str, frozenset[str]]:
        """Order-insensitive identity: slice, type, and the unordered name pair."""
        return (self.slice, self.entity_type, frozenset((self.a.casefold(), self.b.casefold())))


def known_entity_types() -> frozenset[str]:
    """Entity type names from the ontology (imported lazily)."""
    from backend.knowledge.ontologies.v1_0_0 import ALL_NODE_TYPES

    return frozenset(nt.type_name for nt in ALL_NODE_TYPES)


def _parse_pair(index: int, raw: object, problems: list[str]) -> LabelledPair | None:
    where = f"pairs[{index}]"
    if not isinstance(raw, dict):
        problems.append(f"{where}: expected an object, got {type(raw).__name__}")
        return None
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if missing:
        problems.append(f"{where}: missing keys {missing}")
        return None
    unknown = sorted(set(raw) - set(_REQUIRED_KEYS) - set(_OPTIONAL_KEYS))
    if unknown:
        problems.append(f"{where}: unknown keys {unknown}")
        return None
    for key in _REQUIRED_KEYS:
        value = raw[key]
        if not isinstance(value, str) or not value.strip():
            problems.append(f"{where}: `{key}` must be a non-empty string")
            return None
    hard = raw.get("hard", False)
    if not isinstance(hard, bool):
        problems.append(f"{where}: `hard` must be a boolean")
        return None
    b_description = raw.get("b_description")
    if b_description is not None and (not isinstance(b_description, str) or not b_description):
        problems.append(f"{where}: `b_description` must be a non-empty string when present")
        return None
    return LabelledPair(
        slice=raw["slice"],
        entity_type=raw["entity_type"],
        label=raw["label"],
        a=raw["a"],
        b=raw["b"],
        hard=hard,
        b_description=b_description,
    )


def validate_pairs(
    pairs: list[LabelledPair],
    *,
    allowed_types: frozenset[str] | None = None,
    enforce_minimums: bool = True,
) -> list[str]:
    """Return every validation problem found (empty list means valid).

    Args:
        pairs: Parsed pairs.
        allowed_types: Permitted entity types. Defaults to the ontology's.
        enforce_minimums: Apply the count floors. Unit tests of individual
            rules turn this off so they can use tiny datasets.
    """
    problems: list[str] = []
    types = allowed_types if allowed_types is not None else known_entity_types()
    seen: dict[tuple[str, str, frozenset[str]], int] = {}

    for i, pair in enumerate(pairs):
        where = f"pairs[{i}] ({pair.a!r} / {pair.b!r})"
        if pair.slice not in SLICES:
            problems.append(f"{where}: slice {pair.slice!r} not in {list(SLICES)}")
        if pair.label not in LABELS:
            problems.append(f"{where}: label {pair.label!r} not in {list(LABELS)}")
        if pair.entity_type not in types:
            problems.append(f"{where}: entity_type {pair.entity_type!r} is not an ontology type")
        if pair.a.casefold() == pair.b.casefold():
            problems.append(f"{where}: both sides are the same name")
        if pair.hard and pair.label != LABEL_DISTINCT:
            problems.append(f"{where}: `hard` is only meaningful on distinct pairs")
        if pair.b_description is not None and pair.slice != SLICE_EXTRACTED_VS_SEED:
            problems.append(
                f"{where}: `b_description` is only valid in slice {SLICE_EXTRACTED_VS_SEED}"
            )
        first = seen.setdefault(pair.key, i)
        if first != i:
            problems.append(f"{where}: duplicates pairs[{first}] (same slice, type and names)")

    if enforce_minimums:
        problems.extend(_count_problems(pairs))
    return problems


def _count_problems(pairs: list[LabelledPair]) -> list[str]:
    problems: list[str] = []
    dup = sum(1 for p in pairs if p.label == LABEL_DUPLICATE)
    distinct = sum(1 for p in pairs if p.label == LABEL_DISTINCT)
    hard = sum(1 for p in pairs if p.label == LABEL_DISTINCT and p.hard)
    if dup < MIN_DUPLICATE_PAIRS:
        problems.append(f"only {dup} duplicate pairs; need at least {MIN_DUPLICATE_PAIRS}")
    if distinct < MIN_DISTINCT_PAIRS:
        problems.append(f"only {distinct} distinct pairs; need at least {MIN_DISTINCT_PAIRS}")
    if hard < MIN_HARD_DISTINCT_PAIRS:
        problems.append(
            f"only {hard} hard distinct pairs; need at least {MIN_HARD_DISTINCT_PAIRS} "
            "(same-type near misses with overlapping tokens)"
        )
    for slice_name in SLICES:
        in_slice = [p for p in pairs if p.slice == slice_name]
        s_dup = sum(1 for p in in_slice if p.label == LABEL_DUPLICATE)
        s_distinct = sum(1 for p in in_slice if p.label == LABEL_DISTINCT)
        if s_dup < MIN_DUPLICATE_PAIRS_PER_SLICE:
            problems.append(
                f"slice {slice_name}: only {s_dup} duplicate pairs; "
                f"need at least {MIN_DUPLICATE_PAIRS_PER_SLICE}"
            )
        if s_distinct < MIN_DISTINCT_PAIRS_PER_SLICE:
            problems.append(
                f"slice {slice_name}: only {s_distinct} distinct pairs; "
                f"need at least {MIN_DISTINCT_PAIRS_PER_SLICE}"
            )
    return problems


def parse_dataset(document: object, *, enforce_minimums: bool = True) -> list[LabelledPair]:
    """Parse and validate an already-decoded dataset document.

    Raises:
        DatasetError: With every problem found, when the document is invalid.
    """
    problems: list[str] = []
    if not isinstance(document, dict):
        raise DatasetError(["top level must be a JSON object"])
    if document.get("schema_version") != _SUPPORTED_SCHEMA_VERSION:
        problems.append(f"schema_version must be {_SUPPORTED_SCHEMA_VERSION}")
    raw_pairs = document.get("pairs")
    if not isinstance(raw_pairs, list):
        raise DatasetError([*problems, "`pairs` must be a list"])

    pairs: list[LabelledPair] = []
    for index, raw in enumerate(raw_pairs):
        parsed = _parse_pair(index, raw, problems)
        if parsed is not None:
            pairs.append(parsed)
    if not problems:
        problems.extend(validate_pairs(pairs, enforce_minimums=enforce_minimums))
    if problems:
        raise DatasetError(problems)
    return pairs


def load_dataset(path: Path, *, enforce_minimums: bool = True) -> list[LabelledPair]:
    """Read, parse and validate the dataset at `path`.

    Raises:
        DatasetError: The file is unreadable, not JSON, or fails validation.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DatasetError([f"cannot read dataset {path}: {exc}"]) from exc
    try:
        document = json.loads(text)
    except json.JSONDecodeError as exc:
        raise DatasetError([f"dataset {path} is not valid JSON: {exc}"]) from exc
    return parse_dataset(document, enforce_minimums=enforce_minimums)
