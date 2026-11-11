"""The pre-LLM interpreters: what a pack's rules do before the model is called.

Two kinds run here. `item_filter` drops the units in a structural state the pack names,
so the model never sees them; `plan_vocabulary` is what the turn plan's templates may
name. Both read a pack (`backend.knowledge.ontologies.pack`) and know no type or edge of
their own. A `prompt_fragment` rule is data the prompt assembler places; nothing here
interprets it.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

from backend.knowledge.extraction.structure_splitter import InputUnit
from backend.knowledge.ontologies.pack import PRE_LLM, Pack

# How the structure splitter writes each item state at the start of a unit's text.
_STATE_MARKS: dict[str, str] = {"unchecked": "[ ] ", "checked": "[x] "}
_FILTERED_KIND = "list_item"


@dataclass(frozen=True, slots=True)
class PlanWords:
    """What the turn plan may name, from the pack's `plan_vocabulary` rules.

    Attributes:
        entity_types: The types a plan entity may have.
        predicates: The relationship a template's lead-in may state.
        fan_out_predicates: The relationships a template may give each line under it.
    """

    entity_types: tuple[str, ...]
    predicates: tuple[str, ...]
    fan_out_predicates: tuple[str, ...]


@lru_cache(maxsize=64)
def _item_filters(pack: Pack) -> tuple[tuple[str, re.Pattern | None], ...]:
    """Each `item_filter` rule as (the state's mark, its keep words as one pattern)."""
    filters: list[tuple[str, re.Pattern | None]] = []
    for rule in pack.rules_of("item_filter", PRE_LLM):
        words = [re.escape(word.strip()) for word in rule.params.keep_context_words]
        keep = None
        if words:
            # A whole word or phrase, singular or plural, in any case.
            keep = re.compile(
                rf"(?<![A-Za-z0-9])(?:{'|'.join(words)})(?:s|es)?(?![A-Za-z0-9])", re.IGNORECASE
            )
        filters.append((_STATE_MARKS[rule.params.state], keep))
    return tuple(filters)


def filter_units(units: Sequence[InputUnit], pack: Pack) -> tuple[InputUnit, ...]:
    """Drop the list items in a state an `item_filter` rule names.

    An item is kept when a heading, lead-in or parent item above it holds one of the
    rule's keep words; the item's own text is not read for them, since an item that
    mentions a word is not thereby under it.

    Args:
        units: The structure splitter's units, in order.
        pack: The pack whose rules run.

    Returns:
        The units the model is given, in order.
    """
    filters = _item_filters(pack)
    kept: list[InputUnit] = []
    for unit in units:
        dropped = unit.kind == _FILTERED_KIND and any(
            unit.text.startswith(mark)
            and not (keep is not None and any(keep.search(line) for line in unit.context))
            for mark, keep in filters
        )
        if not dropped:
            kept.append(unit)
    return tuple(kept)


def _merged(pack: Pack, field: str) -> tuple[str, ...]:
    words: list[str] = []
    for rule in pack.rules_of("plan_vocabulary", PRE_LLM):
        for word in getattr(rule.params, field):
            if word not in words:
                words.append(word)
    return tuple(words)


def plan_vocabulary(pack: Pack) -> PlanWords:
    """The pack's `plan_vocabulary` rules as one vocabulary.

    A later rule adds its words after an earlier rule's; a repeated word is kept once.
    """
    return PlanWords(
        entity_types=_merged(pack, "entity_types"),
        predicates=_merged(pack, "predicates"),
        fan_out_predicates=_merged(pack, "fan_out_predicates"),
    )
