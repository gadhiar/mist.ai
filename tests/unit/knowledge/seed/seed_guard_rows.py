"""Answers for the seed guard's two read-only statements, for connection fakes.

`apply_seed_documents` and `reseed` run `admin.RESET_GUARD_CYPHER` and
`applier.SEED_GUARD_CYPHER` before any write (MIS-177 D1) and fail closed on
an empty result. A fake that returns `[]` for every read therefore now stops
the seed before its first write, which is correct. A test about something
other than the guard answers the two statements with a clean row through
`answer_seed_guard`; every other read still falls through to the fake's own
behaviour, so nothing else a test asserts changes.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from backend.knowledge import admin
from backend.knowledge.seed import applier

CLEAN_RESET_GUARD_ROW: dict[str, int] = {"nodes": 0, "relationships": 0}
CLEAN_SEED_GUARD_ROW: dict[str, int] = {name: 0 for name in applier.SEED_GUARD_COLUMNS}

Router = Callable[[str, dict | None], list[dict[str, Any]] | None]


def answer_seed_guard(query: str) -> list[dict[str, Any]] | None:
    """A clean row for either guard statement, or None for any other query."""
    if query == admin.RESET_GUARD_CYPHER:
        return [dict(CLEAN_RESET_GUARD_ROW)]
    if query == applier.SEED_GUARD_CYPHER:
        return [dict(CLEAN_SEED_GUARD_ROW)]
    return None


def clean_seed_guard_router(query: str, params: dict | None = None) -> list[dict[str, Any]] | None:
    """A `FakeNeo4jConnection(query_router=...)` that passes the seed guard."""
    return answer_seed_guard(query)


def seed_guard_router(
    *,
    reset_guard_rows: list[dict[str, Any]] | None = None,
    seed_guard_rows: list[dict[str, Any]] | None = None,
) -> Router:
    """A router answering each guard statement with the given rows.

    A statement whose rows are not given gets its clean row. Pass `[]` to
    return no row at all.
    """
    reset_rows = [dict(CLEAN_RESET_GUARD_ROW)] if reset_guard_rows is None else reset_guard_rows
    seed_rows = [dict(CLEAN_SEED_GUARD_ROW)] if seed_guard_rows is None else seed_guard_rows

    def route(query: str, params: dict | None = None) -> list[dict[str, Any]] | None:
        if query == admin.RESET_GUARD_CYPHER:
            return reset_rows
        if query == applier.SEED_GUARD_CYPHER:
            return seed_rows
        return None

    return route
