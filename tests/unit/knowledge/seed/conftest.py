"""Fixtures for the seed tests.

`fake_connection` here overrides the `tests/unit/conftest.py` one so it answers
the seed guard's two statements with a clean row (MIS-177 D1): these tests
exercise the applier's writes, wipe and validation, not the guard, and the
guard fails closed on the `[]` an unconfigured fake returns. Every other read
still gets `[]`. The guard itself is tested in `test_seed_guard.py`.
"""

from __future__ import annotations

import pytest

from tests.mocks.neo4j import FakeNeo4jConnection
from tests.unit.knowledge.seed.seed_guard_rows import clean_seed_guard_router


@pytest.fixture
def fake_connection():
    """A FakeNeo4jConnection whose only answers are clean seed guard rows."""
    return FakeNeo4jConnection(query_router=clean_seed_guard_router)
