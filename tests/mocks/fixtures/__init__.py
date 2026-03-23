"""Shared test fixtures.

Import fixtures from submodules into local conftest.py files:
    from tests.mocks.fixtures.knowledge import sample_ontology  # noqa: F401
"""

from tests.mocks.fixtures.events import make_turn_event  # noqa: F401
from tests.mocks.fixtures.knowledge import (  # noqa: F401
    make_extraction_result,
    sample_ontology,
)
