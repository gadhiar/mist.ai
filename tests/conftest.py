"""Root test configuration.

Auto-applies pytest markers based on test directory:
- tests/unit/ -> @pytest.mark.unit
- tests/integration/ -> @pytest.mark.integration
"""

import pytest


def pytest_collection_modifyitems(items):
    """Auto-apply markers based on test file location."""
    for item in items:
        path = str(item.fspath)
        if "/unit/" in path or "\\unit\\" in path:
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in path or "\\integration\\" in path:
            item.add_marker(pytest.mark.integration)
