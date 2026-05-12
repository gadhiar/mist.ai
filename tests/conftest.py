"""Root test configuration.

Auto-applies pytest markers based on test directory:
- tests/unit/ -> @pytest.mark.unit
- tests/integration/ -> @pytest.mark.integration

Provides shared fixtures:
- isolated_test_vault: ephemeral copy of tests/fixtures/test-vault/ with
  MIST_VAULT_ROOT env var redirected to the copy. Used by any test that
  exercises vault code paths so the real mist-memory/ is never touched.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

TESTS_ROOT = Path(__file__).resolve().parent
TEST_VAULT_BASELINE = TESTS_ROOT / "fixtures" / "test-vault"


def pytest_collection_modifyitems(items):
    """Auto-apply markers based on test file location."""
    for item in items:
        path = str(item.fspath)
        if "/unit/" in path or "\\unit\\" in path:
            item.add_marker(pytest.mark.unit)
        elif "/integration/" in path or "\\integration\\" in path:
            item.add_marker(pytest.mark.integration)


@pytest.fixture
def isolated_test_vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Provide an isolated test vault copied from tests/fixtures/test-vault/.

    Every gauntlet, integration test, and any unit test that touches vault
    code paths MUST use this fixture instead of the real `mist-memory/`
    directory. The real vault is the user's canonical memory; gauntlet
    runs and test extractions would otherwise pollute it.

    The fixture:

    1. Copies tests/fixtures/test-vault/ to an ephemeral location under
       pytest's tmp_path (auto-cleaned at session end).
    2. Sets the MIST_VAULT_ROOT environment variable to the ephemeral
       path. Backend code reads this env var when constructing
       VaultConfig.from_env() (see backend/knowledge/config.py:395).
    3. Drops the global config singleton so a subsequent get_config()
       call re-reads the env var.

    Teardown reverts both the env var and the singleton via pytest's
    monkeypatch fixture; the ephemeral directory is auto-cleaned.

    Yields:
        Path: absolute path to the ephemeral test vault root.

    Usage:
        def test_vault_write(isolated_test_vault: Path):
            assert (isolated_test_vault / "MIST.md").exists()
            # vault writes during this test land in the ephemeral copy

        def test_via_config(isolated_test_vault: Path):
            from backend.knowledge.config import get_config
            cfg = get_config()
            assert cfg.vault.root == str(isolated_test_vault)
    """
    if not TEST_VAULT_BASELINE.exists():
        raise RuntimeError(
            f"Test vault baseline missing at {TEST_VAULT_BASELINE}. "
            f"Did you delete tests/fixtures/test-vault/? Restore it from git."
        )

    target = tmp_path / "test-vault"
    shutil.copytree(TEST_VAULT_BASELINE, target)

    monkeypatch.setenv("MIST_VAULT_ROOT", str(target))

    # Drop the config singleton so get_config() picks up the new env var.
    # monkeypatch.setattr auto-restores on teardown.
    from backend.knowledge import config as config_module

    monkeypatch.setattr(config_module, "_config", None)

    return target
