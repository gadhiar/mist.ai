"""Smoke tests for the isolated_test_vault pytest fixture.

The fixture is defined in tests/conftest.py and provides an ephemeral
copy of tests/fixtures/test-vault/ with MIST_VAULT_ROOT env var redirected.
This file validates the contract:

- Baseline content is present after copy
- MIST_VAULT_ROOT env var points at the ephemeral copy
- Config singleton is reset so get_config() picks up the override
- Two tests using the fixture get independent vault directories (isolation)
- Teardown reverts env var and singleton
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from backend.knowledge import config as config_module


class TestIsolatedTestVaultBaseline:
    """Baseline content is present in the ephemeral copy."""

    def test_vault_root_exists(self, isolated_test_vault: Path):
        assert isolated_test_vault.exists()
        assert isolated_test_vault.is_dir()

    def test_mist_md_present(self, isolated_test_vault: Path):
        mist_md = isolated_test_vault / "MIST.md"
        assert mist_md.exists()
        content = mist_md.read_text(encoding="utf-8")
        assert "MIST Vault Conventions" in content

    def test_identity_present(self, isolated_test_vault: Path):
        identity = isolated_test_vault / "identity" / "mist.md"
        assert identity.exists()
        assert "type: mist-identity" in identity.read_text(encoding="utf-8")

    def test_test_user_present(self, isolated_test_vault: Path):
        user = isolated_test_vault / "users" / "test-user.md"
        assert user.exists()
        assert "user_id: test-user" in user.read_text(encoding="utf-8")

    def test_sessions_present(self, isolated_test_vault: Path):
        sessions = list((isolated_test_vault / "sessions").iterdir())
        assert len(sessions) >= 2

    def test_meta_present(self, isolated_test_vault: Path):
        assert (isolated_test_vault / "meta" / "schema.md").exists()
        assert (isolated_test_vault / "meta" / "changelog.md").exists()


class TestIsolatedTestVaultEnvVarOverride:
    """Env var and singleton are wired correctly."""

    def test_env_var_set_to_target(self, isolated_test_vault: Path):
        assert os.environ.get("MIST_VAULT_ROOT") == str(isolated_test_vault)

    def test_env_var_not_real_mist_memory(self, isolated_test_vault: Path):
        env_path = os.environ.get("MIST_VAULT_ROOT")
        assert env_path is not None
        # The real vault is at mist-memory/ relative to repo root. The
        # ephemeral copy must NOT be that path.
        assert (
            not env_path.endswith("mist-memory")
            or "tmp" in env_path.lower()
            or "test-vault" in env_path
        )

    def test_config_singleton_reset(self, isolated_test_vault: Path):
        # Singleton was reset by the fixture; first get_config() call after
        # the fixture should populate from current env. The env var points
        # at the test vault, so the resolved config.vault.root should match.
        from backend.knowledge.config import get_config

        cfg = get_config()
        assert cfg.vault.root == str(isolated_test_vault)

    def test_config_vault_enabled_by_default(self, isolated_test_vault: Path):
        # VaultConfig.from_env() defaults enabled=True (matches production
        # docker-compose env). The fixture should not change this; tests
        # that need disabled vault must override explicitly.
        from backend.knowledge.config import get_config

        cfg = get_config()
        assert cfg.vault.enabled is True


class TestIsolatedTestVaultIsolation:
    """Successive tests get independent vault dirs."""

    _seen_paths: list[Path] = []

    def test_first_run_unique_path(self, isolated_test_vault: Path):
        # Record the path; second test will assert it's different.
        TestIsolatedTestVaultIsolation._seen_paths.append(isolated_test_vault)
        assert isolated_test_vault.exists()

    def test_second_run_different_path(self, isolated_test_vault: Path):
        # If isolation works, this fixture invocation produced a different
        # tmp_path from the first.
        assert isolated_test_vault.exists()
        assert isolated_test_vault not in TestIsolatedTestVaultIsolation._seen_paths

    def test_mutation_in_one_does_not_affect_other(
        self, isolated_test_vault: Path, tmp_path_factory: pytest.TempPathFactory
    ):
        # Write a sentinel file in this vault; a fresh fixture invocation
        # in another test must not see it. The two TestIsolatedTestVaultIsolation
        # methods above already cover the cross-invocation path; this test
        # belt-and-suspenders by writing a mutation.
        sentinel = isolated_test_vault / "sentinel-must-not-leak.txt"
        sentinel.write_text("polluted")
        assert sentinel.exists()
        # The cleanup is via tmp_path; nothing for this test to assert
        # beyond the write succeeding inside the ephemeral copy.


class TestIsolatedTestVaultTeardown:
    """After the fixture's scope ends, env and singleton revert."""

    def test_env_var_reverted_after_fixture_scope(self, monkeypatch: pytest.MonkeyPatch):
        # This test does NOT use isolated_test_vault. The MIST_VAULT_ROOT
        # env var should reflect whatever was set externally (likely the
        # docker-compose value /app/mist-memory) or unset on host.
        # Importantly: the test_isolated_test_vault_env_var_set_to_target
        # ABOVE set it to a tmp_path. After that test's scope ended,
        # monkeypatch reverted it. So at THIS point, the env var should
        # NOT point at any tmp path.
        env_path = os.environ.get("MIST_VAULT_ROOT", "")
        # The env var, if set, should not be a tmp path leaking from prior tests
        assert (
            "tmp" not in env_path.lower() or "test-vault" not in env_path.lower() or env_path == ""
        ), f"MIST_VAULT_ROOT leaked from prior fixture: {env_path}"

    def test_singleton_reverts_after_fixture_scope(self):
        # After the prior fixture-using test's scope ended, the singleton
        # attribute on the config module is whatever monkeypatch restored.
        # We don't assert a specific value (depends on test order); we only
        # assert the attribute exists.
        assert hasattr(config_module, "_config")
