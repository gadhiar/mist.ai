"""R1.3 Task 8: the one surviving guard from the Phase 3 vault-rebuild suite.

This file originally covered the async `cmd_vault_rebuild(scope, retry_orphaned,
ctx)` signature added in Phase 3 Task 22 (graph-aware --scope / --retry-orphaned
rebuild modes) plus the argparse dispatch that routed between it and the legacy
sidecar-only handler. R1.3 Task 8 retired `cmd_vault_rebuild`,
`_build_vault_rebuild_ctx`, and `_dispatch_vault_rebuild`: a vault edit no
longer produces graph facts, so there is no vault-derived subgraph left to
rebuild from a vault file. Graph rebuilds now run from the utterance log via
`graph-rebuild-from-log` (R1.2).

Eleven of the twelve tests that lived here lost their subject along with those
functions and were deleted. The remaining test guards this task's own
deliverable: the bare `vault-rebuild` subcommand must bind
`_cmd_vault_rebuild_sidecar` directly in `build_parser`, not through a dispatch
layer. `tests/unit/scripts/test_mist_admin_vault_rebuild.py` covers the
retirement itself (the three functions are gone; the parser rejects --scope
and --retry-orphaned).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# scripts/ is not a package; insert repo root so mist_admin is importable.
_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT / "scripts"))

import mist_admin  # noqa: E402  -- after sys.path insertion


class TestVaultRebuildDispatch:
    """The bare `vault-rebuild` subcommand resolves to the sidecar handler."""

    def test_no_flags_routes_to_legacy_sidecar(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """vault-rebuild (no flags) invokes _cmd_vault_rebuild_sidecar.

        `new_path_calls` is retained from the pre-retirement version of this
        test for continuity, but it is now vacuously empty: there is no
        longer any graph-aware function to divert into, so nothing could ever
        append to it. The meaningful assertion is `legacy_calls != []` --
        confirming the subcommand still resolves to _cmd_vault_rebuild_sidecar.
        """
        new_path_calls: list[dict] = []
        legacy_calls: list[object] = []

        def fake_sidecar(args: object) -> int:
            legacy_calls.append(args)
            return 0

        monkeypatch.setattr(mist_admin, "_cmd_vault_rebuild_sidecar", fake_sidecar)

        parser = mist_admin.build_parser()
        args = parser.parse_args(["vault-rebuild"])

        args.func(args)

        assert legacy_calls != []
        assert new_path_calls == []
