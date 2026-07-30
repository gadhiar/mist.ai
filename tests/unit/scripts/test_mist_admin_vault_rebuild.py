"""R1.3: vault-rebuild loses its graph modes, keeps the sidecar rebuild.

--scope and --retry-orphaned drove the deleted curation GraphRegenerator: a
vault edit no longer produces graph facts, so there is no vault-derived
subgraph to rebuild from a vault file. Graph rebuilds now run from the
utterance log via `graph-rebuild-from-log` (R1.2). This file pins the
retired surface and the trimmed parser contract.
"""

from __future__ import annotations

import pytest

import scripts.mist_admin as mist_admin


def test_graph_rebuild_entrypoints_are_retired() -> None:
    """--scope and --retry-orphaned drove the deleted curation regenerator."""
    assert not hasattr(mist_admin, "cmd_vault_rebuild")
    assert not hasattr(mist_admin, "_build_vault_rebuild_ctx")
    assert not hasattr(mist_admin, "_dispatch_vault_rebuild")


def test_vault_rebuild_parser_accepts_only_confirm() -> None:
    parser = mist_admin.build_parser()
    args = parser.parse_args(["vault-rebuild", "--confirm"])
    assert args.confirm is True
    assert not hasattr(args, "scope")
    assert not hasattr(args, "retry_orphaned")
    assert args.func is mist_admin._cmd_vault_rebuild_sidecar


def test_vault_rebuild_rejects_retired_flags() -> None:
    parser = mist_admin.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["vault-rebuild", "--scope", "all"])
    with pytest.raises(SystemExit):
        parser.parse_args(["vault-rebuild", "--retry-orphaned"])
