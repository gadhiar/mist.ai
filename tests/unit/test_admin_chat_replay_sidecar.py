"""Unit tests for vault-sidecar wiring in the mist_admin chat/replay CLI path.

Regression guard for the fidelity gap where `cmd_chat` / `cmd_replay` built the
production ConversationHandler WITHOUT a vault sidecar -- structurally disabling
the vault auto-RAG ("Relevant Documents") retrieval and making the CLI
misrepresent the server's WebSocket path. The server lifespan wires a sidecar
(backend/server.py: build_sidecar_index -> vault_sidecar= into
build_conversation_handler); the CLI must mirror that so chat/replay validate
the same retrieval behavior.

These tests assert the WIRING contract only -- not retrieval quality (that is
covered behaviorally in-container). They confirm:
- when the sidecar is enabled, cmd_chat/cmd_replay build a sidecar and forward
  it as vault_sidecar= into build_conversation_handler
- when the sidecar is disabled, the forwarded vault_sidecar is None (the
  enablement guard is respected, mirroring the server)
- the sidecar handle is released (closed) after the command, even on success

Patching is used as a last resort per tests/CLAUDE.md: cmd_chat/cmd_replay are
module-level CLI glue calling other module-level functions. Spies are explicit
(record args / return scripted values), never bare MagicMock at I/O boundaries.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Make `scripts` importable without installing the repo as a package.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import mist_admin  # noqa: E402

MODULE = "mist_admin"


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class SpySidecar:
    """Stand-in for VaultSidecarIndex that records lifecycle calls.

    Truthy and identity-distinct so tests can assert it is the exact object
    threaded into build_conversation_handler and later closed.
    """

    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class SpyHandler:
    """Captures the vault_sidecar passed into build_conversation_handler.

    Provides the minimal async surface run_chat/run_replay touch so the real
    Tier-3 coroutines can run unchanged against this double.
    """

    def __init__(self, vault_sidecar: object) -> None:
        self.vault_sidecar = vault_sidecar
        self.calls: list[dict] = []

    async def handle_message(
        self,
        user_message: str,
        session_id: str,
        user_id: str = "User",
        max_history: int = 10,
    ) -> str:
        self.calls.append(
            {"user_message": user_message, "session_id": session_id, "user_id": user_id}
        )
        return f"echo: {user_message}"


class WiringSpies:
    """Bundle of installed patches recording how the CLI wired the sidecar.

    `built_sidecar` is what `_build_cli_sidecar` returned; `handler_kwargs`
    captures the kwargs `build_conversation_handler` was called with;
    `closed_sidecars` records every object handed to `_close_cli_sidecar`.
    """

    def __init__(self) -> None:
        self.built_sidecar: object | None = None
        self.handler_kwargs: dict | None = None
        self.closed_sidecars: list[object] = []


@pytest.fixture
def sidecar_enabled(request):
    """Indirect param: True -> sidecar enabled, False -> disabled.

    Routed via @parametrize(..., indirect=True) so each test declares the
    config-enablement state it exercises without a real config object.
    """
    return request.param


@pytest.fixture
def wiring(monkeypatch, sidecar_enabled):
    """Patch the CLI's collaborators with explicit spies.

    `sidecar_enabled` (indirect param) controls whether `_build_cli_sidecar`
    returns a SpySidecar (enabled) or None (disabled) -- mirroring the real
    helper's config.sidecar_index.enabled guard without touching SQLite.
    """
    spies = WiringSpies()

    # _load_backend() -> object exposing get_config(); config value is opaque
    # to these wiring tests (the spies ignore it), so a sentinel suffices.
    fake_config = SimpleNamespace(_sentinel="cfg")
    fake_backend = SimpleNamespace(get_config=lambda: fake_config)
    monkeypatch.setattr(f"{MODULE}._load_backend", lambda: fake_backend)

    def fake_build_cli_sidecar(config):
        assert config is fake_config  # CLI forwards the loaded config verbatim
        spies.built_sidecar = SpySidecar() if sidecar_enabled else None
        return spies.built_sidecar

    monkeypatch.setattr(f"{MODULE}._build_cli_sidecar", fake_build_cli_sidecar)

    def fake_close_cli_sidecar(sidecar):
        spies.closed_sidecars.append(sidecar)
        if sidecar is not None:
            sidecar.close()

    monkeypatch.setattr(f"{MODULE}._close_cli_sidecar", fake_close_cli_sidecar)

    # cmd_chat/cmd_replay do `from backend.factories import build_conversation_handler`
    # at call time, so patch the source attribute on the real factories module.
    import backend.factories as factories

    def fake_build_conversation_handler(config, **kwargs):
        spies.handler_kwargs = kwargs
        return SpyHandler(vault_sidecar=kwargs.get("vault_sidecar"))

    monkeypatch.setattr(factories, "build_conversation_handler", fake_build_conversation_handler)

    return spies


# ---------------------------------------------------------------------------
# cmd_chat
# ---------------------------------------------------------------------------


class TestCmdChatSidecarWiring:
    @pytest.mark.parametrize("sidecar_enabled", [True], indirect=True)
    def test_passes_built_sidecar_into_handler_when_enabled(self, wiring):
        # Arrange
        args = SimpleNamespace(session_id="s1", user_id="user", utterance="hello", output=None)

        # Act
        rc = mist_admin.cmd_chat(args)

        # Assert: the exact sidecar object built was forwarded as vault_sidecar=
        assert rc == 0
        assert wiring.built_sidecar is not None
        assert wiring.handler_kwargs is not None
        assert wiring.handler_kwargs["vault_sidecar"] is wiring.built_sidecar

    @pytest.mark.parametrize("sidecar_enabled", [False], indirect=True)
    def test_forwards_none_when_sidecar_disabled(self, wiring):
        # Arrange
        args = SimpleNamespace(session_id="s1", user_id="user", utterance="hello", output=None)

        # Act
        rc = mist_admin.cmd_chat(args)

        # Assert: enablement guard respected -- no sidecar forced when disabled
        assert rc == 0
        assert wiring.built_sidecar is None
        assert wiring.handler_kwargs["vault_sidecar"] is None

    @pytest.mark.parametrize("sidecar_enabled", [True], indirect=True)
    def test_closes_sidecar_after_run(self, wiring):
        # Arrange
        args = SimpleNamespace(session_id="s1", user_id="user", utterance="hello", output=None)

        # Act
        mist_admin.cmd_chat(args)

        # Assert: the SQLite handle is released even on the success path
        assert wiring.closed_sidecars == [wiring.built_sidecar]
        assert wiring.built_sidecar.closed is True


# ---------------------------------------------------------------------------
# cmd_replay
# ---------------------------------------------------------------------------


class TestCmdReplaySidecarWiring:
    @staticmethod
    def _write_inputs(tmp_path: Path) -> Path:
        path = tmp_path / "inputs.jsonl"
        path.write_text('{"utterance": "hi"}\n{"utterance": "bye"}\n', encoding="utf-8")
        return path

    @pytest.mark.parametrize("sidecar_enabled", [True], indirect=True)
    def test_passes_built_sidecar_into_handler_when_enabled(self, wiring, tmp_path):
        # Arrange
        args = SimpleNamespace(
            input=str(self._write_inputs(tmp_path)),
            session_id="s1",
            user_id="user",
            output=None,
        )

        # Act
        rc = mist_admin.cmd_replay(args)

        # Assert
        assert rc == 0
        assert wiring.built_sidecar is not None
        assert wiring.handler_kwargs["vault_sidecar"] is wiring.built_sidecar

    @pytest.mark.parametrize("sidecar_enabled", [False], indirect=True)
    def test_forwards_none_when_sidecar_disabled(self, wiring, tmp_path):
        # Arrange
        args = SimpleNamespace(
            input=str(self._write_inputs(tmp_path)),
            session_id="s1",
            user_id="user",
            output=None,
        )

        # Act
        rc = mist_admin.cmd_replay(args)

        # Assert
        assert rc == 0
        assert wiring.handler_kwargs["vault_sidecar"] is None

    @pytest.mark.parametrize("sidecar_enabled", [True], indirect=True)
    def test_closes_sidecar_after_run(self, wiring, tmp_path):
        # Arrange
        args = SimpleNamespace(
            input=str(self._write_inputs(tmp_path)),
            session_id="s1",
            user_id="user",
            output=None,
        )

        # Act
        mist_admin.cmd_replay(args)

        # Assert
        assert wiring.closed_sidecars == [wiring.built_sidecar]
        assert wiring.built_sidecar.closed is True
