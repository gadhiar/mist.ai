"""Tests for the capability wiring in `backend.server`: the lifespan, the probe and the endpoint.

`tests/unit/capabilities/test_factory.py` covers what `build_capability_host` builds. What is
left is the seam: the order `lifespan` builds, starts, attaches and stops the host in, which
object the decode tiers read, the `capabilities` health check, and `GET /capabilities`.

The endpoint tests use `TestClient` without the `with` block, so the ASGI lifespan does not
run; they set the module globals `lifespan` would set. The lifespan test runs the real
`lifespan` with every builder that would reach the model stack, Neo4j or the vault replaced,
as `tests/unit/test_server_health_wiring.py` does.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend import server
from backend.capabilities.config import load_capabilities_config
from backend.capabilities.contracts import CapabilityStatus
from backend.capabilities.power import FixedPower
from backend.capabilities.secrets import Secret, SecretRegistry
from backend.factories import CapabilityBuild
from backend.health import BroadcasterState, HealthProbeSettings
from tests.unit.capabilities.fakes import make_tool_spec

AT = datetime(2026, 1, 5, 7, 0, tzinfo=UTC)
TOKEN = "ha-secret-token-0123456789"  # noqa: S105 - a fake token for the redaction check
TAURI = "http://tauri.localhost"


class FakeCapability:
    """Only what the endpoint reads from a capability."""

    def __init__(self, name: str, tools: tuple[str, ...], provider_tools: tuple[str, ...] = ()):
        self.name = name
        self._tools = tuple(make_tool_spec(tool) for tool in tools)
        self._provider_tools = provider_tools

    def tools(self):
        return self._tools

    def provider_tools(self) -> tuple[str, ...]:
        return self._provider_tools


class FakeHost:
    """Only what the endpoint and the probe read from a host."""

    def __init__(self, *statuses: CapabilityStatus, decode: bool = True) -> None:
        self.current = list(statuses)
        self.decode = decode

    def statuses(self) -> list[CapabilityStatus]:
        return list(self.current)

    def decode_available(self) -> bool:
        return self.decode

    def stats(self) -> dict[str, object]:
        return {"since": AT.isoformat(), "invocations": {"ok": 2}, "status_watcher": "running"}


def _status(name: str, state: str = "available", reason: str | None = None) -> CapabilityStatus:
    return CapabilityStatus(name, state, reason, AT)  # type: ignore[arg-type]


@pytest.fixture
def endpoint(monkeypatch):
    """Set the capability globals the endpoint reads, and return a client and a secret registry."""
    registry = SecretRegistry()
    monkeypatch.setattr(server, "_capability_secrets", registry)
    monkeypatch.setattr(server, "_capability_build", None)
    monkeypatch.setattr(server, "_capability_config", None)
    monkeypatch.setattr(server, "_capability_power", FixedPower("low"))
    monkeypatch.delenv("MIST_ALLOWED_ORIGINS", raising=False)

    def install(build: CapabilityBuild, env: dict[str, str] | None = None) -> None:
        monkeypatch.setattr(server, "_capability_build", build)
        if env is not None:
            monkeypatch.setattr(server, "_capability_config", load_capabilities_config(env))

    return SimpleNamespace(client=TestClient(server.app), registry=registry, install=install)


def _lights_build(reason: str | None = None) -> CapabilityBuild:
    lights = FakeCapability(
        "lights",
        ("lights.power", "lights.adjust", "lights.status"),
        provider_tools=("GetLiveContext", "HassTurnOff", "HassTurnOn"),
    )
    clock = FakeCapability("clock", ("clock.now",))
    host = FakeHost(
        _status("clock"),
        _status("lights", "unavailable" if reason else "available", reason),
        decode=True,
    )
    return CapabilityBuild(host=host, disabled_reason=None, capabilities=(clock, lights))  # type: ignore[arg-type]


# -- GET /capabilities ---------------------------------------------------------------------------


class TestEndpointShape:
    def test_an_enabled_build_answers_the_documented_shape(self, endpoint, tmp_path):
        token_file = tmp_path / "spotify_token.json"
        token_file.write_text("{}")
        endpoint.install(
            _lights_build(),
            {
                "MIST_CAPABILITIES_ENABLED": "true",
                "MIST_CAPABILITIES": "clock,lights,music",
                "MIST_HA_MCP_URL": "http://homeassistant.local:8123/api/mcp",
                "MIST_HA_TOKEN": TOKEN,
                "MIST_SPOTIFY_CLIENT_ID": "spotify-client-id",
                "MIST_SPOTIFY_TOKEN_PATH": str(token_file),
            },
        )

        response = endpoint.client.get("/capabilities")

        assert response.status_code == 200
        body = response.json()
        assert body == {
            "enabled": True,
            "reason": None,
            "power_level": "low",
            "decode_available": True,
            "capabilities": [
                {
                    "name": "clock",
                    "state": "available",
                    "reason": None,
                    "tools": ["clock.now"],
                    "provider_tools": [],
                },
                {
                    "name": "lights",
                    "state": "available",
                    "reason": None,
                    "tools": ["lights.power", "lights.adjust", "lights.status"],
                    "provider_tools": ["GetLiveContext", "HassTurnOff", "HassTurnOn"],
                },
            ],
            "secrets": [
                {"name": "MIST_HA_TOKEN", "present": True},
                {"name": "MIST_SPOTIFY_TOKEN_PATH", "present": True},
            ],
            "stats": {
                "since": AT.isoformat(),
                "invocations": {"ok": 2},
                "status_watcher": "running",
            },
        }

    def test_a_disabled_build_answers_enabled_false_with_its_reason(self, endpoint):
        endpoint.install(
            CapabilityBuild(host=None, disabled_reason="MIST_ALLOWED_ORIGINS allows every origin"),
            {"MIST_CAPABILITIES_ENABLED": "false"},
        )

        body = endpoint.client.get("/capabilities").json()

        assert body == {
            "enabled": False,
            "reason": "MIST_ALLOWED_ORIGINS allows every origin",
            "power_level": "low",
            "decode_available": False,
            "capabilities": [],
            "secrets": [],
            "stats": {},
        }

    def test_a_missing_spotify_token_reads_absent(self, endpoint, tmp_path):
        endpoint.install(
            CapabilityBuild(host=None, disabled_reason="capabilities disabled"),
            {
                "MIST_CAPABILITIES": "music",
                "MIST_SPOTIFY_CLIENT_ID": "spotify-client-id",
                "MIST_SPOTIFY_TOKEN_PATH": str(tmp_path / "absent.json"),
            },
        )

        body = endpoint.client.get("/capabilities").json()

        assert body["secrets"] == [{"name": "MIST_SPOTIFY_TOKEN_PATH", "present": False}]


class TestEndpointRedaction:
    def test_no_secret_value_reaches_the_body(self, endpoint):
        endpoint.registry.register(Secret(TOKEN))
        endpoint.install(
            _lights_build(reason=f"auth rejected for Bearer {TOKEN}"),
            {
                "MIST_CAPABILITIES_ENABLED": "true",
                "MIST_CAPABILITIES": "clock,lights",
                "MIST_HA_MCP_URL": "http://homeassistant.local:8123/api/mcp",
                "MIST_HA_TOKEN": TOKEN,
            },
        )

        response = endpoint.client.get("/capabilities")

        assert TOKEN not in response.text
        lights = response.json()["capabilities"][1]
        assert lights["reason"] == "auth rejected for Bearer ***"

    def test_a_disabled_reason_is_redacted(self, endpoint):
        endpoint.registry.register(Secret(TOKEN))
        endpoint.install(CapabilityBuild(host=None, disabled_reason=f"bad value {TOKEN}"))

        response = endpoint.client.get("/capabilities")

        assert TOKEN not in response.text
        assert response.json()["reason"] == "bad value ***"

    def test_each_reason_is_capped_at_200_characters(self, endpoint):
        endpoint.install(_lights_build(reason="x" * 500))

        lights = endpoint.client.get("/capabilities").json()["capabilities"][1]
        endpoint.install(CapabilityBuild(host=None, disabled_reason="y" * 500))
        disabled = endpoint.client.get("/capabilities").json()

        assert lights["reason"] == "x" * 200
        assert disabled["reason"] == "y" * 200


class TestEndpointOrigin:
    def test_an_origin_the_allowlist_refuses_gets_403(self, endpoint, monkeypatch):
        monkeypatch.setenv("MIST_ALLOWED_ORIGINS", TAURI)
        endpoint.install(_lights_build())

        response = endpoint.client.get("/capabilities", headers={"Origin": "https://evil.example"})

        assert response.status_code == 403
        assert "lights" not in response.text

    @pytest.mark.parametrize("headers", [{"Origin": TAURI}, {"Origin": TAURI.upper()}, {}])
    def test_a_listed_origin_or_none_gets_the_body(self, endpoint, monkeypatch, headers):
        monkeypatch.setenv("MIST_ALLOWED_ORIGINS", TAURI)
        endpoint.install(_lights_build())

        response = endpoint.client.get("/capabilities", headers=headers)

        assert response.status_code == 200
        assert response.json()["enabled"] is True

    def test_the_default_allowlist_refuses_another_site(self, endpoint):
        endpoint.install(_lights_build())

        response = endpoint.client.get("/capabilities", headers={"Origin": "https://evil.example"})

        assert response.status_code == 403


# -- Decode tiers, desktop presence and the health check ------------------------------------------


class TestPorts:
    def test_the_decode_is_unavailable_until_the_provider_exists(self, monkeypatch):
        monkeypatch.setattr(server, "voice_processor", SimpleNamespace())
        tiers = server._capability_tiers(FixedPower("full"))

        before = tiers.provider("interaction")
        provider = object()
        server.voice_processor._llm_provider = provider

        assert before is None
        assert tiers.provider("interaction") is provider
        assert tiers.provider("smart") is provider

    def test_no_tier_serves_at_off(self, monkeypatch):
        monkeypatch.setattr(server, "voice_processor", SimpleNamespace(_llm_provider=object()))

        assert server._capability_tiers(FixedPower("off")).provider("interaction") is None

    def test_the_desktop_is_present_while_a_connection_is_open(self, monkeypatch):
        connections: set[object] = set()
        monkeypatch.setattr(server, "active_connections", connections)

        assert server._desktop_present() is False
        connections.add(object())
        assert server._desktop_present() is True


@pytest.fixture
def executor():
    pool = ThreadPoolExecutor(max_workers=1)
    yield pool
    pool.shutdown(wait=False)


class TestHealthCheck:
    def _registry(self, executor, capability_host):
        return server._build_health_registry(
            executor,
            BroadcasterState(),
            SimpleNamespace(),  # type: ignore[arg-type]  # called only with a dispatcher
            HealthProbeSettings(),
            capability_host=capability_host,
        )

    def test_without_a_host_the_check_is_absent(self, executor):
        assert "capabilities" not in self._registry(executor, None).snapshot()["checks"]

    @pytest.mark.asyncio
    async def test_the_check_reads_the_host_at_each_run(self, executor, monkeypatch):
        monkeypatch.setattr(server, "voice_processor", None)
        monkeypatch.setattr(server, "extraction_dispatcher", None)
        monkeypatch.setattr(server, "_capability_build", None)
        registry = self._registry(executor, server._capability_host)
        check = registry.snapshot()["checks"]["capabilities"]
        assert (check["severity"], check["restart_repairs"]) == ("degrades", False)

        host = FakeHost(_status("clock"), _status("alarms", "unavailable", "MIST_TIMEZONE unset"))
        monkeypatch.setattr(server, "_capability_build", CapabilityBuild(host, None))  # type: ignore[arg-type]
        await registry.run_once()
        down = registry.snapshot()["checks"]["capabilities"]
        host.current = [_status("clock"), _status("alarms")]
        await registry.run_once()
        up = registry.snapshot()["checks"]["capabilities"]

        assert (down["status"], down["reason"]) == ("down", "unreachable")
        assert up["status"] == "up"


# -- The lifespan --------------------------------------------------------------------------------


class FakeHandler:
    """The handler attributes `lifespan` reads, and a recording `attach_capability_host`."""

    def __init__(self, events: list[str]) -> None:
        self.event_store = None
        self._tool_usage_tracker = None
        self._provider = None
        self._events = events
        self.attached: tuple[object, object] | None = None

    def attach_capability_host(self, host, *, tiers=None) -> None:
        self._events.append("attach")
        self.attached = (host, tiers)


class FakeModels:
    def __init__(self, handler: FakeHandler) -> None:
        self.knowledge = SimpleNamespace(conversation_handler=handler)

    def shutdown(self) -> None:
        return None


class FakeScheduler:
    async def start(self) -> bool:
        return False

    async def stop(self) -> None:
        return None


@pytest.fixture
def lifespan_rig(monkeypatch, tmp_path):
    """Run the real `lifespan` with capabilities enabled and nothing else behind it."""
    for name in (
        "voice_processor",
        "curation_scheduler",
        "log_handler",
        "extraction_dispatcher",
        "vault_writer",
        "vault_sidecar",
        "vault_filewatcher",
        "vault_invalidation_bus",
        "_broadcaster_state",
        "_health_registry",
        "_health_probe_executor",
        "_extraction_health_client",
        "_extraction_dispatcher_start_failed",
        "_gpu_yield",
        "_models_ready",
        "_capability_build",
        "_capability_config",
        "_capability_power",
    ):
        monkeypatch.setattr(server, name, getattr(server, name))
    events: list[str] = []
    rig = SimpleNamespace(events=events, processors=[], handler=FakeHandler(events), builds=[])

    class FakeVoiceProcessor:
        def __init__(self, config, message_queue, **kwargs) -> None:
            host = server._capability_host()
            events.append("voice_processor")
            self.host_watcher_at_construction = host.stats()["status_watcher"] if host else None
            self.models = None
            self.decode_during_initialize: bool | None = None
            rig.processors.append(self)

        async def initialize(self, *, yielded: bool = False) -> None:
            host = server._capability_host()
            self.decode_during_initialize = host.decode_available() if host else None
            self._llm_provider = object()
            self.models = FakeModels(rig.handler)

    real_build = server.build_capability_host

    def recording_build(config, **kwargs):
        events.append("build")
        build = real_build(config, **kwargs)
        rig.builds.append(build)
        return build

    real_stop = server._stop_capability_host

    async def recording_stop():
        events.append("stop")
        await real_stop()

    monkeypatch.setattr(server, "install_log_redaction", lambda registry: events.append("redact"))
    monkeypatch.setattr(server, "build_capability_host", recording_build)
    monkeypatch.setattr(server, "_stop_capability_host", recording_stop)
    monkeypatch.setattr(server, "VoiceProcessor", FakeVoiceProcessor)
    monkeypatch.setattr(server, "message_queue", asyncio.Queue())
    monkeypatch.setattr(server, "configure_file_logging", lambda log_dir: None)
    monkeypatch.setattr(server, "build_vault_writer", lambda *a, **k: None)
    monkeypatch.setattr(server, "build_sidecar_index", lambda *a, **k: None)
    monkeypatch.setattr(server, "build_phase3_components", lambda *a, **k: None)
    monkeypatch.setattr(server, "build_curation_scheduler", lambda config, **k: FakeScheduler())
    system_status = server.config.system_status.model_copy(update={"enabled": False})
    monkeypatch.setattr(
        server,
        "config",
        server.config.model_copy(update={"tts_enabled": False, "system_status": system_status}),
    )
    for name in ("MIST_OPS_DIR", "MIST_BACKUP_STATUS_PATH", "MIST_ALLOWED_ORIGINS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("MIST_EVAL_ISOLATION", "0")
    monkeypatch.setenv("MIST_CAPABILITIES_ENABLED", "true")
    monkeypatch.setenv("MIST_CAPABILITIES", "clock,timers")
    monkeypatch.setenv("MIST_CAPABILITIES_DB_PATH", str(tmp_path / "capabilities.db"))
    return rig


class TestLifespan:
    @pytest.mark.asyncio
    async def test_the_host_is_built_started_attached_probed_and_stopped_in_order(
        self, lifespan_rig
    ):
        async with server.lifespan(server.app):
            host = server._capability_host()
            processor = lifespan_rig.processors[0]
            decode_after_initialize = host.decode_available()
            checks = server._health_registry.snapshot()["checks"]
            body = server._capabilities_payload()
            lifespan_rig.events.append("serving")

        (build,) = lifespan_rig.builds
        assert host is build.host
        assert lifespan_rig.events == [
            "redact",
            "build",
            "voice_processor",
            "attach",
            "serving",
            "stop",
        ]
        assert processor.host_watcher_at_construction == "running"
        assert processor.decode_during_initialize is False
        assert decode_after_initialize is True
        attached_host, attached_tiers = lifespan_rig.handler.attached
        assert attached_host is host
        assert attached_tiers.provider("interaction") is processor._llm_provider
        assert checks["capabilities"]["restart_repairs"] is False
        assert body["enabled"] is True
        assert [c["name"] for c in body["capabilities"]] == ["clock", "timers"]
        assert host.stats()["status_watcher"] == "stopped"
        json.dumps(body)

    @pytest.mark.asyncio
    async def test_a_disabled_config_builds_no_host_and_registers_no_check(
        self, lifespan_rig, monkeypatch
    ):
        monkeypatch.setenv("MIST_CAPABILITIES_ENABLED", "false")

        async with server.lifespan(server.app):
            checks = server._health_registry.snapshot()["checks"]
            body = server._capabilities_payload()

        assert server._capability_host() is None
        assert "capabilities" not in checks
        assert lifespan_rig.handler.attached is None
        assert (body["enabled"], body["reason"]) == (False, "capabilities disabled")

    @pytest.mark.asyncio
    async def test_an_invalid_config_builds_no_host_and_says_why(self, lifespan_rig, monkeypatch):
        monkeypatch.setenv("MIST_CAPABILITIES", "clock,teleport")

        async with server.lifespan(server.app):
            body = server._capabilities_payload()

        assert lifespan_rig.builds == []
        assert body["enabled"] is False
        assert body["reason"].startswith("capability config invalid: ")
        assert "teleport" in body["reason"]
