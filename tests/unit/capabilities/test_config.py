"""Tests for `load_capabilities_config` over every capability environment variable."""

from __future__ import annotations

import dataclasses

import pytest

from backend.capabilities.config import (
    AlarmSettings,
    CapabilitiesConfig,
    LightsSettings,
    MusicSettings,
    RouterSettings,
    load_capabilities_config,
)
from backend.errors import CapabilityConfigError

FAKE_TOKEN = "fake-ha-token-for-tests"

FULL_ENV = {
    "MIST_CAPABILITIES_ENABLED": "true",
    "MIST_CAPABILITIES": "clock, lights ,music",
    "MIST_WS_ALLOWED_ORIGINS": " http://localhost:1420 , tauri://localhost ",
    "MIST_BACKEND_BIND_ADDR": "127.0.0.1",
    "MIST_TIMEZONE": "Etc/UTC",
    "MIST_CAPABILITIES_DB_PATH": "/tmp/caps.db",
    "MIST_CAPABILITY_POLICY_FILE": "/tmp/policy.yaml",
    "MIST_CAPABILITY_TOOL_TIMEOUT_S": "4.5",
    "MIST_CAPABILITY_PROPOSAL_TTL_S": "30",
    "MIST_CAPABILITY_TOOL_TOKEN_BUDGET": "900",
    "MIST_CAPABILITY_FAST_PATH": "false",
    "MIST_CAPABILITY_DECODE_TIMEOUT_S": "12",
    "MIST_CAPABILITY_WAKE_WORDS": "mist , hey mist",
    "MIST_CAPABILITY_ROUTER_STICKY_TURNS": "3",
    "MIST_CAPABILITY_ROUTER_EMBEDDING_THRESHOLD": "0.42",
    "MIST_ALARM_RING_TIMEOUT_S": "300",
    "MIST_ALARM_SNOOZE_MINUTES": "5",
    "MIST_ALARM_MISSED_GRACE_S": "120",
    "MIST_HA_MCP_URL": "http://ha.example.invalid:8123/api/mcp",
    "MIST_HA_TOKEN": FAKE_TOKEN,
    "MIST_LIGHTS_DEFAULT_TARGET": "living room",
    "MIST_SPOTIFY_CLIENT_ID": "fake-client-id",
    "MIST_SPOTIFY_REDIRECT_URI": "http://127.0.0.1:9999/callback",
    "MIST_SPOTIFY_TOKEN_PATH": "/tmp/spotify_token.json",
    "MIST_SPOTIFY_DEVICE": "Desk speaker",
}


class TestDefaults:
    def test_empty_environment_gives_every_default(self):
        config = load_capabilities_config({})

        assert config == CapabilitiesConfig(
            enabled=False,
            capabilities=("clock", "alarms", "timers", "lights", "music"),
            ws_allowed_origins=(),
            backend_bind_addr="0.0.0.0",
            timezone=None,
            db_path="data/capabilities.db",
            policy_file=None,
            tool_timeout_s=8.0,
            proposal_ttl_s=60.0,
            tool_token_budget=1200,
            fast_path=True,
            decode_timeout_s=10.0,
            wake_words=("mist", "hey mist", "ok mist", "okay mist"),
            router=RouterSettings(sticky_turns=2, embedding_threshold=None),
            alarms=AlarmSettings(ring_timeout_s=600.0, snooze_minutes=9, missed_grace_s=900.0),
            lights=LightsSettings(ha_mcp_url=None, ha_token_configured=False, default_target=None),
            music=MusicSettings(
                spotify_client_id=None,
                spotify_redirect_uri="http://127.0.0.1:8888/callback",
                spotify_token_path="data/secrets/spotify_token.json",
                spotify_device=None,
            ),
        )

    def test_blank_values_count_as_unset(self):
        env = {name: "  " for name in FULL_ENV}

        assert load_capabilities_config(env) == load_capabilities_config({})

    def test_config_is_frozen(self):
        config = load_capabilities_config({})

        with pytest.raises(dataclasses.FrozenInstanceError):
            config.enabled = True  # type: ignore[misc]


class TestEveryVariable:
    def test_full_environment_is_parsed(self):
        config = load_capabilities_config(FULL_ENV)

        assert config == CapabilitiesConfig(
            enabled=True,
            capabilities=("clock", "lights", "music"),
            ws_allowed_origins=("http://localhost:1420", "tauri://localhost"),
            backend_bind_addr="127.0.0.1",
            timezone="Etc/UTC",
            db_path="/tmp/caps.db",
            policy_file="/tmp/policy.yaml",
            tool_timeout_s=4.5,
            proposal_ttl_s=30.0,
            tool_token_budget=900,
            fast_path=False,
            decode_timeout_s=12.0,
            wake_words=("mist", "hey mist"),
            router=RouterSettings(sticky_turns=3, embedding_threshold=0.42),
            alarms=AlarmSettings(ring_timeout_s=300.0, snooze_minutes=5, missed_grace_s=120.0),
            lights=LightsSettings(
                ha_mcp_url="http://ha.example.invalid:8123/api/mcp",
                ha_token_configured=True,
                default_target="living room",
            ),
            music=MusicSettings(
                spotify_client_id="fake-client-id",
                spotify_redirect_uri="http://127.0.0.1:9999/callback",
                spotify_token_path="/tmp/spotify_token.json",
                spotify_device="Desk speaker",
            ),
        )

    def test_token_file_variable_also_counts_as_configured(self):
        config = load_capabilities_config({"MIST_HA_TOKEN_FILE": "/run/secrets/ha_token"})

        assert config.lights.ha_token_configured is True

    def test_config_never_holds_the_token_value(self):
        config = load_capabilities_config(FULL_ENV)

        assert FAKE_TOKEN not in repr(config)


class TestValidation:
    def test_unknown_capability_name_is_refused(self):
        with pytest.raises(CapabilityConfigError, match="weather"):
            load_capabilities_config({"MIST_CAPABILITIES": "clock,weather"})

    def test_duplicate_capability_names_collapse_in_order(self):
        config = load_capabilities_config({"MIST_CAPABILITIES": "music,clock,music"})

        assert config.capabilities == ("music", "clock")

    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            pytest.param("true", True, id="true"),
            pytest.param("TRUE", True, id="upper"),
            pytest.param("1", True, id="one"),
            pytest.param("yes", True, id="yes"),
            pytest.param("on", True, id="on"),
            pytest.param("false", False, id="false"),
            pytest.param("0", False, id="zero"),
            pytest.param("no", False, id="no"),
            pytest.param("off", False, id="off"),
        ],
    )
    def test_fast_path_reads_as_a_boolean(self, raw, expected):
        config = load_capabilities_config({"MIST_CAPABILITY_FAST_PATH": raw})

        assert config.fast_path is expected

    def test_invalid_boolean_is_refused(self):
        with pytest.raises(CapabilityConfigError, match="MIST_CAPABILITIES_ENABLED"):
            load_capabilities_config({"MIST_CAPABILITIES_ENABLED": "maybe"})

    @pytest.mark.parametrize(
        ("name", "raw"),
        [
            pytest.param("MIST_CAPABILITY_TOOL_TIMEOUT_S", "fast", id="timeout-not-a-number"),
            pytest.param("MIST_CAPABILITY_TOOL_TIMEOUT_S", "0", id="timeout-zero"),
            pytest.param("MIST_CAPABILITY_PROPOSAL_TTL_S", "-1", id="ttl-negative"),
            pytest.param("MIST_CAPABILITY_TOOL_TOKEN_BUDGET", "12.5", id="budget-not-an-int"),
            pytest.param("MIST_CAPABILITY_TOOL_TOKEN_BUDGET", "0", id="budget-zero"),
            pytest.param("MIST_CAPABILITY_DECODE_TIMEOUT_S", "inf", id="decode-infinite"),
            pytest.param("MIST_CAPABILITY_ROUTER_STICKY_TURNS", "-1", id="sticky-negative"),
            pytest.param("MIST_CAPABILITY_ROUTER_EMBEDDING_THRESHOLD", "1.5", id="threshold-high"),
            pytest.param("MIST_ALARM_RING_TIMEOUT_S", "0", id="ring-zero"),
            pytest.param("MIST_ALARM_SNOOZE_MINUTES", "61", id="snooze-over-60"),
            pytest.param("MIST_ALARM_MISSED_GRACE_S", "-5", id="grace-negative"),
        ],
    )
    def test_invalid_number_is_refused_naming_the_variable(self, name, raw):
        with pytest.raises(CapabilityConfigError, match=name):
            load_capabilities_config({name: raw})

    def test_sticky_turns_may_be_zero(self):
        config = load_capabilities_config({"MIST_CAPABILITY_ROUTER_STICKY_TURNS": "0"})

        assert config.router.sticky_turns == 0
