"""Tests for backend.memory.config.MemoryConfig.

Most zone tests pass a fake zone lookup, so they need no tz database; the production lookup
test uses the image's tzdata. The resolver's own cases are in tests/unit/test_user_timezone.py.
"""

import re
import zoneinfo
from datetime import UTC, timedelta, timezone, tzinfo
from pathlib import Path

import pytest

import backend.user_timezone as user_timezone
from backend.errors import MemoryConfigError, MistError
from backend.memory.config import (
    MEMORY_ENV_VARS,
    RETIRED_MEMORY_ENV_VARS,
    MemoryConfig,
    NeedEmbedderName,
)
from backend.memory.models import ServingProfile
from backend.memory.need_embedders import NEED_EMBEDDERS
from backend.memory.retrieval_view import DEFAULT_FACT_VIEW, FACT_VIEWS

_REPO_ROOT = Path(__file__).resolve().parents[3]
_NEED_VARIABLES = (
    "MIST_MEMORY_NEED_GATE",
    "MIST_MEMORY_NEED_EMBEDDER",
    "MIST_MEMORY_NEED_CONTEXT",
)


def _documented(text: str) -> dict[str, str]:
    """Each `# NAME=value` or `NAME=value` line of a memory variable, first value per name."""
    documented: dict[str, str] = {}
    pattern = r"^#?\s*(MIST_MEMORY_[A-Z_]+|MIST_TIMEZONE)=(\S*)"
    for match in re.finditer(pattern, text, re.MULTILINE):
        documented.setdefault(match.group(1), match.group(2))
    return documented


@pytest.fixture(autouse=True)
def _clear_memory_env(monkeypatch):
    """Start every test with no memory variable set and the unset-zone warning unsent.

    The warning is once per process, so an earlier test that built a config would otherwise
    have spent it; monkeypatch restores the flag afterwards, so nothing leaks between tests.
    """
    for name in (*MEMORY_ENV_VARS, *RETIRED_MEMORY_ENV_VARS):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(user_timezone, "_unset_warned", False)


def _never_called(name: str) -> tzinfo:
    raise AssertionError(f"the resolver must not be called, got {name!r}")


def _rejecting(name: str) -> tzinfo:
    raise zoneinfo.ZoneInfoNotFoundError(f"No time zone found with key {name}")


class _RecordingResolver:
    def __init__(self) -> None:
        self.names: list[str] = []
        self.zone = timezone(timedelta(hours=-5), "TEST-5")

    def __call__(self, name: str) -> tzinfo:
        self.names.append(name)
        return self.zone


class TestDefaults:
    def test_defaults_match_the_design_doc(self):
        config = MemoryConfig()

        assert config.enabled is True
        assert config.block_tokens == 500
        assert config.continuity_share == 0.35
        assert config.select_k == 8
        assert config.facets is True
        assert config.deadline_ms == 50
        assert config.rerank_backend == "local"
        assert config.rerank_model == "/opt/models/ms-marco-MiniLM-L-6-v2"
        assert config.rerank_top_n == 20
        assert config.rerank_device == "cpu"
        assert config.rerank_url is None
        assert config.rerank_remote_model is None
        assert config.rerank_timeout_ms == 40
        assert config.rerank_remote_scores == "logit"
        assert config.index_path is None
        assert config.refresh_debounce_s == 30
        assert config.block_tokens_low == 0
        assert config.list_max_items == 25
        assert config.rescan_interval_s == 300
        assert config.tz is UTC
        assert config.tz_name is None
        assert config.fact_view == "F3"
        assert config.need_gate == "off"
        assert config.need_embedder == "shared"
        assert config.need_context is True

    def test_the_score_threshold_and_weights_are_gone(self):
        """Selection is by rank (plan v7 A37), so no item score is thresholded or weighted."""
        config = MemoryConfig()

        for name in ("min_score", "rerank_weight", "recency_weight", "confidence_weight"):
            assert not hasattr(config, name), name
        assert {"MIST_MEMORY_MIN_SCORE", "MIST_MEMORY_WEIGHTS"}.isdisjoint(MEMORY_ENV_VARS)

    def test_there_is_no_neighbour_setting(self):
        """The gate model's form carries its own neighbour count; no setting can drift from it."""
        assert not hasattr(MemoryConfig(), "need_neighbours")
        assert "MIST_MEMORY_NEED_NEIGHBOURS" not in MEMORY_ENV_VARS

    def test_the_fact_view_default_is_the_index_default(self):
        # Two literals, kept equal: the index's own default and the setting's.
        assert MemoryConfig().fact_view == DEFAULT_FACT_VIEW

    def test_is_frozen(self):
        config = MemoryConfig()
        with pytest.raises(AttributeError):
            config.block_tokens = 1  # type: ignore[misc]

    def test_from_env_with_nothing_set_equals_the_defaults(self):
        assert MemoryConfig.from_env(tz_resolver=_never_called) == MemoryConfig()

    @pytest.mark.parametrize("raw", [None, "", "  "])
    def test_the_need_gate_is_off_when_unset_or_blank(self, monkeypatch, raw):
        """In shadow the gate put fetches past their deadline at H4: by default none is built."""
        if raw is not None:
            monkeypatch.setenv("MIST_MEMORY_NEED_GATE", raw)

        assert MemoryConfig.from_env(tz_resolver=_never_called).need_gate == "off"


class TestFromEnv:
    def test_reads_every_scalar_variable(self, monkeypatch):
        values = {
            "MIST_MEMORY_ENABLED": "false",
            "MIST_MEMORY_BLOCK_TOKENS": "300",
            "MIST_MEMORY_CONTINUITY_SHARE": "0.5",
            "MIST_MEMORY_SELECT_K": "5",
            "MIST_MEMORY_DEADLINE_MS": "80",
            "MIST_MEMORY_RERANK_BACKEND": "remote",
            "MIST_MEMORY_RERANK_MODEL": "/opt/models/ms-marco-MiniLM-L-12-v2",
            "MIST_MEMORY_RERANK_TOP_N": "12",
            "MIST_MEMORY_RERANK_DEVICE": "cuda",
            "MIST_MEMORY_RERANK_URL": "http://rerank.lan:8080",
            "MIST_MEMORY_RERANK_REMOTE_MODEL": "bge-reranker",
            "MIST_MEMORY_RERANK_TIMEOUT_MS": "25",
            "MIST_MEMORY_RERANK_REMOTE_SCORES": "probability",
            "MIST_MEMORY_INDEX_PATH": ":memory:",
            "MIST_MEMORY_REFRESH_DEBOUNCE_S": "5",
            "MIST_MEMORY_BLOCK_TOKENS_LOW": "120",
            "MIST_MEMORY_LIST_MAX_ITEMS": "6",
            "MIST_MEMORY_FACETS": "off",
            "MIST_MEMORY_RESCAN_INTERVAL_S": "45",
            "MIST_MEMORY_FACT_VIEW": "F2",
            "MIST_MEMORY_NEED_GATE": "on",
            "MIST_MEMORY_NEED_EMBEDDER": "bge-small-en-v1.5",
            "MIST_MEMORY_NEED_CONTEXT": "false",
        }
        for name, value in values.items():
            monkeypatch.setenv(name, value)

        config = MemoryConfig.from_env(tz_resolver=_never_called)

        assert config.enabled is False
        assert config.block_tokens == 300
        assert config.continuity_share == 0.5
        assert config.select_k == 5
        assert config.deadline_ms == 80
        assert config.rerank_backend == "remote"
        assert config.rerank_model == "/opt/models/ms-marco-MiniLM-L-12-v2"
        assert config.rerank_top_n == 12
        assert config.rerank_device == "cuda"
        assert config.rerank_url == "http://rerank.lan:8080"
        assert config.rerank_remote_model == "bge-reranker"
        assert config.rerank_timeout_ms == 25
        assert config.rerank_remote_scores == "probability"
        assert config.index_path == ":memory:"
        assert config.refresh_debounce_s == 5
        assert config.block_tokens_low == 120
        assert config.list_max_items == 6
        assert config.facets is False
        assert config.rescan_interval_s == 45
        assert config.fact_view == "F2"
        assert config.need_gate == "on"
        assert config.need_embedder == "bge-small-en-v1.5"
        assert config.need_context is False

    @pytest.mark.parametrize("mode", ["on", "shadow", "off"])
    def test_every_need_gate_mode_is_read(self, monkeypatch, mode):
        monkeypatch.setenv("MIST_MEMORY_NEED_GATE", mode)

        assert MemoryConfig.from_env(tz_resolver=_never_called).need_gate == mode

    @pytest.mark.parametrize("name", sorted(NEED_EMBEDDERS))
    def test_every_need_embedder_name_is_read(self, monkeypatch, name):
        monkeypatch.setenv("MIST_MEMORY_NEED_EMBEDDER", name)

        assert MemoryConfig.from_env(tz_resolver=_never_called).need_embedder == name

    def test_a_blank_variable_counts_as_unset(self, monkeypatch):
        """Compose passes `${VAR:-}` through as an empty string, which must mean the default."""
        monkeypatch.setenv("MIST_MEMORY_BLOCK_TOKENS", "")
        monkeypatch.setenv("MIST_MEMORY_RERANK_URL", "  ")

        config = MemoryConfig.from_env(tz_resolver=_never_called)

        assert config.block_tokens == 500
        assert config.rerank_url is None

    @pytest.mark.parametrize("raw", ["true", "TRUE", "1", "yes", "on"])
    def test_enabled_accepts_true_spellings(self, monkeypatch, raw):
        monkeypatch.setenv("MIST_MEMORY_ENABLED", raw)
        assert MemoryConfig.from_env(tz_resolver=_never_called).enabled is True

    @pytest.mark.parametrize("raw", ["false", "0", "no", "off"])
    def test_enabled_accepts_false_spellings(self, monkeypatch, raw):
        monkeypatch.setenv("MIST_MEMORY_ENABLED", raw)
        assert MemoryConfig.from_env(tz_resolver=_never_called).enabled is False

    @pytest.mark.parametrize(
        ("name", "raw"),
        [
            ("MIST_MEMORY_ENABLED", "maybe"),
            ("MIST_MEMORY_BLOCK_TOKENS", "five hundred"),
            ("MIST_MEMORY_CONTINUITY_SHARE", "a third"),
            ("MIST_MEMORY_SELECT_K", "eight"),
            ("MIST_MEMORY_FACETS", "some"),
            ("MIST_MEMORY_NEED_CONTEXT", "sometimes"),
        ],
    )
    def test_an_unparsable_value_names_its_variable(self, monkeypatch, name, raw):
        monkeypatch.setenv(name, raw)

        with pytest.raises(MemoryConfigError, match=name):
            MemoryConfig.from_env(tz_resolver=_never_called)


class TestZone:
    def test_unset_zone_is_utc_and_never_calls_the_lookup(self, caplog):
        """Unset needs no zone database: the resolver returns `datetime.timezone.utc`."""
        with caplog.at_level("WARNING", logger=user_timezone.__name__):
            config = MemoryConfig.from_env(tz_resolver=_never_called)

        assert config.tz is UTC
        assert config.tz_name is None
        assert any("MIST_TIMEZONE" in record.getMessage() for record in caplog.records)

    def test_unset_zone_warns_once_per_process(self, caplog):
        """Startup builds the config several times and health probes can rebuild it; warn once."""
        with caplog.at_level("WARNING", logger=user_timezone.__name__):
            configs = [MemoryConfig.from_env(tz_resolver=_never_called) for _ in range(3)]

        warnings = [r for r in caplog.records if "MIST_TIMEZONE" in r.getMessage()]
        assert len(warnings) == 1
        assert all(config.tz is UTC for config in configs)

    def test_blank_zone_counts_as_unset(self, monkeypatch):
        monkeypatch.setenv("MIST_TIMEZONE", "")

        config = MemoryConfig.from_env(tz_resolver=_never_called)

        assert config.tz is UTC
        assert config.tz_name is None

    def test_set_zone_resolves_once_at_config_time(self, monkeypatch):
        monkeypatch.setenv("MIST_TIMEZONE", "America/Chicago")
        resolver = _RecordingResolver()

        config = MemoryConfig.from_env(tz_resolver=resolver)

        assert resolver.names == ["America/Chicago"]
        assert config.tz is resolver.zone
        assert config.tz_name == "America/Chicago"

    def test_a_rejected_name_raises_memory_config_error_naming_the_variable(self, monkeypatch):
        monkeypatch.setenv("MIST_TIMEZONE", "Mars/Olympus_Mons")

        with pytest.raises(MemoryConfigError, match="MIST_TIMEZONE") as excinfo:
            MemoryConfig.from_env(tz_resolver=_rejecting)

        assert "Mars/Olympus_Mons" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, ValueError)
        assert isinstance(excinfo.value.__cause__.__cause__, zoneinfo.ZoneInfoNotFoundError)

    def test_a_malformed_name_raises_memory_config_error(self, monkeypatch):
        """A key like an absolute path makes zoneinfo raise ValueError, not a not-found error."""

        def malformed(name: str) -> tzinfo:
            raise ValueError(f"ZoneInfo keys must be normalized relative paths, got: {name}")

        monkeypatch.setenv("MIST_TIMEZONE", "/etc/localtime")

        with pytest.raises(MemoryConfigError, match="MIST_TIMEZONE"):
            MemoryConfig.from_env(tz_resolver=malformed)

    def test_a_missing_zone_database_gives_utc_without_a_name(self, monkeypatch):
        monkeypatch.setenv("MIST_TIMEZONE", "America/Chicago")
        monkeypatch.setattr(user_timezone, "available_timezones", set)
        monkeypatch.setattr(user_timezone, "_missing_database_warned", set())

        config = MemoryConfig.from_env(tz_resolver=_rejecting)

        assert config.tz is UTC
        assert config.tz_name is None

    def test_production_lookup_resolves_an_iana_name(self, monkeypatch):
        monkeypatch.setenv("MIST_TIMEZONE", "America/Chicago")

        config = MemoryConfig.from_env()

        assert config.tz == zoneinfo.ZoneInfo("America/Chicago")
        assert config.tz_name == "America/Chicago"

    def test_production_lookup_rejects_an_unknown_name(self, monkeypatch):
        monkeypatch.setenv("MIST_TIMEZONE", "Mars/Olympus_Mons")

        with pytest.raises(MemoryConfigError, match="Mars/Olympus_Mons"):
            MemoryConfig.from_env()


class TestBlockTokensFor:
    def test_full_gets_the_full_budget(self):
        config = MemoryConfig(block_tokens=500, block_tokens_low=120)

        assert config.block_tokens_for(ServingProfile(level="full", tier="top")) == 500

    def test_low_gets_the_low_budget(self):
        config = MemoryConfig(block_tokens=500, block_tokens_low=120)

        assert config.block_tokens_for(ServingProfile(level="low", tier="interaction")) == 120

    @pytest.mark.parametrize("tier", ["top", "interaction"])
    def test_off_gets_nothing(self, tier):
        config = MemoryConfig(block_tokens=500, block_tokens_low=120)

        assert config.block_tokens_for(ServingProfile(level="off", tier=tier)) == 0

    def test_the_default_low_budget_is_zero(self):
        assert MemoryConfig().block_tokens_for(ServingProfile(level="low", tier="top")) == 0

    def test_the_tier_does_not_change_the_budget(self):
        config = MemoryConfig(block_tokens=400)

        top = config.block_tokens_for(ServingProfile(level="full", tier="top"))
        interaction = config.block_tokens_for(ServingProfile(level="full", tier="interaction"))

        assert top == interaction == 400


class TestRetiredVariables:
    @pytest.mark.parametrize(
        ("name", "value", "replacement"),
        [
            ("MIST_MEMORY_BLOCK_ROLE", "system_tail", "MIST_CTX_BUDGET_DELTA_PLACEMENT"),
            ("MIST_MEMORY_TZ", "Europe/London", "MIST_TIMEZONE"),
            ("MIST_MEMORY_MIN_SCORE", "0.35", "MIST_MEMORY_SELECT_K"),
            ("MIST_MEMORY_WEIGHTS", "0.8,0.1,0.1", "MIST_MEMORY_SELECT_K"),
        ],
    )
    def test_a_set_retired_variable_fails_naming_its_replacement(
        self, monkeypatch, name, value, replacement
    ):
        monkeypatch.setenv(name, value)

        with pytest.raises(MemoryConfigError, match=name) as excinfo:
            MemoryConfig.from_env(tz_resolver=_never_called)

        assert replacement in str(excinfo.value)

    @pytest.mark.parametrize(
        "name",
        [
            "MIST_MEMORY_BLOCK_ROLE",
            "MIST_MEMORY_TZ",
            "MIST_MEMORY_MIN_SCORE",
            "MIST_MEMORY_WEIGHTS",
        ],
    )
    def test_a_blank_retired_variable_counts_as_unset(self, monkeypatch, name):
        """Compose forwards `${VAR:-}` as an empty string, which must not fail startup."""
        monkeypatch.setenv(name, "  ")

        assert MemoryConfig.from_env(tz_resolver=_never_called) == MemoryConfig()

    def test_retired_variables_leave_the_memory_variables(self):
        assert set(RETIRED_MEMORY_ENV_VARS).isdisjoint(MEMORY_ENV_VARS)

    def test_the_shared_zone_variable_joins_the_memory_variables(self):
        assert "MIST_TIMEZONE" in MEMORY_ENV_VARS

    def test_the_config_has_no_block_role(self):
        assert not hasattr(MemoryConfig(), "block_role")


class TestValidation:
    @pytest.mark.parametrize(
        ("overrides", "fragment"),
        [
            ({"block_tokens": 0}, "block_tokens"),
            ({"block_tokens": -1}, "block_tokens"),
            ({"continuity_share": -0.01}, "continuity_share"),
            ({"continuity_share": 1.01}, "continuity_share"),
            ({"select_k": 0}, "select_k"),
            ({"select_k": 26, "rerank_top_n": 30}, "select_k"),
            ({"select_k": 21}, "select_k"),
            ({"select_k": 10, "rerank_top_n": 9}, "select_k"),
            ({"facets": "on"}, "facets"),
            ({"deadline_ms": 0}, "deadline_ms"),
            ({"rerank_top_n": 0}, "rerank_top_n"),
            ({"rerank_device": "mps"}, "rerank_device"),
            ({"rerank_backend": "gpu"}, "rerank_backend"),
            ({"rerank_remote_scores": "raw"}, "rerank_remote_scores"),
            ({"rerank_backend": "remote"}, "rerank_url"),
            (
                {"rerank_backend": "remote", "rerank_url": "http://r:1", "rerank_timeout_ms": 0},
                "rerank_timeout_ms",
            ),
            ({"refresh_debounce_s": -1}, "refresh_debounce_s"),
            ({"block_tokens_low": -1}, "block_tokens_low"),
            ({"block_tokens_low": 301}, "block_tokens_low"),
            ({"list_max_items": 1}, "list_max_items"),
            ({"list_max_items": 26}, "list_max_items"),
            ({"rescan_interval_s": 29}, "rescan_interval_s"),
            ({"fact_view": "F4"}, "fact_view"),
            ({"fact_view": "f3"}, "fact_view"),
            ({"fact_view": "N0"}, "fact_view"),
            ({"need_gate": "auto"}, "need_gate"),
            ({"need_gate": "ON"}, "need_gate"),
            ({"need_gate": True}, "need_gate"),
            ({"need_embedder": "all-MiniLM-L6-v2"}, "need_embedder"),
            ({"need_embedder": ""}, "need_embedder"),
            ({"need_context": "yes"}, "need_context"),
        ],
    )
    def test_rejects_invalid_values(self, overrides, fragment):
        with pytest.raises(MemoryConfigError, match=fragment):
            MemoryConfig(**overrides)

    @pytest.mark.parametrize(
        ("overrides", "field"),
        [
            ({"block_tokens_low": 0}, "block_tokens_low"),
            ({"block_tokens_low": 300}, "block_tokens_low"),
            ({"list_max_items": 2}, "list_max_items"),
            ({"list_max_items": 25}, "list_max_items"),
            ({"rescan_interval_s": 30}, "rescan_interval_s"),
            ({"select_k": 1}, "select_k"),
            ({"select_k": 25, "rerank_top_n": 25}, "select_k"),
            ({"select_k": 9, "rerank_top_n": 9}, "select_k"),
        ],
    )
    def test_range_bounds_are_inclusive(self, overrides, field):
        assert getattr(MemoryConfig(**overrides), field) == overrides[field]

    @pytest.mark.parametrize(
        ("name", "raw"),
        [
            ("MIST_MEMORY_BLOCK_TOKENS_LOW", "301"),
            ("MIST_MEMORY_LIST_MAX_ITEMS", "30"),
            ("MIST_MEMORY_RESCAN_INTERVAL_S", "10"),
            ("MIST_MEMORY_LIST_MAX_ITEMS", "ten"),
            ("MIST_MEMORY_SELECT_K", "0"),
            ("MIST_MEMORY_SELECT_K", "21"),
        ],
    )
    def test_from_env_rejects_out_of_range_variables(self, monkeypatch, name, raw):
        monkeypatch.setenv(name, raw)

        with pytest.raises(MemoryConfigError):
            MemoryConfig.from_env(tz_resolver=_never_called)

    def test_share_bounds_are_inclusive(self):
        assert MemoryConfig(continuity_share=0.0).continuity_share == 0.0
        assert MemoryConfig(continuity_share=1.0).continuity_share == 1.0

    @pytest.mark.parametrize("raw", ["on", "true", "off", "false"])
    def test_facets_accept_on_and_off(self, monkeypatch, raw):
        monkeypatch.setenv("MIST_MEMORY_FACETS", raw)

        expected = raw in ("on", "true")
        assert MemoryConfig.from_env(tz_resolver=_never_called).facets is expected

    @pytest.mark.parametrize("name", ["min_score", "rerank_weight"])
    def test_a_retired_field_is_refused_by_the_constructor(self, name):
        with pytest.raises(TypeError, match=name):
            MemoryConfig(**{name: 0.5})

    def test_remote_with_url_and_timeout_is_valid(self):
        config = MemoryConfig(rerank_backend="remote", rerank_url="http://rerank.lan:8080")
        assert config.rerank_timeout_ms == 40

    def test_local_and_off_need_no_url(self):
        assert MemoryConfig(rerank_backend="off").rerank_url is None
        assert MemoryConfig(rerank_backend="local").rerank_url is None

    def test_validation_runs_on_from_env(self, monkeypatch):
        monkeypatch.setenv("MIST_MEMORY_RERANK_BACKEND", "remote")

        with pytest.raises(MemoryConfigError, match="rerank_url"):
            MemoryConfig.from_env(tz_resolver=_never_called)

    @pytest.mark.parametrize("view", list(FACT_VIEWS))
    def test_every_fact_view_is_accepted(self, view):
        assert MemoryConfig(fact_view=view).fact_view == view

    def test_from_env_rejects_an_unknown_fact_view(self, monkeypatch):
        monkeypatch.setenv("MIST_MEMORY_FACT_VIEW", "F9")

        with pytest.raises(MemoryConfigError, match="fact_view"):
            MemoryConfig.from_env(tz_resolver=_never_called)

    @pytest.mark.parametrize(
        ("name", "raw", "fragment"),
        [
            ("MIST_MEMORY_NEED_GATE", "true", "need_gate"),
            ("MIST_MEMORY_NEED_EMBEDDER", "bge-small", "need_embedder"),
        ],
    )
    def test_from_env_rejects_an_unknown_need_setting(self, monkeypatch, name, raw, fragment):
        monkeypatch.setenv(name, raw)

        with pytest.raises(MemoryConfigError, match=fragment):
            MemoryConfig.from_env(tz_resolver=_never_called)

    @pytest.mark.parametrize("name", sorted(NEED_EMBEDDERS))
    def test_every_need_embedder_name_is_accepted(self, name):
        assert MemoryConfig(need_embedder=name).need_embedder == name

    def test_the_need_embedder_type_names_the_need_embedders(self):
        """The setting's Literal and the gate's table stay one list of names."""
        assert NeedEmbedderName.__args__ == tuple(NEED_EMBEDDERS)  # type: ignore[attr-defined]

    @pytest.mark.parametrize(
        "name",
        [
            "MemoryIndexError",
            "MemoryRerankError",
            "MemoryCorrectionError",
            "MemoryConfigError",
            "ColdSummaryError",
        ],
    )
    def test_memory_errors_are_mist_errors(self, name):
        import backend.errors

        error = getattr(backend.errors, name)

        assert issubclass(error, MistError)
        assert error is not MistError


class TestRerankUrlIsPrivate:
    """`rerank_url` must name a host on the local network, so memory text stays on it."""

    @pytest.mark.parametrize(
        "url",
        [
            # loopback, IPv4 and IPv6
            "http://127.0.0.1:8080",
            "http://127.200.3.4",
            "http://[::1]:8080",
            # RFC 1918, at both ends of each range
            "http://10.0.0.1",
            "http://10.255.255.254",
            "http://172.16.0.1",
            "http://172.31.255.254",
            "http://192.168.0.1",
            "http://192.168.255.254",
            # CGNAT 100.64.0.0/10, the Tailscale address range
            "http://100.64.0.1",
            "http://100.127.255.254",
            # an IPv4-mapped IPv6 address and a legacy IPv4 spelling are read as IPv4
            "http://[::ffff:10.0.0.7]:8080",
            "http://0x7f000001:8080",
            # names
            "http://localhost:8080",
            "http://LOCALHOST",
            "http://rerank:8080",
            "http://rerank-host.tail1234.ts.net:8080",
            "http://rerank.lan",
            "http://rerank.local",
            "http://rerank.internal",
            "http://rerank.lan.:8080",
            "https://rerank.lan:8443/base/",
        ],
    )
    def test_accepts_a_private_host(self, url):
        config = MemoryConfig(rerank_backend="remote", rerank_url=url)

        assert config.rerank_url == url

    @pytest.mark.parametrize(
        ("url", "host"),
        [
            ("http://rerank.example.com:8080", "rerank.example.com"),
            ("https://api.openai.com", "api.openai.com"),
            ("http://lan.example.com", "lan.example.com"),
            ("http://rerank.local.example.com", "rerank.local.example.com"),
            ("http://evilts.net", "evilts.net"),
            ("http://localhost.example.com", "localhost.example.com"),
            # a single label with a trailing dot is a top-level name, resolved publicly
            ("http://rerank.:8080", "rerank."),
        ],
    )
    def test_rejects_a_public_host_naming_it(self, url, host):
        with pytest.raises(MemoryConfigError, match=re.escape(repr(host))):
            MemoryConfig(rerank_backend="remote", rerank_url=url)

    @pytest.mark.parametrize(
        ("url", "host"),
        [
            ("http://8.8.8.8", "8.8.8.8"),
            ("http://11.0.0.1", "11.0.0.1"),
            ("http://172.15.255.255", "172.15.255.255"),
            ("http://172.32.0.1", "172.32.0.1"),
            ("http://192.169.0.1", "192.169.0.1"),
            ("http://100.63.255.255", "100.63.255.255"),
            ("http://100.128.0.1", "100.128.0.1"),
            ("http://169.254.1.1", "169.254.1.1"),
            ("http://0.0.0.0", "0.0.0.0"),  # nosec B104 - a refused input, not a bind
            ("http://[2001:4860:4860::8888]", "2001:4860:4860::8888"),
            ("http://[fd00::1]", "fd00::1"),
            ("http://[::ffff:8.8.8.8]", "::ffff:8.8.8.8"),
            # legacy IPv4 spellings of 8.8.8.8 that the resolver would accept
            ("http://134744072", "134744072"),
            ("http://0x08080808", "0x08080808"),
        ],
    )
    def test_rejects_an_ip_outside_the_private_ranges_naming_it(self, url, host):
        with pytest.raises(MemoryConfigError, match=re.escape(repr(host))):
            MemoryConfig(rerank_backend="remote", rerank_url=url)

    @pytest.mark.parametrize(
        ("url", "host"),
        [
            ("ftp://rerank.lan", "rerank.lan"),
            ("ws://localhost:8080", "localhost"),
            ("file://localhost/etc/passwd", "localhost"),
        ],
    )
    def test_rejects_a_scheme_other_than_http_or_https_naming_the_host(self, url, host):
        with pytest.raises(MemoryConfigError, match=r"scheme.*" + re.escape(repr(host))):
            MemoryConfig(rerank_backend="remote", rerank_url=url)

    @pytest.mark.parametrize("url", ["http://", "rerank.lan", "http://[::1"])
    def test_rejects_a_url_without_a_usable_host(self, url):
        with pytest.raises(MemoryConfigError, match="rerank_url"):
            MemoryConfig(rerank_backend="remote", rerank_url=url)

    def test_credentials_in_the_url_stay_out_of_the_error(self):
        with pytest.raises(MemoryConfigError, match=re.escape("'8.8.8.8'")) as exc:
            MemoryConfig(rerank_backend="remote", rerank_url="http://user:hunter2@8.8.8.8")

        assert "hunter2" not in str(exc.value)

    def test_a_public_url_fails_even_when_the_backend_does_not_use_it(self):
        """A set URL is checked whatever the backend, so switching to remote cannot leak."""
        with pytest.raises(MemoryConfigError, match="rerank.example.com"):
            MemoryConfig(rerank_backend="local", rerank_url="http://rerank.example.com")

    def test_from_env_rejects_a_public_rerank_url(self, monkeypatch):
        monkeypatch.setenv("MIST_MEMORY_RERANK_BACKEND", "remote")
        monkeypatch.setenv("MIST_MEMORY_RERANK_URL", "https://rerank.example.com/v1")

        with pytest.raises(MemoryConfigError, match="rerank.example.com"):
            MemoryConfig.from_env(tz_resolver=_never_called)


class TestEnvExampleDocumentsEveryVariable:
    def test_every_phase_a_variable_appears_with_its_default(self):
        """Each variable has a `# NAME=value` or `NAME=value` line in .env.example.

        For a variable with a plain default the documented value must parse back to it, so the
        file cannot drift from the dataclass silently.
        """
        text = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")
        documented = _documented(text)

        missing = sorted(set(MEMORY_ENV_VARS) - set(documented))
        assert not missing, f".env.example does not document {missing}"

        defaults = MemoryConfig()
        assert documented["MIST_MEMORY_ENABLED"] == "true"
        assert int(documented["MIST_MEMORY_BLOCK_TOKENS"]) == defaults.block_tokens
        assert float(documented["MIST_MEMORY_CONTINUITY_SHARE"]) == defaults.continuity_share
        assert int(documented["MIST_MEMORY_SELECT_K"]) == defaults.select_k
        assert int(documented["MIST_MEMORY_DEADLINE_MS"]) == defaults.deadline_ms
        assert documented["MIST_MEMORY_RERANK_BACKEND"] == defaults.rerank_backend
        assert documented["MIST_MEMORY_RERANK_MODEL"] == defaults.rerank_model
        assert int(documented["MIST_MEMORY_RERANK_TOP_N"]) == defaults.rerank_top_n
        assert documented["MIST_MEMORY_RERANK_DEVICE"] == defaults.rerank_device
        assert int(documented["MIST_MEMORY_RERANK_TIMEOUT_MS"]) == defaults.rerank_timeout_ms
        assert documented["MIST_MEMORY_RERANK_REMOTE_SCORES"] == defaults.rerank_remote_scores
        assert int(documented["MIST_MEMORY_REFRESH_DEBOUNCE_S"]) == defaults.refresh_debounce_s
        assert int(documented["MIST_MEMORY_BLOCK_TOKENS_LOW"]) == defaults.block_tokens_low
        assert int(documented["MIST_MEMORY_LIST_MAX_ITEMS"]) == defaults.list_max_items
        assert documented["MIST_MEMORY_FACETS"] == "on"
        assert int(documented["MIST_MEMORY_RESCAN_INTERVAL_S"]) == defaults.rescan_interval_s
        assert documented["MIST_MEMORY_FACT_VIEW"] == defaults.fact_view
        assert documented["MIST_MEMORY_NEED_GATE"] == defaults.need_gate
        assert documented["MIST_MEMORY_NEED_EMBEDDER"] == defaults.need_embedder
        assert documented["MIST_MEMORY_NEED_CONTEXT"] == "true"

    def test_the_need_settings_follow_the_fact_view_in_order(self):
        """The new variables join `MEMORY_ENV_VARS` in `.env.example` order."""
        text = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")
        names = ["MIST_MEMORY_FACT_VIEW", *_NEED_VARIABLES]

        positions = [text.index(f"# {name}=") for name in names]

        assert positions == sorted(positions)
        index = MEMORY_ENV_VARS.index("MIST_MEMORY_FACT_VIEW")
        assert MEMORY_ENV_VARS[index + 1 : index + 4] == _NEED_VARIABLES

    def test_the_variables_are_listed_in_env_example_order(self):
        text = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")
        names = [name for name in MEMORY_ENV_VARS if name.startswith("MIST_MEMORY_")]

        positions = [text.index(f"# {name}=") for name in names]

        assert positions == sorted(positions)

    def test_the_need_embedder_comment_names_every_embedder(self):
        text = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")
        section = text[
            text.index("# MIST_MEMORY_FACT_VIEW=") : text.index("# MIST_MEMORY_NEED_CONTEXT=")
        ]

        for name in NEED_EMBEDDERS:
            assert name in section, name

    def test_retired_variables_are_not_documented(self):
        text = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")

        assert set(RETIRED_MEMORY_ENV_VARS).isdisjoint(_documented(text))

    def test_the_shared_zone_variable_is_documented_once(self):
        """Usage counting and memory share `MIST_TIMEZONE`; one entry documents both."""
        text = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")

        assert len(re.findall(r"^#?\s*MIST_TIMEZONE=", text, re.MULTILINE)) == 1

    def test_window_comment_recommends_8192_for_e4b_and_keeps_auto(self):
        text = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")

        assert "# MIST_CTX_BUDGET_WINDOW=auto" in text
        assert "MIST_CTX_BUDGET_WINDOW=8192" in text
        assert "E4B" in text
