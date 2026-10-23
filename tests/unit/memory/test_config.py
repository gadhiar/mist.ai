"""Tests for backend.memory.config.MemoryConfig.

The zone tests pass a fake resolver, so they need no tz database. The single test of the
production resolver (`zoneinfo.ZoneInfo`) skips while the worker image lacks tzdata.
"""

import re
import zoneinfo
from datetime import UTC, timedelta, timezone, tzinfo
from pathlib import Path

import pytest

from backend.errors import MemoryConfigError, MistError
from backend.memory.config import MEMORY_ENV_VARS, MemoryConfig

_REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def _clear_memory_env(monkeypatch):
    """Start every test with no MIST_MEMORY_* variable set, whatever the process holds."""
    for name in MEMORY_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


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
        assert config.block_role == "user_prefix"
        assert config.continuity_share == 0.35
        assert config.min_score == 0.35
        assert config.deadline_ms == 50
        assert config.rerank_backend == "local"
        assert config.rerank_model == "/opt/models/ms-marco-MiniLM-L-6-v2"
        assert config.rerank_top_n == 20
        assert config.rerank_device == "cpu"
        assert config.rerank_url is None
        assert config.rerank_remote_model is None
        assert config.rerank_timeout_ms == 40
        assert config.rerank_remote_scores == "logit"
        assert (config.rerank_weight, config.recency_weight, config.confidence_weight) == (
            0.8,
            0.1,
            0.1,
        )
        assert config.index_path is None
        assert config.refresh_debounce_s == 30
        assert config.tz is UTC
        assert config.tz_name is None

    def test_is_frozen(self):
        config = MemoryConfig()
        with pytest.raises(AttributeError):
            config.block_tokens = 1  # type: ignore[misc]

    def test_from_env_with_nothing_set_equals_the_defaults(self):
        assert MemoryConfig.from_env(tz_resolver=_never_called) == MemoryConfig()


class TestFromEnv:
    def test_reads_every_scalar_variable(self, monkeypatch):
        values = {
            "MIST_MEMORY_ENABLED": "false",
            "MIST_MEMORY_BLOCK_TOKENS": "300",
            "MIST_MEMORY_BLOCK_ROLE": "system_tail",
            "MIST_MEMORY_CONTINUITY_SHARE": "0.5",
            "MIST_MEMORY_MIN_SCORE": "0.4",
            "MIST_MEMORY_DEADLINE_MS": "80",
            "MIST_MEMORY_RERANK_BACKEND": "remote",
            "MIST_MEMORY_RERANK_MODEL": "/opt/models/ms-marco-MiniLM-L-12-v2",
            "MIST_MEMORY_RERANK_TOP_N": "12",
            "MIST_MEMORY_RERANK_DEVICE": "cuda",
            "MIST_MEMORY_RERANK_URL": "http://rerank.example:8080",
            "MIST_MEMORY_RERANK_REMOTE_MODEL": "bge-reranker",
            "MIST_MEMORY_RERANK_TIMEOUT_MS": "25",
            "MIST_MEMORY_RERANK_REMOTE_SCORES": "probability",
            "MIST_MEMORY_WEIGHTS": "0.6, 0.3, 0.1",
            "MIST_MEMORY_INDEX_PATH": ":memory:",
            "MIST_MEMORY_REFRESH_DEBOUNCE_S": "5",
        }
        for name, value in values.items():
            monkeypatch.setenv(name, value)

        config = MemoryConfig.from_env(tz_resolver=_never_called)

        assert config.enabled is False
        assert config.block_tokens == 300
        assert config.block_role == "system_tail"
        assert config.continuity_share == 0.5
        assert config.min_score == 0.4
        assert config.deadline_ms == 80
        assert config.rerank_backend == "remote"
        assert config.rerank_model == "/opt/models/ms-marco-MiniLM-L-12-v2"
        assert config.rerank_top_n == 12
        assert config.rerank_device == "cuda"
        assert config.rerank_url == "http://rerank.example:8080"
        assert config.rerank_remote_model == "bge-reranker"
        assert config.rerank_timeout_ms == 25
        assert config.rerank_remote_scores == "probability"
        assert (config.rerank_weight, config.recency_weight, config.confidence_weight) == (
            0.6,
            0.3,
            0.1,
        )
        assert config.index_path == ":memory:"
        assert config.refresh_debounce_s == 5

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
            ("MIST_MEMORY_WEIGHTS", "0.8,0.2"),
            ("MIST_MEMORY_WEIGHTS", "0.8,x,0.1"),
        ],
    )
    def test_an_unparsable_value_names_its_variable(self, monkeypatch, name, raw):
        monkeypatch.setenv(name, raw)

        with pytest.raises(MemoryConfigError, match=name):
            MemoryConfig.from_env(tz_resolver=_never_called)


class TestZone:
    def test_unset_zone_is_utc_and_never_calls_the_resolver(self, caplog):
        """`zoneinfo.ZoneInfo("UTC")` raises in an image without tzdata (Design point 5)."""
        with caplog.at_level("WARNING", logger="backend.memory.config"):
            config = MemoryConfig.from_env(tz_resolver=_never_called)

        assert config.tz is UTC
        assert config.tz_name is None
        assert any("MIST_MEMORY_TZ" in record.getMessage() for record in caplog.records)

    def test_blank_zone_counts_as_unset(self, monkeypatch):
        monkeypatch.setenv("MIST_MEMORY_TZ", "")

        config = MemoryConfig.from_env(tz_resolver=_never_called)

        assert config.tz is UTC

    def test_set_zone_resolves_once_at_config_time(self, monkeypatch):
        monkeypatch.setenv("MIST_MEMORY_TZ", "America/Chicago")
        resolver = _RecordingResolver()

        config = MemoryConfig.from_env(tz_resolver=resolver)

        assert resolver.names == ["America/Chicago"]
        assert config.tz is resolver.zone
        assert config.tz_name == "America/Chicago"

    def test_a_rejected_name_raises_memory_config_error_naming_the_variable(self, monkeypatch):
        monkeypatch.setenv("MIST_MEMORY_TZ", "Mars/Olympus_Mons")

        with pytest.raises(MemoryConfigError, match="MIST_MEMORY_TZ") as excinfo:
            MemoryConfig.from_env(tz_resolver=_rejecting)

        assert "Mars/Olympus_Mons" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, zoneinfo.ZoneInfoNotFoundError)

    def test_a_malformed_name_raises_memory_config_error(self, monkeypatch):
        """A key like an absolute path makes zoneinfo raise ValueError, not a not-found error."""

        def malformed(name: str) -> tzinfo:
            raise ValueError(f"ZoneInfo keys must be normalized relative paths, got: {name}")

        monkeypatch.setenv("MIST_MEMORY_TZ", "/etc/localtime")

        with pytest.raises(MemoryConfigError, match="MIST_MEMORY_TZ"):
            MemoryConfig.from_env(tz_resolver=malformed)

    def test_production_resolver_resolves_an_iana_name(self, monkeypatch):
        try:
            expected = zoneinfo.ZoneInfo("America/Chicago")
        except zoneinfo.ZoneInfoNotFoundError:
            pytest.skip("tzdata absent until mist-capabilities-build T1's deps PR is in the image")
        monkeypatch.setenv("MIST_MEMORY_TZ", "America/Chicago")

        config = MemoryConfig.from_env()

        assert config.tz == expected
        assert config.tz_name == "America/Chicago"


class TestValidation:
    @pytest.mark.parametrize(
        ("overrides", "fragment"),
        [
            ({"block_tokens": 0}, "block_tokens"),
            ({"block_tokens": -1}, "block_tokens"),
            ({"continuity_share": -0.01}, "continuity_share"),
            ({"continuity_share": 1.01}, "continuity_share"),
            ({"min_score": 1.5}, "min_score"),
            ({"deadline_ms": 0}, "deadline_ms"),
            ({"rerank_top_n": 0}, "rerank_top_n"),
            ({"block_role": "assistant"}, "block_role"),
            ({"rerank_device": "mps"}, "rerank_device"),
            ({"rerank_backend": "gpu"}, "rerank_backend"),
            ({"rerank_remote_scores": "raw"}, "rerank_remote_scores"),
            ({"rerank_weight": 0.8, "recency_weight": 0.2, "confidence_weight": 0.1}, "weights"),
            ({"rerank_weight": 1.1, "recency_weight": -0.2, "confidence_weight": 0.1}, "weights"),
            ({"rerank_backend": "remote"}, "rerank_url"),
            (
                {"rerank_backend": "remote", "rerank_url": "http://r:1", "rerank_timeout_ms": 0},
                "rerank_timeout_ms",
            ),
            ({"refresh_debounce_s": -1}, "refresh_debounce_s"),
        ],
    )
    def test_rejects_invalid_values(self, overrides, fragment):
        with pytest.raises(MemoryConfigError, match=fragment):
            MemoryConfig(**overrides)

    def test_share_bounds_are_inclusive(self):
        assert MemoryConfig(continuity_share=0.0).continuity_share == 0.0
        assert MemoryConfig(continuity_share=1.0).continuity_share == 1.0

    def test_weights_sum_within_tolerance(self):
        config = MemoryConfig(rerank_weight=0.7, recency_weight=0.2, confidence_weight=0.1000005)
        assert config.confidence_weight == 0.1000005

    def test_weights_outside_tolerance_fail(self):
        with pytest.raises(MemoryConfigError, match="weights"):
            MemoryConfig(rerank_weight=0.7, recency_weight=0.2, confidence_weight=0.10001)

    def test_remote_with_url_and_timeout_is_valid(self):
        config = MemoryConfig(rerank_backend="remote", rerank_url="http://rerank.example:8080")
        assert config.rerank_timeout_ms == 40

    def test_local_and_off_need_no_url(self):
        assert MemoryConfig(rerank_backend="off").rerank_url is None
        assert MemoryConfig(rerank_backend="local").rerank_url is None

    def test_validation_runs_on_from_env(self, monkeypatch):
        monkeypatch.setenv("MIST_MEMORY_RERANK_BACKEND", "remote")

        with pytest.raises(MemoryConfigError, match="rerank_url"):
            MemoryConfig.from_env(tz_resolver=_never_called)

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


class TestEnvExampleDocumentsEveryVariable:
    def test_every_phase_a_variable_appears_with_its_default(self):
        """Each variable has a `# NAME=value` or `NAME=value` line in .env.example.

        For a variable with a plain default the documented value must parse back to it, so the
        file cannot drift from the dataclass silently.
        """
        text = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")
        documented: dict[str, str] = {}
        for match in re.finditer(r"^#?\s*(MIST_MEMORY_[A-Z_]+)=(\S*)", text, re.MULTILINE):
            documented.setdefault(match.group(1), match.group(2))

        missing = sorted(set(MEMORY_ENV_VARS) - set(documented))
        assert not missing, f".env.example does not document {missing}"

        defaults = MemoryConfig()
        assert documented["MIST_MEMORY_ENABLED"] == "true"
        assert int(documented["MIST_MEMORY_BLOCK_TOKENS"]) == defaults.block_tokens
        assert documented["MIST_MEMORY_BLOCK_ROLE"] == defaults.block_role
        assert float(documented["MIST_MEMORY_CONTINUITY_SHARE"]) == defaults.continuity_share
        assert float(documented["MIST_MEMORY_MIN_SCORE"]) == defaults.min_score
        assert int(documented["MIST_MEMORY_DEADLINE_MS"]) == defaults.deadline_ms
        assert documented["MIST_MEMORY_RERANK_BACKEND"] == defaults.rerank_backend
        assert documented["MIST_MEMORY_RERANK_MODEL"] == defaults.rerank_model
        assert int(documented["MIST_MEMORY_RERANK_TOP_N"]) == defaults.rerank_top_n
        assert documented["MIST_MEMORY_RERANK_DEVICE"] == defaults.rerank_device
        assert int(documented["MIST_MEMORY_RERANK_TIMEOUT_MS"]) == defaults.rerank_timeout_ms
        assert documented["MIST_MEMORY_RERANK_REMOTE_SCORES"] == defaults.rerank_remote_scores
        assert documented["MIST_MEMORY_WEIGHTS"] == "0.8,0.1,0.1"
        assert int(documented["MIST_MEMORY_REFRESH_DEBOUNCE_S"]) == defaults.refresh_debounce_s

    def test_window_comment_recommends_8192_for_e4b_and_keeps_auto(self):
        text = (_REPO_ROOT / ".env.example").read_text(encoding="utf-8")

        assert "# MIST_CTX_BUDGET_WINDOW=auto" in text
        assert "MIST_CTX_BUDGET_WINDOW=8192" in text
        assert "E4B" in text
