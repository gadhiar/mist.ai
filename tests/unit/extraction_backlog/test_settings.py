"""The dispatcher's client timeout for one `/v1/extract` request.

Medium effort is the deployed default, and one medium extract call that runs
to its cap costs about 651 s, so a 300 s client timeout would cut a long list
turn and dead-letter it after `max_attempts`. The timeout bounds the whole
request, which can hold several capped calls when a batch is split and
retried; the service's own `EXTRACTION_LLM_TIMEOUT_SECONDS` bounds one call.
"""

from __future__ import annotations

import pytest

from backend.extraction_backlog.settings import DispatcherSettings

_ENV = "MIST_EXTRACTION_REQUEST_TIMEOUT_S"


class TestRequestTimeout:
    def test_the_dataclass_default_is_1800(self):
        assert DispatcherSettings().request_timeout_s == 1800.0

    def test_from_env_defaults_to_1800(self, monkeypatch):
        monkeypatch.delenv(_ENV, raising=False)

        assert DispatcherSettings.from_env().request_timeout_s == 1800.0

    def test_an_empty_value_keeps_the_default(self, monkeypatch):
        monkeypatch.setenv(_ENV, "  ")

        assert DispatcherSettings.from_env().request_timeout_s == 1800.0

    def test_the_environment_overrides_it(self, monkeypatch):
        monkeypatch.setenv(_ENV, "300")

        assert DispatcherSettings.from_env().request_timeout_s == 300.0

    def test_an_unparsable_value_is_refused(self, monkeypatch):
        monkeypatch.setenv(_ENV, "half an hour")

        with pytest.raises(ValueError):
            DispatcherSettings.from_env()
