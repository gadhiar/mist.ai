"""Tests for secret loading, masking and log-record redaction."""

from __future__ import annotations

import io
import logging
from collections.abc import Iterator

import pytest

from backend.capabilities.secrets import (
    EnvSecretSource,
    Secret,
    SecretRegistry,
    install_log_redaction,
    redact,
)
from backend.errors import SecretMissingError

FAKE_VALUE = "fake-secret-value-0001"
LATE_VALUE = "fake-refreshed-token-0002"


def _no_file(path: str) -> str:
    raise AssertionError(f"no file read expected, got {path}")


@pytest.fixture()
def restore_record_factory() -> Iterator[None]:
    """Put back the process-wide record factory, so one test's install never leaks."""
    original = logging.getLogRecordFactory()
    yield
    logging.setLogRecordFactory(original)


@pytest.fixture()
def captured() -> Iterator[tuple[logging.Logger, io.StringIO]]:
    """A child logger whose records reach a formatting handler on its parent.

    The parent does not propagate, so pytest's own root handlers, which fail a test on any
    record that cannot be formatted, never see these records; a formatting error then takes
    the standard `Handler.handleError` path to stderr, as it does in production.
    """
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(levelname)s %(message)s"))
    parent = logging.getLogger("mist_test_redaction")
    parent.addHandler(handler)
    parent.setLevel(logging.DEBUG)
    parent.propagate = False
    child = logging.getLogger("mist_test_redaction.child")
    yield child, stream
    parent.removeHandler(handler)
    parent.setLevel(logging.NOTSET)
    parent.propagate = True


class _RaisingStr:
    """A log argument whose `__str__` fails with an error logging does not anticipate."""

    def __str__(self) -> str:
        raise RuntimeError("str failed")


class TestSecret:
    def test_reveal_returns_the_value(self):
        assert Secret(FAKE_VALUE).reveal() == FAKE_VALUE

    @pytest.mark.parametrize(
        "render",
        [
            pytest.param(repr, id="repr"),
            pytest.param(str, id="str"),
            pytest.param(lambda s: f"{s}", id="f-string"),
            pytest.param(lambda s: f"{s!r}", id="f-string-repr"),
            pytest.param(lambda s: f"{[s]}", id="inside-a-list"),
        ],
    )
    def test_rendering_never_contains_the_value(self, render):
        text = render(Secret(FAKE_VALUE))

        assert FAKE_VALUE not in text
        assert "Secret('***')" in text


class TestEnvSecretSource:
    def test_reads_the_variable_before_the_file(self):
        source = EnvSecretSource(
            {"MIST_HA_TOKEN": FAKE_VALUE, "MIST_HA_TOKEN_FILE": "/x"}, _no_file
        )

        secret = source.get("MIST_HA_TOKEN")

        assert secret is not None
        assert secret.reveal() == FAKE_VALUE

    def test_reads_the_file_and_strips_one_trailing_newline(self):
        files = {"/run/secrets/ha": f"{FAKE_VALUE}\n\n"}
        source = EnvSecretSource({"MIST_HA_TOKEN_FILE": "/run/secrets/ha"}, files.__getitem__)

        secret = source.get("MIST_HA_TOKEN")

        assert secret is not None
        assert secret.reveal() == f"{FAKE_VALUE}\n"

    def test_strips_a_trailing_crlf_as_one_newline(self):
        source = EnvSecretSource({"MIST_HA_TOKEN_FILE": "/f"}, lambda path: f"{FAKE_VALUE}\r\n")

        secret = source.get("MIST_HA_TOKEN")

        assert secret is not None
        assert secret.reveal() == FAKE_VALUE

    def test_returns_none_when_neither_is_set(self):
        assert EnvSecretSource({}, _no_file).get("MIST_HA_TOKEN") is None

    def test_returns_none_for_an_empty_value(self):
        source = EnvSecretSource({"MIST_HA_TOKEN": "", "MIST_HA_TOKEN_FILE": "/f"}, lambda p: "\n")

        assert source.get("MIST_HA_TOKEN") is None

    def test_unreadable_file_raises_secret_missing_naming_the_variable(self):
        def unreadable(path: str) -> str:
            raise FileNotFoundError(path)

        source = EnvSecretSource({"MIST_HA_TOKEN_FILE": "/missing"}, unreadable)

        with pytest.raises(SecretMissingError, match="MIST_HA_TOKEN_FILE"):
            source.get("MIST_HA_TOKEN")


class TestRedact:
    def test_replaces_every_registered_value(self):
        registry = SecretRegistry()
        registry.register(Secret(FAKE_VALUE))

        assert redact(f"a {FAKE_VALUE} b {FAKE_VALUE}", registry) == "a *** b ***"

    def test_longer_value_is_replaced_before_a_value_it_contains(self):
        registry = SecretRegistry()
        registry.register(Secret("fake-abc"))
        registry.register(Secret("fake-abc-def"))

        assert redact("x fake-abc-def y", registry) == "x *** y"

    def test_empty_value_is_ignored(self):
        registry = SecretRegistry()
        registry.register(Secret(""))

        assert redact("unchanged", registry) == "unchanged"


@pytest.mark.usefixtures("restore_record_factory")
class TestLogRedaction:
    def test_value_in_the_message_is_redacted(self, captured):
        logger, stream = captured
        registry = SecretRegistry()
        registry.register(Secret(FAKE_VALUE))
        install_log_redaction(registry)

        logger.warning(f"token is {FAKE_VALUE}")

        assert FAKE_VALUE not in stream.getvalue()
        assert "token is ***" in stream.getvalue()

    def test_value_in_the_arguments_is_redacted(self, captured):
        logger, stream = captured
        registry = SecretRegistry()
        registry.register(Secret(FAKE_VALUE))
        install_log_redaction(registry)

        logger.info("token is %s, again %r", FAKE_VALUE, FAKE_VALUE)

        assert FAKE_VALUE not in stream.getvalue()
        assert "token is ***, again '***'" in stream.getvalue()

    def test_value_in_a_logged_exception_traceback_is_redacted(self, captured):
        logger, stream = captured
        registry = SecretRegistry()
        registry.register(Secret(FAKE_VALUE))
        install_log_redaction(registry)

        try:
            raise ValueError(f"refused token {FAKE_VALUE}")
        except ValueError:
            logger.exception("call failed")

        output = stream.getvalue()
        assert FAKE_VALUE not in output
        assert "Traceback" in output
        assert "ValueError: refused token ***" in output

    def test_secret_registered_after_install_is_redacted(self, captured):
        logger, stream = captured
        registry = SecretRegistry()
        install_log_redaction(registry)

        registry.register(Secret(LATE_VALUE))
        logger.warning("refreshed %s", LATE_VALUE)

        assert LATE_VALUE not in stream.getvalue()
        assert "refreshed ***" in stream.getvalue()

    def test_second_install_is_a_no_op(self, captured):
        logger, stream = captured
        registry = SecretRegistry()
        registry.register(Secret(FAKE_VALUE))
        install_log_redaction(registry)
        installed = logging.getLogRecordFactory()

        install_log_redaction(SecretRegistry())
        logger.warning("value %s", FAKE_VALUE)

        assert logging.getLogRecordFactory() is installed
        assert "value ***" in stream.getvalue()

    def test_mismatched_arguments_do_not_raise_or_leak(self, captured, capsys):
        logger, stream = captured
        registry = SecretRegistry()
        registry.register(Secret(FAKE_VALUE))
        install_log_redaction(registry)

        logger.warning("two slots %s %s", FAKE_VALUE)

        error_output = capsys.readouterr().err
        assert "Logging error" in error_output
        assert FAKE_VALUE not in error_output
        assert FAKE_VALUE not in stream.getvalue()

    def test_argument_whose_str_raises_does_not_raise_or_leak(self, captured, capsys):
        logger, stream = captured
        registry = SecretRegistry()
        registry.register(Secret(FAKE_VALUE))
        install_log_redaction(registry)

        logger.warning("token %s then %s", FAKE_VALUE, _RaisingStr())

        error_output = capsys.readouterr().err
        assert "RuntimeError: str failed" in error_output
        assert FAKE_VALUE not in error_output
        assert FAKE_VALUE not in stream.getvalue()

    def test_secret_registered_after_the_record_is_redacted_when_formatted(self, captured):
        logger, stream = captured
        registry = SecretRegistry()
        install_log_redaction(registry)
        record = logging.getLogRecordFactory()(
            "mist_test_redaction.child", logging.INFO, __file__, 1, "late %s", (LATE_VALUE,), None
        )

        registry.register(Secret(LATE_VALUE))
        logger.handle(record)

        assert LATE_VALUE not in stream.getvalue()
        assert "late ***" in stream.getvalue()

    def test_record_without_secrets_is_unchanged(self, captured):
        logger, stream = captured
        install_log_redaction(SecretRegistry())

        logger.info("plain %d items", 3)

        assert stream.getvalue() == "INFO plain 3 items\n"
