"""Secret values for capability providers, and their removal from logs and outcomes.

Tokens (Home Assistant, Spotify) come only from the environment or Docker secret files (ADR-025
section 2.5). A `Secret` never renders its value, and every loaded or refreshed value is
registered with a `SecretRegistry`. `install_log_redaction` then changes the log records
themselves rather than any handler, so the console, file and WebSocket log handlers are covered
alike, including the one that streams formatted records with tracebacks to the client.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from typing import Protocol

from backend.errors import SecretMissingError

MASK = "***"
_MASKED_REPR = "Secret('***')"

# Set on the installed factory so a second install can recognise it.
_INSTALLED_MARKER = "_mist_secret_redaction"
_install_lock = threading.Lock()


class Secret:
    """A secret string that renders as `Secret('***')` in repr, str and f-strings."""

    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        self._value = value

    def reveal(self) -> str:
        """Return the value; call it only where the value is sent to its provider."""
        return self._value

    def __repr__(self) -> str:
        return _MASKED_REPR

    def __str__(self) -> str:
        return _MASKED_REPR

    def __format__(self, format_spec: str) -> str:
        return format(_MASKED_REPR, format_spec)


class SecretSource(Protocol):
    """Where named secrets come from."""

    def get(self, name: str) -> Secret | None:
        """Return the secret called `name`, or None when it is not configured."""
        ...


class EnvSecretSource:
    """Reads `NAME` from the environment, else the file named by `NAME_FILE`.

    The file form serves Docker secrets. One trailing newline is stripped from file content,
    because editors and `echo` add one that is not part of the token.
    """

    def __init__(self, env: Mapping[str, str], read_file: Callable[[str], str]) -> None:
        self._env = env
        self._read_file = read_file

    def get(self, name: str) -> Secret | None:
        """Return the secret called `name`, or None when neither form is set or it is empty.

        Raises:
            SecretMissingError: `NAME_FILE` is set but the file cannot be read.
        """
        value = self._env.get(name, "")
        if value:
            return Secret(value)
        file_variable = f"{name}_FILE"
        path = self._env.get(file_variable, "").strip()
        if not path:
            return None
        try:
            content = self._read_file(path)
        except OSError as exc:
            raise SecretMissingError(
                f"{file_variable} names a file that cannot be read: {type(exc).__name__}"
            ) from exc
        content = _strip_one_newline(content)
        return Secret(content) if content else None


class SecretRegistry:
    """Every secret value MIST holds, including tokens refreshed after startup.

    Thread-safe: a token refresh registers from the event loop while log records are created on
    any thread.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._values: frozenset[str] = frozenset()
        self._ordered: tuple[str, ...] = ()

    def register(self, secret: Secret) -> None:
        """Add a secret's value; empty values are ignored, since they would match everywhere."""
        value = secret.reveal()
        if not value:
            return
        with self._lock:
            if value in self._values:
                return
            self._values = self._values | {value}
            # Longest first, so a value that contains another is masked whole.
            self._ordered = tuple(sorted(self._values, key=lambda v: (-len(v), v)))

    def _snapshot(self) -> tuple[str, ...]:
        return self._ordered


def redact(text: str, registry: SecretRegistry) -> str:
    """Replace every registered secret value in `text` with `***`."""
    for value in registry._snapshot():
        if value in text:
            text = text.replace(value, MASK)
    return text


def install_log_redaction(registry: SecretRegistry) -> None:
    """Wrap the logging record factory so no record can carry a secret to a handler.

    At creation the wrapper redacts the parts that can be redacted without running user code:
    a string message, string arguments, and any exception or stack information, which it
    formats into redacted text and puts in place of `exc_info`. The full message is formatted
    and redacted later, when a handler calls `getMessage`, so formatting runs exactly where it
    runs without redaction: an argument whose `__str__` raises makes the handler's `emit`
    fail into `Handler.handleError`, as before, and never makes the logging call raise. Before
    such an error propagates, the record's message and arguments are reduced to redacted or
    type-only text, because `handleError` prints both.

    Values registered after installation apply to every message formatted after their
    registration. A second call is a no-op, whichever registry it names, so wiring that runs
    twice never stacks wrappers.
    """
    global _active_registry
    with _install_lock:
        current = logging.getLogRecordFactory()
        if getattr(current, _INSTALLED_MARKER, False):
            return
        _active_registry = registry

        def factory(*args: object, **kwargs: object) -> logging.LogRecord:
            record = current(*args, **kwargs)
            _redact_at_creation(record, registry)
            record.__class__ = _redacting_class(type(record))
            return record

        setattr(factory, _INSTALLED_MARKER, True)
        logging.setLogRecordFactory(factory)


_FORMATTER = logging.Formatter()
_active_registry: SecretRegistry | None = None
_redacting_classes: dict[type[logging.LogRecord], type[logging.LogRecord]] = {}


def _redacted_message(record: logging.LogRecord, base: type[logging.LogRecord]) -> str:
    formatted = False
    try:
        message = base.getMessage(record)
        formatted = True
    finally:
        if not formatted:
            # No except clause: the formatting error propagates to `Handler.handleError`
            # unchanged, which prints `msg` and `args`, so neither may hold a secret.
            _make_error_report_safe(record)
    registry = _active_registry
    return redact(message, registry) if registry is not None else message


class _RedactingLogRecord(logging.LogRecord):
    """A stock `LogRecord` whose formatted message is redacted; module-level so it pickles."""

    def getMessage(self) -> str:  # noqa: N802 - overrides the stdlib method name
        return _redacted_message(self, logging.LogRecord)


def _redacting_class(base: type[logging.LogRecord]) -> type[logging.LogRecord]:
    if base is logging.LogRecord:
        return _RedactingLogRecord
    cached = _redacting_classes.get(base)
    if cached is None:
        # A custom record factory installed earlier keeps its own record class underneath.
        def get_message(self: logging.LogRecord) -> str:
            return _redacted_message(self, base)

        cached = type(f"Redacting{base.__name__}", (base,), {"getMessage": get_message})
        _redacting_classes[base] = cached
    return cached


def _redact_at_creation(record: logging.LogRecord, registry: SecretRegistry) -> None:
    if not registry._snapshot():
        return
    if isinstance(record.msg, str):
        record.msg = redact(record.msg, registry)
    if isinstance(record.args, tuple):
        record.args = tuple(
            redact(arg, registry) if isinstance(arg, str) else arg for arg in record.args
        )
    if record.exc_info:
        record.exc_text = redact(_FORMATTER.formatException(record.exc_info), registry)
        record.exc_info = None
    elif record.exc_text:
        record.exc_text = redact(record.exc_text, registry)
    if record.stack_info:
        record.stack_info = redact(record.stack_info, registry)


def _make_error_report_safe(record: logging.LogRecord) -> None:
    registry = _active_registry or SecretRegistry()
    if isinstance(record.msg, str):
        record.msg = redact(record.msg, registry)
    else:
        record.msg = f"<unformattable {type(record.msg).__name__}>"
    if isinstance(record.args, tuple):
        record.args = tuple(_safe_argument(arg, registry) for arg in record.args)
    elif record.args:
        record.args = (f"<{type(record.args).__name__}>",)


def _safe_argument(arg: object, registry: SecretRegistry) -> object:
    """Return `arg` in a form whose repr cannot raise and cannot show a secret."""
    if isinstance(arg, str):
        return redact(arg, registry)
    if arg is None or isinstance(arg, bool | int | float):
        return arg
    return f"<{type(arg).__name__}>"


def _strip_one_newline(content: str) -> str:
    if content.endswith("\r\n"):
        return content[:-2]
    if content.endswith("\n"):
        return content[:-1]
    return content
