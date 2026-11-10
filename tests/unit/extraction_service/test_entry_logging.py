"""The entry point sets up logging so the engine's INFO call line reaches `docker logs`.

uvicorn's default logging configures its own loggers only, which leaves the
root logger at WARNING with no handler: a WARNING line such as the length
stop reaches stderr through logging's last-resort handler, an INFO line does
not. `configure_logging()` gives the root logger one stderr handler and lets
the engine logger alone through at INFO, so `app.py`'s INFO lines (job
outcomes that carry error text) stay off.

`capfd` reads what reaches file descriptor 2, which is what `docker logs`
shows; `caplog` alone would pass without the setup, because pytest attaches
its own capturing handler. Each test restores the root logger's handlers and
the loggers' levels, so no later test captures through a stale handler.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from backend.extraction_service import __main__ as entry

_ENGINE = "backend.extraction_service.engine"
_APP = "backend.extraction_service.app"
_UVICORN = ("uvicorn", "uvicorn.error", "uvicorn.access")


@pytest.fixture
def restored_logging():
    """Put the root logger's handlers and every touched logger's level back."""
    root = logging.getLogger()
    root_handlers = root.handlers[:]
    names = (_ENGINE, _APP, *_UVICORN)
    saved = {
        name: (
            logging.getLogger(name).level,
            logging.getLogger(name).handlers[:],
            logging.getLogger(name).propagate,
        )
        for name in names
    }
    root_level = root.level
    yield
    root.handlers[:] = root_handlers
    root.setLevel(root_level)
    for name, (level, handlers, propagate) in saved.items():
        logger = logging.getLogger(name)
        logger.setLevel(level)
        logger.handlers[:] = handlers
        logger.propagate = propagate


def _added_handlers(before: list[logging.Handler]) -> list[logging.Handler]:
    return [h for h in logging.getLogger().handlers if h not in before]


class TestConfigureLogging:
    def test_the_engine_info_line_reaches_stderr_and_the_app_info_line_does_not(
        self, capfd, restored_logging
    ):
        capfd.readouterr()

        entry.configure_logging()
        logging.getLogger(_ENGINE).info("extraction call: cap %d finish %s", 2048, "stop")
        logging.getLogger(_APP).info("job complete: %s", "app-info-marker")

        err = capfd.readouterr().err
        assert "extraction call: cap 2048 finish stop" in err
        assert "app-info-marker" not in err

    def test_a_warning_from_any_logger_still_reaches_stderr(self, capfd, restored_logging):
        capfd.readouterr()

        entry.configure_logging()
        logging.getLogger(_APP).warning("job failed: %s", "app-warning-marker")

        assert "app-warning-marker" in capfd.readouterr().err

    def test_the_levels_and_the_one_handler(self, restored_logging):
        before = logging.getLogger().handlers[:]

        entry.configure_logging()

        root = logging.getLogger()
        assert root.level == logging.WARNING
        assert logging.getLogger(_ENGINE).level == logging.INFO
        assert logging.getLogger(_ENGINE).isEnabledFor(logging.INFO)
        assert not logging.getLogger(_APP).isEnabledFor(logging.INFO)
        (handler,) = _added_handlers(before)
        assert isinstance(handler, logging.StreamHandler)
        # No level of its own, so the engine's INFO record is not dropped here.
        assert handler.level == logging.NOTSET

    def test_a_second_call_adds_no_second_handler(self, restored_logging):
        before = logging.getLogger().handlers[:]

        entry.configure_logging()
        entry.configure_logging()

        assert len(_added_handlers(before)) == 1

    def test_uvicorns_own_logging_setup_leaves_the_line_reaching_stderr(
        self, capfd, restored_logging
    ):
        """Uvicorn applies its logging config when it builds its Config, after ours."""
        import uvicorn

        capfd.readouterr()

        entry.configure_logging()
        uvicorn.Config(app=object())
        logging.getLogger(_ENGINE).info("extraction call: cap %d finish %s", 400, "stop")

        assert "extraction call: cap 400 finish stop" in capfd.readouterr().err


class TestMain:
    def test_main_configures_logging_before_uvicorn_runs(self, monkeypatch, restored_logging):
        seen: list[tuple[str, object]] = []

        def fake_run(app, **kwargs):
            seen.append(("run", logging.getLogger(_ENGINE).isEnabledFor(logging.INFO)))

        settings = SimpleNamespace(from_env=lambda: SimpleNamespace(port=8090))
        monkeypatch.setattr(entry, "ServiceSettings", settings)
        monkeypatch.setattr(entry, "build_app_from_env", lambda: "app")
        monkeypatch.setattr(entry.uvicorn, "run", fake_run)
        logging.getLogger(_ENGINE).setLevel(logging.NOTSET)
        logging.getLogger().setLevel(logging.WARNING)

        entry.main()

        assert seen == [("run", True)]
