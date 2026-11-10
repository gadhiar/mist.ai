"""Standalone entry point: `python -m backend.extraction_service`.

Binds uvicorn to `EXTRACTION_PORT` (default 8090) and serves the app
`build_app_from_env()` wires from `EXTRACTION_*` env vars.
"""

from __future__ import annotations

import logging
import sys

import uvicorn

from backend.extraction_service.app import build_app_from_env
from backend.extraction_service.settings import ServiceSettings

_HANDLER_NAME = "mist-extraction-stderr"
_ENGINE_LOGGER = "backend.extraction_service.engine"


def configure_logging() -> None:
    """Send WARNING from every logger, and the engine's INFO, to stderr.

    uvicorn's default logging configures its own loggers only, which leaves
    the root logger at WARNING with no handler, so the engine's INFO
    `extraction call:` line never reached `docker logs`. The root logger
    stays at WARNING with one stderr handler that has no level of its own,
    and only the engine logger is opened to INFO: the engine logs nothing
    else at INFO, while `app.py`'s INFO lines (job outcomes with error text)
    stay at the root's level and off. Calling it twice adds no second
    handler. uvicorn's own logging setup, applied after this, keeps the
    handler and the levels, since it names neither the root nor the engine
    logger.
    """
    root = logging.getLogger()
    root.setLevel(logging.WARNING)
    if not any(handler.get_name() == _HANDLER_NAME for handler in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.set_name(_HANDLER_NAME)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
    logging.getLogger(_ENGINE_LOGGER).setLevel(logging.INFO)


def main() -> None:
    configure_logging()
    settings = ServiceSettings.from_env()
    app = build_app_from_env()
    # All interfaces inside the container: the backend reaches the service over
    # the compose network on the workstation (docker-compose.extraction.yml,
    # `extraction-local` profile) or, on the remote host, through the Tailscale
    # sidecar's shared network namespace (docker/extraction/compose.host.yml,
    # `network_mode: service:mist-extraction-ts`). Exposure is decided by
    # those compose files: the host file publishes no host port, and the local
    # one publishes 127.0.0.1 only.
    uvicorn.run(app, host="0.0.0.0", port=settings.port)  # nosec B104


if __name__ == "__main__":
    main()
