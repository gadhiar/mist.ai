"""Standalone entry point: `python -m backend.extraction_service`.

Binds uvicorn to `EXTRACTION_PORT` (default 8090) and serves the app
`build_app_from_env()` wires from `EXTRACTION_*` env vars.
"""

from __future__ import annotations

import logging
import sys

import uvicorn

from backend.extraction_service.app import build_app_from_env
from backend.extraction_service.engine import LIST_OUTPUTS_LOGGER
from backend.extraction_service.settings import ServiceSettings


def _configure_list_output_log(settings: ServiceSettings) -> None:
    """Give the list output log a stderr handler and the INFO level, only when it is on.

    Under uvicorn's default logging configuration the root logger has no handler and
    the effective level is WARNING, so an INFO record would be dropped. The logger
    does not propagate, so its lines reach stderr once. uvicorn applies its own
    configuration later, in `uvicorn.run`, with `disable_existing_loggers` false, so
    both settings survive it.
    """
    if not settings.log_list_outputs:
        return
    log = logging.getLogger(LIST_OUTPUTS_LOGGER)
    log.setLevel(logging.INFO)
    log.addHandler(logging.StreamHandler(sys.stderr))
    log.propagate = False


def main() -> None:
    settings = ServiceSettings.from_env()
    _configure_list_output_log(settings)
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
